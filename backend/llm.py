"""OpenAI-compatible streaming LLM adapter.

Wraps an OpenAI-compatible async client (works for both vLLM and TGI) rather
than hand-rolling a transport - see BACKEND_COMPLETION.md Sec3.1. This keeps the
model swappable via LlmSettings alone: llm.py and conversation.py never know
whether they are talking to Llama 3.1 8B, Qwen2.5-7B, or anything else.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging

from .settings import LlmSettings

logger = logging.getLogger("aica.llm")


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class LlmReply:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class TextDelta:
    """A fragment of assistant text, as it comes off the wire."""

    text: str


@dataclass(frozen=True)
class ReplyComplete:
    """The reassembled reply, yielded last by stream() once the wire closes."""

    reply: LlmReply


# What stream() yields: zero or more TextDelta, then exactly one ReplyComplete.
StreamEvent = TextDelta | ReplyComplete


class LlmClient:
    """Loads an OpenAI-compatible async client once and streams chat completions."""

    def __init__(self, settings: LlmSettings) -> None:
        self.settings = settings
        self._client = None

    @property
    def ready(self) -> bool:
        return self._client is not None

    async def reachable(self) -> bool:
        """Whether anything is answering at LLM_BASE_URL right now.

        `ready` only means the client is configured; the LLM is a separate
        service (vLLM) that may not be mounted yet. Any HTTP answer counts -
        only a failed connection means "not mounted". Checked per turn, so
        mounting the LLM later needs no restart here.
        """
        if self._client is None:
            return False
        import httpx

        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                await client.get(f"{self.settings.base_url.rstrip('/')}/models")
            return True
        except httpx.HTTPError:
            return False

    def load(self) -> None:
        """Construct the client during startup, keeping call-time latency low."""
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(base_url=self.settings.base_url, api_key=self.settings.api_key)
        logger.info("LLM client configured for %s at %s", self.settings.model, self.settings.base_url)

    async def complete(self, messages: list[dict], tools: list[dict] | None = None) -> LlmReply:
        """Run one chat completion and return the whole reassembled reply.

        Convenience wrapper over stream() for callers that have nothing useful
        to do with a half-finished reply - the eval scripts, and the tool-call
        iterations of conversation.py's loop, whose output is a tool call
        rather than something anybody speaks.
        """
        async for event in self.stream(messages, tools):
            if isinstance(event, ReplyComplete):
                return event.reply
        raise RuntimeError("LLM stream ended without a ReplyComplete event")

    async def stream(self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int | None = None):
        """Yield assistant text as it arrives, then the reassembled reply.

        This is what makes clause-level TTS possible (BACKEND_COMPLETION.md
        Sec3.2: "chunk at sentence/clause boundaries, not full LLM
        completions"). Waiting for the whole completion before synthesizing
        anything puts the entire generation time in front of the first sound
        the caller hears - on CPU inference that is tens of seconds of silence
        per turn. Yielding deltas lets conversation.py close the first clause
        and hand it to TTS while the model is still writing the rest.

        Yields zero or more TextDelta, then exactly one ReplyComplete. Callers
        that only want the finished reply should use complete() instead.

        `max_tokens` overrides the configured cap for one call. Its only user is
        ConversationManager.prewarm(), which wants the server to EVALUATE the
        prompt (filling its cache) and generate essentially nothing.
        """
        if self._client is None:
            raise RuntimeError("LLM client is not loaded")

        # Omitted entirely, not passed as [] - an empty tools array still
        # makes some servers emit the tool-calling scaffolding into the
        # prompt, which is the 1778 tokens this agent stopped paying for.
        optional = {"tools": tools, "tool_choice": "auto"} if tools else {}
        stream = await self._client.chat.completions.create(
            model=self.settings.model,
            messages=messages,
            **optional,
            temperature=self.settings.temperature,
            max_tokens=max_tokens or self.settings.max_tokens,
            stream=True,
        )

        content_parts: list[str] = []
        # Tool-call deltas are index-addressed and build up incrementally across
        # chunks: id/name usually arrive in the first delta for that index,
        # argument fragments (partial JSON) keep arriving after.
        pending_calls: dict[int, dict[str, str]] = {}

        async for chunk in stream:
            if not chunk.choices:
                # Some OpenAI-compatible servers emit a final usage-only chunk
                # with an empty choices list; indexing [0] on it would raise
                # mid-turn.
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
                yield TextDelta(delta.content)
            for tool_call_delta in delta.tool_calls or []:
                slot = pending_calls.setdefault(tool_call_delta.index, {"id": "", "name": "", "arguments": ""})
                if tool_call_delta.id:
                    slot["id"] = tool_call_delta.id
                function = tool_call_delta.function
                if function and function.name:
                    slot["name"] = function.name
                if function and function.arguments:
                    slot["arguments"] += function.arguments

        tool_calls = tuple(
            ToolCall(id=slot["id"], name=slot["name"], arguments=_parse_arguments(slot["arguments"]))
            for _, slot in sorted(pending_calls.items())
            if slot["name"]
        )
        yield ReplyComplete(LlmReply(content="".join(content_parts), tool_calls=tool_calls))


def _parse_arguments(raw: str) -> dict:
    """Parse a tool call's accumulated argument JSON, tolerating a bad one.

    A truncated or malformed argument blob is a model/serving fault, not a
    reason to end the call: execute_tool() already turns a bad call into an
    error result the model can react to (see tools.py), and that path only
    works if the call reaches it. Raising here instead would surface as
    "agent turn failed" and drop the caller's turn on the floor.
    """
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("tool call arguments were not valid JSON: %r", raw[:200])
        return {}
