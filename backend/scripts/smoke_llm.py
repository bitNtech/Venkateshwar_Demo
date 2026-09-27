"""Manual smoke test: one real start_call() + handle_utterance() round trip
against whatever LLM_BASE_URL is actually configured to.

Not a pytest test - conversation.py/llm.py are already covered by
test_conversation.py/test_llm.py against a fake client. This script exists to
visually confirm a *real* OpenAI-compatible server (vLLM/TGI/etc.) accepts the
tool schemas in tools.py and produces a sensible reply, before building
anything on top of it. It imports and uses ConversationManager/LlmClient
exactly as they exist today - no wrapping of their internals, only a thin
logging shim around LlmClient.complete() so every request/response in the
tool-call loop is visible.

Usage:
    python -m backend.scripts.smoke_llm ["caller utterance in Tamil/English"]

Requires LLM_BASE_URL (default http://localhost:8001/v1) to point at a
running OpenAI-compatible chat-completions server.
"""

from __future__ import annotations

import asyncio
import sys

# Windows consoles default to cp1252, which can't print Tamil script - force
# UTF-8 so this doesn't crash mid-run on the exact platform it's meant for.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.conversation import ConversationManager
from backend.llm import LlmClient, LlmReply, ReplyComplete
from backend.settings import ConversationSettings, LlmSettings

DEFAULT_UTTERANCE = "வணக்கம் மேடம். Cardiology-ல ஒரு appointment book பண்ணணும்."

# Caller metadata a real call would carry in from telephony/CRM context
# (SIP headers, a CRM lookup by caller ID, or a campaign dial list).
SAMPLE_METADATA = {
    "agent_name": "Gayathri",
    "caller_mobile": "9840721534",
    "mrn": "ARV-118342",
    "caller_name": "Murugesan",
    "campaign": "",
    "last_visit": "",
    "patient_name": "Murugesan",
}


class _LoggingLlmClient:
    """Wraps a real LlmClient so every request/reply in the tool-call loop is printed."""

    def __init__(self, inner: LlmClient) -> None:
        self._inner = inner

    async def stream(self, messages: list[dict], tools: list[dict]):
        """Proxy the streaming interface conversation.py actually consumes.

        This shim has to mirror LlmClient's whole surface, not just complete():
        stream_utterance() calls stream(), so a shim that only wrapped
        complete() silently turned every replay into an AttributeError and
        every flow into a FAIL that said nothing about the model.
        """
        print(f"\n>>> LLM request: {len(messages)} messages, {len(tools)} tool schemas")
        for message in messages:
            _print_message(message)

        async for event in self._inner.stream(messages, tools):
            if isinstance(event, ReplyComplete):
                _print_reply(event.reply)
            yield event

    async def complete(self, messages: list[dict], tools: list[dict]) -> LlmReply:
        print(f"\n>>> LLM request: {len(messages)} messages, {len(tools)} tool schemas")
        for message in messages:
            _print_message(message)

        reply = await self._inner.complete(messages, tools)
        _print_reply(reply)
        return reply


def _print_reply(reply: LlmReply) -> None:
    print("\n<<< LLM reply")
    print(f"    content: {reply.content!r}")
    if reply.tool_calls:
        for call in reply.tool_calls:
            print(f"    tool_call: {call.name}({call.arguments}) [id={call.id}]")
    else:
        print("    tool_call: (none)")


def _print_message(message: dict) -> None:
    role = message.get("role")
    content = message.get("content")
    if isinstance(content, str) and len(content) > 200:
        content = content[:200] + f"... [{len(content)} chars total]"
    print(f"    [{role}] {content!r}")
    for tool_call in message.get("tool_calls") or []:
        function = tool_call["function"]
        print(f"        tool_call -> {function['name']}({function['arguments']}) [id={tool_call['id']}]")
    if role == "tool":
        print(f"        tool_call_id: {message.get('tool_call_id')}")


async def main() -> None:
    utterance = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_UTTERANCE

    llm_settings = LlmSettings()
    print(f"LLM_BASE_URL={llm_settings.base_url}  LLM_MODEL={llm_settings.model}")

    llm = LlmClient(llm_settings)
    llm.load()
    logging_llm = _LoggingLlmClient(llm)

    manager = ConversationManager(ConversationSettings())
    manager.load()

    connection_id = "smoke-test"
    greeting = manager.start_call(connection_id, **SAMPLE_METADATA)
    print(f"\n=== start_call() greeting ===\n{greeting}")

    print(f"\n=== caller says ===\n{utterance}")

    try:
        reply = await manager.handle_utterance(connection_id, logging_llm, utterance)
    except Exception:
        print(
            "\nRound trip failed - is a real OpenAI-compatible server actually running at "
            f"{llm_settings.base_url}? (set LLM_BASE_URL / LLM_MODEL / LLM_API_KEY to point at one)"
        )
        raise

    print(f"\n=== final agent reply ===\n{reply}")


if __name__ == "__main__":
    asyncio.run(main())
