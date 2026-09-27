"""Self-check for LlmClient's stream -> reassembled-reply path.

Stubs the OpenAI-shaped async streaming client so chunk reassembly (content,
single and simultaneous tool-call deltas, empty streams) is exercised without
a real model or network call.
"""

from __future__ import annotations

from types import SimpleNamespace

from .llm import LlmClient
from .settings import LlmSettings


def _chunk(content: str | None = None, tool_calls: list | None = None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=tool_calls))])


def _tool_call_delta(index: int, *, id: str | None = None, name: str | None = None, arguments: str | None = None):
    function = SimpleNamespace(name=name, arguments=arguments) if (name is not None or arguments is not None) else None
    return SimpleNamespace(index=index, id=id, function=function)


class _FakeStream:
    def __init__(self, chunks: list) -> None:
        self._chunks = chunks

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for chunk in self._chunks:
            yield chunk


class _FakeCompletions:
    def __init__(self, chunks: list) -> None:
        self._chunks = chunks
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeStream(self._chunks)


class _FakeClient:
    def __init__(self, chunks: list) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(chunks))


def _make_llm(chunks: list) -> tuple[LlmClient, _FakeClient]:
    llm = LlmClient(LlmSettings())
    client = _FakeClient(chunks)
    llm._client = client
    return llm, client


async def test_content_only_stream_has_no_tool_calls() -> None:
    llm, client = _make_llm([_chunk(content="வண"), _chunk(content="க்கம்")])

    reply = await llm.complete([{"role": "user", "content": "hi"}], tools=[])

    assert reply.content == "வணக்கம்"
    assert reply.tool_calls == ()
    assert client.chat.completions.calls[0]["model"] == LlmSettings().model
    assert client.chat.completions.calls[0]["stream"] is True


async def test_single_tool_call_split_across_chunks_is_reassembled() -> None:
    llm, _ = _make_llm(
        [
            _chunk(tool_calls=[_tool_call_delta(0, id="call_1", name="lookupPatient", arguments='{"mob')]),
            _chunk(tool_calls=[_tool_call_delta(0, arguments='ile": "98')]),
            _chunk(tool_calls=[_tool_call_delta(0, arguments='40721534"}')]),
        ]
    )

    reply = await llm.complete([], tools=[])

    assert reply.content == ""
    assert len(reply.tool_calls) == 1
    call = reply.tool_calls[0]
    assert call.id == "call_1" and call.name == "lookupPatient"
    assert call.arguments == {"mobile": "9840721534"}


async def test_two_simultaneous_tool_calls_assemble_independently() -> None:
    llm, _ = _make_llm(
        [
            _chunk(
                tool_calls=[
                    _tool_call_delta(0, id="call_1", name="lookupPatient", arguments='{"mrn": "ARV-1"}'),
                    _tool_call_delta(1, id="call_2", name="searchSlots", arguments='{"department": "Cardiology"}'),
                ]
            )
        ]
    )

    reply = await llm.complete([], tools=[])

    assert [c.name for c in reply.tool_calls] == ["lookupPatient", "searchSlots"]
    assert reply.tool_calls[0].arguments == {"mrn": "ARV-1"}
    assert reply.tool_calls[1].arguments == {"department": "Cardiology"}


async def test_empty_stream_is_empty_reply() -> None:
    llm, _ = _make_llm([])

    reply = await llm.complete([], tools=[])

    assert reply.content == "" and reply.tool_calls == ()


async def test_unloaded_client_raises() -> None:
    llm = LlmClient(LlmSettings())
    try:
        await llm.complete([], tools=[])
    except RuntimeError:
        return
    raise AssertionError("complete() must refuse to run before load()")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_content_only_stream_has_no_tool_calls())
    asyncio.run(test_single_tool_call_split_across_chunks_is_reassembled())
    asyncio.run(test_two_simultaneous_tool_calls_assemble_independently())
    asyncio.run(test_empty_stream_is_empty_reply())
    asyncio.run(test_unloaded_client_raises())
    print("ok")
