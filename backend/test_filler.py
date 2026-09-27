"""Self-check for filler timing, cancellation, and echo handling.

Same style as test_barge_in.py: plain coroutines stand in for the websocket,
so the timing rules are tested without a socket, VAD or TTS model.
"""

from __future__ import annotations

import asyncio
import contextlib

from .barge_in import is_probably_self_echo
from .filler import PHRASES, FillerBank, TurnFiller
from .honorific import MADAM, SIR


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _Client:
    """Records what TurnFiller asked the transport to do."""

    def __init__(self, seconds: float = 0.6) -> None:
        self.seconds = seconds
        self.played = 0
        self.stopped = 0

    async def play(self) -> float:
        self.played += 1
        return self.seconds

    async def stop(self) -> None:
        self.stopped += 1


async def test_filler_fires_once_the_threshold_passes_without_real_audio() -> None:
    client = _Client()
    filler = TurnFiller(0.01, client.play, client.stop)
    filler.start()

    await asyncio.sleep(0.05)

    assert client.played == 1
    assert filler.played


async def test_no_filler_when_real_audio_beats_the_threshold() -> None:
    client = _Client()
    filler = TurnFiller(0.2, client.play, client.stop)
    filler.start()

    await filler.real_audio_ready()
    await asyncio.sleep(0.25)

    assert client.played == 0
    assert client.stopped == 0


async def test_real_audio_cuts_a_filler_that_is_still_playing() -> None:
    clock = _Clock()
    client = _Client(seconds=1.0)
    filler = TurnFiller(0.0, client.play, client.stop, clock=clock)
    filler.start()
    await asyncio.sleep(0.01)

    clock.now = 0.3  # 300 ms into a 1 s filler
    await filler.real_audio_ready()
    # Called before every audio frame of the turn: must stop only once.
    await filler.real_audio_ready()

    assert client.played == 1
    assert client.stopped == 1


async def test_a_filler_that_already_finished_is_not_stopped() -> None:
    clock = _Clock()
    client = _Client(seconds=0.4)
    filler = TurnFiller(0.0, client.play, client.stop, clock=clock)
    filler.start()
    await asyncio.sleep(0.01)

    clock.now = 2.0
    await filler.real_audio_ready()

    assert client.stopped == 0


async def test_barge_in_before_the_threshold_means_no_filler_ever() -> None:
    client = _Client()
    filler = TurnFiller(0.02, client.play, client.stop)
    filler.start()

    filler.cancel()
    await asyncio.sleep(0.05)

    assert client.played == 0


async def test_cancelling_the_turn_is_not_swallowed_while_waiting_on_the_filler() -> None:
    """A barge-in cancelling speak() mid-real_audio_ready() must still cancel it."""
    release = asyncio.Event()

    async def slow_play() -> float:
        await release.wait()
        return 0.5

    async def stop() -> None:
        pass

    filler = TurnFiller(0.0, slow_play, stop)
    filler.start()
    await asyncio.sleep(0.01)  # timer is now mid-play

    turn = asyncio.create_task(filler.real_audio_ready())
    await asyncio.sleep(0.01)
    turn.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await turn

    assert turn.cancelled()
    release.set()


def test_the_bank_speaks_the_decided_honorific_and_never_repeats_itself() -> None:
    bank = FillerBank()
    bank.warm(lambda text: b"\x01\x00")

    for honorific in (None, SIR, MADAM):
        previous = None
        for _ in range(20):
            text, audio = bank.pick(honorific, previous)
            assert text in PHRASES[honorific]
            assert text != previous
            assert audio
            previous = text

    # Honorific-free unless decided: the wrong one is worse than none.
    assert not any(SIR in t or MADAM in t for t in PHRASES[None])
    assert not any(MADAM in t for t in PHRASES[SIR])
    assert not any(SIR in t for t in PHRASES[MADAM])


def test_an_uncached_phrase_is_skipped_never_synthesized_on_demand() -> None:
    bank = FillerBank()
    assert bank.pick(None) is None

    def flaky(text: str) -> bytes | None:
        if text == "சரி…":
            raise RuntimeError("edge endpoint blip")
        return None if text == "ம்ம்…" else b"\x01\x00"

    bank.warm(flaky)
    for _ in range(10):
        text, _ = bank.pick(None)
        assert text == "சரிங்க…"


def test_filler_echo_is_discarded_but_the_caller_is_still_heard() -> None:
    agent = "நன்றி சார். எந்த department-க்கு வேணும்?"
    filler = "சரி சார்…"

    # The filler coming back through the speakers, heard while it played.
    assert is_probably_self_echo("சரி சார்", agent, filler_text=filler)
    assert is_probably_self_echo("சரி", agent, filler_text=filler)

    # Anything of the caller's own is still a caller turn.
    assert not is_probably_self_echo("சரி, Cardiology வேணும்", agent, filler_text=filler)
    assert not is_probably_self_echo("ஆமாம்", agent, filler_text=filler)

    # Outside the filler's window main.py passes no filler_text, and the
    # existing short-turn rule stands: a caller saying "சரி சார்" is heard.
    assert not is_probably_self_echo("சரி சார்", agent)
    assert not is_probably_self_echo("சரி சார்", "")
