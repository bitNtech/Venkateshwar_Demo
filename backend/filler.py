"""Short acknowledgements played while the agent's real reply is still coming.

clause_chunker.py's first clause is the only one whose latency is not hidden
behind audio already playing, and it is long: ASR, then the model's
time-to-first-clause (median 2.7 s, LLM_TEST_RESULTS.txt), then one TTS
round-trip (0.9-1.5 s). A caller hears all of it as dead air. A person on a
hospital desk fills that gap with "சரி…" or "ம்ம்…" - so this does too.

THE ONE UNMEASURED SYSTEM IN THIS PIPELINE. Everything else here was tuned on
recorded calls; this was not. FILLER_ENABLED=false (settings.FillerSettings)
removes it entirely without a code change.

What a filler may say is narrow on purpose:

  - Acknowledgement only. runtime_core.txt forbids narrating a lookup
    ("ஒரு நிமிஷம் சார், system-ல check பண்றேன்" promises something the agent
    cannot do), and "ஆமாம்" would pre-agree with whatever comes next. "சரி" and
    "ம்ம்" commit to nothing.
  - Tamil only. caller_is_speaking_english() was built, measured and left
    unwired - English replies made the agent worse - so fillers match the
    register the agent actually speaks.
  - சார்/மேடம் only once honorific.py is sure. Honorific-free is never wrong.
  - Short (0.4-1.0 s measured). A filler is cut off the moment real audio is
    ready, and a short one is usually finished by then.

Audio is synthesized ONCE per process and cached (FillerBank.warm): on the
measured 0.96-1.49 s synthesis time a filler made on demand would arrive after
the reply it was meant to cover. A phrase that is not cached yet is skipped,
never synthesized on the hot path.

What is spoken as a filler is NOT part of the agent's turn: it never reaches
recent_agent_text, the conversation history, or the repeat breaker. It is
tracked separately so its echo can be recognised - see
barge_in.is_probably_self_echo(filler_text=...).
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Awaitable, Callable

from .honorific import MADAM, SIR

logger = logging.getLogger("aica.filler")

# Fillers are Tamil; a call in any other language gets none.
FILLER_LANGUAGE = "ta"

# Measured with edge ta-IN-PallaviNeural after trim_padding, audio seconds:
#   சரி… 0.58   ம்ம்… 0.40   சரிங்க… 0.72   சரி சார்… 0.92   சரி மேடம்… 1.02
# "ம்ம், சரி சார்." (1.42 s) and longer were dropped: past ~1 s a filler starts
# to be cut mid-word by the reply it is covering.
PHRASES: dict[str | None, tuple[str, ...]] = {
    None: ("சரி…", "ம்ம்…", "சரிங்க…"),
    SIR: ("சரி சார்…", "ம்ம்…", "சரிங்க…"),
    MADAM: ("சரி மேடம்…", "ம்ம்…", "சரிங்க…"),
}


class FillerBank:
    """Process-wide cache of synthesized filler audio (int16 PCM bytes)."""

    def __init__(self) -> None:
        self._audio: dict[str, bytes] = {}

    def warm(self, synthesize: Callable[[str], bytes | None]) -> None:
        """Synthesize every phrase not already cached. Blocking - run in a thread."""
        for text in dict.fromkeys(p for phrases in PHRASES.values() for p in phrases):
            if text in self._audio:
                continue
            try:
                audio = synthesize(text)
            except Exception as error:
                # One missing phrase just narrows the rotation.
                logger.warning("filler %r failed to synthesize: %s", text, error)
                continue
            if audio:
                self._audio[text] = audio
        logger.info("filler bank ready: %d phrases cached", len(self._audio))

    def pick(self, honorific: str | None, avoid: str | None = None) -> tuple[str, bytes] | None:
        """A cached phrase for this honorific, never the same one twice running."""
        cached = [text for text in PHRASES.get(honorific, PHRASES[None]) if text in self._audio]
        fresh = [text for text in cached if text != avoid] or cached
        if not fresh:
            return None
        text = random.choice(fresh)
        return text, self._audio[text]


class TurnFiller:
    """At most one filler for one agent turn.

    Fires once `delay` seconds pass without real audio, and is stopped the
    moment real audio is ready. Kept free of the websocket for the same reason
    barge_in.ActiveSpeech is: `play` and `stop` are the only I/O, so the
    timing is unit-testable with plain coroutines.

      play()  send the filler; return its duration in seconds (0 = nothing sent)
      stop()  tell the client to drop the filler still playing
    """

    def __init__(
        self,
        delay: float,
        play: Callable[[], Awaitable[float]],
        stop: Callable[[], Awaitable[None]],
        clock=time.monotonic,
    ) -> None:
        self._delay = delay
        self._play = play
        self._stop = stop
        self._clock = clock
        self._timer: asyncio.Task | None = None
        self._playing = False
        self._audible_until = 0.0
        self.played = False

    def start(self) -> None:
        self._timer = asyncio.create_task(self._fire())

    async def _fire(self) -> None:
        await asyncio.sleep(self._delay)
        # Past this point the send is not cancelled halfway: a frame half
        # written to the socket is worse than a filler cut a moment later.
        self._playing = True
        seconds = await self._play()
        if seconds > 0:
            self.played = True
            self._audible_until = self._clock() + seconds

    async def real_audio_ready(self) -> None:
        """Real audio is about to go out: stop the timer, cut any filler still playing.

        Idempotent - safe to call before every audio frame of the turn.
        """
        timer, self._timer = self._timer, None
        if timer is not None:
            if not self._playing:
                timer.cancel()
            # wait(), not `await timer`: awaiting a cancelled task inside
            # suppress(CancelledError) would also swallow a barge-in cancelling
            # the task that is calling this.
            await asyncio.wait([timer])
        if self._clock() < self._audible_until:
            self._audible_until = 0.0
            await self._stop()

    def cancel(self) -> None:
        """Barge-in or a turn that produced no audio: just never fire.

        No stop() - on barge-in the client's agent_interrupted already drops
        everything scheduled, and a turn with no reply may as well let a
        half-second acknowledgement finish.

        Cancels even a send in progress: on barge-in, filler audio landing
        after agent_interrupted would play over the caller.
        """
        if self._timer is not None:
            self._timer.cancel()
        self._timer = None
