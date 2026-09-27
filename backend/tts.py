"""TTS adapters. Three engines behind one interface, chosen by TTS_ENGINE.

`fish` (FishTts) is the default: a client for the self-hosted Fish Speech
S2-Pro server - local GPU, cloned Tamil voice, audio streamed as it decodes.
It is the only engine with stream(); main.py uses that when it exists and
falls back to whole-clause synthesize() otherwise. See its class docstring.

`edge` (EdgeTts) is the fallback: Microsoft's neural Tamil voices, no
API key, no model download, and it runs on CPU - which is what makes a voice
demo possible on this hardware at all. ta-IN-PallaviNeural is female.

PRIVACY - read before pointing this at real callers. edge sends the text to be
spoken to a Microsoft endpoint. That is fine for local testing with the
fictional Aruvi data in golden/, and NOT fine for real patient speech: this is
a hospital agent and the text is PHI-adjacent (see BACKEND_COMPLETION.md Sec4,
which already flags exactly this gap). For deployment use a self-hosted engine
- AI4Bharat Indic Parler-TTS is the recommended first try (SETUP.md) - behind
the same interface, and the rest of the pipeline does not change.

`svara` (SvaraTts) is the original placeholder, kept because it is what
BACKEND_COMPLETION.md Sec3.2 specifies; its load() still raises until a real
svara-TTS reference exists.

Both keep asr.py's load()/ready shape - loaded once at startup (see main.py's
lifespan) so per-call synthesis never pays model-load latency - and both return
mono int16 PCM at the engine's native rate. Resampling stays out of these
adapters so one code path serves a 16 kHz browser socket today and an 8 kHz SIP
leg later without knowing which it is talking to.

Original module docstring follows.

svara-TTS streaming adapter (voice-cloned, sentence-chunked synthesis).

Mirrors asr.py's load()/ready shape: the model and cloned-voice reference are
meant to be loaded once at startup (see main.py's lifespan) so per-call
synthesis never pays model-load latency. Output stays at the model's native
sample rate - BACKEND_COMPLETION.md Sec3.2 keeps resampling out of this
adapter so the same code path can serve a 16 kHz browser socket today and an
8 kHz SIP leg later without knowing which transport it's talking to.

load() is a placeholder: no real svara-TTS package/API reference exists yet.
synthesize()'s call into self._model assumes a plausible shape (text in,
float32 waveform in [-1, 1] out, at whatever rate self._model reports) so the
rest of this class - and main.py's wiring - is real and testable now; that
assumed shape is the only thing likely to need adjusting once the real
package lands.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import io
import logging
import re

import numpy as np

from .settings import TtsSettings

logger = logging.getLogger("aica.tts")


@dataclass(frozen=True)
class SynthesisResult:
    samples: np.ndarray  # mono int16 PCM at `sample_rate`
    sample_rate: int


# One retry, briefly delayed. Enough to ride out a connect timeout without
# adding a perceptible stall to a turn that was going to fail anyway.
EDGE_ATTEMPTS = 2
EDGE_RETRY_DELAY_SECONDS = 0.4

# Repeated clause text is the common case here, not a rare one worth
# optimising for: EVERY call opens with the same four greeting clauses and
# every silent tool turn says the same holding line. Caching the fetched MP3
# makes those instant (measured 0.89s -> 0s for the greeting) and, more
# importantly, makes them survive the endpoint being unreachable - which is
# the failure that otherwise leaves a caller with no voice at all.
#
# ponytail: in-memory and unbounded-LRU-free - a plain dict with a cap, since
# a call's vocabulary of repeated lines is tiny. Persist it to disk if the
# greeting must also survive a cold start on a dead link.
EDGE_CACHE_MAX_ENTRIES = 64

# Written Tamil-English code-mix glues an English word to its Tamil case
# suffix with a hyphen - "Cardiology-ல", "department-க்கு", "bill-ல" - and the
# prompt teaches exactly that style, so it is in almost every reply.
#
# The Tamil neural voice SILENTLY DROPS the English half of such a token.
# Measured, voiced-audio duration (total duration is useless here, edge pads
# every short clip to ~1.78s):
#
#     department          0.68s   spoken
#     department-க்கு      0.30s   the English word is GONE, only "க்கு" is said
#     department க்கு      0.80s   spoken
#     Cardiology          0.82s   spoken
#     Cardiology-ல         0.28s   GONE
#     Cardiology ல         1.10s   spoken
#
# So the hyphen, not the script mixing, is what breaks it. Replacing just that
# one hyphen with a space restores the word. This is a SPEECH-ONLY transform:
# the transcript, the call log and the model's own history keep the hyphen,
# which is how the language is actually written.
#
# Latin-to-Latin hyphens ("pre-auth", "co-pay", "follow-up") are untouched -
# the pattern requires a Tamil character on the right-hand side.
_LATIN_TAMIL_HYPHEN_RE = re.compile(r"([A-Za-z0-9])-(?=[\u0b80-\u0bff])")


def speakable(text: str) -> str:
    """Rewrite one clause into what the voice can actually pronounce."""
    return _LATIN_TAMIL_HYPHEN_RE.sub(r"\1 ", text)


def trim_padding(
    samples: np.ndarray,
    sample_rate: int,
    lead_seconds: float,
    pause_seconds: float,
    threshold: float,
) -> np.ndarray:
    """Cut edge's silent padding down to one deliberate inter-clause pause.

    Edge returns every clip padded, and a SHORT clip is padded hardest -
    measured, a clause is padded out to exactly 1.78s however brief it is, so
    "சரி சார்." arrives as 0.72s of speech behind 0.16s of silence
    and in front of 0.90s more. A turn's clauses are synthesized separately and
    played back to back, so all of that padding lands at the clause boundaries
    and the caller hears a long gap after every full stop.

    Detects speech on 20ms frame peaks rather than a per-sample threshold: a
    single sample crossing it mid-silence would otherwise defeat the whole
    trim, and Tamil word-final vowels trail off quietly enough that a
    per-sample test on a soft frame reads as speech.

    Returns the input untouched if nothing crosses the threshold - a clause
    that is genuinely all silence is a TTS failure, and this is not the place
    to turn it into an empty array the sender would treat differently.
    """
    if samples.size == 0:
        return samples

    frame = max(1, int(sample_rate * 0.02))
    usable = samples.size // frame * frame
    if usable == 0:
        return samples

    peaks = np.abs(samples[:usable].astype(np.float32) / 32767.0).reshape(-1, frame).max(axis=1)
    voiced = np.flatnonzero(peaks > threshold)
    if voiced.size == 0:
        return samples

    start = max(0, int((voiced[0] * frame) - lead_seconds * sample_rate))
    end = min(samples.size, int((voiced[-1] + 1) * frame))
    pause = np.zeros(int(pause_seconds * sample_rate), dtype=samples.dtype)
    return np.concatenate([samples[start:end], pause])


# Microsoft's neural voices for the languages settings.py already accepts.
# Female by default: the golden transcripts are written around a woman at the
# desk (Gayathri, Deepa, Kavitha...), and conversation.OPENING_LINE names her.
_EDGE_VOICES: dict[str, str] = {
    "ta": "ta-IN-PallaviNeural",
    "hi": "hi-IN-SwaraNeural",
    "te": "te-IN-ShrutiNeural",
    "ml": "ml-IN-SobhanaNeural",
    "kn": "kn-IN-SapnaNeural",
    "bn": "bn-IN-TanishaaNeural",
    "mr": "mr-IN-AarohiNeural",
    "gu": "gu-IN-DhwaniNeural",
    "pa": "pa-IN-OjasNeural",
}


def _run_blocking(coro):
    """Run `coro` to completion from synchronous code, loop or no loop.

    synthesize() is a sync method with an async engine underneath. main.py
    reaches it through asyncio.to_thread, where the thread has no loop and
    asyncio.run() is fine - but any caller that is itself async (the transcript
    logger, a notebook, a future streaming path) would hit "asyncio.run()
    cannot be called from a running event loop". Handing the coroutine to a
    private loop on its own thread works in both cases.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class EdgeTts:
    """Microsoft neural TTS. No API key, no model download, runs on CPU.

    See this module's docstring for the privacy constraint before using it with
    anything but the fictional test data.
    """

    def __init__(self, settings: TtsSettings) -> None:
        self.settings = settings
        self._voice: str | None = None
        # text -> complete MP3 body. Only ever written from the success path
        # in _stream_mp3(), never from the partial-on-timeout one, so a
        # clipped clause can't be served for the rest of the process's life.
        self._mp3_cache: dict[str, bytes] = {}
        # edge streams MP3; libsndfile 1.2+ decodes it, and every clip so far
        # has come back at 24 kHz. Read from the decoder rather than assumed,
        # since main.py tells the client the rate before the first clause.
        self._sample_rate: int = 24_000

    @property
    def ready(self) -> bool:
        return self._voice is not None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def load(self) -> None:
        import edge_tts  # noqa: F401  - fail loudly at startup, not mid-call
        import soundfile

        if "MP3" not in soundfile.available_formats():
            raise RuntimeError(
                "libsndfile has no MP3 decoder (needs >= 1.1); "
                f"found {soundfile.__libsndfile_version__}. pip install -U soundfile"
            )

        voice = self.settings.voice or _EDGE_VOICES.get(self.settings.language)
        if voice is None:
            raise RuntimeError(f"no edge voice mapped for language {self.settings.language!r}")

        self._voice = voice
        logger.info("edge TTS ready: voice=%s rate=%s", voice, self.settings.rate)

    def synthesize(self, text: str, language: str) -> SynthesisResult:
        if self._voice is None:
            raise RuntimeError("TTS model is not loaded")

        text = text.strip()
        if not text:
            return SynthesisResult(np.array([], dtype=np.int16), self._sample_rate)

        mp3 = _run_blocking(self._stream_mp3(speakable(text)))
        if not mp3:
            logger.warning("edge TTS returned no audio for %r", text[:60])
            return SynthesisResult(np.array([], dtype=np.int16), self._sample_rate)

        import soundfile

        waveform, sample_rate = soundfile.read(io.BytesIO(mp3), dtype="float32", always_2d=True)
        self._sample_rate = int(sample_rate)
        mono = waveform.mean(axis=1)
        samples = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
        # After decode, not before: the MP3 cache stores the body edge sent, so
        # the same clip stays reusable if the pause is ever retuned.
        samples = trim_padding(
            samples,
            self._sample_rate,
            self.settings.clause_lead_seconds,
            self.settings.clause_pause_seconds,
            self.settings.silence_threshold,
        )
        return SynthesisResult(samples=samples, sample_rate=self._sample_rate)

    async def _stream_mp3(self, text: str) -> bytes:
        """Fetch one clause's audio, retrying a transient network failure.

        This engine is a network call to a Microsoft endpoint, so a single
        dropped connection is an ordinary event, not a fault - and losing it
        costs the caller the voice for that clause while the text still goes
        out, which reads as the agent going silent mid-sentence. One quick
        retry converts the common case (a timeout on connect) into a small
        delay. A second failure is reported to the caller of synthesize()
        rather than retried further: on a voice channel, late audio is worth
        less than a turn that moves on.
        """
        import edge_tts

        cached = self._mp3_cache.get(text)
        if cached is not None:
            return cached

        timeout = self.settings.timeout_seconds
        last_error: Exception | None = None
        for attempt in range(EDGE_ATTEMPTS):
            # Outside the try: a timeout must be able to keep whatever already
            # arrived. MP3 is a stream of independent frames, so a truncated
            # body still decodes to the part of the clause that got through -
            # a clipped word is worth more to a caller than silence, and the
            # old code threw all of it away.
            chunks = bytearray()
            try:
                async with asyncio.timeout(timeout):
                    stream = edge_tts.Communicate(text, self._voice, rate=self.settings.rate).stream()
                    async for chunk in stream:
                        if chunk["type"] == "audio":
                            chunks += chunk["data"]
                    body = bytes(chunks)
                    if body and len(self._mp3_cache) < EDGE_CACHE_MAX_ENTRIES:
                        self._mp3_cache[text] = body
                    return body
            except TimeoutError as error:
                # Deliberately NOT retried. A timeout means the endpoint is
                # stalling rather than refusing, and the second attempt stalls
                # the same way: measured 10s + 10s = 21s on one clause, during
                # which the whole turn's remaining audio sat behind it.
                if chunks:
                    logger.warning(
                        "edge TTS timed out after %.1fs for %r - keeping the %d bytes that arrived",
                        timeout, text[:40], len(chunks),
                    )
                    return bytes(chunks)
                logger.warning("edge TTS timed out after %.1fs for %r", timeout, text[:40])
                raise RuntimeError(f"edge TTS timed out after {timeout:.1f}s") from error
            except Exception as error:  # aiohttp raises a family of these
                last_error = error
                logger.warning(
                    "edge TTS attempt %d/%d failed for %r: %s",
                    attempt + 1,
                    EDGE_ATTEMPTS,
                    text[:40],
                    error,
                )
                if attempt + 1 < EDGE_ATTEMPTS:
                    await asyncio.sleep(EDGE_RETRY_DELAY_SECONDS)

        raise RuntimeError(f"edge TTS failed after {EDGE_ATTEMPTS} attempts: {last_error}")


class FishTts:
    """Client for the self-hosted Fish Speech S2-Pro streaming server.

    The server is its own process (its own GPU venv, ~4.9 GB checkpoint, see
    SETUP.md) and speaks MessagePack over /v1/tts/live. Everything about the
    VOICE lives there: POST /v1/voice once and every session speaks in it, so
    this client never sends a reference clip.

    Why it beats edge on this pipeline:
      - first audio ~70-100 ms after a clause is sent (RTX 2050, measured by
        the server's own probe) against edge's 0.9-1.5 s per clause;
      - audio STREAMS: stream() yields PCM as each frame is decoded, and
        main.py forwards it the moment it arrives instead of waiting for the
        whole clause;
      - nothing leaves the machine, which closes the PHI gap edge leaves open;
      - barge-in is felt within one decode step: closing the socket cancels
        the in-flight generation on the server.

    One session per clause, deliberately, rather than one per turn fed raw LLM
    deltas: conversation.py rewrites and suppresses whole clauses (repeat
    breaker, grounding) before they may be spoken, so only a finished clause
    is ever safe to send. The server runs one generation at a time in arrival
    order, so clauses queued ahead by main.py's lookahead still come out in
    order, each one prefilled while the previous one plays.
    """

    streaming = True

    def __init__(self, settings: TtsSettings) -> None:
        self.settings = settings
        base = settings.fish_url.rstrip("/")
        self._health_url = f"{base}/health"
        self._ws_url = base.replace("http", "ws", 1) + "/v1/tts/live"  # https -> wss too
        self._sample_rate: int | None = None
        # Same reasoning as edge's MP3 cache: the greeting, the recovery lines
        # and fillers are the same text on every call, and a cached clause
        # plays with zero GPU time. Only complete clauses are stored.
        self._pcm_cache: dict[str, bytes] = {}

    @property
    def ready(self) -> bool:
        return self._sample_rate is not None

    @property
    def sample_rate(self) -> int:
        if self._sample_rate is None:
            raise RuntimeError("TTS model is not loaded")
        return self._sample_rate

    def load(self) -> None:
        import json
        import time
        import urllib.error
        import urllib.request

        import ormsgpack  # noqa: F401  - fail loudly at startup, not mid-call
        import websockets  # noqa: F401

        # A server that is up but still loading answers {"status": "loading"}
        # and is worth waiting for; one that refuses the connection is not
        # running, and startup must not stall minutes on it.
        deadline = time.monotonic() + self.settings.fish_startup_wait_seconds
        refused_deadline = time.monotonic() + 3
        while True:
            try:
                with urllib.request.urlopen(self._health_url, timeout=5) as response:
                    status = json.loads(response.read()).get("status")
            except urllib.error.HTTPError as error:
                raise RuntimeError(f"Fish TTS server failed to load its model: {error.read()[:200]!r}") from error
            except OSError as error:
                if time.monotonic() > refused_deadline:
                    raise RuntimeError(
                        f"Fish TTS server not reachable at {self._health_url} ({error}) - start it first"
                    ) from error
                status = None
            if status == "ok":
                break
            if time.monotonic() > deadline:
                raise RuntimeError(f"Fish TTS server still {status or 'unreachable'} after waiting")
            if status == "loading":
                logger.info("waiting for the Fish TTS server to finish loading its model")
            time.sleep(2)

        # One real clause end to end: proves the socket, the voice and the GPU
        # path, and is where the sample rate comes from.
        pcm, sample_rate = _run_blocking(self._collect("வணக்கம்.", self.settings.language))
        if not pcm:
            raise RuntimeError("Fish TTS server returned no audio for its warm-up clause")
        self._sample_rate = sample_rate
        logger.info("fish TTS ready: %s, %d Hz", self._ws_url, sample_rate)

    async def stream(self, text: str, language: str):
        """Yield one clause as int16 PCM byte chunks, as the server decodes them."""
        if self._sample_rate is None:
            raise RuntimeError("TTS model is not loaded")
        text = speakable(text.strip())
        if not text:
            return
        cached = self._pcm_cache.get(text)
        if cached is not None:
            yield cached
            return
        async for chunk in self._frames(text, language):
            yield chunk

    def synthesize(self, text: str, language: str) -> SynthesisResult:
        """Whole-clause synthesis, for the startup warmers (fixed lines, fillers)."""
        if self._sample_rate is None:
            raise RuntimeError("TTS model is not loaded")

        async def collect() -> bytes:
            return b"".join([chunk async for chunk in self.stream(text, language)])

        pcm = _run_blocking(collect())
        return SynthesisResult(np.frombuffer(pcm, dtype="<i2").copy(), self._sample_rate)

    async def _collect(self, text: str, language: str) -> tuple[bytes, int]:
        pcm = b"".join([chunk async for chunk in self._frames(text, language)])
        return pcm, self._sample_rate or 0

    async def _frames(self, text: str, language: str):
        import ormsgpack
        import websockets

        request: dict[str, object] = {"format": "pcm", "language": language}
        if self.settings.fish_seed:
            request["seed"] = int(self.settings.fish_seed)
        timeout = self.settings.fish_timeout_seconds
        pieces: list[bytes] = []
        # Leaving this block for ANY reason - including main.py cancelling the
        # task on barge-in - closes the socket, and the server cancels the
        # generation within one decode step.
        async with websockets.connect(self._ws_url, max_size=None, open_timeout=timeout) as ws:
            for frame in (
                {"event": "start", "request": request},
                {"event": "text", "text": text},
                {"event": "stop"},
            ):
                await ws.send(ormsgpack.packb(frame))
            while True:
                async with asyncio.timeout(timeout):
                    message = ormsgpack.unpackb(await ws.recv())
                event = message.get("event")
                if event == "started":
                    self._sample_rate = int(message["sample_rate"])
                elif event == "audio":
                    chunk = message.get("audio") or b""
                    if chunk:
                        pieces.append(chunk)
                        yield chunk
                elif event == "finish":
                    if message.get("reason") != "stop":
                        raise RuntimeError(f"Fish TTS finished with {message.get('reason')}: {message.get('error')}")
                    break
        if pieces and len(self._pcm_cache) < EDGE_CACHE_MAX_ENTRIES:
            self._pcm_cache[text] = b"".join(pieces)


class SvaraTts:
    """Loads the svara-TTS model + cloned-voice reference once and synthesizes text."""

    def __init__(self, settings: TtsSettings) -> None:
        self.settings = settings
        self._model = None
        self._sample_rate: int | None = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def sample_rate(self) -> int:
        if self._sample_rate is None:
            raise RuntimeError("TTS model is not loaded")
        return self._sample_rate

    def load(self) -> None:
        """Load model weights and the cloned-voice reference during startup.

        PLACEHOLDER: replace this method's body with the real svara-TTS model
        + voice-reference load once that reference is available. Everything
        else in this class - ready gating, synthesize()'s output shape,
        main.py's wiring - is written against the interface below and should
        not need to change when this does.
        """
        raise NotImplementedError(
            "SvaraTts.load() is a placeholder - wire in the real svara-TTS model "
            "and cloned-voice reference here (see backend/tts.py)."
        )

    def synthesize(self, text: str, language: str) -> SynthesisResult:
        """Return mono int16 PCM for one clause, at the model's native rate."""
        if self._model is None:
            raise RuntimeError("TTS model is not loaded")

        text = text.strip()
        if not text:
            return SynthesisResult(samples=np.array([], dtype=np.int16), sample_rate=self.sample_rate)

        waveform = np.asarray(
            self._model.synthesize(
                text, language=language, voice_reference_path=self.settings.voice_reference_path
            ),
            dtype=np.float32,
        )
        samples = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
        return SynthesisResult(samples=samples, sample_rate=self.sample_rate)


def create_tts(settings: TtsSettings):
    """Return the adapter named by TTS_ENGINE.

    main.py holds this as `SvaraTts` for typing purposes only - both adapters
    expose the same load()/ready/sample_rate/synthesize surface, which is the
    whole point of selecting between them here rather than at the call site.
    """
    if settings.engine == "fish":
        return FishTts(settings)
    if settings.engine == "edge":
        return EdgeTts(settings)
    if settings.engine == "svara":
        return SvaraTts(settings)
    raise ValueError(f"unknown TTS_ENGINE: {settings.engine!r} (expected 'fish', 'edge' or 'svara')")
