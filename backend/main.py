"""Low-latency browser audio WebSocket: PCM -> TEN VAD -> IndicConformer ASR."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import time
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from .asr import IndicConformerAsr
from .barge_in import AUDIBLE_TAIL_SECONDS, ActiveSpeech, is_probably_self_echo
from .clause_chunker import ClauseChunker
from .conversation import (
    AgentClause,
    AgentTurn,
    ConversationManager,
    split_reply_into_clauses as _split_reply_into_clauses,
)
from .filler import FILLER_LANGUAGE, FillerBank, TurnFiller
from .honorific import HonorificDetector, estimate_pitches
from .llm import LlmClient
from .persistence import CallEventStore
from .settings import (
    AudioSettings,
    ConversationSettings,
    FillerSettings,
    LlmSettings,
    PersistenceSettings,
    SecuritySettings,
    SUPPORTED_LANGUAGES,
    TtsSettings,
)
from .tasks import shutdown_worker
from .telephony import router as telephony_router
from .tts import SvaraTts, create_tts
from .vad import TenVadSegmenter, VadUpdate

# LOG_LEVEL=DEBUG surfaces the per-clause timing lines; WARNING quiets a
# production box. An unrecognised value falls back to INFO rather than
# refusing to start - a typo in a log knob must never cost a deployment.
#
# TIMESTAMPED, because the default format is not. Every latency question this
# project has asked was answered out of this log, and "TTS clause 37 chars:
# 0.00s synth" three lines under an LLM request tells you nothing about the
# gap between them without a clock. Millisecond resolution: the things being
# measured here are tenths of a second apart.
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format=os.getenv("LOG_FORMAT", "%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s"),
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aica.audio")

# BACKEND_COMPLETION.md Sec3.4: how often (in VAD hops) to run an interim CTC
# decode while an utterance is still in progress. 30 hops * 16ms/hop = ~480ms
# between partial_transcript updates - frequent enough to feel responsive,
# far below a rate that would make repeated CTC decodes a latency problem.
PARTIAL_TRANSCRIPT_INTERVAL_FRAMES = int(os.getenv("ASR_PARTIAL_INTERVAL_FRAMES", "30"))

# How many clauses may be mid-synthesis at once. The audible gap between the
# agent's sentences was one whole synthesis round-trip, because each clause was
# only synthesized after the previous one had been sent - and the default
# engine is a network call to Microsoft (measured 849ms to 10.8s per clause,
# see LLM_TEST_RESULTS.txt). Starting the next clause while the current one is
# still in flight hides all but the first of those behind audio the caller is
# already hearing. Real concurrency is still bounded by the TTS semaphore; this
# only decides how far ahead of playback we are willing to queue work.
SYNTH_LOOKAHEAD = int(os.getenv("TTS_SYNTH_LOOKAHEAD", "3"))

# Sent once per turn when TTS is unavailable. The React dashboard matches on
# this exact string to collapse the repeat into a single banner instead of one
# error pill per turn, so it is a wire constant - do not reword it casually.
TTS_UNAVAILABLE_MESSAGE = (
    "TTS is unavailable, so agent replies are text-only. Check TTS_ENGINE and the "
    "server log for the load() failure."
)


@dataclass
class ConversationTurnOutcome:
    """What one agent turn produced, once its speech has finished going out."""

    spoken: list[str] = field(default_factory=list)
    interrupted: bool = False


def _warm_tts_cache(tts) -> None:
    """Synthesize the fixed lines once so the first caller does not pay for them."""
    try:
        _warm(tts)
    except Exception:
        # Pure optimisation - a cold cache only costs latency, so nothing here
        # may stop the server coming up.
        logger.warning("TTS cache warm failed", exc_info=True)


def _warm(tts) -> None:
    from .conversation import (
        _CANNOT_RECALL,
        _GO_AHEAD,
        _STUCK_REPLIES,
        OPENING_LINE,
        render_template,
    )

    lines = [
        *_split_reply_into_clauses(render_template(OPENING_LINE, {"agent_name": "Gayathri"})),
        _GO_AHEAD,
        _CANNOT_RECALL,
        *_STUCK_REPLIES,
        # The openers the model reaches for on almost every call.
        "சரி சார்.",
        "கண்டிப்பா சார்.",
        "நன்றி சார்.",
        "புரியுது சார்.",
    ]
    warmed = 0
    for line in lines:
        try:
            tts.synthesize(line, tts.settings.language)
            warmed += 1
        except Exception:
            # Pure optimisation - a cold cache only costs latency, so a failure
            # here must never stop the server coming up.
            logger.warning("TTS cache warm failed for %r", line[:40], exc_info=True)
            break
    logger.info("TTS cache warmed with %d of %d fixed lines", warmed, len(lines))


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = AudioSettings()
    tts_settings = TtsSettings()
    asr = IndicConformerAsr(settings)
    conversation = ConversationManager(ConversationSettings())
    llm = LlmClient(LlmSettings())
    tts = create_tts(tts_settings)
    call_store = CallEventStore(PersistenceSettings())

    try:
        # Preload once so model startup never delays a live call.
        await asyncio.to_thread(asr.load)
    except Exception:
        logger.exception("ASR model failed to load")

    try:
        conversation.load()
    except Exception:
        logger.exception("Conversation Manager failed to load master prompt")

    try:
        llm.load()
    except Exception:
        logger.exception("LLM client failed to configure")

    filler_settings = FillerSettings()
    filler_bank = FillerBank()

    def warm_voice() -> None:
        # The lines the server itself can say verbatim - the scripted opening,
        # conversation.py's recovery lines, a few constant openers - and the
        # filler phrases, synthesized once so no live turn waits on them.
        _warm_tts_cache(tts)
        if filler_settings.enabled:
            def synthesize_filler(text: str) -> bytes | None:
                samples = tts.synthesize(text, FILLER_LANGUAGE).samples
                return samples.tobytes() if samples.size else None

            filler_bank.warm(synthesize_filler)

    async def attach_tts() -> None:
        """Load TTS now, or whenever its service comes up.

        TTS is a separate service (the Fish server) that may be mounted after
        this one starts, so a failed load is retried rather than final - and
        until it succeeds, speak() degrades to text-only agent_clause events,
        which is what the ready gate is for. Runs in the background so the
        rest of the pipeline (VAD, ASR, transcripts) is up immediately.
        """
        while True:
            try:
                await asyncio.to_thread(tts.load)
                break
            except Exception as error:
                if tts_settings.engine != "fish":
                    logger.exception("TTS engine %r failed to load", tts_settings.engine)
                    return
                logger.warning("TTS not mounted yet (%s); retrying in 15s", error)
                await asyncio.sleep(15)
        try:
            await asyncio.to_thread(warm_voice)
        except Exception:
            logger.warning("TTS warm-up failed", exc_info=True)

    app.state.tts_attach = asyncio.create_task(attach_tts())

    try:
        call_store.load()
    except Exception:
        logger.exception("Call-event store failed to initialize")

    app.state.settings = settings
    app.state.asr = asr
    # BACKEND_COMPLETION.md Sec3.5: bounded worker slots, not one global mutex
    # serializing every concurrent call's ASR/TTS work - see settings.py's
    # ASR_MAX_CONCURRENCY / TTS_MAX_CONCURRENCY.
    app.state.asr_semaphore = asyncio.Semaphore(settings.asr_max_concurrency)
    app.state.conversation = conversation
    app.state.llm = llm
    app.state.tts = tts
    app.state.tts_semaphore = asyncio.Semaphore(tts_settings.max_concurrency)
    app.state.call_store = call_store
    app.state.security = SecuritySettings()
    app.state.filler_settings = filler_settings
    app.state.filler_bank = filler_bank
    yield


app = FastAPI(title="AICA Audio Pipeline", lifespan=lifespan)

# Added at import time, not in lifespan(): Starlette builds its middleware stack
# on the first request, and a middleware appended after that is silently ignored.
# Read here rather than from app.state.security, which does not exist yet.
_cors_origins = SecuritySettings().cors_allow_origins
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(_cors_origins),
        allow_credentials="*" not in _cors_origins,  # the browser forbids both
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

# BACKEND_COMPLETION.md Sec3.3: telephony ingress is a separate router (its
# own file, its own tests) sharing this app's lifespan-loaded app.state.* -
# see backend/telephony.py's module docstring.
app.include_router(telephony_router)

_CONSOLE_HTML = Path(__file__).resolve().parent / "console.html"


@app.get("/console", response_class=HTMLResponse)
async def console() -> HTMLResponse:
    """Single-page test console for the whole pipeline.

    Served from the backend itself so it shares this app's origin and can open
    /ws/audio without CORS or a second dev server. It drives the same socket a
    real caller does; typing a turn uses the `user_text` path, which skips only
    VAD/ASR and exercises the identical conversation -> LLM -> tool -> TTS
    chain. That is what makes it useful when the gated ASR model is not
    installed - the part most likely to be missing on a fresh machine.
    """
    return HTMLResponse(_CONSOLE_HTML.read_text(encoding="utf-8"))


_LAB_HTML = Path(__file__).resolve().parent / "lab.html"


@app.get("/lab", response_class=HTMLResponse)
async def lab() -> HTMLResponse:
    """Minimal voice test page: speak, see what was heard and how fast.

    Same socket and protocol as /console, stripped to what a tester checks:
    live and final transcripts with their latency, fillers, and the சார்/மேடம்
    decision. For the noise-floor measurement use VAD_DIAGNOSTICS_MS and
    scripts/noise_floor_report.py instead.
    """
    return HTMLResponse(_LAB_HTML.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health() -> dict:
    """Which pieces of the pipeline actually came up.

    lifespan() logs each component's load failure and carries on, so a server
    that answers requests is not evidence that ASR, the LLM or TTS is usable.
    This is the one place to look to find out which - and it is what the
    console's status pills read before a socket is even opened.
    """
    llm: LlmClient = app.state.llm
    return {
        "asr_ready": app.state.asr.ready,
        "conversation_ready": app.state.conversation.ready,
        # Answering, not merely configured: the LLM is a separate service.
        "llm_ready": await llm.reachable(),
        "llm_model": llm.settings.model,
        "llm_base_url": llm.settings.base_url,
        "tts_ready": app.state.tts.ready,
        "tts_engine": type(app.state.tts).__name__,
    }


@app.get("/api/calls")
async def list_calls(limit: int = 50) -> dict:
    """Call Log: every call this server has handled, newest first.

    BACKEND_COMPLETION.md Sec3.6 - the call events were already being persisted
    but nothing could read them back, so a disconnect still lost the call as
    far as any client was concerned.
    """
    call_store: CallEventStore = app.state.call_store
    limit = max(1, min(limit, 500))
    calls = await asyncio.to_thread(call_store.recent_calls, limit)
    return {"calls": calls}


@app.get("/api/calls/{connection_id}")
async def get_call(connection_id: str) -> dict:
    """One call's full event history, oldest first - the transcript view."""
    call_store: CallEventStore = app.state.call_store
    events = await asyncio.to_thread(call_store.events_for_call, connection_id)
    return {"connection_id": connection_id, "events": events}


def _validate_start_event(payload: dict[str, object], settings: AudioSettings) -> str:
    if payload.get("audio_format") != "pcm_s16le":
        raise ValueError("audio_format must be pcm_s16le")
    if payload.get("sample_rate") != settings.sample_rate:
        raise ValueError(f"sample_rate must be {settings.sample_rate}")
    if payload.get("channels") != 1:
        raise ValueError("channels must be 1")

    language = str(payload.get("language", settings.language))
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported language: {language}")
    return language


@app.websocket("/ws/audio")
async def capture_browser_audio(websocket: WebSocket) -> None:
    """Process one browser call's raw 16 kHz PCM stream."""
    security: SecuritySettings = websocket.app.state.security
    if security.ws_auth_token:
        # Checked before accept(): a WS client (browser or otherwise) has no
        # way to set a custom header on the handshake, so the token travels
        # as a query param instead - see BACKEND_COMPLETION.md Sec4. Reject
        # before accept() rather than accept-then-close, so an unauthorized
        # caller never gets a "connected" event at all.
        token = websocket.query_params.get("token")
        if token != security.ws_auth_token:
            await websocket.close(code=4401)
            return

    await websocket.accept()
    connection_id = str(uuid4())
    settings: AudioSettings = websocket.app.state.settings
    asr: IndicConformerAsr = websocket.app.state.asr
    conversation: ConversationManager = websocket.app.state.conversation
    llm: LlmClient = websocket.app.state.llm
    tts: SvaraTts = websocket.app.state.tts
    call_store: CallEventStore = websocket.app.state.call_store
    send_lock = asyncio.Lock()
    language = settings.language
    started = False
    chunks_received = 0
    bytes_received = 0
    pcm_buffer = bytearray()
    partial_frame_count = 0
    # AudioSettings.vad_diagnostics_ms: one sample of the VAD's view of the
    # room per interval, persisted for scripts/noise_floor_report.py. The
    # accumulators cover exactly one interval and are reset after each sample.
    diagnostics_every = settings.vad_diagnostics_ms // 16 if settings.vad_diagnostics_ms > 0 else 0
    diag_frames = 0
    diag_rms_sum = 0.0
    diag_prob_max = 0.0
    diag_speech_hops = 0
    diag_refusals_seen = 0
    frames_seen = 0

    call_store.start_call(connection_id)

    # Persistence runs behind a queue, not inline in send_event. Writing each
    # event to SQLite before returning put a disk round-trip between every
    # clause and the synthesis of the next one, and made a slow or contended
    # write able to stall the agent mid-sentence - exactly what persistence.py's
    # "never take down a live call" contract rules out. The queue is unbounded
    # because dropping call history to keep audio flowing is the right trade in
    # the only direction that matters, and an event is a few hundred bytes.
    event_queue: asyncio.Queue[dict | None] = asyncio.Queue()

    async def persist_events() -> None:
        while True:
            payload = await event_queue.get()
            try:
                if payload is None:
                    return
                await asyncio.to_thread(call_store.record, connection_id, payload)
            except Exception:
                logger.exception("failed to persist call event for %s", connection_id)
            finally:
                event_queue.task_done()

    async def send_event(payload: dict[str, object]) -> None:
        async with send_lock:
            await websocket.send_json(payload)
        event_queue.put_nowait(payload)

    try:
        segmenter = TenVadSegmenter(settings)
    except Exception as error:
        logger.exception("TEN VAD failed to initialize")
        await send_event({"type": "pipeline_error", "stage": "vad", "message": str(error)})
        await websocket.close(code=1011)
        return

    # (samples, language, endpoint reason, onset time, agent audible at onset)
    transcription_queue: asyncio.Queue[tuple[np.ndarray, str, str, float, bool] | None] = asyncio.Queue()
    # ("greeting" | "utterance", text) - both go through one queue/worker so a
    # fast caller turn can never overtake the call's opening greeting; see
    # handle_conversation_turns().
    conversation_queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
    # Tracks whichever speak() task is currently synthesizing/sending audio, so
    # queue_segment() can cancel it the moment the caller barges in - see
    # BACKEND_COMPLETION.md Sec3.2 and backend/barge_in.py.
    active_speech = ActiveSpeech(
        settings.barge_in_speech_frames, settings.barge_in_frames_while_audible
    )
    # The last thing the agent said out loud, kept so a transcript that is
    # really the agent hearing itself can be recognised - see
    # barge_in.is_probably_self_echo(). Only the most recent turn matters:
    # echo arrives within seconds of being spoken.
    recent_agent_text = ""
    # Fillers (filler.py) and the caller's honorific (honorific.py). Read with
    # a default so a server whose lifespan never set them - the unit tests -
    # simply runs without either.
    filler_settings: FillerSettings = getattr(
        websocket.app.state, "filler_settings", FillerSettings(enabled=False, honorific_detection=False)
    )
    filler_bank: FillerBank | None = getattr(websocket.app.state, "filler_bank", None)
    honorific = HonorificDetector() if filler_settings.honorific_detection else None
    # The last filler played and when it stops being audible. Kept apart from
    # recent_agent_text on purpose - see is_probably_self_echo(filler_text=).
    last_filler_text = ""
    filler_audible_until = 0.0
    # When the utterance being captured began, and whether the agent could be
    # heard at that moment. Travels with the utterance to the transcriber.
    utterance_onset = 0.0
    utterance_onset_audible = False
    # The interim decode in flight, if any, and which utterance it belongs to.
    partial_task: asyncio.Task | None = None
    utterance_seq = 0

    async def synthesize_clause(clause: str, out: asyncio.Queue) -> None:
        """Synthesize one clause into `out` as PCM byte chunks, then None.

        Deliberately does NOT send anything. Sending is ordered and must follow
        the clause sequence exactly; synthesis is the slow part and is what
        speak() overlaps across clauses - see SYNTH_LOOKAHEAD.

        A streaming engine (tts.FishTts) puts each chunk the moment it is
        decoded, so the sender forwards the start of a clause while the rest is
        still being generated - first audio ~0.1s instead of a whole clause's
        synthesis. Other engines put the clause as one chunk. The None always
        comes, failure or not, so the sender can never wait forever.
        """
        try:
            if not tts.ready:
                return
            queued_at = time.perf_counter()
            async with websocket.app.state.tts_semaphore:
                waited = time.perf_counter() - queued_at
                started_at = time.perf_counter()
                if getattr(tts, "streaming", False):
                    first_audio = None
                    async for chunk in tts.stream(clause, language):
                        if first_audio is None:
                            first_audio = time.perf_counter() - started_at
                        out.put_nowait(chunk)
                    logger.info(
                        "TTS clause %d chars: first audio %.2fs, done %.2fs, %.2fs queued behind semaphore",
                        len(clause),
                        first_audio or 0.0,
                        time.perf_counter() - started_at,
                        waited,
                    )
                    return
                result = await asyncio.to_thread(tts.synthesize, clause, language)
                # The gap the caller hears between sentences is exactly this
                # number minus whatever audio was still playing. Logged per
                # clause because the engine's latency is wildly variable
                # (849ms-10.8s measured) and an average hides that.
                logger.info(
                    "TTS clause %d chars: %.2fs synth, %.2fs queued behind semaphore",
                    len(clause),
                    time.perf_counter() - started_at,
                    waited,
                )
                if result.samples.size:
                    out.put_nowait(result.samples.tobytes())
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # One clause losing its audio must not lose the rest of the turn.
            # A failure here is usually a blip: the text has already gone out,
            # so the caller sees the reply either way, whatever audio arrived
            # before the failure still plays, and the next clause gets its own
            # attempt rather than the whole turn being abandoned.
            logger.warning("TTS failed for a clause on %s: %s", connection_id, error)
            await send_event({"type": "agent_audio_error", "message": str(error)})
        finally:
            out.put_nowait(None)

    async def speak(clauses, with_filler: bool = False) -> ConversationTurnOutcome:
        """Synthesize and send an agent turn clause by clause as it is produced.

        `clauses` is an async iterator, not a finished string: the first clause
        goes out while the LLM is still generating the rest
        (BACKEND_COMPLETION.md Sec3.2), so the caller stops waiting on total
        generation time and starts waiting only on time-to-first-clause.

        Cancellable mid-clause for barge-in: a caller speaking over the agent
        cancels the task running this coroutine (see queue_segment()). There is
        no separate outbound audio queue to flush - this loop is what's in
        flight, so cancelling it stops both any further synthesis and, now,
        the LLM generation feeding it.

        `with_filler` plays a short acknowledgement if no real audio is ready
        within FILLER_THRESHOLD_MS - see filler.py. It lives inside this task,
        so a barge-in that cancels the turn cancels the filler with it.
        """
        nonlocal recent_agent_text
        outcome = ConversationTurnOutcome()
        started = False

        async def announce_start() -> None:
            nonlocal started
            if started:
                return
            # Deferred until there is something to say: announcing
            # "speaking" and then producing only a tool call would
            # leave the UI's speaking indicator stuck on.
            started = True
            await send_event(
                {"type": "agent_speaking_start", "sample_rate": tts.sample_rate if tts.ready else None}
            )
            if not tts.ready:
                await send_event({"type": "agent_error", "message": TTS_UNAVAILABLE_MESSAGE})

        async def play_filler() -> float:
            nonlocal last_filler_text, filler_audible_until
            if active_speech.audible:
                # Something is still playing; a filler would only queue behind it.
                return 0.0
            choice = filler_bank.pick(honorific.honorific if honorific else None, last_filler_text)
            if choice is None:
                return 0.0
            text, audio = choice
            await announce_start()
            # Its own event, never agent_clause: a filler is not part of the
            # reply, and the client tags the next audio frame as the filler so
            # filler_stop can cut exactly it.
            await send_event({"type": "agent_filler", "text": text})
            async with send_lock:
                await websocket.send_bytes(audio)
            seconds = len(audio) / 2 / tts.sample_rate
            active_speech.note_audio_sent(seconds)
            last_filler_text = text
            filler_audible_until = time.monotonic() + seconds + AUDIBLE_TAIL_SECONDS
            logger.info("filler %r played for %s", text, connection_id)
            return seconds

        async def stop_filler() -> None:
            nonlocal filler_audible_until
            # Only the filler has been sent this turn, so nothing is audible
            # once the client drops it; real audio re-arms this right after.
            active_speech.silence()
            filler_audible_until = time.monotonic() + AUDIBLE_TAIL_SECONDS
            await send_event({"type": "filler_stop"})

        filler: TurnFiller | None = None
        if (
            with_filler
            and filler_settings.enabled
            and filler_bank is not None
            and tts.ready
            and language == FILLER_LANGUAGE
        ):
            filler = TurnFiller(filler_settings.threshold_ms / 1000, play_filler, stop_filler)
            filler.start()
        # Synthesis tasks in clause order, handed to a sender that runs
        # INDEPENDENTLY of clause arrival. Draining them inline from the loop
        # below looks equivalent and is not: the loop is blocked awaiting the
        # LLM's next clause, so clause one's finished audio sat undelivered
        # until clause three showed up (measured - first audio 7.5s when the
        # audio had been ready since 5.9s). A separate consumer sends it the
        # moment it is ready. Order is preserved because the queue is FIFO and
        # each item is awaited in turn.
        # One chunk queue per clause, in clause order; see synthesize_clause().
        audio_queue: asyncio.Queue[asyncio.Queue | None] = asyncio.Queue()
        synth_tasks: list[asyncio.Task] = []

        async def send_audio_in_order() -> None:
            while True:
                chunks = await audio_queue.get()
                if chunks is None:
                    return
                blocked_at = time.perf_counter()
                first = True
                while (audio := await chunks.get()) is not None:
                    if first:
                        first = False
                        # How long the sender sat idle on synthesis already
                        # started - the silence the caller actually hears
                        # between sentences. Trends to zero for every clause
                        # after the first when the pipeline is keeping up.
                        logger.info(
                            "TTS send waited %.2fs for the next clause", time.perf_counter() - blocked_at
                        )
                    if filler is not None:
                        # Cut any filler still playing the instant real audio
                        # is ready - not when the clause's TEXT arrives, which
                        # is a whole synthesis earlier and would leave silence.
                        await filler.real_audio_ready()
                    async with send_lock:
                        await websocket.send_bytes(audio)
                    # int16 mono: 2 bytes per sample. Tracking how much audio
                    # has been handed over is how the server knows its own
                    # voice is still playing - agent_speaking_end fires when
                    # SENDING finishes, which is well before the caller stops
                    # hearing it.
                    if tts.ready and tts.sample_rate:
                        active_speech.note_audio_sent(len(audio) / 2 / tts.sample_rate)

        sender = asyncio.create_task(send_audio_in_order())

        try:
            async for item in clauses:
                if isinstance(item, AgentTurn):
                    if item.ungrounded:
                        # Reported, never silently swallowed: the caller has
                        # already heard these, so the only useful thing left is
                        # to make them visible in the console and the call log.
                        await send_event(
                            {"type": "grounding_warning", "identifiers": list(item.ungrounded)}
                        )
                    if item.unbacked_claims:
                        # A separate event rather than more identifiers: this
                        # one is about something the server did NOT do, which
                        # the console has to say differently.
                        await send_event(
                            {"type": "action_claim_warning", "claims": list(item.unbacked_claims)}
                        )
                    continue

                await announce_start()

                outcome.spoken.append(item.text)
                recent_agent_text = " ".join(outcome.spoken)
                # Text first, always: the transcript must not lag the audio,
                # and when TTS is unavailable this is the whole of the output.
                await send_event({"type": "agent_clause", "text": item.text})
                chunks: asyncio.Queue[bytes | None] = asyncio.Queue()
                synth_tasks.append(asyncio.create_task(synthesize_clause(item.text, chunks)))
                audio_queue.put_nowait(chunks)
                # Backpressure: don't run further than SYNTH_LOOKAHEAD clauses
                # ahead of what the caller is actually hearing.
                while sum(not t.done() for t in synth_tasks) > SYNTH_LOOKAHEAD:
                    await asyncio.sleep(0.05)
            audio_queue.put_nowait(None)
            await sender
            if filler is not None:
                # A turn that produced no audio at all must not fire late.
                filler.cancel()
        except asyncio.CancelledError:
            # Barge-in: drop the audio the caller talked over rather than
            # letting already-started synthesis land on top of their voice.
            if filler is not None:
                filler.cancel()
            sender.cancel()
            for task in synth_tasks:
                task.cancel()
            logger.info("agent speech interrupted by barge-in: %s", connection_id)
            outcome.interrupted = True
            with suppress(WebSocketDisconnect, RuntimeError):
                await send_event({"type": "agent_interrupted"})
            return outcome
        except Exception:
            # The turn failed (an unreachable LLM, say). A filler may already
            # have announced agent_speaking_start, and without an end the UI's
            # speaking indicator stays on; a filler still waiting must never
            # fire after the error. Measured live against a server with no LLM.
            if filler is not None:
                filler.cancel()
            # Let the sender finish the clauses whose text already went out,
            # then exit - left alone it waited on audio_queue forever and was
            # garbage-collected mid-await ("Task was destroyed but it is pending").
            audio_queue.put_nowait(None)
            if started:
                with suppress(WebSocketDisconnect, RuntimeError):
                    await send_event({"type": "agent_speaking_end"})
            raise

        if started:
            await send_event({"type": "agent_speaking_end"})
        return outcome

    async def greeting_clauses(text: str):
        """Adapt the scripted greeting to the same async shape a turn produces."""
        for clause in _split_reply_into_clauses(text):
            yield AgentClause(clause)

    async def handle_conversation_turns() -> None:
        while True:
            item = await conversation_queue.get()
            try:
                if item is None:
                    return
                kind, payload = item
                if kind == "utterance" and not await llm.reachable():
                    # The LLM is a separate service and is not mounted: the
                    # caller's transcript already went out, and there is no
                    # reply to give. Said once per turn, as a notice rather
                    # than an error - nothing is broken.
                    await send_event(
                        {"type": "agent_unavailable", "message": "LLM not mounted - transcript only, no reply."}
                    )
                    continue
                clauses = (
                    greeting_clauses(payload)
                    if kind == "greeting"
                    else conversation.stream_utterance(connection_id, llm, payload)
                )
                # Fillers cover the LLM's wait; the scripted greeting has none.
                speak_task = asyncio.create_task(speak(clauses, with_filler=kind == "utterance"))
                active_speech.set(speak_task)
                try:
                    outcome = await speak_task
                finally:
                    active_speech.clear(speak_task)
                    # Closing the generator runs stream_utterance's own cleanup
                    # even when the task above was cancelled mid-yield.
                    with suppress(Exception):
                        await clauses.aclose()

                if outcome.interrupted:
                    conversation.record_interrupted_turn(connection_id, " ".join(outcome.spoken))
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.exception("agent turn failed for %s", connection_id)
                with suppress(WebSocketDisconnect, RuntimeError):
                    await send_event({"type": "agent_error", "message": str(error)})
            finally:
                conversation_queue.task_done()

    async def send_partial(buffer: np.ndarray, seq: int) -> None:
        try:
            partial_text = await asyncio.to_thread(asr.transcribe_partial, buffer, language)
        except Exception:
            logger.warning("interim decode failed for %s", connection_id, exc_info=True)
            return
        # Dropped if the utterance ended meanwhile: an interim landing after
        # the final transcript would overwrite it in the UI.
        if partial_text and seq == utterance_seq and segmenter.in_speech:
            await send_event({"type": "partial_transcript", "text": partial_text})

    async def queue_segment(update: VadUpdate) -> None:
        nonlocal partial_frame_count, utterance_onset, utterance_onset_audible, partial_task, utterance_seq
        if update.speech_started:
            utterance_seq += 1
            utterance_onset = time.monotonic()
            utterance_onset_audible = active_speech.audible
            await send_event({"type": "vad_start", "probability": round(update.probability, 4)})
            logger.info("speech started: %s", connection_id)
        # Not on speech_started: one flagged 16 ms hop is a cough, not a
        # caller interrupting. See ActiveSpeech.note_speech().
        if active_speech.note_speech(update.speech_frame, update.speech_started):
            logger.info("barge-in: cancelling in-flight agent speech for %s", connection_id)

        if not update.speech_ended:
            if segmenter.in_speech and asr.ready:
                # BACKEND_COMPLETION.md Sec3.4: interim transcripts while the
                # caller is still mid-utterance, via the fast CTC branch -
                # throttled (not every 16ms VAD hop) since a rolling-buffer
                # decode isn't free, and interim results don't need to update
                # faster than a caller can perceive anyway.
                #
                # In the background, never awaited here: this runs inside the
                # audio receive loop, and awaiting a ~0.15s CPU decode every
                # 480ms stalled VAD on the frames behind it - endpointing, and
                # so the final decode, slipped by up to one decode per turn.
                # One at a time; a tick that finds one still running skips.
                partial_frame_count += 1
                if partial_frame_count % PARTIAL_TRANSCRIPT_INTERVAL_FRAMES == 0 and (
                    partial_task is None or partial_task.done()
                ):
                    buffer = segmenter.peek_utterance()
                    if buffer is not None:
                        partial_task = asyncio.create_task(send_partial(buffer, utterance_seq))
            return
        partial_frame_count = 0

        await send_event(
            {
                "type": "vad_end",
                "probability": round(update.probability, 4),
                "reason": update.end_reason,
            }
        )
        if update.samples is not None:
            await transcription_queue.put(
                (
                    update.samples,
                    language,
                    update.end_reason or "silence",
                    utterance_onset,
                    utterance_onset_audible,
                )
            )
        logger.info("speech ended: %s (%s)", connection_id, update.end_reason)

    async def transcribe_segments() -> None:
        while True:
            item = await transcription_queue.get()
            try:
                if item is None:
                    return
                samples, segment_language, reason, onset, onset_audible = item
                duration_ms = round(len(samples) / settings.sample_rate * 1000)
                if not asr.ready:
                    await send_event(
                        {
                            "type": "asr_error",
                            "message": "ASR model is unavailable. Accept the model terms and set HF_TOKEN, then restart the server.",
                        }
                    )
                    continue

                await send_event({"type": "asr_start", "duration_ms": duration_ms, "language": segment_language})
                # Pitch runs alongside the ASR, never after it: a few ms of
                # numpy in its own thread, done long before the decode is.
                # Skipped once decided, and for any utterance that began while
                # the agent was audible - its own (female) voice leaking in
                # would otherwise vote on the caller's honorific.
                pitch_task = None
                if honorific is not None and honorific.honorific is None and not onset_audible:
                    pitch_task = asyncio.create_task(
                        asyncio.to_thread(estimate_pitches, samples, settings.sample_rate)
                    )
                async with websocket.app.state.asr_semaphore:
                    transcript = await asyncio.to_thread(asr.transcribe, samples, segment_language)
                transcript = transcript.strip()
                if is_probably_self_echo(
                    transcript,
                    recent_agent_text,
                    # Only an utterance that began while a filler was audible
                    # can be that filler's echo.
                    filler_text=last_filler_text if onset <= filler_audible_until else "",
                ):
                    # The agent hearing itself through the speakers. Unlike
                    # background noise this transcribes to real words, so the
                    # empty-transcript check below cannot catch it - and acting
                    # on it means the agent answering its own question.
                    logger.info(
                        "discarded self-echo for %s: %r", connection_id, transcript[:60]
                    )
                    await send_event({"type": "echo_discarded", "text": transcript})
                    continue
                await send_event(
                    {
                        "type": "transcript",
                        "text": transcript,
                        "language": segment_language,
                        "duration_ms": duration_ms,
                        "endpoint_reason": reason,
                    }
                )
                if transcript and pitch_task is not None:
                    # Before the turn is queued, so this turn's own filler can
                    # already use the honorific.
                    try:
                        pitches = await pitch_task
                        decided = honorific.observe(pitches)
                        # A number per utterance, never audio: this is how the
                        # decision band gets checked against real callers.
                        logger.info(
                            "pitch for %s: %d voiced frames, median %s Hz -> %s",
                            connection_id,
                            len(pitches),
                            f"{float(np.median(pitches)):.0f}" if len(pitches) else "-",
                            decided or "undecided",
                        )
                    except Exception:
                        # A heuristic must never cost the caller their turn.
                        logger.exception("pitch estimate failed for %s", connection_id)
                        decided = None
                    if decided is not None:
                        conversation.set_caller_honorific(connection_id, decided)
                        await send_event({"type": "caller_honorific", "honorific": decided})
                        logger.info("caller honorific for %s: %s", connection_id, decided)
                if transcript:
                    await conversation_queue.put(("utterance", transcript))
                logger.info("transcribed %s ms as %s for %s", duration_ms, segment_language, connection_id)
            except Exception as error:
                logger.exception("ASR failed for %s", connection_id)
                with suppress(WebSocketDisconnect, RuntimeError):
                    await send_event({"type": "asr_error", "message": str(error)})
            finally:
                transcription_queue.task_done()

    worker = asyncio.create_task(transcribe_segments())
    conversation_worker = asyncio.create_task(handle_conversation_turns())
    persistence_worker = asyncio.create_task(persist_events())
    # Started when the call opens, to evaluate the prompt under the greeting.
    # Held so teardown can cancel it if the call ends while it is still running.
    prewarm_task: asyncio.Task | None = None
    logger.info("audio pipeline connected: %s", connection_id)
    await send_event(
        {
            "type": "ready",
            "connection_id": connection_id,
            "audio_format": "pcm_s16le",
            "sample_rate": settings.sample_rate,
            "vad_hop_size": settings.vad_hop_size,
            "asr_ready": asr.ready,
            "asr_language": language,
            "asr_decoding": settings.decoding,
            "conversation_ready": conversation.ready,
            "llm_ready": await llm.reachable(),
            "tts_ready": tts.ready,
        }
    )

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            text_message = message.get("text")
            if text_message is not None:
                try:
                    payload = json.loads(text_message)
                    event_type = payload.get("type")
                    if event_type == "call_started":
                        if started:
                            # A second call_started used to reset the session
                            # and overwrite prewarm_task, orphaning the first
                            # one - teardown then cancelled only the last, and
                            # the greeting was spoken twice into the same call.
                            # One call per socket; the client reconnects for a
                            # new one.
                            await send_event(
                                {
                                    "type": "protocol_error",
                                    "message": "call_started already received on this connection",
                                }
                            )
                            continue
                        language = _validate_start_event(payload, settings)
                        started = True
                        await send_event({"type": "pipeline_configured", "language": language})
                        if conversation.ready:
                            greeting = conversation.start_call(connection_id, agent_name=conversation.settings.agent_name)
                            await conversation_queue.put(("greeting", greeting))
                            # Evaluate the prompt while the greeting is being
                            # spoken. The first turn of a call otherwise pays
                            # 6-8s to evaluate ~2.7k tokens cold, and the
                            # greeting is ~3s of audio during which the model
                            # is idle and the caller is occupied. Fire and
                            # forget: it is cancelled with the call, and any
                            # failure just leaves the first turn as slow as it
                            # was before.
                            prewarm_task = asyncio.create_task(
                                conversation.prewarm(connection_id, llm)
                            )
                        else:
                            await send_event(
                                {
                                    "type": "agent_error",
                                    "message": "Conversation Manager is unavailable (master prompt failed to load).",
                                }
                            )
                        logger.info("audio capture started: %s (%s)", connection_id, language)
                    elif event_type == "mic_settings":
                        # What getUserMedia actually applied (track.getSettings()),
                        # which is not always what was asked for. Persisted next to
                        # the vad_diagnostics samples it explains.
                        logger.info("mic settings for %s: %s", connection_id, payload.get("settings"))
                        event_queue.put_nowait({"type": "mic_settings", "settings": payload.get("settings")})
                    elif event_type == "call_ended":
                        final_update = segmenter.flush()
                        if final_update:
                            await queue_segment(final_update)
                        logger.info("audio capture ended: %s", connection_id)
                    elif event_type == "user_text":
                        # Typed-caller input for manual testing (frontend's
                        # DirectTestingPanel) - bypasses VAD/ASR entirely and
                        # feeds the same conversation_queue a real transcript
                        # would, so it drives the identical LLM/tool/TTS path
                        # as a spoken turn.
                        if not started:
                            await send_event(
                                {"type": "protocol_error", "message": "send call_started before user_text"}
                            )
                        else:
                            text = str(payload.get("text", "")).strip()
                            if text:
                                await conversation_queue.put(("utterance", text))
                    else:
                        logger.info("audio event %s: %s", connection_id, event_type)
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    await send_event({"type": "protocol_error", "message": str(error)})
                continue

            audio_chunk = message.get("bytes")
            if audio_chunk is None:
                continue
            if not started:
                await send_event({"type": "protocol_error", "message": "send call_started before audio"})
                continue

            chunks_received += 1
            bytes_received += len(audio_chunk)
            pcm_buffer.extend(audio_chunk)
            frame_bytes = settings.vad_hop_size * np.dtype("<i2").itemsize
            while len(pcm_buffer) >= frame_bytes:
                frame = np.frombuffer(pcm_buffer[:frame_bytes], dtype="<i2").copy()
                del pcm_buffer[:frame_bytes]
                update = segmenter.process(frame)
                frames_seen += 1
                if diagnostics_every:
                    diag_frames += 1
                    diag_rms_sum += TenVadSegmenter._rms(frame)
                    diag_prob_max = max(diag_prob_max, update.probability)
                    diag_speech_hops += update.speech_frame
                    if diag_frames >= diagnostics_every:
                        # Persist-only, never sent: a measurement must not add
                        # traffic the client has to understand.
                        event_queue.put_nowait(
                            {
                                "type": "vad_diagnostics",
                                "t": round(frames_seen * settings.vad_hop_size / settings.sample_rate, 2),
                                "noise_floor": round(segmenter.noise_floor, 1),
                                "onset_bar": round(segmenter.onset_bar, 1),
                                "room_rms": round(diag_rms_sum / diag_frames, 1),
                                "p_max": round(diag_prob_max, 3),
                                "in_speech": segmenter.in_speech,
                                "agent_audible": active_speech.audible,
                                "speech_hops": diag_speech_hops,
                                "onset_refusals": segmenter.onset_refusals - diag_refusals_seen,
                            }
                        )
                        diag_refusals_seen = segmenter.onset_refusals
                        diag_frames = diag_speech_hops = 0
                        diag_rms_sum = diag_prob_max = 0.0
                await queue_segment(update)

            if chunks_received % 100 == 0:
                logger.info(
                    "audio capture %s: %d chunks, %d bytes received",
                    connection_id,
                    chunks_received,
                    bytes_received,
                )
    except WebSocketDisconnect:
        pass
    finally:
        final_update = segmenter.flush()
        if final_update:
            with suppress(WebSocketDisconnect, RuntimeError):
                await queue_segment(final_update)
        # Sentinel first, then a bounded wait: a worker mid-turn gets the
        # chance to finish and record how the turn ended, instead of being
        # cancelled on the spot and truncating the call log there. See
        # backend/tasks.py.
        # The prewarm is pure optimisation and nothing waits on its result, so
        # it is cancelled outright rather than drained - unlike the workers
        # below, it has no call state to finish writing.
        if partial_task is not None:
            partial_task.cancel()
        if prewarm_task is not None and not prewarm_task.done():
            prewarm_task.cancel()
            with suppress(asyncio.CancelledError):
                await prewarm_task
        await transcription_queue.put(None)
        await conversation_queue.put(None)
        await shutdown_worker(worker, "transcription worker")
        await shutdown_worker(conversation_worker, "conversation worker")
        # Drained last, so it also captures whatever the two workers above
        # emitted on their way out.
        event_queue.put_nowait(None)
        await shutdown_worker(persistence_worker, "persistence worker")
        conversation.end_call(connection_id)
        call_store.end_call(connection_id)
        logger.info(
            "audio pipeline disconnected: %s (%d chunks, %d bytes)",
            connection_id,
            chunks_received,
            bytes_received,
        )
