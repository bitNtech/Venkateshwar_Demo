"""Low-latency browser audio WebSocket: 16 kHz PCM -> TEN VAD -> IndicConformer ASR."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import json
import logging
import os
from pathlib import Path
import time
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse

from .asr import IndicConformerAsr
from .persistence import CallEventStore
from .settings import AudioSettings, PersistenceSettings, SecuritySettings, SUPPORTED_LANGUAGES
from .tasks import shutdown_worker
from .vad import TenVadSegmenter, VadUpdate

# LOG_LEVEL=DEBUG for more, WARNING to quiet a production box. An unknown value
# falls back to INFO rather than refusing to start. Millisecond timestamps,
# because the things measured in this log are tenths of a second apart.
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format=os.getenv("LOG_FORMAT", "%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s"),
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline.audio")

# How often (in VAD hops) to run an interim CTC decode while an utterance is
# still in progress. 30 hops * 16 ms = ~480 ms between partial_transcript
# updates - responsive, and far below a rate that would starve the final decode.
PARTIAL_TRANSCRIPT_INTERVAL_FRAMES = int(os.getenv("ASR_PARTIAL_INTERVAL_FRAMES", "30"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = AudioSettings()
    asr = IndicConformerAsr(settings)
    call_store = CallEventStore(PersistenceSettings())

    try:
        # Preload once so model startup never delays a live call.
        await asyncio.to_thread(asr.load)
    except Exception:
        # Not fatal: the socket still runs VAD and reports asr_error per
        # utterance, and /api/health says why transcripts are missing.
        logger.exception("ASR model failed to load")

    try:
        call_store.load()
    except Exception:
        logger.exception("Call-event store failed to initialize")

    app.state.settings = settings
    app.state.asr = asr
    # Bounded concurrent ASR inference across calls, not one global lock.
    app.state.asr_semaphore = asyncio.Semaphore(settings.asr_max_concurrency)
    app.state.call_store = call_store
    app.state.security = SecuritySettings()
    yield


app = FastAPI(title="VAD + ASR Pipeline", lifespan=lifespan)

# Added at import time, not in lifespan(): Starlette builds its middleware stack
# on the first request, and a middleware appended after that is silently ignored.
_cors_origins = SecuritySettings().cors_allow_origins
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(_cors_origins),
        allow_credentials="*" not in _cors_origins,  # the browser forbids both
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

_CONSOLE_HTML = Path(__file__).resolve().parent / "console.html"


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse("/console")


@app.get("/console", response_class=HTMLResponse)
async def console() -> HTMLResponse:
    """Test console: live mic -> VAD -> ASR, with partial and final transcripts.

    Served from the backend itself so it shares this app's origin and can open
    /ws/audio without CORS or a second dev server.
    """
    return HTMLResponse(_CONSOLE_HTML.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health() -> dict:
    """Whether ASR actually loaded - lifespan() logs a failure and carries on."""
    settings: AudioSettings = app.state.settings
    return {
        "asr_ready": app.state.asr.ready,
        "asr_language": settings.language,
        "asr_decoding": settings.decoding,
    }


@app.get("/api/calls")
async def list_calls(limit: int = 50) -> dict:
    """Every call this server has handled, newest first."""
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
        # Checked before accept(): a browser cannot set a custom header on the
        # WS handshake, so the token travels as a query param. Rejecting before
        # accept() means an unauthorized client never gets a "ready" event.
        if websocket.query_params.get("token") != security.ws_auth_token:
            await websocket.close(code=4401)
            return

    await websocket.accept()
    connection_id = str(uuid4())
    settings: AudioSettings = websocket.app.state.settings
    asr: IndicConformerAsr = websocket.app.state.asr
    call_store: CallEventStore = websocket.app.state.call_store
    send_lock = asyncio.Lock()
    language = settings.language
    started = False
    chunks_received = 0
    bytes_received = 0
    pcm_buffer = bytearray()
    partial_frame_count = 0
    # The interim decode in flight, if any, and which utterance it belongs to.
    partial_task: asyncio.Task | None = None
    utterance_seq = 0
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

    # Persistence runs behind a queue, not inline in send_event: a slow SQLite
    # write must never stall the audio path. Unbounded, because dropping call
    # history to keep audio flowing is the right trade, and an event is small.
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

    # (samples, language, endpoint reason, monotonic time the speech ended)
    transcription_queue: asyncio.Queue[tuple[np.ndarray, str, str, float] | None] = asyncio.Queue()

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
        nonlocal partial_frame_count, partial_task, utterance_seq
        if update.speech_started:
            utterance_seq += 1
            await send_event({"type": "vad_start", "probability": round(update.probability, 4)})
            logger.info("speech started: %s", connection_id)

        if not update.speech_ended:
            if segmenter.in_speech and asr.ready:
                # Interim transcripts via the fast CTC branch, in the
                # background and never awaited here: this runs inside the
                # audio receive loop, and awaiting a ~0.15 s CPU decode every
                # 480 ms stalled VAD on the frames behind it - endpointing, and
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
                (update.samples, language, update.end_reason or "silence", time.monotonic())
            )
        logger.info("speech ended: %s (%s)", connection_id, update.end_reason)

    async def transcribe_segments() -> None:
        while True:
            item = await transcription_queue.get()
            try:
                if item is None:
                    return
                samples, segment_language, reason, ended_at = item
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
                async with websocket.app.state.asr_semaphore:
                    transcript = await asyncio.to_thread(asr.transcribe, samples, segment_language)
                await send_event(
                    {
                        "type": "transcript",
                        "text": transcript.strip(),
                        "language": segment_language,
                        "duration_ms": duration_ms,
                        "endpoint_reason": reason,
                        # Speech end -> transcript ready: what the caller waits.
                        "latency_ms": round((time.monotonic() - ended_at) * 1000),
                    }
                )
                logger.info("transcribed %s ms as %s for %s", duration_ms, segment_language, connection_id)
            except Exception as error:
                logger.exception("ASR failed for %s", connection_id)
                with suppress(WebSocketDisconnect, RuntimeError):
                    await send_event({"type": "asr_error", "message": str(error)})
            finally:
                transcription_queue.task_done()

    worker = asyncio.create_task(transcribe_segments())
    persistence_worker = asyncio.create_task(persist_events())
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
                            # One call per socket; the client reconnects for a new one.
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
        if partial_task is not None:
            partial_task.cancel()
        # Sentinel first, then a bounded wait, so an in-flight decode can
        # finish and be recorded instead of being cut off (see tasks.py).
        await transcription_queue.put(None)
        await shutdown_worker(worker, "transcription worker")
        # Drained last, so it also captures whatever the worker emitted on its way out.
        event_queue.put_nowait(None)
        await shutdown_worker(persistence_worker, "persistence worker")
        call_store.end_call(connection_id)
        logger.info(
            "audio pipeline disconnected: %s (%d chunks, %d bytes)",
            connection_id,
            chunks_received,
            bytes_received,
        )
