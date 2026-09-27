"""Self-check for the /ws/audio handler: ready event, call_started validation,
auth gate, and the VAD -> ASR flow end to end (partials, final transcript,
diagnostics).

app.state is populated by hand (TestClient never enters the `with` form, so
lifespan()'s real model loading never runs), and the native TEN VAD model is
swapped for a scripted stub at `backend.vad.TenVad` - the WS handler builds its
own TenVadSegmenter, so there is no other injection point.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from . import vad as vad_module
from .main import app
from .persistence import CallEventStore
from .settings import AudioSettings, PersistenceSettings, SecuritySettings


class _FakeAsr:
    def __init__(self, transcript: str = "", partial_transcript: str = "", ready: bool = True) -> None:
        self._transcript = transcript
        self._partial_transcript = partial_transcript
        self.ready = ready
        self.calls: list[tuple] = []
        self.partial_calls: list[tuple] = []

    def transcribe(self, samples, language: str) -> str:
        self.calls.append((samples, language))
        return self._transcript

    def transcribe_partial(self, samples, language: str) -> str:
        self.partial_calls.append((samples, language))
        return self._partial_transcript


class _FakeTenVad:
    """Replaces the native ten_vad model with a scripted flag sequence."""

    def __init__(self, hop_size: int, threshold: float) -> None:
        self._flags = iter(_CURRENT_FLAGS)

    def process(self, frame):
        return (1.0, next(self._flags, 0))


_CURRENT_FLAGS: list[int] = []


@contextlib.contextmanager
def _scripted_vad(flags: list[int]):
    global _CURRENT_FLAGS
    _CURRENT_FLAGS = flags
    original = vad_module.TenVad
    vad_module.TenVad = _FakeTenVad
    try:
        yield
    finally:
        vad_module.TenVad = original


def _set_app_state(*, asr=None, security: SecuritySettings | None = None) -> None:
    app.state.settings = AudioSettings()
    app.state.asr = asr if asr is not None else _FakeAsr()
    app.state.asr_semaphore = asyncio.Semaphore(2)
    store = CallEventStore(PersistenceSettings(db_path=Path(tempfile.mkdtemp()) / "call_events.db"))
    store.load()
    app.state.call_store = store
    app.state.security = security if security is not None else SecuritySettings()


def _next_json(ws) -> dict:
    return json.loads(ws.receive_text())


_VALID_CALL_STARTED = {
    "type": "call_started",
    "audio_format": "pcm_s16le",
    "sample_rate": 16_000,
    "channels": 1,
    "language": "ta",
}

# Onset is loudness-gated (AudioSettings.vad_onset_min_rms), so the PCM behind a
# scripted "speech" frame has to be at a speaking level; silence opens nothing.
_SPOKEN_SAMPLE = 2600
_ROOM_SAMPLE = 40


def _audio_for(flags: list[int]) -> bytes:
    """PCM matching the scripted flags: speaking level where the VAD says speech."""
    hop = AudioSettings().vad_hop_size
    return np.concatenate(
        [np.full(hop, _SPOKEN_SAMPLE if flag else _ROOM_SAMPLE, dtype="<i2") for flag in flags]
    ).tobytes()


def _send_like_a_browser(ws, audio: bytes) -> None:
    """One VAD hop per message, the way the console streams mic frames. The
    interim decode runs in the background and needs the receive loop to
    yield, which one giant message never does."""
    hop = AudioSettings().vad_hop_size * 2
    for offset in range(0, len(audio), hop):
        ws.send_bytes(audio[offset : offset + hop])


def _one_turn_flags() -> list[int]:
    # Read from settings, so tuning either knob cannot silently stop exercising a turn.
    s = AudioSettings()
    return [1] * s.vad_start_frames + [0] * (s.endpoint_silence_frames + 1)


def test_ready_event_reports_component_status() -> None:
    _set_app_state(asr=_FakeAsr(ready=False))
    with TestClient(app).websocket_connect("/ws/audio") as ws:
        ready = _next_json(ws)

    assert ready["type"] == "ready"
    assert ready["asr_ready"] is False
    assert ready["sample_rate"] == 16_000


def test_call_started_with_wrong_sample_rate_is_a_protocol_error() -> None:
    _set_app_state()
    with TestClient(app).websocket_connect("/ws/audio") as ws:
        _next_json(ws)  # ready
        ws.send_json({**_VALID_CALL_STARTED, "sample_rate": 8_000})
        error = _next_json(ws)

    assert error["type"] == "protocol_error"
    assert "sample_rate" in error["message"]


def test_audio_before_call_started_is_a_protocol_error() -> None:
    _set_app_state()
    with TestClient(app).websocket_connect("/ws/audio") as ws:
        _next_json(ws)  # ready
        ws.send_bytes(b"\x00" * 512)
        error = _next_json(ws)

    assert error["type"] == "protocol_error"
    assert "call_started" in error["message"]


def test_a_second_call_started_is_refused() -> None:
    _set_app_state()
    with TestClient(app).websocket_connect("/ws/audio") as ws:
        _next_json(ws)  # ready
        ws.send_json(_VALID_CALL_STARTED)
        assert _next_json(ws) == {"type": "pipeline_configured", "language": "ta"}
        ws.send_json(_VALID_CALL_STARTED)
        error = _next_json(ws)

    assert error["type"] == "protocol_error"
    assert "already" in error["message"]


def test_auth_rejects_connection_with_wrong_or_missing_token() -> None:
    _set_app_state(security=SecuritySettings(ws_auth_token="secret-token"))
    client = TestClient(app)
    for url in ("/ws/audio", "/ws/audio?token=wrong"):
        try:
            with client.websocket_connect(url):
                raise AssertionError(f"{url} must be rejected")
        except Exception:
            pass  # server closes before accept() - the client sees a handshake failure


def test_auth_accepts_connection_with_correct_token() -> None:
    _set_app_state(security=SecuritySettings(ws_auth_token="secret-token"))
    with TestClient(app).websocket_connect("/ws/audio?token=secret-token") as ws:
        assert _next_json(ws)["type"] == "ready"


def test_speech_flows_through_vad_into_a_final_transcript() -> None:
    asr = _FakeAsr(transcript="  Cardiology appointment வேணும்  ")
    _set_app_state(asr=asr)
    flags = _one_turn_flags()

    with _scripted_vad(flags), TestClient(app).websocket_connect("/ws/audio") as ws:
        _next_json(ws)  # ready
        ws.send_json(_VALID_CALL_STARTED)
        _next_json(ws)  # pipeline_configured
        ws.send_bytes(_audio_for(flags))
        vad_start, vad_end, asr_start, transcript = (_next_json(ws) for _ in range(4))

    assert vad_start["type"] == "vad_start"
    assert vad_end == {"type": "vad_end", "probability": vad_end["probability"], "reason": "silence"}
    assert asr_start["type"] == "asr_start"
    assert transcript["type"] == "transcript"
    assert transcript["text"] == "Cardiology appointment வேணும்"
    assert transcript["endpoint_reason"] == "silence"
    assert transcript["latency_ms"] >= 0
    assert asr.calls and asr.calls[0][1] == "ta"


def test_long_utterance_emits_partial_transcripts_before_the_turn_ends() -> None:
    asr = _FakeAsr(transcript="final text", partial_transcript="interim text")
    _set_app_state(asr=asr)
    # PARTIAL_TRANSCRIPT_INTERVAL_FRAMES is 30 - 65 speech frames guarantees at
    # least one interim decode fires mid-utterance, before the silence closes it.
    flags = [1] * 65 + [0] * (AudioSettings().endpoint_silence_frames + 1)

    with _scripted_vad(flags), TestClient(app).websocket_connect("/ws/audio") as ws:
        _next_json(ws)  # ready
        ws.send_json(_VALID_CALL_STARTED)
        _next_json(ws)  # pipeline_configured
        _send_like_a_browser(ws, _audio_for(flags))

        event = _next_json(ws)
        assert event["type"] == "vad_start"
        partials = []
        while event["type"] != "vad_end":
            event = _next_json(ws)
            if event["type"] == "partial_transcript":
                partials.append(event)
        _next_json(ws)  # asr_start
        transcript = _next_json(ws)

    assert partials, "a 65-frame utterance should have produced at least one partial_transcript"
    assert all(p["text"] == "interim text" for p in partials)
    assert transcript == {**transcript, "type": "transcript", "text": "final text"}


def test_asr_not_ready_reports_asr_error_instead_of_transcript() -> None:
    _set_app_state(asr=_FakeAsr(ready=False))
    flags = _one_turn_flags()

    with _scripted_vad(flags), TestClient(app).websocket_connect("/ws/audio") as ws:
        _next_json(ws)  # ready
        ws.send_json(_VALID_CALL_STARTED)
        _next_json(ws)  # pipeline_configured
        ws.send_bytes(_audio_for(flags))
        _next_json(ws)  # vad_start
        _next_json(ws)  # vad_end
        error = _next_json(ws)

    assert error["type"] == "asr_error"
    assert "unavailable" in error["message"]


def test_vad_diagnostics_are_persisted_but_never_sent_to_the_client() -> None:
    """VAD_DIAGNOSTICS_MS is a measurement: it goes to the call log for
    scripts/noise_floor_report.py and adds nothing the client must parse."""
    _set_app_state()
    app.state.settings = AudioSettings(vad_diagnostics_ms=160)  # every 10 hops
    flags = [0] * 40

    with _scripted_vad(flags), TestClient(app).websocket_connect("/ws/audio") as ws:
        connection_id = _next_json(ws)["connection_id"]
        ws.send_json(_VALID_CALL_STARTED)
        _next_json(ws)  # pipeline_configured
        ws.send_json({"type": "mic_settings", "settings": {"autoGainControl": True, "label": "Headset"}})
        ws.send_bytes(_audio_for(flags))
        # Malformed on purpose: the protocol_error reply is a sentinel that
        # arrives only after everything sent before it has been processed.
        ws.send_text("not json")
        seen = []
        while not seen or seen[-1] != "protocol_error":
            seen.append(_next_json(ws)["type"])
        # Read while the socket is open: persistence is a background queue.
        deadline = time.time() + 5
        events = []
        while time.time() < deadline:
            events = app.state.call_store.events_for_call(connection_id)
            if any(e.get("type") == "protocol_error" for e in events):
                break
            time.sleep(0.05)

    assert "vad_diagnostics" not in seen
    samples = [e for e in events if e.get("type") == "vad_diagnostics"]
    assert [s["t"] for s in samples] == [0.16, 0.32, 0.48, 0.64]
    assert {"noise_floor", "onset_bar", "room_rms", "in_speech", "onset_refusals"} <= set(samples[0])
    assert {"type": "mic_settings", "settings": {"autoGainControl": True, "label": "Headset"}} in events


def test_the_console_is_served_and_the_root_points_at_it() -> None:
    _set_app_state()
    client = TestClient(app)

    assert client.get("/console").status_code == 200
    root = client.get("/", follow_redirects=False)
    assert root.status_code in (302, 307) and root.headers["location"] == "/console"


def test_health_reports_whether_asr_loaded() -> None:
    _set_app_state(asr=_FakeAsr(ready=False))

    assert TestClient(app).get("/api/health").json() == {
        "asr_ready": False,
        "asr_language": "ta",
        "asr_decoding": AudioSettings().decoding,
    }
