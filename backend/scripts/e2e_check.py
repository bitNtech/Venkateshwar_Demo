"""End-to-end check of a running server: spoken turn and typed turn.

This is the test that could never be run before - LLM_TEST_RESULTS.txt Part 5
lists "THE MICROPHONE PATH ... VAD -> ASR -> LLM has never run end to end" as
untested, because the ASR model would not load. It does now.

Rather than needing a human at a microphone, the caller's voice is synthesized
with the same TTS engine the agent speaks through, resampled to the 16 kHz the
socket requires, and streamed in as raw PCM exactly the way the browser console
streams mic frames. That exercises the real path - VAD segmentation, endpointing,
IndicConformer decoding, the conversation manager, the LLM, the tool loop and
TTS back out - against the real running server, with nothing stubbed.

A synthesized voice is cleaner than a real one, so a pass here is a floor: it
proves the pipeline is connected and the decode is sane, not that ASR is robust
to a noisy phone line. Read the transcript it prints.

Usage:
    python -m backend.scripts.e2e_check                     # both checks
    python -m backend.scripts.e2e_check --text-only         # skip the audio leg
    python -m backend.scripts.e2e_check --say "..."         # different spoken line

Requires the server to be running:
    uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from backend.audio_transcode import resample_pcm
from backend.settings import AudioSettings, TtsSettings
from backend.tts import EdgeTts

DEFAULT_SPOKEN_LINE = "வணக்கம், எனக்கு ஒரு appointment book பண்ணணும்."
DEFAULT_TYPED_LINE = "Visiting hours என்ன?"

# Frame size the console sends: 4096 samples at 16 kHz (~256 ms). Matching it
# means this drives the server's buffering exactly as a browser does.
CHUNK_SAMPLES = 4096


def synthesize_caller_audio(line: str, sample_rate: int) -> np.ndarray:
    """Render the caller's line to mono int16 PCM at the socket's sample rate."""
    engine = EdgeTts(TtsSettings())
    engine.load()
    result = engine.synthesize(line, "ta")
    if not result.samples.size:
        raise SystemExit("TTS produced no audio for the caller line - is the network up?")
    return resample_pcm(result.samples, result.sample_rate, sample_rate)


async def run(server: str, spoken_line: str, typed_line: str, text_only: bool) -> int:
    import websockets

    settings = AudioSettings()
    audio = None
    if not text_only:
        print(f"Synthesizing the caller's voice: {spoken_line!r}")
        audio = synthesize_caller_audio(spoken_line, settings.sample_rate)
        print(f"  {audio.size} samples @ {settings.sample_rate} Hz = {audio.size / settings.sample_rate:.1f}s\n")

    url = f"ws://{server}/ws/audio"
    print(f"Connecting to {url}")
    failures: list[str] = []

    async with websockets.connect(url, max_size=None) as ws:
        ready = json.loads(await ws.recv())
        print(f"ready: asr={ready['asr_ready']} conversation={ready['conversation_ready']} tts={ready['tts_ready']}")
        if not ready["conversation_ready"]:
            raise SystemExit("conversation manager is not loaded - check the server log")

        await ws.send(json.dumps({
            "type": "call_started", "audio_format": "pcm_s16le",
            "sample_rate": settings.sample_rate, "channels": 1, "language": "ta",
        }))

        print("\n--- greeting ---")
        await drain_turn(ws)

        if audio is not None and ready["asr_ready"]:
            print(f"\n--- spoken turn ---")
            transcript = await speak_and_transcribe(ws, audio, settings.sample_rate)
            if transcript:
                print(f"  TRANSCRIPT: {transcript}")
            else:
                failures.append("the spoken turn produced no transcript")
            await drain_turn(ws)
        elif audio is not None:
            print("\n--- spoken turn SKIPPED: server reports asr_ready=false ---")
            failures.append("asr_ready is false, so the microphone path is unavailable")

        print(f"\n--- typed turn: {typed_line!r} ---")
        started = time.time()
        await ws.send(json.dumps({"type": "user_text", "text": typed_line}))
        clauses, audio_bytes = await drain_turn(ws)
        if not clauses:
            failures.append("the typed turn produced no agent reply")
        if ready["tts_ready"] and not audio_bytes:
            failures.append("TTS is ready but the reply carried no audio")
        print(f"  ({time.time() - started:.1f}s, {audio_bytes} bytes of audio)")

        await ws.send(json.dumps({"type": "call_ended"}))

    print("\n" + "=" * 60)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: greeting, spoken turn and typed turn all produced text and audio.")
    return 0


async def speak_and_transcribe(ws, audio: np.ndarray, sample_rate: int) -> str:
    """Stream `audio` in browser-sized chunks and return the final transcript."""
    for start in range(0, audio.size, CHUNK_SAMPLES):
        await ws.send(audio[start : start + CHUNK_SAMPLES].astype("<i2").tobytes())
        # Real time-ish pacing: VAD endpointing counts consecutive silent
        # frames, so firehosing the whole utterance would still segment, but
        # pacing keeps the partial_transcript timing representative.
        await asyncio.sleep(CHUNK_SAMPLES / sample_rate / 4)

    # Trailing silence so VAD's endpoint_silence_frames closes the turn, the
    # same way a caller falling quiet does.
    silence = np.zeros(sample_rate, dtype=np.int16)
    for start in range(0, silence.size, CHUNK_SAMPLES):
        await ws.send(silence[start : start + CHUNK_SAMPLES].astype("<i2").tobytes())
        await asyncio.sleep(CHUNK_SAMPLES / sample_rate / 4)

    while True:
        message = await asyncio.wait_for(ws.recv(), timeout=180)
        if isinstance(message, bytes):
            continue
        event = json.loads(message)
        if event["type"] == "partial_transcript":
            print(f"  partial: {event['text']}")
        elif event["type"] == "transcript":
            return event["text"]
        elif event["type"] in {"asr_error", "pipeline_error"}:
            print(f"  {event['type']}: {event.get('message')}")
            return ""


async def drain_turn(ws) -> tuple[list[str], int]:
    """Read one agent turn, returning its clauses and how many audio bytes came with it."""
    clauses: list[str] = []
    audio_bytes = 0
    speaking = False
    while True:
        try:
            message = await asyncio.wait_for(ws.recv(), timeout=300)
        except asyncio.TimeoutError:
            print("  (timed out waiting for the agent - CPU inference can be very slow)")
            return clauses, audio_bytes

        if isinstance(message, bytes):
            audio_bytes += len(message)
            continue

        event = json.loads(message)
        kind = event["type"]
        if kind == "agent_speaking_start":
            speaking = True
        elif kind == "agent_clause":
            clauses.append(event["text"])
            print(f"  AGENT: {event['text']}")
        elif kind == "agent_tool_call":
            print(f"  TOOL: {event['name']}({event['arguments']}) -> {str(event['result'])[:100]}")
        elif kind == "grounding_warning":
            print(f"  !! UNGROUNDED (fabricated): {', '.join(event['identifiers'])}")
        elif kind == "call_control":
            print(f"  CALL CONTROL: {event['action']} {event.get('detail', '')}")
        elif kind in {"agent_error", "asr_error", "pipeline_error", "protocol_error"}:
            print(f"  {kind}: {event.get('message')}")
            # A turn that failed before saying anything (LLM unreachable) sends
            # no agent_speaking_end - waiting for one just hangs the check.
            if kind == "agent_error" and not speaking:
                return clauses, audio_bytes
        elif kind == "agent_speaking_end":
            return clauses, audio_bytes
        elif kind == "agent_interrupted":
            return clauses, audio_bytes
        elif kind == "pipeline_configured" and not speaking:
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="127.0.0.1:8000")
    parser.add_argument("--say", default=DEFAULT_SPOKEN_LINE, help="the line the caller speaks")
    parser.add_argument("--type", dest="typed", default=DEFAULT_TYPED_LINE, help="the line the caller types")
    parser.add_argument("--text-only", action="store_true", help="skip the synthesized-audio leg")
    args = parser.parse_args()

    raise SystemExit(asyncio.run(run(args.server, args.say, args.typed, args.text_only)))


if __name__ == "__main__":
    main()
