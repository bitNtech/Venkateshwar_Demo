# VAD + ASR pipeline (Tamil)

Streaming speech-to-text over a WebSocket: browser mic (16 kHz PCM) → **TEN VAD**
(turn segmentation) → **AI4Bharat IndicConformer** ASR → live partial + final transcripts.
Extracted from the AICA voice agent; the LLM and TTS run as separate services.

## Run

```bash
./run.sh          # Python 3.10/3.11; builds .venv, installs deps, starts on :8000
```

Open **http://localhost:8000/console**, press the mic, speak Tamil. You'll see the partial
transcript while speaking (~every 0.5 s) and the final transcript once you pause (~350 ms).

First run needs:
- `git clone --depth 1 https://github.com/AI4Bharat/NeMo.git NeMo_ai4bharat` (run.sh installs it)
- `HF_TOKEN` in `.env` after accepting the model terms on Hugging Face (first download only, ~500 MB)

Every knob (VAD thresholds, endpointing, decoding, auth, CORS) is in `.env.example`.

## WebSocket protocol — `ws://host:8000/ws/audio[?token=…]`

Client → server: JSON `{"type":"call_started","audio_format":"pcm_s16le","sample_rate":16000,"channels":1,"language":"ta"}`,
then binary int16 PCM frames; optional `{"type":"mic_settings",...}`, `{"type":"call_ended"}`.

Server → client (JSON): `ready`, `pipeline_configured`, `vad_start`, `partial_transcript {text}`,
`vad_end {reason}`, `asr_start {duration_ms}`, `transcript {text, duration_ms, endpoint_reason, latency_ms}`,
`asr_error`, `protocol_error`, `pipeline_error`.

HTTP: `GET /api/health` (is ASR loaded), `GET /api/calls`, `GET /api/calls/{id}` (persisted call log).

## Mounting TTS and the LLM (separate services)

This repo stops at the transcript. A full voice agent adds two services, each run on
its own and connected by consuming the `transcript` events above:

```
/ws/audio  --transcript-->  LLM (vLLM)  --reply text, clause by clause-->  TTS (Fish)  --PCM-->  caller
```

### LLM: vLLM, OpenAI-compatible

Any OpenAI-compatible `/v1/chat/completions` endpoint works. For vLLM:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name aruvi-base \
  --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes
```

- **Tool calling must be on** (`--enable-auto-tool-choice` plus `--tool-call-parser hermes` for
  Qwen, or `llama3_json` for Llama 3.x). The AICA agent sends `tool_choice: auto`, which
  vLLM rejects otherwise.
- The client needs: base URL `http://<vllm-host>:8000/v1`, the `--served-model-name`, and
  the context length (`--max-model-len`), so that call history can be trimmed to fit.
- Stream the reply (`stream: true`) and cut it into clauses, so TTS can start on the first
  clause while the model is still generating.

### TTS: Fish Speech S2-Pro streaming server

A separate repo/process (GPU, ~3.4 GB VRAM, ~4.9 GB checkpoint). Start it with its own `./run.sh`
(default `http://127.0.0.1:8080`).

- **Set the voice once:** `POST /v1/voice` with `reference_audio` plus its exact
  `reference_text`. Every session then speaks in that voice.
- **Synthesize:** WebSocket `ws://<host>:8080/v1/tts/live`, MessagePack frames.
  - Send `{"event":"start","request":{"format":"pcm","language":"ta","seed":1234}}`, then
    `{"event":"text","text":"<clause>"}`, then `{"event":"stop"}`.
  - Receive `started {sample_rate}` (44.1 kHz), then `audio {audio: int16 PCM bytes}` frames,
    then `finish`.
  - First audio arrives about 0.1 s after a clause is sent.
- **Barge-in:** close the socket (or send `cancel`) when `vad_start` fires while the
  agent is speaking. Generation stops within one decode step.
- **Pin `seed`**, so that every clause sounds like the same take of the voice.
- **GPU sharing:** on a 4 GB GPU, Fish and a local LLM cannot run together. Host vLLM on
  another machine.

### Ready-made wiring

Commit `46e034a` in this repo's history has the full AICA backend with both services
already connected: `backend/llm.py` (client, with a per-turn reachability check),
`backend/tts.py` (`FishTts` streaming client), and `backend/main.py` (the clause-by-clause
speak path, fillers and barge-in). Settings: `LLM_BASE_URL`, `LLM_MODEL`, `LLM_NUM_CTX`,
`TTS_ENGINE=fish`, `TTS_FISH_URL`. Either service can be started after the backend without
restarting it. See `git show 46e034a:SETUP.md`, section "Mounting the LLM".

## Layout

| File | What |
| --- | --- |
| `backend/main.py` | FastAPI app, `/ws/audio` socket, console, call-log API |
| `backend/vad.py` | TEN VAD segmenter: onset debounce, loudness gate, endpoint + watchdog |
| `backend/asr.py` | IndicConformer loader, RNNT final / CTC interim decode |
| `backend/transcript_norm.py` | Tamil-script English words → Latin (`data/asr_lexicon.json`) |
| `backend/persistence.py` | SQLite call-event log (optional Fernet encryption) |
| `backend/scripts/noise_floor_report.py` | Reads `VAD_DIAGNOSTICS_MS` samples back from the log |

## Test

```bash
.venv/Scripts/python -m pytest -q      # no model or mic needed; VAD is stubbed
```
