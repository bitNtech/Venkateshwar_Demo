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
