# AICA — backend

The Python/FastAPI backend powering AICA, an AI front-desk attendant that answers inbound calls,
handles minimal front-desk tasks itself (scheduling, hours, simple intake), and redirects anything
past that scope to staff.

```
Browser mic/keyboard --WebSocket--> FastAPI /ws/audio --> TEN VAD --> IndicConformer ASR (Tamil)
      --> transcript normaliser (English words + dictated numbers back into the agent's register)
      --> Conversation Manager (flow playbook + exemplars) --> LLM (OpenAI-compatible, e.g. Ollama)
      --> clause-by-clause TTS
      --> agent reply streams back as text AND 24 kHz PCM audio
```

The LLM is consumed as a token stream: each clause is synthesized and sent as soon as
it closes, so audio starts before the model has finished the turn. A caller speaking
over the agent cancels the in-flight turn (barge-in).

The React admin dashboard that talks to this backend's `/ws/audio` contract lives on the separate
[`frontend-aica-ui`](../../tree/frontend-aica-ui) branch — see its README for the UI side. See
[SETUP.md](SETUP.md) for running both together locally.

## Setup

```bash
./run.sh
```

That is the whole thing. It finds a supported Python, builds `.venv` if it is missing,
installs requirements only when they are actually absent, builds the `aruvi-base` Ollama
model if Ollama does not have it, starts the API, waits until every component reports
ready, prints which of the four came up, and gives you the console URL. `Ctrl+C` stops
it. `BACKEND_PORT=9000 ./run.sh` to move it, `RELOAD=1` for autoreload, `REINSTALL=1` to
force a dependency pass.

Two things it cannot do for you the first time:

```bash
git clone --depth 1 https://github.com/AI4Bharat/NeMo.git NeMo_ai4bharat   # microphone
cp .env.example .env                                                       # add HF_TOKEN
```

Without them the microphone is unavailable, but typed turns in the console still drive
the full conversation → LLM → TTS chain. See [SETUP.md](SETUP.md) for the long way round
and for choosing a model.

**Why the model is built rather than pulled:** `LLM_MODEL` must name a model with an
explicitly pinned `num_ctx`. Pointed at a bare tag, Ollama picks a VRAM-derived default
that silently truncates the system prompt before its language rules, and the agent then
answers in English and invents identifiers. `Modelfile` pins it;
`python -m backend.scripts.setup_model` is the manual equivalent of what `run.sh` does,
and additionally picks the best base you actually have installed.

`run.sh` reports which of the four components loaded, because the server starts even
when ASR, the LLM or TTS did not — "it responds" is not evidence the pipeline is whole.
`GET /api/health` says the same thing at any time.

## Testing

**The console — `http://localhost:8000/console`** (printed by `./run.sh`)

A self-contained page served by the backend itself, so it shares the app's origin and
opens `/ws/audio` with no CORS and no second dev server. Typed turns use the socket's
`user_text` path, which skips only VAD/ASR and drives the identical
conversation → LLM → TTS chain — so **it works without `HF_TOKEN` or the NeMo
fork**, the pieces most likely missing on a fresh machine. The microphone button enables
itself only when ASR actually loaded. One-click sample turns come from the golden flows.

```bash
pytest                                     # unit tests; no model, no network, no GPU
python -m backend.scripts.e2e_check        # drives a RUNNING server: spoken turn + typed turn
python -m backend.scripts.register_eval    # Tamil/English register on UNSEEN scenarios
python -m backend.scripts.safety_eval      # the three clinical refusals, under pressure
python -m backend.scripts.transcript_log --tts   # writes a readable .txt of prompts + replies
```

`e2e_check` is the one that proves the microphone path. It synthesizes the *caller's*
line, resamples it to 16 kHz and streams it into `/ws/audio` as raw PCM exactly the way
the browser does, then asserts a transcript came back and the reply carried both text
and audio. A synthesized voice is cleaner than a real one, so passing is a floor, not
proof of robustness on a noisy line.

`register_eval` is the one that catches memorisation: it scores Tamil/English word
ratio, wrong-script leakage, unspeakable symbols, turn length, questions per turn and
**fabricated identifiers** on scenarios absent from `golden/`, so a model that merely
memorised the 20 flows scores badly. Its fabrication check is now
[`backend/grounding.py`](backend/grounding.py), which compares every ID the agent says
against what it could legitimately know — see below. All of these are floors:
read the transcripts.

`safety_eval` covers the three highest-consequence behaviours: never read or grade a
lab value, never name or rule out a condition, never authorise a medicine (including
saying yes to aspirin during chest pain). Each scenario escalates — asks, insists, then
makes it personal — because a refusal that collapses on the second ask is not a refusal.
Its detectors are unit-tested in both directions (`backend/test_safety_eval.py`): a
missed violation reports an unsafe reply as fine, and a false positive on "fee ₹800"
makes the eval noise. A violation it reports is real; silence means only that nothing
obviously wrong was said.

### Grounding

The agent has no access to any hospital system, so every MRN, appointment ID or bill
amount it could say would be invented. The prompt says so; a small model does it
anyway. Given a few-shot exemplar containing an MRN it will say "ஒரு நிமிஷம் சார்,
system-ல check பண்றேன்..." and read that MRN back. The transcript looks perfect.

`backend/grounding.py` compares every structured ID and phone number in the agent's
reply against the caller's own turns and the call's established facts — deliberately
*not* the system prompt, since that is where the exemplars live, and treating them as
provenance is exactly how a parroted MRN passes for a real one. Anything unaccounted
for is logged, sent to the client as a `grounding_warning` event, shown in red in the
console, and counted by `register_eval`. Relatedly, the exemplars in
`golden/flow_exemplars.json` are kept disjoint from `golden/flows/*.txt` and from the
seeded records in `tools.py` (two separate tests, two different failure modes) so that
a copied fact is a *wrong* fact this check can catch.

## Known limitations

- **Voice sends text off-device** — `TTS_ENGINE=edge` (default) uses Microsoft's neural
  Tamil voice (`ta-IN-PallaviNeural`, female). Fine for the fictional Aruvi data, **not**
  for real patient speech; swap in self-hosted Indic Parler-TTS behind the same adapter
  interface first. `TTS_ENGINE=svara` is still an unimplemented placeholder. Because it
  is a network call, a clause can lose its audio to a blip; one retry is automatic, and
  a second failure sends `agent_audio_error` and keeps the turn going with text.
- **The agent talks; it does not transact.** There is deliberately no tool layer — it
  takes the request down, remembers what the caller said, and hands off to the desk.
  Measured: the 22 tool schemas cost 1778 of 5290 prompt tokens AND dropped generation
  from 12.5 to 10.2 tok/s, while scoring 10/14 clean turns against 12/14 without them.
  `backend/tools.py` is kept as the reference for restoring them per-flow later, but
  nothing in the runtime imports it. See [LLM_STACK.md](LLM_STACK.md) §5.
- **Model quality is the binding constraint, not the plumbing.** On a 3B model the agent
  produces degenerate repetition; `qwen3:4b-instruct-2507` is the smallest that holds
  the Tamil/English register. Run `register_eval` after any model change — the
  grounding counter is the number that matters.
- **The ASR is Tamil-only**, so English hospital words come back transliterated
  ("appointment" as "அப்பாயின்மென்ட்") and a dictated phone number comes back as number
  words. `backend/transcript_norm.py` rewrites both over the hospital's closed
  vocabulary; a word it does not know is passed through untouched.
- **CPU latency** — no GPU here, so a turn is seconds to tens of seconds. Clause-level
  streaming hides much of it (audio starts at the first clause, not the last token), but
  a GPU is still required for a natural-feeling call.
- **No real phone line** — `backend/telephony.py` is a code-complete, unit-tested adapter for
  Twilio-Media-Streams-shaped input, but has never been wired to an actual SIP trunk/DID; only the
  browser WebSocket (`/ws/audio`) is a live transport today, and live telephony is
  explicitly out of scope for now.
- **Single-process concurrency** — bounded ASR/TTS semaphore pools; fine for one caller at a
  time, not load-tested for many simultaneous callers.

See [BACKEND_COMPLETION.md](BACKEND_COMPLETION.md) for the fuller design/progress log.
