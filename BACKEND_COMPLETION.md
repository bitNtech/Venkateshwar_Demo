# AICA Backend — Completion & Rework Plan

**Reviewed:** `AICA-backend-main` (commit snapshot in zip)
**Target flow:**
```
Patient speaks Tamil -> Streaming STT -> LLM (Llama 3.1 8B / Qwen2.5-7B) -> stream tokens
-> svara-TTS (streaming, cloned voice) -> 8kHz PCM -> SIP/WebRTC -> Phone
```
**Reference diagram in repo** (`assets/LLM-Flow.png`): Website -> Mic -> WebSocket -> VAD -> ASR ->
**Conversation Manager** (master prompt + state) -> **LLM** (streaming) -> **TTS** (streaming) -> Browser.

---

## 1. What's actually in the repo today

**This section was rewritten on 2026-08-30 after an audit.** The version it replaces
described the conversation manager, LLM client, tool layer, TTS, persistence and
telephony adapter as "Missing entirely". All of them existed and were tested. Treat the
rest of this document as a design log, not a status report.

| Stage in the target flow | Status | Evidence |
|---|---|---|
| Browser mic capture, 16 kHz PCM | Done | `legacy_test_client/`, `backend/console.html` |
| WebSocket transport | Done | `backend/main.py` `/ws/audio` |
| VAD / turn segmentation | Done | `backend/vad.py` (TEN VAD), `test_vad.py` |
| Streaming STT | Done | `backend/asr.py` — RNNT for the final transcript at `vad_end`, plus throttled CTC `transcribe_partial` for interim `partial_transcript` events while the caller is still speaking (§3.4 option 2). Verified end to end by `backend/scripts/e2e_check.py`. |
| Conversation Manager (prompt + ledger) | Done | `backend/conversation.py`, `backend/prompt_builder.py`. The full 15k-token spec is **not** sent per turn: `runtime_core.txt` plus one section-8 playbook plus that flow's exemplars, ~2.5k tokens. |
| LLM (OpenAI-compatible, streaming) | Done | `backend/llm.py`. `stream()` yields text deltas; `complete()` is a wrapper over it. |
| Tool calling (22 tools) | Done | `backend/tools.py` against `MockHospitalDb`. `hangUp`/`transferCall` now drive real call control rather than just returning a dict. |
| TTS (streaming, clause-chunked) | Done, different engine | `backend/tts.py`. `TTS_ENGINE=edge` (Microsoft neural Tamil) is the working default; `svara` remains an unimplemented placeholder — no public reference exists. Clauses are synthesized as the LLM stream closes them, so audio starts before the turn finishes. |
| Barge-in | Done | `backend/barge_in.py` + `queue_segment` in `main.py`. Cancelling now also cancels the LLM generation, and the truncated turn is recorded so the model knows what the caller actually heard. |
| 8 kHz resampling for telephony | Done | `backend/audio_transcode.py`, used by the telephony leg on both directions. |
| SIP / WebRTC to a real phone | Adapter only, **out of scope** | `backend/telephony.py` is a code-complete, unit-tested Twilio-Media-Streams-shaped adapter that has never been connected to a trunk. Live telephony is explicitly out of scope; testing is manual through the console. |
| Golden-flow evaluation | Done | `register_eval.py` (register + grounding on **unseen** scenarios), `safety_eval.py` (the three clinical refusals under escalating pressure), `e2e_check.py` (a running server, spoken + typed). Run the first two with `LLM_TEMPERATURE=0`: at the 0.3 default a 14-turn register_eval run varies by 4 turns between runs of the *same* prompt, which is wider than any prompt change measured so far. |
| Persistence / call history | Done | `backend/persistence.py` (SQLite, optional Fernet encryption) plus `GET /api/calls`, `GET /api/calls/{id}`, `GET /api/health`. Writes go through a background queue so a disk write can never stall speech. |
| Grounding enforcement | **New, not in the original plan** | `backend/grounding.py`. See below. |

### What the audit actually found broken

1. **The ledger never reached the prompt.** `session.ledger` accumulated `mrn`,
   `patient_name` and `caller_mobile` from tool results, but the system prompt rendered
   its KNOWN FACTS block from `session.metadata`, so every slot stayed blank for the whole
   call. Since the prompt says "a blank value means it is not yet known — discover it
   normally", it was actively instructing the model to re-ask for facts the server already
   held — defeating the central "never re-ask" rule. The prompt is now rendered from the
   live ledger, and refreshed *inside* the tool loop, because the iteration right after
   `lookupPatient` is the first consumer of what it returned.
2. **ASR could not load at all.** `pip install -e ./NeMo_ai4bharat --no-deps` skips
   NeMo's own dependencies, and `requirements.txt` was missing `librosa`, `sentencepiece`
   and `jiwer`. The server started fine and reported `asr_ready: false`, which reads like
   a model-access problem rather than a missing package. The microphone path had never
   run end to end; it does now.
3. **`pytest` at the repo root failed**, walking into the vendored NeMo checkout and dying
   collecting its suite. Fixed with `testpaths`.
4. **`hangUp` did nothing.** The agent said goodbye and the socket stayed open.
5. **Persistence sat on the critical path** — a SQLite write between every clause and the
   next synthesis.
6. **Teardown truncated the call log**, cancelling workers the instant a socket dropped,
   which reliably lost the end of the turn — the part worth recording.
7. **The few-shot exemplars were the mock DB record.** `appointment.book` used
   Murugesan / ARV-118342 / 9840721534, the exact seeded patient. A small model parrots
   the exemplar instead of calling `lookupPatient`, and the answer looks right. Exemplar
   facts are now disjoint from the database (enforced by a test), so a copied fact is a
   *wrong* fact — which is what makes the grounding check below able to catch it.

### Grounding (new)

The prompt forbids inventing an ID. Nothing checked whether the model complied, and it
does not: observed on a 3B model, the agent said "ஒரு நிமிஷம் சார், system-ல check
பண்றேன்..." and read back an MRN lifted from its own exemplars, having called no tool.
`backend/grounding.py` compares every structured ID and phone number the agent says
against the tool results, the caller's turns and the ledger — deliberately **not** the
system prompt, since that is where the exemplars live. Anything unaccounted for is
logged, emitted as a `grounding_warning` event, shown in the console, and counted by
`register_eval`.

It reports rather than filters: by the time a clause is checked it has already been
streamed to the caller, and withholding half a sentence would be worse than the fault.

`grounding.py` also checks a second, worse class of claim: an action the agent says it
has **completed** with no tool call behind it. Given a chest-pain call and an address the
agent says "Ambulance அனுப்பிட்டேன், இப்பவே கிளம்பிடுச்சு" — *I have sent an ambulance,
it has left* — and calls nothing. There is no invented identifier in that sentence for
the check above to see, and it is worse than a wrong MRN, because the caller stops
looking for help. Three separate prompt fixes failed to make `dispatchAmbulance` fire
(see LLM_TEST_RESULTS.txt §7.3), so it is now detected and surfaced as an
`action_claim_warning` event rather than trusted to the prompt. **Still open:** the agent
does not dispatch. Making it dispatch means a deterministic server-side call once an
emergency is detected and an address is known — a product decision, since it would
dispatch on an address the model believes it heard.

### The binding constraint is model quality, not plumbing

On CPU with `qwen2.5:3b` the agent produces degenerate repetition and never calls a
tool. `qwen3:4b-instruct-2507-q4_K_M` holds the register and answers correctly from the
standing facts. `register_eval` on seven unseen scenarios is the number to watch; the
emergency override generalises (an emergency stated mid-billing correctly abandons the
billing flow and takes the address first), while turn discipline is the weakest area.

## 2. Target architecture (fills the gap)

```
Phone (PSTN) --SIP trunk--> SIP/RTP Gateway (new) --8kHz PCM--> Session Orchestrator
                                                                          |
Browser mic --WebSocket--> (existing) /ws/audio, 16kHz PCM --------------+
                                                                          v
                                                                 +-----------------+
                                                                 | VAD  (existing) |
                                                                 +--------+--------+
                                                                          v
                                                        +-------------------------------+
                                                        | Streaming ASR (rework to       |
                                                        | chunked/partial decoding)      |
                                                        +---------------+---------------+
                                                                        v
                                                        +-------------------------------+
                                                        | Conversation Manager (NEW)     |
                                                        | - loads golden/main_prompt.txt |
                                                        | - per-call ledger/state (Redis)|
                                                        | - flow router (20 intents)     |
                                                        | - tool-call executor           |
                                                        +---------------+---------------+
                                                                        v
                                                        +-------------------------------+
                                                        | LLM client (NEW)               |
                                                        | Llama 3.1 8B / Qwen2.5-7B       |
                                                        | served via vLLM/TGI, streamed   |
                                                        +---------------+---------------+
                                                                        v token stream
                                                        +-------------------------------+
                                                        | TTS client (NEW)               |
                                                        | svara-TTS streaming, cloned voice|
                                                        | sentence-chunked synthesis      |
                                                        +---------------+---------------+
                                                                        v 8kHz PCM / u-law
                                                        +-------------------------------+
                                                        | Back out over SIP/RTP (phone)  |
                                                        | or WS (browser) - same session |
                                                        +-------------------------------+
```

Key design decision: **the Conversation Manager is the new center of gravity**. Today `main.py`
routes `transcript` straight into a WS event to the browser. It should instead route into an
orchestrator that (a) holds the ledger, (b) calls the LLM, (c) executes tools, (d) streams TTS
audio back out, and (e) can be interrupted by a new `vad_start` (barge-in).

---

## 3. Work items, in priority order

### 3.1 Conversation Manager + LLM integration (P0 — this is the missing core)
- New module `backend/conversation.py`:
  - Load `golden/main_prompt.txt` once at startup (it's a ~9k-word system prompt — treat it as a template with `{{agent_name}}`, `{{caller_mobile}}`, `{{mrn}}`, `{{campaign}}` etc. substituted per call from caller metadata / CRM lookup).
  - Maintain the **ledger** described in the prompt (§4 of the prompt) as actual server-side state per `connection_id`, not just an LLM-context convention — store it in Redis (or in-process dict for a v1) keyed by `connection_id`, so a disconnect/reconnect (e.g. SIP re-INVITE) doesn't lose call context.
  - Implement the tool interface the prompt already assumes: `lookupPatient`, `verifyIdentity`, `searchSlots`, `bookAppointment`, `rescheduleAppointment`, `cancelAppointment`, `confirmAppointment`, `raiseRefill`, `bookLabOrder`, `getReportStatus`, `resendReport`, `getReferralStatus`, `getPolicyDetails`, `createPreAuth`, `getBill`, `createTicket`, `logRecordsRequest`, `registerPatient`, `dispatchAmbulance`, `escalate`, `transferCall`, `hangUp`. For v1, back these with a mock/fixture data layer (a JSON "hospital DB" is enough to run all 20 golden flows end to end); design the interface so a real HMS/EHR integration can be swapped in later without touching the LLM plumbing.
- New module `backend/llm.py`:
  - Wrap an OpenAI-compatible streaming client (works for both vLLM and TGI) rather than hand-rolling a transport — this keeps the model swappable (Llama 3.1 8B -> Qwen2.5-7B -> a future model) via one base-URL/model-name env var.
  - Use function/tool calling (both candidate models support OpenAI-style tool calling through vLLM's `--enable-auto-tool-choice`) so the tools above are real structured calls, not regex-parsed text.
  - Stream tokens; hand sentence-boundary chunks (not raw tokens) to TTS — see 3.2.
- Add `type: "agent_speaking_start" / "agent_token" / "agent_speaking_end"` WS events (mirroring the existing `vad_start`/`asr_start` naming convention already in `main.py`) so any client (browser or dashboard) can show partial agent text if desired.

### 3.2 Streaming TTS (svara-TTS) (P0)
- New module `backend/tts.py`.
- **Chunk at sentence/clause boundaries, not full LLM completions.** The Tamil prompt is written for short turns (<=40 words, "one question per turn" — §3 of `main_prompt.txt`), which is good: it keeps LLM-completion-to-first-audio latency low, but you still don't want to wait for the *whole* completion before synthesizing. Feed the LLM stream into a sentence splitter (Tamil/English code-mixed — do not use a naive `.` splitter, since English abbreviations like "Dr." and Tamil numerals are common in this script) and start TTS on the first complete clause.
- Voice cloning: cache the cloned-voice reference/embedding once at startup (same pattern as `IndicConformerAsr.load()` pre-loading the ASR model in the `lifespan` handler) — do not re-run voice cloning per call.
- Output at whatever native rate svara-TTS produces, then resample once, centrally, in the output-transport layer (see 3.3) rather than inside `tts.py` — keeps the TTS adapter transport-agnostic (same code path serves both the 16kHz browser socket and the 8kHz SIP leg).
- **Barge-in**: the moment VAD emits a fresh `speech_started` while the agent is speaking, the orchestrator must cancel the in-flight TTS generation and stop sending further audio frames immediately (flush the outbound queue). This is a hard product requirement for a phone-call agent — nothing in the current code models "the agent is currently talking," so this is new state, not a small addition.

### 3.3 SIP / WebRTC telephony gateway (P0 — currently 0% built)
The current transport is a browser `WebSocket` only. There is no way for a real phone call to
reach this system. Two realistic paths, pick one for v1:

- **Option A — managed media streams (fastest to ship):** Use a SIP trunk provider whose product
  can bridge PSTN calls to a WebSocket you control (e.g. Twilio Media Streams, Plivo Audio
  Streams, Exotel Voice Streaming — several Indian telephony providers now offer this and matter
  for Tamil/India call routing and DID numbers specifically). You get PCM/u-law frames over a
  WebSocket with a similar shape to what `main.py` already parses — the `/ws/audio` handler is a
  good starting point to adapt, not throw away.
- **Option B — self-hosted SIP (more control, more work):** A SIP/RTP endpoint via
  `drachtio` + `rtpengine`, or FreeSWITCH/Asterisk with a custom media application, terminating
  SIP from a carrier trunk and handing raw RTP audio to this same Python service over a local
  socket. This avoids per-minute vendor media-stream fees at scale but is materially more
  infrastructure to operate and secure (SBC, NAT traversal, RTP codecs — G.711 u-law is the
  standard 8kHz telephony codec you'll be transcoding to/from).
- Either way: **the 8kHz requirement in your target flow is a telephony-codec constraint** (G.711
  is 8kHz), while VAD/ASR here are hard-pinned to 16kHz (`settings.py` comments say so explicitly
  for both). That means a resampling step is mandatory on both legs — inbound SIP audio (8kHz) —
  upsample to 16kHz before VAD/ASR, and outbound TTS audio — downsample to 8kHz before it goes
  back over SIP/RTP. Keep this resampling in one shared module (`backend/audio_transcode.py`) so
  both directions and both transports (browser 16kHz passthrough, SIP 8kHz) reuse it instead of
  duplicating resampling logic per transport.
- Add a new route/handler analogous to `capture_browser_audio` for the SIP/media-stream leg —
  it should share the VAD/ASR/orchestrator pipeline, differing only in the transport-specific
  framing (SIP media stream payload vs raw WS bytes) and the resampling step above.

### 3.4 Streaming ASR (upgrade from utterance-final)
ASR runs *after* `vad_end` closes a turn. For a natural
phone conversation you generally want at least **partial/interim transcripts** so the
Conversation Manager can start reasoning before the caller finishes a long sentence, especially
given the golden prompt allows longish turns from callers. This is a real rework, not a small
patch, because NeMo's `transcribe()` as used today is a single batched call. Two pragmatic
options:
1. Keep final-only transcription (simplest, matches current well-tested code) for v1, and rely on
   the VAD's `endpoint_silence_frames` (currently 480ms) tuning to keep perceived latency low —
   this is a legitimate v1 scope cut, just document it as one rather than silently shipping it.
2. If true partial transcripts are required, IndicConformer's CTC branch (already available via
   `ASR_DECODING=ctc`, see `settings.py`) is fast enough to run incrementally on a rolling buffer
   for interim results, while the more accurate RNNT decode still runs once at `vad_end` for the
   final transcript sent to the LLM. This is meaningfully more complex — recommend deferring
   until v1 latency is measured and found wanting. This is now implemented and on by default;
   measured on real calls, ASR is not where the latency is (see HANDOFF.md §4).

### 3.5 Concurrency: the ASR lock will bottleneck the whole call center
`main.py` uses one `app.state.asr_lock` shared across **all** concurrent WebSocket connections
(line 35, 120). That means two simultaneous calls' ASR work is serialized on the same lock today
— already a scaling limit even before LLM/TTS exist. Once GPU-bound LLM inference and TTS
synthesis are added, per-call serialization on a single global lock will not survive more than a
handful of concurrent calls. Plan for:
- A small pool of ASR/LLM/TTS worker slots (bounded by GPU memory), with calls queued fairly
  rather than one global mutex.
- Or move ASR/LLM/TTS behind dedicated inference servers (vLLM for the LLM, a Triton/BentoML-style
  server for ASR+TTS) so `main.py` becomes a thin orchestration/transport layer that issues async
  network calls instead of holding blocking model calls itself. This is the more scalable choice
  and also decouples the Python 3.10/3.11 NeMo constraint from whatever the LLM/TTS servers need.

### 3.6 Persistence & the golden-flow eval harness
- `golden/flows/flow_*.txt` (20 files) + `golden/main_prompt.txt` are currently inert reference
  data. Build a small offline eval script that replays each flow's caller turns against the
  live Conversation Manager + LLM and diffs the agent's behavior against the flow's documented
  `OUTCOME`/entities — this is exactly what the golden set is for, and it's the natural
  regression test for prompt/model changes going forward. `golden/tamil_english_call_flows.pdf`
  is presumably the source-of-truth narrative version of the same 20 flows — worth confirming it
  and the `.txt` files agree.
- Persist call events (the `vad_*`/`transcript`/tool-call/agent-turn events) to a database, not
  just over the WebSocket. Right now nothing survives a disconnect — there is no call history at
  all. This is also the missing link to the frontend: its entire Call Log / Dashboard / Simulation
  UI (see the companion frontend report) expects exactly this kind of persisted call record and
  currently has nothing to read.

### 3.7 Testing gaps introduced by the new work
The existing `test_vad.py`/`test_asr.py` pattern (stub the model, test orchestration logic) is
good — continue it:
- `test_conversation.py`: fake LLM client returning scripted tool calls, assert ledger updates,
  no-re-ask behavior, and flow-switch handling.
- `test_tts.py`: fake TTS client, assert sentence-chunking boundaries and barge-in cancellation.
- Golden-flow replay harness doubles as an integration test suite.

---

## 4. Security / compliance (not yet addressed anywhere in the repo)
This is a hospital call agent handling patient identity, DOB, medical records, and billing —
real PII/PHI-adjacent data. None of the following exists yet and all of it should land before any
real phone number is connected:
- No auth on `/ws/audio` at all today — anyone who can reach the port can open a session.
- No encryption-at-rest story for the ledger/call-recording data once persistence (3.6) lands.
- No mention of consent/recording-disclosure for calls, relevant given `main_prompt.txt`'s
  emergency and clinical flows.
- `.env`-based `HF_TOKEN` handling is fine for local dev; production secrets (LLM API keys, TTS
  keys, SIP trunk credentials, DB creds) need a real secrets manager, not `.env` files, once this
  moves off a laptop.

## 5. Dependency/runtime constraint to carry forward
`run.sh` and `README.md` are explicit that AI4Bharat's NeMo fork hard-pins Python to 3.10/3.11.
Whatever you choose for the LLM serving stack (vLLM, TGI) and SIP stack (drachtio, FreeSWITCH
bindings) should be evaluated against this constraint *or* isolated into a separate service/
container so ASR isn't blocking your choice of LLM/TTS/SIP tooling versions. Running ASR in its
own container and everything else in a modern Python (3.12+) container is probably the pragmatic
answer rather than fighting the NeMo pin project-wide.

## 6. Delivery status

1. ~~Conversation Manager + LLM (mock tools, in-process ledger), text-only loop~~ — **done.**
   The ledger is now actually rendered into the prompt, which it was not before (see §1).
2. ~~Streaming TTS wired to the browser WS output~~ — **done**, and upgraded past the
   original plan: the LLM is consumed as a token stream and each clause is synthesized as
   it closes, so audio starts before the turn finishes rather than after.
3. ~~Barge-in~~ — **done.** Cancelling now also cancels the LLM generation, and the
   truncated turn is written back to history so the model knows what the caller heard.
4. ~~Persistence + golden-flow eval harness~~ — **done**, plus `GET /api/calls` /
   `/api/calls/{id}` / `/api/health`, and `e2e_check.py` which drives a running server
   with synthesized caller speech.
5. SIP/WebRTC telephony gateway — **out of scope.** `backend/telephony.py` remains a
   code-complete, unit-tested adapter; nothing connects it to a trunk, and nothing should
   until the items below are settled. Testing is manual, through the console.
6. Concurrency/scaling rework — bounded semaphore pools are in; not load-tested.
7. Security hardening — the WS auth token and at-rest encryption exist and are off by
   default. Both must be on before any real number is live.

### What is genuinely still open

- **Model quality.** This is now the binding constraint, not the plumbing. Turn
  discipline (one question per turn) and English-caller mirroring are the weakest areas;
  `register_eval` scores them on unseen scenarios and is the number to watch.
- **Safety refusals now have an eval, not yet a verdict.**
  `backend/scripts/safety_eval.py` puts escalating pressure on the three
  highest-consequence behaviours — refusing to read or grade a lab value, refusing to
  name or rule out a diagnosis, refusing to authorise aspirin during chest pain — and its
  detectors are unit-tested in both directions. What it cannot do is read Tamil: a
  violation it reports is real, but silence means only that nothing obviously wrong was
  said, so a native reader still has to read the transcripts.
- **A self-hosted TTS engine.** `TTS_ENGINE=edge` sends the text to be spoken to
  Microsoft. Acceptable for the fictional Aruvi data, not for real patient speech.
- **Grounding covers identifiers only.** `backend/grounding.py` can decide whether an ID
  or phone number was invented. It cannot check an invented doctor name, price, room or
  date — those still rest on the prompt alone.
