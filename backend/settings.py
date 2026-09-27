"""Runtime settings for the low-latency audio pipeline.

Defaults are the hyperparameters from README.md, which is the configuration
the live mic pipeline was actually tuned on - do not "improve" them casually.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

# .env (gitignored) holds HF_TOKEN and every override in this file - see
# .env.example, which documents all of them.
#
# The load that matters is in backend/__init__.py, which runs before any
# module reads a default; this one is a no-op in the normal path (load_dotenv
# never overrides a name already set).
load_dotenv()


# Languages IndicConformer accepts as `language_id`.
SUPPORTED_LANGUAGES = frozenset({"ta", "hi", "te", "ml", "kn", "bn", "mr", "gu", "pa"})


@dataclass(frozen=True)
class AudioSettings:
    """Tuned for conversational turn-taking, not long-form batch transcription."""

    # Must stay 16 kHz: both TEN VAD and IndicConformer are trained on it.
    sample_rate: int = 16_000
    vad_hop_size: int = 256  # 16 ms at 16 kHz; TEN VAD is tuned around this.

    # Lower = catches softer/quieter onsets but more false triggers on noise.
    vad_threshold: float = float(os.getenv("VAD_THRESHOLD", "0.35"))

    # TEN VAD flags speech per 16 ms hop, and ONE positive hop is not a turn.
    # Measured over the 87 real captured turns in call_events.db, 46 of them
    # transcribed to 3 characters or fewer and 34 to the EMPTY STRING - i.e.
    # 39% of everything the VAD opened contained no speech at all. Requiring
    # 4 consecutive positive hops (64 ms) before a turn opens removes the
    # shortest of those without costing a real caller anything: the candidate
    # frames are kept and prepended, so confirming an onset never clips the
    # first syllable.
    #
    vad_start_frames: int = int(os.getenv("VAD_START_FRAMES", "4"))

    # LOUDNESS AT ONSET. Only speech this loud may OPEN a turn, so a
    # television, a fan or a conversation across the room no longer starts one.
    #
    # Read this before touching it. An energy gate was added here once before
    # and reverted, because it gated EVERY frame: a quiet trailing syllable
    # scored as "silence", the endpoint countdown ran on through the middle of
    # a word, and turns came back as one-character transcripts ("ந", "ப", "க").
    # The lesson was not that loudness is unusable, it is that loudness must
    # never be allowed to END a turn - only to refuse to start one. So these
    # two knobs are read in exactly one place, the not-yet-in-speech branch of
    # TenVadSegmenter.process(). Once a turn is open, endpointing is decided by
    # the VAD flag alone, and no quiet syllable can cut it short.
    #
    # Measured, per 16 ms frame of real Tamil speech at full digital level:
    #
    #     voiced p10   931      voiced p50  2624     peak  9804
    #
    # so 200 sits about 4.6x below the quietest speech frame - low enough to
    # be a backstop rather than a filter, because microphone gain varies wildly
    # between machines and an absolute number cannot be right for all of them.
    # The SNR term is the part that actually adapts: the room's noise floor is
    # learned continuously while nobody is speaking, and onset has to beat a
    # multiple of it. Raise VAD_ONSET_SNR first if a noisy room still opens
    # turns; raise VAD_ONSET_MIN_RMS only if the microphone is unusually hot.
    vad_onset_min_rms: float = float(os.getenv("VAD_ONSET_MIN_RMS", "200"))
    vad_onset_snr: float = float(os.getenv("VAD_ONSET_SNR", "3.0"))

    # THE WATCHDOG. A turn ends when nothing LOUD has arrived for this long,
    # whatever the VAD is flagging. 44 x 16 ms = 704 ms.
    #
    # Without it a turn can be held open forever, and this is not theoretical:
    # the endpoint countdown is restarted by any sustained run of flagged
    # frames, and once the agent has spoken once, residual echo and the
    # browser's automatic gain control produce exactly such runs out of an
    # empty room. Observed live - the first turn of a call endpointed normally
    # and every turn after it listened until the 30 s hard cap.
    #
    # Deliberately TWICE endpoint_silence_frames, not equal to it. Setting the
    # two equal would make a quiet flagged frame identical to silence, which is
    # the reverted behaviour that cut turns off mid-word. Double gives a
    # trailing syllable room to be quiet without giving noise room to hold the
    # microphone open. Someone actually speaking refreshes this constantly.
    vad_quiet_endpoint_frames: int = int(os.getenv("VAD_QUIET_ENDPOINT_FRAMES", "44"))

    # How fast the learned room-noise floor follows the room. Updated only
    # while no turn is open AND the VAD says the frame is not speech, so a
    # caller talking can never raise the bar against themselves.
    vad_noise_ema: float = float(os.getenv("VAD_NOISE_EMA", "0.05"))

    # Once the endpoint countdown has begun, only a sustained run of speech
    # restarts it. Resetting on a SINGLE positive hop is what let background
    # noise hold the microphone open indefinitely - one blip inside every
    # 352 ms window and the turn never closes, which is why a caller ends up
    # toggling their mic by hand after speaking. Real speech produces long
    # runs and clears this instantly; an isolated blip now costs one frame.
    vad_resume_frames: int = int(os.getenv("VAD_RESUME_FRAMES", "3"))

    # 8 x 16 ms = 128 ms kept *before* VAD fires, so VAD onset lag doesn't
    # clip the first syllable.
    pre_roll_frames: int = int(os.getenv("VAD_PRE_ROLL_FRAMES", "8"))

    # 22 x 16 ms = 352 ms of trailing silence before the turn closes. This sits
    # directly in front of every reply the caller hears - it is spent before the
    # ASR has even started - so it is part of the latency budget, not free.
    #
    # Was 30 (480 ms). Lowered because 480 ms is past the point where a listener
    # starts to feel talked-at-by-a-machine: human turn-taking gaps cluster
    # around 200 ms. Not lowered further than 352 ms on purpose - below roughly
    # 300 ms a natural mid-sentence breath starts closing the turn, and the ASR
    # gets half a sentence, which costs far more than the 150 ms it saves.
    #
    # Raise it back with VAD_ENDPOINT_SILENCE_FRAMES if callers who pause to
    # think are being cut off; that failure looks like the agent answering a
    # question the caller had not finished asking.
    endpoint_silence_frames: int = int(os.getenv("VAD_ENDPOINT_SILENCE_FRAMES", "22"))

    # ponytail: not in the reference CLI, which is a trusted local mic. A
    # browser socket is not: without a cap, a stuck speech flag buffers
    # forever. 30 s is far past any real turn, so it never truncates one.
    max_utterance_frames: int = int(os.getenv("ASR_MAX_UTTERANCE_FRAMES", "1875"))

    # MEASUREMENT ONLY, off by default. When > 0, every this-many ms of call
    # audio the learned noise floor, the onset bar it implies, the room level
    # and how many flagged hops the loudness gate refused are written to the
    # call log (never to the client). Read back with
    # `python -m backend.scripts.noise_floor_report`. Exists to test one
    # theory before anything above is retuned: that browser AGC lifts the
    # "quiet" frames _update_noise_floor() learns from, so the floor climbs
    # over a call and onset gets progressively harder. 500 is plenty.
    vad_diagnostics_ms: int = int(os.getenv("VAD_DIAGNOSTICS_MS", "0"))

    language: str = os.getenv("ASR_LANGUAGE", "ta")

    # rnnt is slower than ctc but more accurate on this hybrid model -
    # correctness over latency for transcript quality.
    decoding: str = os.getenv("ASR_DECODING", "rnnt")

    # BACKEND_COMPLETION.md Sec3.5: one global asr_lock serializes every
    # concurrent call's ASR work, which won't scale past a handful of calls.
    # This bounds concurrent ASR inference to whatever the GPU can actually
    # hold rather than either (a) one-at-a-time or (b) fully unbounded.
    asr_max_concurrency: int = int(os.getenv("ASR_MAX_CONCURRENCY", "2"))

    def __post_init__(self) -> None:
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported ASR_LANGUAGE: {self.language}")
        if self.decoding not in {"ctc", "rnnt"}:
            raise ValueError("ASR_DECODING must be either 'ctc' or 'rnnt'")
        if self.asr_max_concurrency <= 0:
            raise ValueError("ASR_MAX_CONCURRENCY must be positive")
        if self.vad_start_frames <= 0 or self.vad_resume_frames <= 0:
            raise ValueError("VAD_START_FRAMES and VAD_RESUME_FRAMES must be positive")
        if self.vad_quiet_endpoint_frames <= self.endpoint_silence_frames:
            raise ValueError(
                "VAD_QUIET_ENDPOINT_FRAMES must exceed VAD_ENDPOINT_SILENCE_FRAMES - "
                "equal makes a quiet syllable indistinguishable from silence"
            )
        if self.vad_onset_min_rms < 0 or self.vad_onset_snr < 1:
            raise ValueError("VAD_ONSET_MIN_RMS must be >= 0 and VAD_ONSET_SNR must be >= 1")
        if not 0 < self.vad_noise_ema <= 1:
            raise ValueError("VAD_NOISE_EMA must be in (0, 1]")


_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SecuritySettings:
    """Access control for the audio WebSocket - BACKEND_COMPLETION.md Sec4.

    No auth existed at all before this: anyone who could reach the port could
    open a session. A shared-secret token is the minimum viable gate for a
    single-tenant v1 deployment; swap for real per-caller auth (SIP trunk
    identity, a signed browser session token, ...) once there is more than one
    trusted caller of this service. Blank token means auth is OFF, matching
    this repo's dev-friendly-by-default env-var pattern (see TTS_VOICE_REFERENCE_PATH)
    - do not deploy to a reachable network with this left blank.
    """

    ws_auth_token: str = os.getenv("AUDIO_WS_AUTH_TOKEN", "")

    # Browser origins allowed to call the /api/* routes. The React dashboard is
    # served from its OWN origin (a Vite dev server, or a static host in
    # production), so without this its fetch of /api/health and /api/calls is
    # blocked by the browser and the whole dashboard reads as "backend down".
    # The /console page is exempt by construction - it is served BY this app,
    # so it is same-origin and never involves CORS.
    #
    # Comma-separated, e.g. "http://localhost:5173,https://aruvi.example.com".
    # "*" is accepted for a closed network but is NOT the default: it would let
    # any page on the internet a staff browser happens to open read the call
    # log, which is patient data. Blank means no cross-origin caller is allowed
    # at all, which is correct for a console-only deployment.
    #
    # This does NOT gate /ws/audio. WebSocket handshakes are exempt from CORS
    # in every browser, so the socket's gate is AUDIO_WS_AUTH_TOKEN above -
    # setting one without the other leaves the audio path open.
    cors_allow_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
        if origin.strip()
    )


@dataclass(frozen=True)
class PersistenceSettings:
    """Config for the call-event/history store - BACKEND_COMPLETION.md Sec3.6.

    SQLite (a single file, no server to run) is enough for v1's single-process
    deployment; swap for a real DB once concurrency (Sec3.5) moves this out of
    one process.
    """

    db_path: Path = Path(os.getenv("CALL_EVENTS_DB_PATH", str(_REPO_ROOT / "call_events.db")))

    # Fernet key (base64-encoded, from Fernet.generate_key()) for encrypting
    # event payloads at rest - BACKEND_COMPLETION.md Sec4 flags no
    # encryption-at-rest story for the ledger/call-recording data; this is
    # what makes that story real once a real key is set. Blank means
    # encryption is OFF (plaintext payloads), matching this repo's
    # dev-friendly-by-default env-var pattern (see TTS_VOICE_REFERENCE_PATH /
    # AUDIO_WS_AUTH_TOKEN) - do not deploy to a reachable network with this
    # left blank once real patient PII/PHI is flowing through the store.
    encryption_key: str = os.getenv("CALL_EVENTS_ENCRYPTION_KEY", "")
