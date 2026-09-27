"""AI4Bharat IndicConformer transcription adapter.

Uses the same model and decoder as the reference live-mic pipeline: the hybrid
CTC/RNNT `.nemo` checkpoint loaded through the AI4Bharat NeMo fork
(`pip install -e ./NeMo_ai4bharat --no-deps`) - vanilla `nemo-toolkit` cannot
load its multilingual tokenizer.

The model emits Tamil script only, so English hospital words come back
transliterated and dictated numbers come back as number words. Both are
rewritten on the way out by transcript_norm.py - see _unwrap().
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

from .settings import AudioSettings
from .transcript_norm import normalize_transcript

logger = logging.getLogger("aica.asr")

MODEL_ID = "ai4bharat/indicconformer_stt_ta_hybrid_ctc_rnnt_large"
MODEL_FILE = "indicconformer_stt_ta_hybrid_rnnt_large.nemo"


class IndicConformerAsr:
    """Loads the IndicConformer checkpoint once and transcribes PCM segments."""

    def __init__(self, settings: AudioSettings) -> None:
        self.settings = settings
        self._model = None
        # THE DECODER IS SHARED MUTABLE STATE ON THE MODEL OBJECT.
        # transcribe_partial() flips `cur_decoder` to ctc and back while the
        # final rnnt decode may be running on the same object from another
        # thread - main.py runs interim decodes during an utterance and the
        # final one the moment it ends. The race surfaces as KeyError('logits')
        # and the caller's whole turn is lost. Observed live.
        #
        # An RLock rather than a Lock: transcribe_partial() holds it across a
        # try/finally that restores the decoder, and re-entering from the same
        # thread must not deadlock.
        self._inference_lock = threading.RLock()

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load model weights during startup, keeping call-time latency low."""
        import nemo.collections.asr as nemo_asr
        import torch
        from huggingface_hub import hf_hub_download

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("loading ASR model %s on %s", MODEL_ID, device)

        nemo_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename=MODEL_FILE,
            token=os.getenv("HF_TOKEN") or None,
        )
        model = nemo_asr.models.ASRModel.restore_from(nemo_path)
        model.freeze()
        self._model = model.to(device)
        self._model.cur_decoder = self.settings.decoding
        logger.info("ASR model loaded")

    @staticmethod
    def _normalize(samples: np.ndarray) -> np.ndarray:
        """int16 PCM -> float32 in [-1, 1], the waveform NeMo expects."""
        return samples.astype(np.float32) / 32768.0

    @staticmethod
    def _unwrap(result) -> str:
        """Flatten NeMo's nested result, then put it in the agent's register.

        This model has a Tamil character vocabulary, so an English hospital
        word can only come back transliterated ("appointment" ->
        "அப்பாயின்மென்ட்") and a dictated phone number comes back as Tamil
        NUMBER WORDS. Both are rewritten here rather than at one call site, so
        the interim transcript, the final transcript, the browser socket and
        the telephony leg all get the same text - see transcript_norm.py.
        """
        # NeMo returns a list of strings, sometimes nested per decoder.
        while isinstance(result, (list, tuple)):
            if not result:
                return ""
            result = result[0]
        return normalize_transcript(str(result).strip())

    def transcribe(self, samples: np.ndarray, language: str) -> str:
        """Return a native-script transcript for one 16 kHz mono int16 segment.

        Runs whatever decoding branch load() configured (self.settings.decoding,
        rnnt by default) - the accurate, final-transcript path used once at
        vad_end. See transcribe_partial() for the fast interim path.
        """
        if self._model is None:
            raise RuntimeError("ASR model is not loaded")

        if samples is None or len(samples) == 0:
            return ""

        # NeMo takes the waveform in memory, so a segment never touches disk:
        # no temp WAV write, no soundfile decode, no unlink per utterance.
        # verbose=False drops the per-call tqdm bar. transcribe() is already
        # wrapped in torch.no_grad() by NeMo.
        with self._inference_lock:
            result = self._model.transcribe(
                [self._normalize(samples)],
                batch_size=1,
                language_id=language,
                verbose=False,
            )
        return self._unwrap(result)

    def transcribe_raw(self, samples: np.ndarray, language: str) -> str:
        """The model's own output, with NO transcript normalisation applied.

        Only backend/scripts/build_asr_lexicon.py wants this. That script
        builds the normaliser's lexicon by round-tripping English words through
        TTS and back through this model, so it must see what the model
        actually emitted - feeding it normalised text would make the result
        depend on how much the normaliser already knows, and the lexicon would
        stop growing at whatever it happened to cover already.
        """
        if self._model is None:
            raise RuntimeError("ASR model is not loaded")
        result = self._model.transcribe(
            [self._normalize(samples)],
            batch_size=1,
            language_id=language,
            verbose=False,
        )
        while isinstance(result, (list, tuple)):
            if not result:
                return ""
            result = result[0]
        return str(result).strip()

    def transcribe_partial(self, samples: np.ndarray, language: str) -> str:
        """Fast interim transcript for a rolling, not-yet-endpointed buffer.

        BACKEND_COMPLETION.md Sec3.4 (option 2): forces the CTC decoding
        branch for this call only, regardless of self.settings.decoding -
        CTC is fast enough to run repeatedly on a growing buffer without
        accumulating latency, unlike the more accurate but slower RNNT
        branch transcribe() uses for the final result at vad_end.

        Stateless like transcribe(): callers own the rolling buffer and may
        call this repeatedly (e.g. once per new chunk of audio) while the
        caller is still mid-utterance. Does not touch self.settings.decoding
        and restores the model's decoder afterwards, so a subsequent
        transcribe() call is unaffected.
        """
        if self._model is None:
            raise RuntimeError("ASR model is not loaded")

        if samples is None or len(samples) == 0:
            return ""

        # cur_decoder is the same mechanism load() uses to select rnnt vs
        # ctc for transcribe() - flip it to ctc for just this call, then put
        # it back so transcribe()'s configured decoding mode is untouched.
        with self._inference_lock:
            previous_decoder = getattr(self._model, "cur_decoder", self.settings.decoding)
            self._model.cur_decoder = "ctc"
            try:
                result = self._model.transcribe(
                    [self._normalize(samples)],
                    batch_size=1,
                    language_id=language,
                    verbose=False,
                )
            finally:
                self._model.cur_decoder = previous_decoder

        return self._unwrap(result)
