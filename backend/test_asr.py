"""Self-check for IndicConformerAsr's segment -> NeMo call -> transcript path.

Stubs the NeMo model so the normalization, call arguments and result
unwrapping are exercised without downloading IndicConformer.
"""

from __future__ import annotations

import numpy as np

from .asr import IndicConformerAsr
from .settings import AudioSettings


class _FakeModel:
    """Records how NeMo was called, including the audio it was handed."""

    def __init__(self, result) -> None:
        self.result = result
        self.calls: list[dict] = []

    def transcribe(self, audio, batch_size, language_id, verbose):
        self.calls.append(
            {
                "audio": audio,
                "batch_size": batch_size,
                "language_id": language_id,
                "verbose": verbose,
                # Recorded at call time so tests can see which decoding
                # branch was selected *during* this specific call, even
                # though asr.py restores cur_decoder afterwards.
                "cur_decoder": getattr(self, "cur_decoder", None),
            }
        )
        return self.result


def _make_asr(result) -> tuple[IndicConformerAsr, _FakeModel]:
    asr = IndicConformerAsr(AudioSettings())
    model = _FakeModel(result)
    asr._model = model
    return asr, model


def test_empty_segment_short_circuits() -> None:
    asr, model = _make_asr(["ignored"])
    assert asr.transcribe(np.array([], dtype=np.int16), "ta") == ""
    assert model.calls == []


def test_unloaded_model_raises() -> None:
    asr = IndicConformerAsr(AudioSettings())
    try:
        asr.transcribe(np.zeros(160, dtype=np.int16), "ta")
    except RuntimeError:
        return
    raise AssertionError("transcribe must refuse to run before load()")


def test_segment_is_passed_in_memory_with_reference_call_args() -> None:
    samples = (np.sin(np.arange(1600) / 8.0) * 20_000).astype(np.int16)
    asr, model = _make_asr(["வணக்கம்"])

    assert asr.transcribe(samples, "ta") == "வணக்கம்"

    call = model.calls[0]
    assert call["batch_size"] == 1 and call["language_id"] == "ta"
    assert call["verbose"] is False, "the per-call tqdm bar is pure latency"
    # One float32 waveform, normalized exactly like the reference pipeline.
    audio = call["audio"]
    assert len(audio) == 1 and audio[0].dtype == np.float32
    assert np.array_equal(audio[0], samples.astype(np.float32) / 32768.0)


def test_native_script_transcript_is_returned_unchanged() -> None:
    asr, _ = _make_asr(["  வணக்கம் நண்பரே  "])
    assert asr.transcribe(np.zeros(320, dtype=np.int16), "ta") == "வணக்கம் நண்பரே"


def test_nested_result_is_unwrapped_and_stripped() -> None:
    asr, _ = _make_asr([["  வணக்கம்  "], ["ctc-branch"]])
    assert asr.transcribe(np.zeros(320, dtype=np.int16), "ta") == "வணக்கம்"


def test_empty_result_list_is_empty_string() -> None:
    asr, _ = _make_asr([])
    assert asr.transcribe(np.zeros(320, dtype=np.int16), "ta") == ""


def test_transcribe_uses_configured_decoding_not_hardcoded_ctc() -> None:
    """transcribe() must keep selecting rnnt vs ctc via self.settings.decoding
    (applied through cur_decoder at load() time) - it must never force ctc
    itself, that is transcribe_partial()'s job only."""
    asr = IndicConformerAsr(AudioSettings(decoding="rnnt"))
    model = _FakeModel(["வணக்கம்"])
    model.cur_decoder = "rnnt"  # what load() would have set for this settings.decoding
    asr._model = model

    assert asr.settings.decoding == "rnnt"
    assert asr.transcribe(np.zeros(320, dtype=np.int16), "ta") == "வணக்கம்"

    call = model.calls[0]
    assert call["cur_decoder"] == "rnnt", "transcribe() must not override the configured decoder"
    # And the model is left exactly as transcribe() found it - no ctc override leaking out.
    assert model.cur_decoder == "rnnt"


def test_transcribe_partial_empty_segment_short_circuits() -> None:
    asr, model = _make_asr(["ignored"])
    assert asr.transcribe_partial(np.array([], dtype=np.int16), "ta") == ""
    assert model.calls == []


def test_transcribe_partial_unloaded_model_raises() -> None:
    asr = IndicConformerAsr(AudioSettings())
    try:
        asr.transcribe_partial(np.zeros(160, dtype=np.int16), "ta")
    except RuntimeError:
        return
    raise AssertionError("transcribe_partial must refuse to run before load()")


def test_transcribe_partial_forces_ctc_regardless_of_configured_decoding() -> None:
    """Sec3.4 option 2: interim decode always uses ctc, even when the final,
    vad_end transcribe() path is configured for the slower/more-accurate rnnt."""
    asr = IndicConformerAsr(AudioSettings(decoding="rnnt"))
    model = _FakeModel(["இது ஒரு"])
    model.cur_decoder = "rnnt"  # simulate load()'s decoding-mode setup
    asr._model = model

    samples = (np.sin(np.arange(800) / 8.0) * 20_000).astype(np.int16)
    assert asr.transcribe_partial(samples, "ta") == "இது ஒரு"

    call = model.calls[0]
    assert call["cur_decoder"] == "ctc", "transcribe_partial must force ctc for its own call"
    assert call["batch_size"] == 1 and call["language_id"] == "ta"
    assert call["verbose"] is False
    audio = call["audio"]
    assert len(audio) == 1 and audio[0].dtype == np.float32
    assert np.array_equal(audio[0], samples.astype(np.float32) / 32768.0)

    # The override must not leak into the model's state for later calls -
    # a subsequent transcribe() at vad_end must see rnnt again, untouched.
    assert model.cur_decoder == "rnnt"
    assert asr.settings.decoding == "rnnt"


def test_transcribe_partial_result_unwrapping_matches_transcribe() -> None:
    asr, _ = _make_asr([["  வணக்கம்  "], ["ctc-branch"]])
    assert asr.transcribe_partial(np.zeros(320, dtype=np.int16), "ta") == "வணக்கம்"


if __name__ == "__main__":
    test_empty_segment_short_circuits()
    test_unloaded_model_raises()
    test_segment_is_passed_in_memory_with_reference_call_args()
    test_native_script_transcript_is_returned_unchanged()
    test_nested_result_is_unwrapped_and_stripped()
    test_empty_result_list_is_empty_string()
    test_transcribe_uses_configured_decoding_not_hardcoded_ctc()
    test_transcribe_partial_empty_segment_short_circuits()
    test_transcribe_partial_unloaded_model_raises()
    test_transcribe_partial_forces_ctc_regardless_of_configured_decoding()
    test_transcribe_partial_result_unwrapping_matches_transcribe()
    print("ok")


def test_model_inference_is_serialized_across_threads() -> None:
    """KeyError('logits'), observed live and reported by the user.

    NeMo's hybrid model keeps its ACTIVE DECODER on the model object.
    transcribe_partial() flips `cur_decoder` to ctc and back, while the final
    rnnt decode can be running on the same object from another thread -
    main.py starts interim decodes during an utterance and the final one the
    moment it ends. Without a lock the two interleave, the decode fails, and
    the caller's entire turn is lost.
    """
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    class _ConcurrentModel(_FakeModel):
        def __init__(self) -> None:
            super().__init__(["வணக்கம்"])
            self.active = 0
            self.max_active = 0
            self.decoders_seen: list = []
            self.guard = threading.Lock()

        def transcribe(self, audio, batch_size, language_id, verbose):
            with self.guard:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.decoders_seen.append(getattr(self, "cur_decoder", None))
            time.sleep(0.02)
            with self.guard:
                self.active -= 1
            return self.result

    asr = IndicConformerAsr(AudioSettings())
    model = _ConcurrentModel()
    asr._model = model
    samples = np.ones(320, dtype=np.int16)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = []
        for index in range(8):
            target = asr.transcribe_partial if index % 2 else asr.transcribe
            futures.append(pool.submit(target, samples, "ta"))
        for future in futures:
            future.result()

    assert model.max_active == 1, (
        f"{model.max_active} decodes ran at once - the decoder race is back"
    )
