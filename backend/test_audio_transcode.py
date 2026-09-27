"""Self-check for resample_pcm using synthetic sine-wave PCM.

No SIP/telephony infra or real audio files needed - a generated tone is enough
to check round-trip fidelity and output length, same spirit as test_vad.py's
synthetic frames.
"""

from __future__ import annotations

import numpy as np

from .audio_transcode import resample_pcm

ORIG_RATE = 16_000
TONE_HZ = 440
DURATION_S = 1.0


def _sine_pcm(sample_rate: int, duration_s: float = DURATION_S, freq_hz: float = TONE_HZ) -> np.ndarray:
    t = np.arange(int(sample_rate * duration_s)) / sample_rate
    waveform = 0.5 * np.sin(2 * np.pi * freq_hz * t)
    return (waveform * 32767).astype(np.int16)


def test_downsample_then_upsample_round_trip_is_close_to_original() -> None:
    original = _sine_pcm(ORIG_RATE)

    down = resample_pcm(original, ORIG_RATE, 8_000)
    back = resample_pcm(down, 8_000, ORIG_RATE)

    # Lengths can differ by a sample or two from rounding; trim to compare.
    n = min(len(original), len(back))
    original_f = original[:n].astype(np.float64)
    back_f = back[:n].astype(np.float64)

    # Round-trip through 8kHz drops content above 4kHz (irrelevant for a
    # 440Hz tone) but resampling filters still perturb phase/amplitude a
    # little, so correlation (not exact equality) is the right check.
    correlation = np.corrcoef(original_f, back_f)[0, 1]
    assert correlation > 0.99


def test_downsampling_halves_the_sample_count() -> None:
    original = _sine_pcm(ORIG_RATE)
    down = resample_pcm(original, ORIG_RATE, 8_000)

    expected = len(original) * 8_000 // ORIG_RATE
    assert abs(len(down) - expected) <= 1


def test_upsampling_doubles_the_sample_count() -> None:
    original = _sine_pcm(8_000)
    up = resample_pcm(original, 8_000, ORIG_RATE)

    expected = len(original) * ORIG_RATE // 8_000
    assert abs(len(up) - expected) <= 1


def test_passthrough_when_rates_match_returns_input_unchanged() -> None:
    original = _sine_pcm(ORIG_RATE)
    result = resample_pcm(original, ORIG_RATE, ORIG_RATE)

    # Same values, same dtype, and literally the same array (no copy, no
    # float round-trip) - proof the fast-path skipped resampling entirely.
    assert result is original
    assert result.dtype == np.int16


def test_empty_input_does_not_raise() -> None:
    empty = np.array([], dtype=np.int16)

    assert len(resample_pcm(empty, ORIG_RATE, 8_000)) == 0
    assert len(resample_pcm(empty, ORIG_RATE, ORIG_RATE)) == 0


if __name__ == "__main__":
    test_downsample_then_upsample_round_trip_is_close_to_original()
    test_downsampling_halves_the_sample_count()
    test_upsampling_doubles_the_sample_count()
    test_passthrough_when_rates_match_returns_input_unchanged()
    test_empty_input_does_not_raise()
    print("ok")
