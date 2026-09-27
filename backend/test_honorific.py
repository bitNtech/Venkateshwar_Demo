"""Self-check for the pitch-based சார்/மேடம் detector.

Synthetic voiced signals (a fundamental plus decaying harmonics, like a
glottal source) stand in for real speech. The detector's calibration against
real neural voices is recorded in honorific.py itself.
"""

from __future__ import annotations

import numpy as np

from .honorific import MADAM, MIN_VOICED_FRAMES, SIR, HonorificDetector, estimate_pitches

RATE = 16_000


def _voice(f0: float, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    wave = sum((0.8**k) * np.sin(2 * np.pi * f0 * (k + 1) * t) for k in range(8))
    return (wave / np.abs(wave).max() * 8000).astype(np.int16)


def test_pitch_is_read_at_the_fundamental_not_an_octave_below() -> None:
    for f0 in (110.0, 150.0, 220.0, 260.0):
        pitches = estimate_pitches(_voice(f0), RATE)
        assert len(pitches) > MIN_VOICED_FRAMES
        # The octave error - reading 220 Hz as 110 Hz - would call a woman சார்.
        assert abs(np.median(pitches) - f0) / f0 < 0.03, f0


def test_silence_and_noise_produce_no_pitch() -> None:
    assert len(estimate_pitches(np.zeros(RATE, dtype=np.int16), RATE)) == 0
    noise = np.random.default_rng(0).normal(0, 300, RATE).astype(np.int16)
    assert len(estimate_pitches(noise, RATE)) < 5


def test_a_low_voice_is_addressed_as_sir_and_a_high_one_as_madam() -> None:
    low, high = HonorificDetector(), HonorificDetector()

    assert low.observe(estimate_pitches(_voice(125.0), RATE)) == SIR
    assert high.observe(estimate_pitches(_voice(225.0), RATE)) == MADAM


def test_an_ambiguous_voice_gets_no_honorific() -> None:
    detector = HonorificDetector()

    assert detector.observe(estimate_pitches(_voice(177.0), RATE)) is None
    assert detector.honorific is None


def test_one_short_word_is_not_enough_to_judge_a_voice() -> None:
    detector = HonorificDetector()

    assert detector.observe(estimate_pitches(_voice(225.0, seconds=0.4), RATE)) is None
    # ...but it accumulates across the call's utterances.
    assert detector.observe(estimate_pitches(_voice(225.0, seconds=1.5), RATE)) == MADAM


def test_once_decided_the_honorific_never_switches_within_a_call() -> None:
    detector = HonorificDetector()
    detector.observe(estimate_pitches(_voice(125.0), RATE))

    # A different voice later in the call (a relative takes the phone) does
    # not flip it: the prompt forbids switching mid-call.
    assert detector.observe(estimate_pitches(_voice(240.0, seconds=6.0), RATE)) == SIR


def test_a_male_voice_through_a_small_mic_is_not_read_an_octave_high() -> None:
    """Found live: a male tester on a Realtek headset was addressed as மேடம்.

    A small mic rolls off a male voice's fundamental, which leaves the even
    harmonics dominant and the waveform looking nearly periodic at HALF its
    true period. Here: a 115 Hz voice, fundamental removed, odd harmonics at
    half strength. The previous autocorrelation picker read this as 232 Hz
    (மேடம்); YIN reads the true period.

    Not bulletproof, and deliberately not pretended to be: with the odd
    harmonics much weaker still (under ~0.3) YIN also reads the octave, because
    the signal then genuinely IS closer to periodic at 230 Hz.
    """
    t = np.arange(RATE * 3) / RATE
    wave = sum(
        (0.5 if (k + 1) % 2 else 1.0) * (0.8**k) * np.sin(2 * np.pi * 115.0 * (k + 1) * t)
        for k in range(1, 10)
    )
    small_mic = (wave / np.abs(wave).max() * 8000).astype(np.int16)

    pitches = estimate_pitches(small_mic, RATE)

    assert abs(np.median(pitches) - 115.0) < 4
    assert HonorificDetector().observe(pitches) == SIR
