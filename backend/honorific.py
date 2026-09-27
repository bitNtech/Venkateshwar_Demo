"""Picks சார் or மேடம் for the caller from the pitch of their voice.

The prompt requires one honorific per call and never a switch (runtime_core.txt
LANGUAGE), but nothing on a browser call tells the server who is speaking - no
CRM, no caller ID. Voice pitch is the one signal available, and it is a
HEURISTIC: an elderly woman, a hoarse caller or a child can sit on the wrong
side of any line. So the detector is built to say nothing rather than guess:

  - it decides only once it has heard enough voiced speech to be stable,
  - only when the median pitch sits clearly outside the overlap band, and
  - once decided it never changes, matching the prompt's no-switch rule.

Undecided is the normal, safe outcome - callers are then addressed without an
honorific (see filler.py), which is never wrong, whereas the wrong one is.

Only utterances that survived the self-echo check are fed in, because the
agent's own voice (ta-IN-PallaviNeural, female) leaking back through the
speakers would otherwise vote மேடம் on every call.
"""

from __future__ import annotations

import numpy as np

SIR = "சார்"
MADAM = "மேடம்"

# 40 ms analysis window at a 20 ms hop: long enough to hold two periods of a
# 70 Hz voice, short enough that pitch is roughly steady within it.
_FRAME = 640
_HOP = 320

# The search range for a speaking voice's fundamental. Wider than any adult's
# speaking range on purpose - clamping it tighter would push outliers INTO the
# decision band instead of letting them fail the voicing test.
_F0_MIN_HZ = 70.0
_F0_MAX_HZ = 350.0

# YIN (de Cheveigne & Kawahara 2002). A frame's period is the FIRST lag whose
# cumulative-mean-normalised difference dips below this; frames with no such
# dip are unvoiced and skipped.
#
# Replaced an autocorrelation picker that took the shortest near-best lag.
# That rule read MALE voices high once a headset/laptop mic had rolled off the
# fundamental - measured with a 4th-order high-pass at 350 Hz: hi-IN-Madhur
# 112 -> 165 Hz, ml-IN-Midhun 101 -> 125 Hz - and a real male tester on a
# Realtek headset was addressed as மேடம். YIN on the same audio: 109 -> 110 Hz
# and 98 -> 115 Hz, and no voice of 18 crossed a decision line under any of
# four filters (clean, 150, 250, 350 Hz).
_YIN_THRESHOLD = 0.15

# Only frames this loud relative to the utterance's own loud frames are
# analysed, so trailing silence and a soft room never contribute a pitch.
_RELATIVE_LEVEL = 0.3

# Decision band; between the two lines nothing is decided. Measured with the
# YIN estimator below on 18 edge neural voices (9 male, 9 female; ta-IN/MY/SG/
# LK plus en/hi/te/ml/kn-IN), one ~6 s sentence each, median F0, clean and
# through a 150/250/350 Hz high-pass standing in for small mics:
#
#     male    98 - 170 Hz   (Tamil male voices sit HIGH: 142 - 170)
#     female  191 - 267 Hz
#
# Of those 72 runs: 67 decided correctly, 5 undecided (low male voices under
# the 250/350 Hz filters left too few voiced frames in one sentence - a real
# call keeps accumulating), and NONE wrong. The male line is 170 rather than a
# textbook 155 because the Tamil male voices sit that high. These are
# synthetic voices: re-measure on real callers - main.py logs every
# utterance's median pitch ("pitch for ...") for exactly that.
MALE_MAX_HZ = 170.0
FEMALE_MIN_HZ = 185.0

# ~1 s of voiced speech (50 x 20 ms hops) before any decision. One short
# "ஆமாம்" is not enough to judge a voice by.
MIN_VOICED_FRAMES = 50

# Enough history for a stable median; older frames add nothing.
_MAX_FRAMES = 500


def estimate_pitches(samples: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
    """Per-frame F0 in Hz for the voiced frames of `samples` (int16 or float), by YIN."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.size < _FRAME:
        return np.empty(0, dtype=np.float32)
    count = 1 + (audio.size - _FRAME) // _HOP
    frames = np.lib.stride_tricks.sliding_window_view(audio, _FRAME)[::_HOP][:count]
    frames = frames - frames.mean(axis=1, keepdims=True)

    rms = np.sqrt(np.mean(frames * frames, axis=1))
    loud = (rms > 0) & (rms >= _RELATIVE_LEVEL * np.percentile(rms, 90))
    frames = frames[loud]
    if not len(frames):
        return np.empty(0, dtype=np.float32)

    lo = int(sample_rate / _F0_MAX_HZ)
    hi = int(sample_rate / _F0_MIN_HZ)
    width = _FRAME // 2  # integration window; width + hi must fit in _FRAME
    lags = np.arange(hi + 2)

    # d(t) = sum_j (x_j - x_{j+t})^2 over the window, expanded as
    # E(head) + E(shifted window) - 2 * cross-correlation, all per frame.
    head = frames[:, :width]
    size = 2 * _FRAME
    cross = np.fft.irfft(
        np.conj(np.fft.rfft(head, n=size, axis=1)) * np.fft.rfft(frames, n=size, axis=1), n=size, axis=1
    )[:, : hi + 2]
    energy = np.concatenate([np.zeros((len(frames), 1)), np.cumsum(frames * frames, axis=1)], axis=1)
    shifted = energy[:, lags + width] - energy[:, lags]
    diff = np.maximum(energy[:, width : width + 1] + shifted - 2 * cross, 0.0)

    # Cumulative-mean normalisation: what makes the first dip, not the
    # deepest, the period - and what keeps a strong 2nd harmonic from winning.
    cmnd = np.ones_like(diff)
    running = np.cumsum(diff[:, 1:], axis=1)
    cmnd[:, 1:] = diff[:, 1:] * lags[1:] / np.maximum(running, 1e-9)

    pitches = []
    for row in cmnd:
        below = np.flatnonzero(row[lo : hi + 1] < _YIN_THRESHOLD)
        if not below.size:
            continue
        tau = lo + int(below[0])
        while tau + 1 <= hi and row[tau + 1] < row[tau]:
            tau += 1
        # Parabolic refinement between integer lags.
        a, b, c = row[tau - 1], row[tau], row[tau + 1]
        denom = a - 2 * b + c
        if denom > 0:
            tau = tau + 0.5 * (a - c) / denom
        pitches.append(sample_rate / tau)
    return np.asarray(pitches, dtype=np.float32)


class HonorificDetector:
    """Accumulates the caller's pitch across a call and commits once, if ever."""

    def __init__(self) -> None:
        self._pitches: list[float] = []
        self._decided: str | None = None

    @property
    def honorific(self) -> str | None:
        """சார், மேடம், or None while the voice is still ambiguous."""
        return self._decided

    def observe(self, pitches: np.ndarray) -> str | None:
        """Add one utterance's voiced-frame pitches; return the honorific if known."""
        if self._decided is not None:
            return self._decided
        self._pitches.extend(float(p) for p in pitches)
        del self._pitches[:-_MAX_FRAMES]
        if len(self._pitches) < MIN_VOICED_FRAMES:
            return None
        median = float(np.median(self._pitches))
        if median <= MALE_MAX_HZ:
            self._decided = SIR
        elif median >= FEMALE_MIN_HZ:
            self._decided = MADAM
        return self._decided
