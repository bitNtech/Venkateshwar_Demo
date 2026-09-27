"""Shared PCM resampling for cross-transport sample-rate conversion.

BACKEND_COMPLETION.md Sec3.3: VAD/ASR are hard-pinned to 16 kHz (see
settings.py's AudioSettings.sample_rate comment) while the telephony leg's
codec (G.711, the SIP/RTP standard) is 8 kHz. That means inbound SIP audio
needs upsampling 8kHz -> 16kHz before it can reach VAD/ASR, and outbound TTS
audio needs downsampling 16kHz -> 8kHz before it can go out over SIP/RTP. Both
directions and both transports (today's 16 kHz browser passthrough and a
future 8 kHz SIP leg) need the exact same conversion, so it lives here once
rather than being duplicated per transport - the browser leg calls this with
orig == target and gets a no-op; the SIP leg (not built yet, see Sec3.3) would
call it on both legs of the pipeline.

Uses torchaudio.functional.resample - torch/torchaudio are already hard
dependencies of this repo via the NeMo-based ASR stack (see asr.py), so this
adds no new dependency.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio


def resample_pcm(samples: np.ndarray, orig_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    """Resample mono int16 PCM from `orig_sample_rate` to `target_sample_rate`.

    Matches the int16 PCM convention used everywhere else in this pipeline
    (vad.py/asr.py/tts.py all pass samples around this way, never float) so
    callers on either side of this function never have to think about the
    float32 conversion happening in between.
    """
    if orig_sample_rate == target_sample_rate:
        # Fast-path passthrough: this is the common case today (browser leg
        # is 16kHz end-to-end), and it also means no resampling artifacts or
        # rounding ever get introduced when no conversion is actually needed.
        return samples

    if len(samples) == 0:
        return samples

    # int16 -> float32 in [-1, 1], the same convention asr.py uses when
    # handing samples to NeMo (samples.astype(np.float32) / 32768.0).
    waveform = torch.from_numpy(samples.astype(np.float32) / 32768.0)
    resampled = torchaudio.functional.resample(waveform, orig_sample_rate, target_sample_rate)

    # Clip-and-rescale back to int16, mirroring the exact pattern tts.py's
    # synthesize() already uses for its own float -> int16 conversion.
    resampled = resampled.numpy()
    return (np.clip(resampled, -1.0, 1.0) * 32767).astype(np.int16)
