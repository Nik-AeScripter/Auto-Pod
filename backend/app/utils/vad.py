"""Silero VAD wrapper.

Silero VAD is a small, fast neural network that classifies short audio frames
as speech vs non-speech. It runs on CPU in real time and is ~30 MB.

We isolate the model load + inference here so the pipeline modules don't carry
torch/silero-vad imports at module top level.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Silero VAD requires 16 kHz mono PCM. Don't change without reading their docs.
VAD_SAMPLE_RATE = 16000

_MODEL = None
_LOCK = threading.Lock()


@dataclass
class VADConfig:
    """Tunables for Silero VAD speech-segment detection."""
    min_speech_ms: int = 250            # ignore speech bursts shorter than this
    min_silence_ms: int = 200           # gaps shorter than this don't split speech
    speech_pad_ms: int = 30             # extend each speech span by this on each side
    threshold: float = 0.5              # 0..1, VAD confidence cutoff


def get_model():
    """Lazy-load and cache the Silero VAD model."""
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            log.info("Loading Silero VAD model…")
            # silero-vad >=5 packages a load helper that handles weight download
            from silero_vad import load_silero_vad  # type: ignore
            _MODEL = load_silero_vad()
        return _MODEL


def _read_audio_16k_mono(path: str | Path) -> np.ndarray:
    """Read any WAV at 16 kHz mono as a float32 numpy array in [-1, 1].

    The file must be 16 kHz mono PCM — the caller is expected to extract it via
    ffmpeg first. This keeps decoding behavior predictable.
    """
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1).astype("float32")
    if sr != VAD_SAMPLE_RATE:
        raise ValueError(
            f"VAD expects {VAD_SAMPLE_RATE} Hz audio; got {sr} Hz. "
            f"Use ffmpeg.extract_audio_wav(..., sample_rate=16000) first."
        )
    return data


def detect_speech_intervals(
    wav_16k_mono_path: str | Path,
    cfg: Optional[VADConfig] = None,
) -> list[tuple[float, float]]:
    """Return a list of (start, end) speech intervals in seconds.

    Input MUST be a 16 kHz mono WAV file. Extract it first with
    ffmpeg.extract_audio_wav().

    Implementation note: we do NOT use silero-vad's `read_audio` helper because
    on recent torchaudio (>=2.9) it raises a torchcodec dependency error. We
    read the WAV with `soundfile` ourselves and hand a plain torch tensor to
    `get_speech_timestamps`, which the silero model accepts directly.
    """
    cfg = cfg or VADConfig()
    model = get_model()

    # We read with soundfile (no torchaudio) and pass a torch tensor directly.
    import torch  # type: ignore
    from silero_vad import get_speech_timestamps  # type: ignore

    samples = _read_audio_16k_mono(wav_16k_mono_path)
    wav = torch.from_numpy(samples)

    timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=VAD_SAMPLE_RATE,
        threshold=cfg.threshold,
        min_speech_duration_ms=cfg.min_speech_ms,
        min_silence_duration_ms=cfg.min_silence_ms,
        speech_pad_ms=cfg.speech_pad_ms,
        return_seconds=True,
    )

    intervals = [(float(t["start"]), float(t["end"])) for t in timestamps]
    log.info("Silero VAD found %d speech intervals", len(intervals))
    return intervals


def total_duration(wav_16k_mono_path: str | Path) -> float:
    """Duration in seconds of the WAV file."""
    import soundfile as sf
    info = sf.info(str(wav_16k_mono_path))
    return float(info.duration)
