"""Whisper transcription via faster-whisper.

We isolate this in a util so the pipeline modules don't depend directly on the
model loader. Models are cached on first use.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..models import TranscriptSegment
from . import ffmpeg as ffu

log = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, object] = {}
_LOCK = threading.Lock()


@dataclass
class TranscribeConfig:
    model_size: str = "small"        # tiny | base | small | medium | large-v3
    language: Optional[str] = None   # autodetect by default
    device: str = "auto"             # auto | cpu | cuda
    compute_type: str = "auto"       # auto | int8 | float16 | float32
    word_timestamps: bool = True
    vad_filter: bool = True
    beam_size: int = 1


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch  # noqa: F401
        import torch.cuda
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _resolve_compute_type(compute_type: str, device: str) -> str:
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


def get_model(cfg: TranscribeConfig):
    """Lazy-load and cache a faster-whisper WhisperModel."""
    from faster_whisper import WhisperModel  # imported lazily — heavy

    device = _resolve_device(cfg.device)
    compute_type = _resolve_compute_type(cfg.compute_type, device)
    key = f"{cfg.model_size}|{device}|{compute_type}"
    with _LOCK:
        if key not in _MODEL_CACHE:
            log.info("Loading faster-whisper model=%s device=%s compute_type=%s",
                     cfg.model_size, device, compute_type)
            _MODEL_CACHE[key] = WhisperModel(
                cfg.model_size,
                device=device,
                compute_type=compute_type,
                download_root=os.environ.get("AUTO_POD_MODELS_DIR"),
            )
    return _MODEL_CACHE[key]


def transcribe(
    media_path: str | Path,
    cfg: Optional[TranscribeConfig] = None,
    *,
    audio_wav_path: Optional[str | Path] = None,
) -> list[TranscriptSegment]:
    """Transcribe a media file. Optionally accept a pre-extracted WAV path."""
    cfg = cfg or TranscribeConfig()
    model = get_model(cfg)

    if audio_wav_path is None:
        # Faster-whisper can read most media via its bundled av/ffmpeg, but we
        # extract WAV explicitly to keep behavior predictable across environments.
        wav_path = Path(media_path).with_suffix(".__autopod__.wav")
        ffu.extract_audio_wav(media_path, wav_path)
    else:
        wav_path = Path(audio_wav_path)

    try:
        segments_iter, info = model.transcribe(
            str(wav_path),
            language=cfg.language,
            beam_size=cfg.beam_size,
            vad_filter=cfg.vad_filter,
            word_timestamps=cfg.word_timestamps,
        )
        log.info("Whisper detected language=%s prob=%.2f",
                 getattr(info, "language", "?"),
                 getattr(info, "language_probability", 0.0))

        out: list[TranscriptSegment] = []
        for s in segments_iter:
            words = []
            if cfg.word_timestamps and getattr(s, "words", None):
                for w in s.words:
                    words.append({
                        "start": float(w.start) if w.start is not None else None,
                        "end": float(w.end) if w.end is not None else None,
                        "word": w.word,
                    })
            out.append(TranscriptSegment(
                start=float(s.start),
                end=float(s.end),
                text=s.text.strip(),
                words=words,
            ))
        return out
    finally:
        # Only delete the WAV if we created it.
        if audio_wav_path is None:
            try:
                Path(wav_path).unlink()
            except OSError:
                pass
