"""Silence removal pipeline.

PRIMARY MODE — `mode="transcript"` (default, recommended):
  1. Transcribe audio with Whisper (word-level timestamps).
  2. The "silent" regions are simply the gaps BETWEEN words.
  3. Cuts are placed exactly between words, never mid-syllable.
  4. Render via segment-concat.

This is more robust than amplitude-based detection because:
  - background noise / music doesn't register as "speech"
  - breaths, lip smacks, keyboard clicks don't survive the transcript
  - cuts always land on word boundaries

FALLBACK MODE — `mode="amplitude"`:
  Uses ffmpeg `silencedetect` (energy threshold). Faster (no model download)
  but less accurate. Available for users who don't want to run Whisper.

ffmpeg/ffprobe are still used to (a) probe duration/streams and (b) render the
final output — Python has no replacement for that. Only the *decision* of where
to cut has been moved to the transcript.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from ..models import CutSegment, EditDecision, TimeRange, TranscriptSegment
from ..utils import ffmpeg
from ..utils.transcribe import TranscribeConfig, transcribe

log = logging.getLogger(__name__)


@dataclass
class SilenceConfig:
    # Mode
    mode: Literal["transcript", "amplitude"] = "transcript"

    # Common
    min_silence_sec: float = 0.5     # gaps shorter than this are NOT removed
    pad_sec: float = 0.05            # keep this much extra audio around each kept span
    min_keep_sec: float = 0.10       # drop tiny kept slivers

    # Amplitude-mode only
    noise_db: float = -30.0

    # Transcript-mode only
    transcribe: TranscribeConfig = None  # type: ignore[assignment]


def _word_intervals(transcript: list[TranscriptSegment]) -> list[tuple[float, float]]:
    """Flatten transcript into a sorted list of (word_start, word_end) intervals.

    Falls back to segment-level intervals if word timestamps are missing.
    """
    intervals: list[tuple[float, float]] = []
    for seg in transcript:
        had_words = False
        for w in seg.words or []:
            ws, we = w.get("start"), w.get("end")
            if ws is None or we is None:
                continue
            if we > ws:
                intervals.append((float(ws), float(we)))
                had_words = True
        if not had_words and seg.end > seg.start:
            intervals.append((float(seg.start), float(seg.end)))
    intervals.sort(key=lambda x: x[0])
    return intervals


def _merge_close_intervals(
    intervals: list[tuple[float, float]],
    *,
    min_gap_sec: float,
) -> list[tuple[float, float]]:
    """Merge two adjacent word/speech intervals if the gap between them is < min_gap_sec."""
    if not intervals:
        return []
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s - pe < min_gap_sec:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _analyze_transcript(
    input_path: str | Path,
    cfg: SilenceConfig,
) -> EditDecision:
    """Use Whisper word timestamps to decide what to keep."""
    info = ffmpeg.probe(input_path)
    if not info.has_audio:
        return EditDecision(
            kept_segments=[CutSegment(source=str(input_path), start=0.0, end=info.duration)],
            removed_ranges=[],
            notes={"reason": "no_audio_track", "duration": info.duration},
        )

    transcript = transcribe(input_path, cfg.transcribe or TranscribeConfig())

    word_intervals = _word_intervals(transcript)
    # Merge words that are closer together than min_silence_sec — those gaps are NOT silence.
    speech = _merge_close_intervals(word_intervals, min_gap_sec=cfg.min_silence_sec)

    if not speech:
        # No speech detected — fall back to amplitude method so we still produce something
        log.warning("Transcript produced no words; falling back to amplitude silencedetect.")
        return _analyze_amplitude(input_path, cfg, _info=info)

    # Pad each speech span and clamp to media duration
    kept_pairs: list[tuple[float, float]] = []
    for s, e in speech:
        s2 = max(0.0, s - cfg.pad_sec)
        e2 = min(info.duration, e + cfg.pad_sec)
        if e2 - s2 >= cfg.min_keep_sec:
            kept_pairs.append((s2, e2))

    # Merge overlaps caused by padding
    kept_pairs = _merge_close_intervals(kept_pairs, min_gap_sec=0.001)

    # Removed = the inverse
    removed_pairs: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in kept_pairs:
        if s > cursor + 0.001:
            removed_pairs.append((cursor, s))
        cursor = e
    if info.duration > cursor + 0.001:
        removed_pairs.append((cursor, info.duration))

    kept = [
        CutSegment(source=str(input_path), start=s, end=e, label="speech")
        for s, e in kept_pairs
    ]
    removed = [TimeRange(start=s, end=e) for s, e in removed_pairs]

    return EditDecision(
        kept_segments=kept,
        removed_ranges=removed,
        transcript=transcript,
        notes={
            "feature": "silence",
            "mode": "transcript",
            "config": {k: v for k, v in vars(cfg).items() if k != "transcribe"},
            "input_duration": info.duration,
            "word_count": sum(len(s.words or []) for s in transcript),
        },
    )


def _analyze_amplitude(
    input_path: str | Path,
    cfg: SilenceConfig,
    *,
    _info=None,
) -> EditDecision:
    """Legacy amplitude-based path using ffmpeg silencedetect."""
    info = _info or ffmpeg.probe(input_path)
    if not info.has_audio:
        return EditDecision(
            kept_segments=[CutSegment(source=str(input_path), start=0.0, end=info.duration)],
            removed_ranges=[],
            notes={"reason": "no_audio_track", "duration": info.duration},
        )

    silences = ffmpeg.detect_silence(
        input_path,
        noise_db=cfg.noise_db,
        min_silence_sec=cfg.min_silence_sec,
    )
    log.info("silencedetect found %d silent ranges", len(silences))

    # shrink silent ranges by pad
    padded: list[tuple[float, float]] = []
    for s, e in silences:
        s2 = max(0.0, s + cfg.pad_sec)
        e2 = min(info.duration, e - cfg.pad_sec)
        if e2 - s2 >= max(0.05, cfg.min_silence_sec * 0.5):
            padded.append((s2, e2))

    # invert
    kept_pairs: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in padded:
        if s > cursor + cfg.min_keep_sec:
            kept_pairs.append((cursor, s))
        cursor = e
    if info.duration > cursor + cfg.min_keep_sec:
        kept_pairs.append((cursor, info.duration))

    kept = [CutSegment(source=str(input_path), start=s, end=e, label="speech") for s, e in kept_pairs]
    removed = [TimeRange(start=s, end=e) for s, e in padded]

    return EditDecision(
        kept_segments=kept,
        removed_ranges=removed,
        notes={
            "feature": "silence",
            "mode": "amplitude",
            "config": {"noise_db": cfg.noise_db,
                       "min_silence_sec": cfg.min_silence_sec,
                       "pad_sec": cfg.pad_sec,
                       "min_keep_sec": cfg.min_keep_sec},
            "input_duration": info.duration,
        },
    )


def analyze(
    input_path: str | Path,
    cfg: Optional[SilenceConfig] = None,
) -> EditDecision:
    cfg = cfg or SilenceConfig()
    if cfg.mode == "amplitude":
        return _analyze_amplitude(input_path, cfg)
    return _analyze_transcript(input_path, cfg)


def render(
    decision: EditDecision,
    output_path: str | Path,
    *,
    reencode: bool = True,
) -> Path:
    if not decision.kept_segments:
        raise RuntimeError("Silence pipeline produced no kept segments; nothing to render.")
    triplets = [(s.source, s.start, s.end) for s in decision.kept_segments]
    return ffmpeg.render_segments(triplets, output_path, reencode=reencode)


def run(
    input_path: str | Path,
    output_path: str | Path,
    cfg: Optional[SilenceConfig] = None,
    *,
    reencode: bool = True,
) -> tuple[EditDecision, Path]:
    decision = analyze(input_path, cfg)
    out = render(decision, output_path, reencode=reencode)
    return decision, out
