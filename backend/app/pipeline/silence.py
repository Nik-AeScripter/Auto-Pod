"""Silence removal pipeline — Silero VAD + dB threshold confirmation.

This is a faithful port of the method used by EduardoAndreu/clean-cut
(https://github.com/EduardoAndreu/clean-cut), adapted to render a standalone
output video instead of pushing cuts into Premiere Pro.

Pipeline:
  1. Extract audio as 16 kHz mono WAV (ffmpeg, decoder only).
  2. Run Silero VAD to get SPEECH intervals.
  3. Invert: candidate silences are the gaps between speech segments
     (plus head/tail).
  4. For each candidate silence, compute RMS dB over that range.
     Confirm it as silence only if dB < silence_threshold_db.
     This is the "safety net" from clean-cut — protects loud non-speech
     (music, applause, noise) from being cut.
  5. Shrink each confirmed silence by `padding_ms` on both sides
     (leaves breathing room around words).
  6. Merge silences less than `merge_gap_sec` apart.
  7. Invert again -> kept segments.
  8. Render via ffmpeg segment-concat.

The decision step is *entirely* timestamp-based. ffmpeg is only used to
decode audio and encode the final video.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..models import CutSegment, EditDecision, TimeRange
from ..utils import ffmpeg as ffu
from ..utils.vad import VAD_SAMPLE_RATE, VADConfig, detect_speech_intervals

log = logging.getLogger(__name__)


@dataclass
class SilenceConfig:
    # dB safety net (matches clean-cut default of -35 dB)
    silence_threshold_db: float = -35.0

    # VAD timing
    min_silence_ms: int = 200       # gaps shorter than this aren't candidate silences
    min_speech_ms: int = 250        # ignore tiny speech bursts (clicks, lip smacks)
    speech_pad_ms: int = 30         # VAD's own padding around each speech span
    vad_threshold: float = 0.5      # VAD confidence cutoff (0..1)

    # Cut padding (matches clean-cut default of 150 ms)
    padding_ms: int = 150           # shrink silences by this much on each side

    # Post-processing
    merge_gap_sec: float = 0.5      # merge silences closer than this together
    min_silence_to_cut_sec: float = 0.30  # don't bother cutting silences shorter than this


def _rms_db_of_range(samples: np.ndarray, sample_rate: int, start_s: float, end_s: float) -> float:
    """Compute RMS dB over [start_s, end_s) of the audio array (full-scale = 0 dB)."""
    i0 = max(0, int(start_s * sample_rate))
    i1 = min(len(samples), int(end_s * sample_rate))
    if i1 <= i0:
        return -90.0
    chunk = samples[i0:i1].astype(np.float64)
    rms = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
    if rms <= 0:
        return -90.0
    # samples are float32 in [-1, 1] (full-scale = 1.0)
    return 20.0 * math.log10(rms)


def _read_wav_float(path: str | Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1).astype("float32")
    return data, int(sr)


def _invert_to_silences(
    speech: list[tuple[float, float]],
    total_duration: float,
) -> list[tuple[float, float]]:
    """Return gaps between speech intervals (plus leading/trailing silence)."""
    if not speech:
        return [(0.0, total_duration)]
    silences: list[tuple[float, float]] = []
    if speech[0][0] > 0:
        silences.append((0.0, speech[0][0]))
    for i in range(len(speech) - 1):
        silences.append((speech[i][1], speech[i + 1][0]))
    if speech[-1][1] < total_duration:
        silences.append((speech[-1][1], total_duration))
    return silences


def _filter_by_db(
    candidates: list[tuple[float, float]],
    samples: np.ndarray,
    sample_rate: int,
    threshold_db: float,
) -> list[tuple[float, float]]:
    """Keep only candidates whose RMS dB is below the threshold."""
    confirmed: list[tuple[float, float]] = []
    for s, e in candidates:
        if e - s <= 0:
            continue
        db = _rms_db_of_range(samples, sample_rate, s, e)
        if db < threshold_db:
            confirmed.append((s, e))
        else:
            log.debug("Skipping silence %.2f-%.2f (%.1f dB >= %.1f dB threshold)",
                      s, e, db, threshold_db)
    return confirmed


def _pad_silences(
    silences: list[tuple[float, float]],
    *,
    padding_ms: int,
    min_silence_to_cut_sec: float,
) -> list[tuple[float, float]]:
    """Shrink each silence by padding_ms on both sides; drop ones too short to cut."""
    pad = padding_ms / 1000.0
    out: list[tuple[float, float]] = []
    for s, e in silences:
        ns, ne = s + pad, e - pad
        if ne - ns >= min_silence_to_cut_sec:
            out.append((ns, ne))
    return out


def _merge_close(
    silences: list[tuple[float, float]],
    *,
    max_gap_sec: float,
) -> list[tuple[float, float]]:
    if not silences:
        return []
    merged = [silences[0]]
    for s, e in silences[1:]:
        ps, pe = merged[-1]
        if s - pe < max_gap_sec:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _silences_to_kept(
    silences: list[tuple[float, float]],
    total_duration: float,
) -> list[tuple[float, float]]:
    """Invert silences back into kept-segment intervals."""
    kept: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in silences:
        if s > cursor + 1e-3:
            kept.append((cursor, s))
        cursor = e
    if total_duration > cursor + 1e-3:
        kept.append((cursor, total_duration))
    return kept


def analyze(
    input_path: str | Path,
    cfg: Optional[SilenceConfig] = None,
) -> EditDecision:
    """Run VAD + dB analysis. Returns an EditDecision; does NOT render."""
    cfg = cfg or SilenceConfig()
    info = ffu.probe(input_path)

    if not info.has_audio or info.duration <= 0:
        whole = [CutSegment(source=str(input_path), start=0.0, end=info.duration)]
        return EditDecision(
            kept_segments=whole,
            removed_ranges=[],
            notes={"feature": "silence", "reason": "no_audio_track",
                   "input_duration": info.duration},
        )

    # Step 1: extract 16 kHz mono WAV (ffmpeg = decoder only)
    wav_path = Path(input_path).with_suffix(".__autopod_silence__.wav")
    try:
        ffu.extract_audio_wav(input_path, wav_path, sample_rate=VAD_SAMPLE_RATE, channels=1)

        # Step 2: VAD speech detection
        vad_cfg = VADConfig(
            min_speech_ms=cfg.min_speech_ms,
            min_silence_ms=cfg.min_silence_ms,
            speech_pad_ms=cfg.speech_pad_ms,
            threshold=cfg.vad_threshold,
        )
        speech = detect_speech_intervals(wav_path, vad_cfg)

        # Step 3: candidate silences = gaps between speech
        samples, sr = _read_wav_float(wav_path)
        wav_duration = len(samples) / sr if sr else info.duration
        candidates = _invert_to_silences(speech, wav_duration)

        # Step 4: dB safety net (clean-cut's idea: confirm silence by amplitude)
        confirmed = _filter_by_db(candidates, samples, sr, cfg.silence_threshold_db)

        # Step 5: padding (shrink each silence to leave breathing room)
        padded = _pad_silences(
            confirmed,
            padding_ms=cfg.padding_ms,
            min_silence_to_cut_sec=cfg.min_silence_to_cut_sec,
        )

        # Step 6: merge nearby silences
        merged = _merge_close(padded, max_gap_sec=cfg.merge_gap_sec)

        # Step 7: invert -> kept segments
        kept_pairs = _silences_to_kept(merged, info.duration)
    finally:
        try:
            wav_path.unlink()
        except OSError:
            pass

    kept = [
        CutSegment(source=str(input_path), start=s, end=e, label="speech")
        for s, e in kept_pairs
    ]
    removed = [TimeRange(start=s, end=e) for s, e in merged]

    return EditDecision(
        kept_segments=kept,
        removed_ranges=removed,
        notes={
            "feature": "silence",
            "method": "silero_vad+db_threshold",
            "config": {
                "silence_threshold_db": cfg.silence_threshold_db,
                "min_silence_ms": cfg.min_silence_ms,
                "min_speech_ms": cfg.min_speech_ms,
                "padding_ms": cfg.padding_ms,
                "merge_gap_sec": cfg.merge_gap_sec,
                "vad_threshold": cfg.vad_threshold,
            },
            "input_duration": info.duration,
            "speech_segments_detected": len(speech),
            "candidate_silences": len(candidates),
            "confirmed_silences": len(confirmed),
            "final_silences": len(merged),
        },
    )


def render(
    decision: EditDecision,
    output_path: str | Path,
    *,
    reencode: bool = True,
) -> Path:
    if not decision.kept_segments:
        raise RuntimeError("Silence pipeline produced no kept segments; nothing to render.")
    triplets = [(s.source, s.start, s.end) for s in decision.kept_segments]
    return ffu.render_segments(triplets, output_path, reencode=reencode)


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
