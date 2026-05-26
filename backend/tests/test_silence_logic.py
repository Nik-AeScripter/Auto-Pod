"""Tests for the *cut math* of the silence pipeline.

These don't require ffmpeg, soundfile, torch, or Silero VAD — they exercise
only the pure-Python logic that turns VAD speech intervals into kept segments.
"""
from __future__ import annotations

import numpy as np

from app.pipeline.silence import (
    _filter_by_db,
    _invert_to_silences,
    _merge_close,
    _pad_silences,
    _silences_to_kept,
)


def test_invert_with_speech_in_middle():
    speech = [(2.0, 4.0), (6.0, 8.0)]
    silences = _invert_to_silences(speech, total_duration=10.0)
    assert silences == [(0.0, 2.0), (4.0, 6.0), (8.0, 10.0)]


def test_invert_with_no_leading_silence():
    speech = [(0.0, 4.0), (6.0, 10.0)]
    silences = _invert_to_silences(speech, total_duration=10.0)
    assert silences == [(4.0, 6.0)]


def test_invert_with_no_speech_returns_full_range():
    silences = _invert_to_silences([], total_duration=5.0)
    assert silences == [(0.0, 5.0)]


def test_filter_by_db_keeps_quiet_drops_loud():
    sr = 16000
    # 4 seconds: 2s loud (full-scale ~ 0 dB), 2s quiet (~-60 dB)
    loud = np.full(sr * 2, 0.5, dtype=np.float32)             # ~ -6 dB
    quiet = np.full(sr * 2, 1e-3, dtype=np.float32)           # ~ -60 dB
    samples = np.concatenate([loud, quiet])
    candidates = [(0.0, 2.0), (2.0, 4.0)]
    kept = _filter_by_db(candidates, samples, sr, threshold_db=-35.0)
    # Only the quiet half should be kept as confirmed silence
    assert kept == [(2.0, 4.0)]


def test_pad_silences_shrinks_and_drops_too_short():
    silences = [(1.0, 2.0), (3.0, 3.4)]   # 1.0s and 0.4s
    out = _pad_silences(silences, padding_ms=150, min_silence_to_cut_sec=0.30)
    # First: 1.15..1.85 = 0.7s -> kept; second: 3.15..3.25 = 0.10s -> dropped
    assert len(out) == 1
    assert abs(out[0][0] - 1.15) < 1e-6
    assert abs(out[0][1] - 1.85) < 1e-6


def test_merge_close_collapses_nearby():
    silences = [(1.0, 2.0), (2.3, 3.0), (10.0, 11.0)]
    merged = _merge_close(silences, max_gap_sec=0.5)
    assert merged == [(1.0, 3.0), (10.0, 11.0)]


def test_merge_close_preserves_distant():
    silences = [(1.0, 2.0), (5.0, 6.0)]
    merged = _merge_close(silences, max_gap_sec=0.5)
    assert merged == [(1.0, 2.0), (5.0, 6.0)]


def test_silences_to_kept_inverts_correctly():
    silences = [(2.0, 4.0), (7.0, 8.0)]
    kept = _silences_to_kept(silences, total_duration=10.0)
    assert kept == [(0.0, 2.0), (4.0, 7.0), (8.0, 10.0)]


def test_silences_to_kept_no_silences_keeps_everything():
    kept = _silences_to_kept([], total_duration=5.0)
    assert kept == [(0.0, 5.0)]


def test_silences_to_kept_full_silence_keeps_nothing():
    kept = _silences_to_kept([(0.0, 5.0)], total_duration=5.0)
    assert kept == []


def test_end_to_end_logic_on_synthetic_intervals():
    """Mini end-to-end of the cut math (no audio decoding required)."""
    speech = [(1.0, 3.0), (5.0, 7.0)]
    total = 8.0

    candidates = _invert_to_silences(speech, total)
    # Pretend dB filter passed everything (we test the filter elsewhere).
    padded = _pad_silences(candidates, padding_ms=100, min_silence_to_cut_sec=0.30)
    merged = _merge_close(padded, max_gap_sec=0.5)
    kept = _silences_to_kept(merged, total)

    # Padded silences: (0.1..0.9), (3.1..4.9), (7.1..7.9)
    # Kept: (0..0.1), (0.9..3.1), (4.9..7.1), (7.9..8.0)
    expected = [(0.0, 0.1), (0.9, 3.1), (4.9, 7.1), (7.9, 8.0)]
    for got, want in zip(kept, expected):
        assert abs(got[0] - want[0]) < 1e-6
        assert abs(got[1] - want[1]) < 1e-6
    assert len(kept) == len(expected)
