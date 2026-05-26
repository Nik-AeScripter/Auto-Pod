"""Tests for the *cut math* of the silence pipeline.

These don't require ffmpeg or Whisper — they exercise the pure-Python
logic that converts transcript word intervals into kept segments.
"""
from __future__ import annotations

from app.pipeline.silence import _merge_close_intervals, _word_intervals
from app.models import TranscriptSegment


def _seg(start: float, end: float, words: list[tuple[float, float, str]]) -> TranscriptSegment:
    return TranscriptSegment(
        start=start,
        end=end,
        text=" ".join(w[2] for w in words),
        words=[{"start": s, "end": e, "word": w} for (s, e, w) in words],
    )


def test_word_intervals_uses_word_timestamps_when_present():
    transcript = [
        _seg(0.0, 2.0, [(0.0, 0.5, "hello"), (0.7, 1.2, "there"), (1.5, 2.0, "friend")]),
    ]
    iv = _word_intervals(transcript)
    assert iv == [(0.0, 0.5), (0.7, 1.2), (1.5, 2.0)]


def test_word_intervals_falls_back_to_segment_when_no_words():
    transcript = [
        TranscriptSegment(start=0.0, end=2.0, text="hello there", words=[]),
        TranscriptSegment(start=3.0, end=4.5, text="again", words=[]),
    ]
    iv = _word_intervals(transcript)
    assert iv == [(0.0, 2.0), (3.0, 4.5)]


def test_merge_close_intervals_collapses_short_gaps():
    iv = [(0.0, 1.0), (1.1, 2.0), (5.0, 6.0)]
    merged = _merge_close_intervals(iv, min_gap_sec=0.5)
    assert merged == [(0.0, 2.0), (5.0, 6.0)]


def test_merge_close_intervals_keeps_long_gaps():
    iv = [(0.0, 1.0), (1.6, 2.0)]  # 0.6s gap
    merged = _merge_close_intervals(iv, min_gap_sec=0.5)
    assert merged == [(0.0, 1.0), (1.6, 2.0)]


def test_merge_close_intervals_handles_overlap():
    iv = [(0.0, 1.0), (0.8, 2.0)]
    merged = _merge_close_intervals(iv, min_gap_sec=0.001)
    assert merged == [(0.0, 2.0)]


def test_merge_close_intervals_empty():
    assert _merge_close_intervals([], min_gap_sec=0.5) == []
