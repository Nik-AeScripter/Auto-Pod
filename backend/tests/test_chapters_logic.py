"""Tests for chapter segmentation logic (no Whisper / ffmpeg needed)."""
from __future__ import annotations

from app.models import TranscriptSegment
from app.pipeline.chapters import ChapterConfig, from_transcript


def _ts(start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text, words=[])


def test_empty_transcript_returns_single_chapter():
    out = from_transcript([], media_duration=120.0)
    assert len(out) == 1
    assert out[0].start == 0.0
    assert out[0].end == 120.0


def test_topic_shift_creates_boundary():
    # First half: cooking. Second half: astronomy. Should split.
    transcript = [
        _ts(0, 20, "today we will be cooking pasta with tomato sauce"),
        _ts(20, 40, "boil the water and add salt then drop the pasta"),
        _ts(40, 60, "stir the tomato sauce and add basil and garlic"),
        _ts(60, 80, "let me tell you about astronomy and the planets"),
        _ts(80, 100, "the planets orbit the sun in elliptical paths"),
        _ts(100, 120, "stars are huge balls of plasma fusing hydrogen"),
    ]
    cfg = ChapterConfig(min_chapter_sec=10.0, similarity_drop=0.6, window_segments=2)
    out = from_transcript(transcript, media_duration=120.0, cfg=cfg)
    assert len(out) >= 2
    # boundaries should be ascending
    starts = [c.start for c in out]
    assert starts == sorted(starts)
    # last chapter should end at media duration
    assert out[-1].end <= 120.0 + 1e-6


def test_min_chapter_length_enforced():
    transcript = [
        _ts(0, 5, "alpha alpha alpha"),
        _ts(5, 10, "beta beta beta"),
        _ts(10, 15, "gamma gamma gamma"),
    ]
    cfg = ChapterConfig(min_chapter_sec=999.0, similarity_drop=0.99)
    out = from_transcript(transcript, media_duration=15.0, cfg=cfg)
    # min length is huge so we can have at most 1 chapter
    assert len(out) == 1
