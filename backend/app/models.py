"""Data models shared across the Auto-Pod pipeline.

All time values are in seconds (float) unless otherwise noted.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TimeRange:
    """A half-open interval [start, end) in seconds."""
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "duration": self.duration}


@dataclass
class CutSegment:
    """A segment of the source we want to KEEP in the output."""
    source: str               # path to source media (or logical id for multicam)
    start: float              # seconds in source
    end: float                # seconds in source
    track_index: int = 0      # for multicam: which input track (0..N-1)
    label: Optional[str] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TranscriptSegment:
    """A line of transcribed speech with timing."""
    start: float
    end: float
    text: str
    words: list[dict] = field(default_factory=list)  # optional [{start,end,word}]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Chapter:
    """An auto-detected chapter."""
    start: float
    end: float
    title: str
    summary: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MediaInfo:
    """Probed info about a media file."""
    path: str
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    audio_sample_rate: int
    container: str
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EditDecision:
    """Result of an analysis pass: which segments to keep, plus optional metadata."""
    kept_segments: list[CutSegment]
    removed_ranges: list[TimeRange] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)
    transcript: list[TranscriptSegment] = field(default_factory=list)
    notes: dict = field(default_factory=dict)

    @property
    def total_kept(self) -> float:
        return sum(s.duration for s in self.kept_segments)

    @property
    def total_removed(self) -> float:
        return sum(r.duration for r in self.removed_ranges)

    def to_dict(self) -> dict:
        return {
            "kept_segments": [s.to_dict() for s in self.kept_segments],
            "removed_ranges": [r.to_dict() for r in self.removed_ranges],
            "chapters": [c.to_dict() for c in self.chapters],
            "transcript": [t.to_dict() for t in self.transcript],
            "notes": self.notes,
            "stats": {
                "total_kept_seconds": self.total_kept,
                "total_removed_seconds": self.total_removed,
            },
        }
