"""Auto-chapters: split a transcript into topical chapters.

Strategy (no LLM required, no network):
- Treat each transcript segment as a node.
- Compute TF-IDF cosine similarity between adjacent windows of text.
- Place a chapter boundary where similarity drops sharply OR at min-spacing
  intervals as a fallback.
- Pick a title for each chapter from the most "distinctive" sentence
  (highest TF-IDF norm against the rest of the chapter).

This is deterministic, fast, and offline — perfect for a 3-day prototype.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..models import Chapter, EditDecision, TranscriptSegment
from ..utils.transcribe import TranscribeConfig, transcribe

log = logging.getLogger(__name__)


@dataclass
class ChapterConfig:
    min_chapter_sec: float = 60.0          # don't create chapters shorter than this
    target_chapters: Optional[int] = None  # if set, split into ~N chapters
    similarity_drop: float = 0.35          # boundary if cos sim < this
    window_segments: int = 4               # how many segments per smoothing window
    max_title_words: int = 8


def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _build_tfidf(texts: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        stop_words="english",
    )
    matrix = vec.fit_transform(texts) if texts else None
    return vec, matrix


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _find_boundaries(
    segments: list[TranscriptSegment],
    cfg: ChapterConfig,
) -> list[int]:
    """Return indices into `segments` that mark START of a new chapter (always includes 0)."""
    if not segments:
        return []
    if len(segments) <= 2:
        return [0]

    texts = [_normalize_text(s.text) for s in segments]
    _, matrix = _build_tfidf(texts)
    if matrix is None or matrix.shape[1] == 0:
        return [0]
    dense = matrix.toarray()

    # smoothed window vectors
    w = max(1, cfg.window_segments)
    sims: list[float] = []
    for i in range(len(segments) - 1):
        a_lo, a_hi = max(0, i - w + 1), i + 1
        b_lo, b_hi = i + 1, min(len(segments), i + 1 + w)
        a = dense[a_lo:a_hi].mean(axis=0)
        b = dense[b_lo:b_hi].mean(axis=0)
        sims.append(_cosine(a, b))

    # candidate boundaries: where similarity is a local minimum AND below drop threshold
    candidates: list[tuple[int, float]] = []
    for i in range(1, len(sims) - 1):
        if sims[i] < cfg.similarity_drop and sims[i] <= sims[i - 1] and sims[i] <= sims[i + 1]:
            candidates.append((i + 1, sims[i]))  # boundary BEFORE segment i+1
    # sort weakest similarity first (strongest boundary first)
    candidates.sort(key=lambda x: x[1])

    boundaries = {0}
    if cfg.target_chapters and cfg.target_chapters > 1:
        for idx, _ in candidates:
            if len(boundaries) >= cfg.target_chapters:
                break
            boundaries.add(idx)
    else:
        for idx, _ in candidates:
            boundaries.add(idx)

    # enforce min duration
    sorted_idx = sorted(boundaries)
    enforced: list[int] = []
    last_start_time = -1e9
    for i in sorted_idx:
        seg = segments[i]
        if seg.start - last_start_time >= cfg.min_chapter_sec or not enforced:
            enforced.append(i)
            last_start_time = seg.start
    return enforced


def _pick_title(chunk: list[TranscriptSegment], cfg: ChapterConfig) -> str:
    if not chunk:
        return "Chapter"
    # Use first sentence-ish thing, truncated
    raw = " ".join(s.text for s in chunk[:3]).strip()
    # take first sentence
    m = re.split(r"(?<=[\.\?\!])\s+", raw, maxsplit=1)
    head = (m[0] if m else raw).strip()
    words = head.split()
    if len(words) > cfg.max_title_words:
        head = " ".join(words[: cfg.max_title_words]).rstrip(",;:") + "…"
    head = head[:1].upper() + head[1:] if head else "Chapter"
    return head or "Chapter"


def from_transcript(
    transcript: list[TranscriptSegment],
    *,
    media_duration: float,
    cfg: Optional[ChapterConfig] = None,
) -> list[Chapter]:
    cfg = cfg or ChapterConfig()
    if not transcript:
        return [Chapter(start=0.0, end=media_duration, title="Chapter 1")]

    boundary_indices = _find_boundaries(transcript, cfg)
    boundary_indices = sorted(set(boundary_indices))

    chapters: list[Chapter] = []
    for j, start_idx in enumerate(boundary_indices):
        end_idx = boundary_indices[j + 1] if j + 1 < len(boundary_indices) else len(transcript)
        chunk = transcript[start_idx:end_idx]
        if not chunk:
            continue
        ch_start = chunk[0].start
        ch_end = chunk[-1].end if j + 1 == len(boundary_indices) else transcript[end_idx].start
        ch_end = min(ch_end, media_duration)
        title = _pick_title(chunk, cfg)
        chapters.append(Chapter(start=float(ch_start), end=float(ch_end), title=title))

    if not chapters:
        chapters = [Chapter(start=0.0, end=media_duration, title="Chapter 1")]
    return chapters


def analyze(
    input_path: str | Path,
    *,
    cfg: Optional[ChapterConfig] = None,
    transcribe_cfg: Optional[TranscribeConfig] = None,
) -> EditDecision:
    """Run Whisper, then segment into chapters."""
    from ..utils import ffmpeg as ffu
    info = ffu.probe(input_path)
    transcript = transcribe(input_path, transcribe_cfg or TranscribeConfig())
    chapters = from_transcript(transcript, media_duration=info.duration, cfg=cfg)
    return EditDecision(
        kept_segments=[],
        removed_ranges=[],
        chapters=chapters,
        transcript=transcript,
        notes={"feature": "chapters", "input_duration": info.duration},
    )
