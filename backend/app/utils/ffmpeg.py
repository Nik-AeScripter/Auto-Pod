"""Thin, well-tested wrappers around ffmpeg / ffprobe.

Design rules:
- All commands raise RuntimeError on non-zero exit, with stderr captured.
- Never construct shell strings; always pass arg lists to subprocess.
- Probe / analysis methods return plain Python types (no exotic objects).
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional

from ..models import MediaInfo

log = logging.getLogger(__name__)


class FFmpegError(RuntimeError):
    """Raised when ffmpeg/ffprobe fails."""


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise FFmpegError(
            f"`{name}` not found on PATH. Install ffmpeg and ensure it is on PATH "
            f"(see README for instructions)."
        )
    return path


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def run(cmd: list[str], *, capture: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess command, log it, and surface a clean error if it fails."""
    log.debug("RUN: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
    )
    if check and proc.returncode != 0:
        raise FFmpegError(
            f"Command failed (exit {proc.returncode}): {' '.join(cmd)}\n"
            f"stderr:\n{(proc.stderr or '')[-2000:]}"
        )
    return proc


def probe(path: str | Path) -> MediaInfo:
    """Probe a media file with ffprobe and return a MediaInfo."""
    ffprobe = _require_binary("ffprobe")
    p = str(path)
    cmd = [
        ffprobe, "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        p,
    ]
    proc = run(cmd)
    data = json.loads(proc.stdout or "{}")

    fmt = data.get("format", {}) or {}
    streams = data.get("streams", []) or []
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if not fmt:
        raise FFmpegError(f"ffprobe returned no format info for {p}")

    duration = float(fmt.get("duration") or (v or {}).get("duration") or 0.0)

    width = int((v or {}).get("width") or 0)
    height = int((v or {}).get("height") or 0)
    fps = _parse_fps((v or {}).get("r_frame_rate", "0/0"))

    has_audio = a is not None
    sr = int((a or {}).get("sample_rate") or 0)

    return MediaInfo(
        path=p,
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        has_audio=has_audio,
        audio_sample_rate=sr,
        container=fmt.get("format_name", ""),
        video_codec=(v or {}).get("codec_name"),
        audio_codec=(a or {}).get("codec_name"),
    )


def _parse_fps(rate: str) -> float:
    try:
        if "/" in rate:
            n, d = rate.split("/", 1)
            n_f, d_f = float(n), float(d)
            return (n_f / d_f) if d_f else 0.0
        return float(rate)
    except Exception:
        return 0.0


def extract_audio_wav(
    input_path: str | Path,
    output_path: str | Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """Extract audio as mono 16kHz WAV (ideal for Whisper / silence detection)."""
    ffmpeg = _require_binary("ffmpeg")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        str(out),
    ]
    run(cmd)
    return out


def detect_silence(
    input_path: str | Path,
    *,
    noise_db: float = -30.0,
    min_silence_sec: float = 0.5,
) -> list[tuple[float, float]]:
    """Run ffmpeg silencedetect filter and return list of (silence_start, silence_end)."""
    ffmpeg = _require_binary("ffmpeg")
    cmd = [
        ffmpeg, "-hide_banner", "-nostats",
        "-i", str(input_path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence_sec}",
        "-f", "null", "-",
    ]
    # silencedetect emits to stderr regardless of exit; tolerate non-zero just in case.
    proc = run(cmd, check=False)
    err = proc.stderr or ""
    starts: list[float] = []
    ends: list[float] = []
    for line in err.splitlines():
        line = line.strip()
        if "silence_start:" in line:
            try:
                starts.append(float(line.split("silence_start:")[1].strip().split()[0]))
            except (ValueError, IndexError):
                continue
        elif "silence_end:" in line:
            # format: ... silence_end: 1.234 | silence_duration: 0.567
            try:
                token = line.split("silence_end:")[1].strip().split("|")[0].strip()
                ends.append(float(token.split()[0]))
            except (ValueError, IndexError):
                continue

    pairs: list[tuple[float, float]] = []
    for i, s in enumerate(starts):
        if i < len(ends):
            pairs.append((s, ends[i]))
        else:
            # silence still ongoing at EOF; ignore — we'd need duration to close it
            pass
    return pairs


def render_segments(
    segments: Iterable[tuple[str, float, float]],
    output_path: str | Path,
    *,
    reencode: bool = True,
    target_video_codec: str = "libx264",
    target_audio_codec: str = "aac",
    crf: int = 20,
    preset: str = "veryfast",
    extra_video_args: Optional[list[str]] = None,
) -> Path:
    """Render a final video by concatenating (source, start, end) segments.

    Strategy: re-encode each segment to a uniform format, then concat-demuxer-join.
    This avoids container/keyframe alignment issues. With reencode=False we attempt
    stream-copy first; on failure caller can retry with reencode=True.
    """
    ffmpeg = _require_binary("ffmpeg")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    seg_list = [s for s in segments if (s[2] - s[1]) > 0.02]
    if not seg_list:
        raise FFmpegError("No segments to render (all empty).")

    tmp_dir = out.parent / f".{out.stem}_segs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    part_paths: list[Path] = []
    try:
        for i, (src, start, end) in enumerate(seg_list):
            part = tmp_dir / f"part_{i:05d}.mp4"
            cmd = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{max(0.0, start):.3f}",
                "-to", f"{end:.3f}",
                "-i", str(src),
            ]
            if reencode:
                cmd += [
                    "-c:v", target_video_codec,
                    "-preset", preset,
                    "-crf", str(crf),
                    "-pix_fmt", "yuv420p",
                    "-c:a", target_audio_codec,
                    "-b:a", "192k",
                    "-movflags", "+faststart",
                ]
                if extra_video_args:
                    cmd += extra_video_args
            else:
                cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
            cmd += [str(part)]
            run(cmd)
            part_paths.append(part)

        # concat list
        list_file = tmp_dir / "concat.txt"
        list_file.write_text(
            "\n".join(f"file '{p.as_posix()}'" for p in part_paths) + "\n",
            encoding="utf-8",
        )

        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            "-movflags", "+faststart",
            str(out),
        ]
        run(cmd)
    finally:
        # cleanup parts but keep output
        for p in part_paths:
            try:
                p.unlink()
            except OSError:
                pass
        try:
            (tmp_dir / "concat.txt").unlink()
        except OSError:
            pass
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    return out


def transcode(
    input_path: str | Path,
    output_path: str | Path,
    *,
    container: str = "mp4",
    video_codec: Optional[str] = None,
    audio_codec: Optional[str] = None,
    crf: int = 20,
    preset: str = "veryfast",
) -> Path:
    """Transcode a finished video to a different container/codec (mp4/mov/webm)."""
    ffmpeg = _require_binary("ffmpeg")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if video_codec is None:
        video_codec = {"mp4": "libx264", "mov": "libx264", "webm": "libvpx-vp9"}.get(container, "libx264")
    if audio_codec is None:
        audio_codec = {"mp4": "aac", "mov": "aac", "webm": "libopus"}.get(container, "aac")

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_path),
        "-c:v", video_codec,
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", audio_codec,
    ]
    if container in ("mp4", "mov"):
        cmd += ["-movflags", "+faststart"]
    cmd += [str(out)]
    run(cmd)
    return out
