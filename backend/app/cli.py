"""Auto-Pod command-line interface.

Examples:
    auto-pod silence input.mp4 output.mp4
    auto-pod silence input.mp4 output.mp4 --mode amplitude --noise-db -28
    auto-pod chapters input.mp4 --json chapters.json
    auto-pod probe input.mp4
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .models import EditDecision
from .pipeline import chapters as chapters_pipe
from .pipeline import silence as silence_pipe
from .utils import ffmpeg as ffu
from .utils.transcribe import TranscribeConfig

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Auto-Pod: local-first AI video editor.",
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )


def _check_ffmpeg() -> None:
    if not ffu.have_ffmpeg():
        console.print(
            "[red]ffmpeg / ffprobe not found on PATH.[/]\n"
            "Install ffmpeg first (https://ffmpeg.org/download.html) and re-run."
        )
        raise typer.Exit(code=2)


@app.command()
def probe(input: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True)):
    """Print media info for a file."""
    _check_ffmpeg()
    info = ffu.probe(input)
    table = Table(title=f"Media info: {input.name}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    for k, v in info.to_dict().items():
        table.add_row(k, str(v))
    console.print(table)


@app.command()
def silence(
    input: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output: Path = typer.Argument(..., dir_okay=False, writable=True),
    mode: str = typer.Option(
        "transcript",
        "--mode",
        help="`transcript` (Whisper, recommended) or `amplitude` (ffmpeg silencedetect).",
    ),
    min_silence: float = typer.Option(0.5, "--min-silence", help="Min silence length to remove (sec)."),
    pad: float = typer.Option(0.05, "--pad", help="Padding to keep around speech (sec)."),
    noise_db: float = typer.Option(-30.0, "--noise-db", help="(amplitude mode) silence threshold dB."),
    model: str = typer.Option("small", "--model", help="(transcript mode) Whisper model size."),
    language: Optional[str] = typer.Option(None, "--language", help="Force language (e.g. 'en')."),
    no_reencode: bool = typer.Option(False, "--no-reencode", help="Stream-copy instead of re-encoding (faster, may shift cuts to keyframes)."),
    json_out: Optional[Path] = typer.Option(None, "--json", help="Write the edit decision to this JSON file."),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Remove silences from a video. Default mode uses Whisper word timestamps."""
    _setup_logging(verbose)
    _check_ffmpeg()

    if mode not in ("transcript", "amplitude"):
        console.print(f"[red]Unknown --mode '{mode}'. Use 'transcript' or 'amplitude'.[/]")
        raise typer.Exit(code=2)

    cfg = silence_pipe.SilenceConfig(
        mode=mode,  # type: ignore[arg-type]
        min_silence_sec=min_silence,
        pad_sec=pad,
        noise_db=noise_db,
        transcribe=TranscribeConfig(model_size=model, language=language),
    )

    console.print(f"[cyan]Analyzing[/] {input} (mode={mode})…")
    decision, out_path = silence_pipe.run(input, output, cfg, reencode=not no_reencode)

    _print_decision_summary(decision, out_path)
    if json_out:
        json_out.write_text(json.dumps(decision.to_dict(), indent=2), encoding="utf-8")
        console.print(f"[green]Wrote edit decision JSON:[/] {json_out}")


@app.command()
def chapters(
    input: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    json_out: Optional[Path] = typer.Option(None, "--json", help="Write chapters JSON."),
    target: Optional[int] = typer.Option(None, "--target", help="Aim for ~N chapters."),
    min_len: float = typer.Option(60.0, "--min-len", help="Minimum chapter length (sec)."),
    model: str = typer.Option("small", "--model", help="Whisper model size."),
    language: Optional[str] = typer.Option(None, "--language"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Generate chapter markers from spoken content."""
    _setup_logging(verbose)
    _check_ffmpeg()

    cfg = chapters_pipe.ChapterConfig(
        min_chapter_sec=min_len,
        target_chapters=target,
    )
    transcribe_cfg = TranscribeConfig(model_size=model, language=language)
    console.print(f"[cyan]Transcribing & chaptering[/] {input}…")
    decision = chapters_pipe.analyze(input, cfg=cfg, transcribe_cfg=transcribe_cfg)

    table = Table(title="Auto-detected chapters")
    table.add_column("#", style="cyan")
    table.add_column("Start", style="green")
    table.add_column("End", style="green")
    table.add_column("Title", style="white")
    for i, ch in enumerate(decision.chapters, 1):
        table.add_row(str(i), _fmt_time(ch.start), _fmt_time(ch.end), ch.title)
    console.print(table)

    if json_out:
        json_out.write_text(json.dumps(decision.to_dict(), indent=2), encoding="utf-8")
        console.print(f"[green]Wrote chapters JSON:[/] {json_out}")


def _fmt_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - (h * 3600 + m * 60)
    if h:
        return f"{h:d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


def _print_decision_summary(decision: EditDecision, out_path: Path) -> None:
    table = Table(title="Edit decision")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Kept segments", str(len(decision.kept_segments)))
    table.add_row("Removed ranges", str(len(decision.removed_ranges)))
    table.add_row("Total kept (sec)", f"{decision.total_kept:.2f}")
    table.add_row("Total removed (sec)", f"{decision.total_removed:.2f}")
    table.add_row("Output", str(out_path))
    console.print(table)


def main() -> None:  # entrypoint
    try:
        app()
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/]")
        sys.exit(130)


if __name__ == "__main__":
    main()
