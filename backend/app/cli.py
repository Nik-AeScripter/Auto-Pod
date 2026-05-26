"""Auto-Pod command-line interface.

Currently implemented:
  auto-pod probe   <input>                        -- probe media info
  auto-pod silence <input> <output>               -- remove silences (Silero VAD + dB)

Coming next: chapters, repeats, multicam, exports.
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
from .pipeline import silence as silence_pipe
from .utils import ffmpeg as ffu

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
    threshold_db: float = typer.Option(
        -35.0, "--threshold-db", "-t",
        help="Confirm silence only if dB level is below this. Default matches clean-cut.",
    ),
    min_silence_ms: int = typer.Option(
        200, "--min-silence-ms",
        help="Ignore silences shorter than this (default 200ms).",
    ),
    min_speech_ms: int = typer.Option(
        250, "--min-speech-ms",
        help="Ignore speech bursts shorter than this (default 250ms).",
    ),
    padding_ms: int = typer.Option(
        150, "--padding-ms", "-p",
        help="Keep this much audio around each kept span (default 150ms).",
    ),
    merge_gap: float = typer.Option(
        0.5, "--merge-gap",
        help="Merge silences that are within this many seconds of each other.",
    ),
    vad_threshold: float = typer.Option(
        0.5, "--vad-threshold",
        help="Silero VAD confidence threshold (0..1). Higher = stricter speech detection.",
    ),
    no_reencode: bool = typer.Option(
        False, "--no-reencode",
        help="Stream-copy instead of re-encoding (faster, may shift cuts to keyframes).",
    ),
    json_out: Optional[Path] = typer.Option(
        None, "--json",
        help="Write the edit decision (cuts, kept segments, stats) to this JSON file.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Analyze only — don't render the output video.",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Remove silences from a video using Silero VAD + dB threshold (clean-cut method)."""
    _setup_logging(verbose)
    _check_ffmpeg()

    cfg = silence_pipe.SilenceConfig(
        silence_threshold_db=threshold_db,
        min_silence_ms=min_silence_ms,
        min_speech_ms=min_speech_ms,
        padding_ms=padding_ms,
        merge_gap_sec=merge_gap,
        vad_threshold=vad_threshold,
    )

    console.print(f"[cyan]Analyzing[/] {input}…")
    decision = silence_pipe.analyze(input, cfg)

    _print_decision_summary(decision, output if not dry_run else None)

    if json_out:
        json_out.write_text(json.dumps(decision.to_dict(), indent=2), encoding="utf-8")
        console.print(f"[green]Wrote edit decision JSON:[/] {json_out}")

    if dry_run:
        console.print("[yellow]Dry run — skipping render.[/]")
        return

    console.print(f"[cyan]Rendering[/] {output}…")
    out_path = silence_pipe.render(decision, output, reencode=not no_reencode)
    console.print(f"[green]Done.[/] Output: {out_path}")


def _fmt_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - (h * 3600 + m * 60)
    if h:
        return f"{h:d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


def _print_decision_summary(decision: EditDecision, out_path: Optional[Path]) -> None:
    notes = decision.notes or {}
    input_dur = float(notes.get("input_duration") or 0.0)
    saved = decision.total_removed
    saved_pct = (100.0 * saved / input_dur) if input_dur else 0.0

    table = Table(title="Edit decision")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Method", str(notes.get("method", "?")))
    table.add_row("Input duration", _fmt_time(input_dur))
    table.add_row("Speech segments (VAD)", str(notes.get("speech_segments_detected", "—")))
    table.add_row("Candidate silences", str(notes.get("candidate_silences", "—")))
    table.add_row("Confirmed by dB", str(notes.get("confirmed_silences", "—")))
    table.add_row("Final cuts", str(notes.get("final_silences", "—")))
    table.add_row("Total removed", f"{_fmt_time(saved)}  ({saved_pct:.1f}% of input)")
    table.add_row("Total kept", _fmt_time(decision.total_kept))
    if out_path is not None:
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
