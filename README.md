# Auto-Pod

A local-first AI video editor for podcasts and talking-head video.

Built incrementally — one feature at a time. **Currently shipped: silence removal.**

## Status

| Feature | Status | Try it |
|---|---|---|
| Silence remover (Silero VAD + dB) | ✅ Ready | `auto-pod silence in.mp4 out.mp4` |
| Auto chapters | 🚧 Next | — |
| Repeat-takes remover | 🚧 Planned | — |
| Auto multicam editor | 🚧 Planned | — |
| Exports (MP4 / MOV / WebM / FCPXML / Premiere XML / EDL) | 🚧 Planned | — |
| Web UI | 🚧 Planned | — |

## How silence removal works

This is a faithful port of the method from [EduardoAndreu/clean-cut](https://github.com/EduardoAndreu/clean-cut), adapted to render a standalone output video instead of pushing cuts into Premiere Pro. Approach (rephrased for compliance with licensing restrictions):

1. **Decode** — ffmpeg extracts the audio as 16 kHz mono WAV.
2. **VAD** — [Silero VAD](https://github.com/snakers4/silero-vad) (a small neural net) classifies each 16 kHz frame as speech or non-speech and returns a list of speech intervals.
3. **dB safety net** — for each gap between speech segments, RMS dB is computed. The gap is *confirmed* as silence only if it's below your threshold (default `-35 dB`). This protects loud non-speech (music, applause, noise) from being cut.
4. **Padding** — confirmed silences are shrunk by `padding_ms` on each side (default 150 ms) so we don't clip word starts/ends.
5. **Merge** — silences within 0.5 s of each other are merged into one.
6. **Render** — the inverse (kept ranges) is concatenated by ffmpeg into a new video, frame-accurate by default.

ffmpeg is **only** used as a decoder/encoder. All cut decisions are timestamp-based.

## Requirements

- **ffmpeg / ffprobe** on PATH.
- **Python 3.10+**.
- ~30 MB disk for Silero VAD on first run (auto-downloaded).
- ~500 MB RAM during processing. CPU is fine; GPU optional.

### Install ffmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows: https://ffmpeg.org/download.html  (add bin/ to PATH)
```

## Install

```bash
git clone https://github.com/Nik-AeScripter/Auto-Pod
cd Auto-Pod/backend
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

> First run downloads the Silero VAD weights (~30 MB) and PyTorch (~100 MB).

## Try it (silence removal)

```bash
# Probe a file (sanity check ffmpeg is wired up)
auto-pod probe path/to/your-video.mp4

# Remove silences with defaults (matches clean-cut: -35 dB, 150 ms padding)
auto-pod silence path/to/your-video.mp4 path/to/output.mp4

# More aggressive (quieter threshold, less padding)
auto-pod silence in.mp4 out.mp4 --threshold-db -40 --padding-ms 80

# Less aggressive (catch more loud noise as "kept")
auto-pod silence in.mp4 out.mp4 --threshold-db -25 --padding-ms 200

# See exactly what would be cut, without rendering (fast)
auto-pod silence in.mp4 out.mp4 --dry-run --json cuts.json

# Stream-copy for faster but keyframe-aligned cuts
auto-pod silence in.mp4 out.mp4 --no-reencode

# Verbose logs (shows VAD progress, ffmpeg cmds, dB checks)
auto-pod silence in.mp4 out.mp4 -v
```

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--threshold-db, -t` | `-35.0` | Silence is confirmed only if quieter than this |
| `--min-silence-ms` | `200` | Ignore silences shorter than this |
| `--min-speech-ms` | `250` | Ignore speech bursts shorter than this |
| `--padding-ms, -p` | `150` | Keep this much audio on each side of cuts |
| `--merge-gap` | `0.5` | Merge silences within this many seconds |
| `--vad-threshold` | `0.5` | Silero VAD confidence (0..1, higher = stricter) |
| `--no-reencode` | off | Stream-copy cuts at keyframes (faster, less precise) |
| `--dry-run` | off | Analyze only, skip rendering |
| `--json` | — | Write the full edit decision (cuts, kept, stats) |

## What the JSON looks like

`auto-pod silence in.mp4 out.mp4 --dry-run --json cuts.json` produces:

```json
{
  "kept_segments": [
    {"source": "in.mp4", "start": 0.0,   "end": 4.85,  "duration": 4.85,  "track_index": 0, "label": "speech"},
    {"source": "in.mp4", "start": 6.32,  "end": 12.10, "duration": 5.78,  "track_index": 0, "label": "speech"}
  ],
  "removed_ranges": [
    {"start": 4.85, "end": 6.32, "duration": 1.47}
  ],
  "notes": {
    "method": "silero_vad+db_threshold",
    "input_duration": 12.10,
    "speech_segments_detected": 2,
    "candidate_silences": 3,
    "confirmed_silences": 1,
    "final_silences": 1,
    "config": { "silence_threshold_db": -35.0, "padding_ms": 150, ... }
  },
  "stats": { "total_kept_seconds": 10.63, "total_removed_seconds": 1.47 }
}
```

That JSON is what later features (multicam, exports) will consume.

## Project layout

```
auto-pod/
├── backend/
│   ├── app/
│   │   ├── cli.py             # Typer CLI
│   │   ├── models.py          # Shared dataclasses
│   │   ├── pipeline/
│   │   │   └── silence.py     # Silero VAD + dB silence removal
│   │   └── utils/
│   │       ├── ffmpeg.py      # ffmpeg/ffprobe wrappers (decode/encode only)
│   │       └── vad.py         # Silero VAD wrapper
│   ├── tests/                 # Pure-Python logic tests (no ffmpeg/VAD needed)
│   └── requirements.txt
└── README.md
```

## Run the tests

```bash
cd backend
PYTHONPATH=. pytest tests/ -v
```

These cover the cut math (interval inversion, dB filtering, padding, merging) without needing ffmpeg, torch, or audio files.

## Credits

Silence-removal method ported from [EduardoAndreu/clean-cut](https://github.com/EduardoAndreu/clean-cut) (MIT). Speech detection by [Silero VAD](https://github.com/snakers4/silero-vad).

## License

MIT.
