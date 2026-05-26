# Auto-Pod

A local-first AI video editor for podcasts and talking-head video. Three interfaces, one pipeline:

- **Web UI** — drag-and-drop, configure, preview, download.
- **CLI** — `auto-pod silence in.mp4 out.mp4`.
- **REST API** — JSON in, video out. Fully documented.

Everything runs on your machine. No cloud, no upload, no account.

## Features

| Feature | Status (Day 1) | How it works |
|---|---|---|
| **Silence remover** | Working | Whisper word timestamps → cut between words. Falls back to ffmpeg `silencedetect` if you opt out. |
| **Auto chapters** | Working | TF-IDF similarity over the transcript locates topic shifts; titles picked from the most distinctive sentence. |
| **Repeat takes remover** | Day 2 | Sentence-embedding similarity over the transcript; keep the last/best take. |
| **Auto multicam editor** | Day 2 | Audio cross-correlation for sync; per-window speaker activity for switching. |
| **Exports** | Day 2 | MP4 / MOV / WebM, plus FCPXML, Premiere XML, EDL. |

## Why transcript-driven cuts?

Cuts are placed using Whisper's word-level timestamps, not amplitude. This means:

- Background noise, music, breaths, lip smacks don't fool the cutter.
- Cuts always land between words, never mid-syllable.
- One transcription pass powers silence + repeats + chapters.

Amplitude mode (`--mode amplitude`) is still available for users who don't want to run Whisper.

## Requirements

- **ffmpeg / ffprobe** on PATH (used for decoding & rendering only — Python has no equivalent).
- **Python 3.10+**.
- ~500 MB disk for the default Whisper `small` model on first run.
- (Optional) NVIDIA GPU for faster transcription.

### Install ffmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add bin/ to PATH
```

## Quick start (CLI)

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Probe a file
auto-pod probe input.mp4

# Remove silences (transcript mode, recommended)
auto-pod silence input.mp4 output.mp4

# Faster, model-free silence removal
auto-pod silence input.mp4 output.mp4 --mode amplitude --noise-db -28

# Auto chapters
auto-pod chapters input.mp4 --json chapters.json
```

## Project layout

```
auto-pod/
├── backend/
│   ├── app/
│   │   ├── cli.py             # Typer CLI
│   │   ├── models.py          # Shared dataclasses
│   │   ├── pipeline/
│   │   │   ├── silence.py     # Silence removal (transcript + amplitude modes)
│   │   │   ├── chapters.py    # Auto chapters
│   │   │   ├── repeats.py     # Day 2: repeat-takes
│   │   │   ├── multicam.py    # Day 2: multicam
│   │   │   └── render.py      # Day 2: export & XML writers
│   │   └── utils/
│   │       ├── ffmpeg.py      # ffmpeg/ffprobe wrappers
│   │       └── transcribe.py  # faster-whisper wrapper
│   ├── tests/                 # Pure-Python logic tests
│   └── requirements.txt
├── frontend/                  # Day 3: Next.js UI
└── README.md
```

## How the silence pipeline works

1. **Probe** — `ffprobe` extracts duration, codecs, audio info.
2. **Transcribe** — `faster-whisper` (local) produces word-level timestamps.
3. **Decide** — words are merged into "speech spans" wherever the gap is shorter than `--min-silence`. Spans get padded by `--pad`.
4. **Render** — kept spans are concatenated. We re-encode by default for frame-accurate cuts; `--no-reencode` falls back to stream-copy at keyframes.

The `EditDecision` returned from `analyze()` is fully serializable JSON — every cut and every chapter is auditable.

## Roadmap

- **Day 1 (now)** — silence + chapters, CLI, logic tests.
- **Day 2** — repeats, multicam, FastAPI, FCPXML/Premiere XML/EDL exports.
- **Day 3** — Next.js UI with shadcn/Tailwind, Docker compose, polish.

## License

MIT.
