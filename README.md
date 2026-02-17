# whisperx

A skill for your OpenClaw agent that uses [WhisperX](https://github.com/m-bain/whisperX) for speech-to-text with word-level timestamps, speaker diarization, and forced alignment.

WhisperX builds on faster-whisper and adds:
- **Word-level timestamps** via forced alignment (phoneme ASR models)
- **Speaker diarization** — label who said what (via pyannote.audio)
- **Batched inference** — up to 70x realtime speed on GPU
- **Subtitle generation** — SRT, VTT with accurate word timing

## Quick Start

```bash
# Basic transcription (word-aligned by default)
./scripts/transcribe audio.mp3

# With speaker diarization
./scripts/transcribe audio.mp3 --diarize --hf-token YOUR_HF_TOKEN

# Generate SRT subtitles
./scripts/transcribe audio.mp3 --srt -o subtitles.srt

# JSON output with full metadata
./scripts/transcribe audio.mp3 --json

# Translate to English
./scripts/transcribe audio.mp3 --translate
```

## Setup

```bash
# Auto-setup (detects GPU, installs dependencies)
./setup.sh
```

Or install whisperx manually: `pip install whisperx`

### Speaker Diarization

Requires a free [Hugging Face token](https://huggingface.co/settings/tokens) and accepting the [pyannote speaker-diarization model](https://huggingface.co/pyannote/speaker-diarization-community-1).

## When to Use This vs faster-whisper

| Need | Use |
|------|-----|
| Simple transcription | faster-whisper (faster than whisperx as fewer features need to be loaded) |
| Word-level timestamps | **whisperx** |
| Speaker identification | **whisperx** |
| Subtitle generation | **whisperx** |
| Lightest setup | faster-whisper |

## Requirements

- Python 3.10+
- ffmpeg
- NVIDIA GPU with CUDA (recommended for practical speed)

## License

MIT
