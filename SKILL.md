---
name: whisperx
description: Speech-to-text with word-level timestamps, speaker diarization, and forced alignment using WhisperX. Built on faster-whisper with batched inference for 70x realtime speed.
version: 1.0.0
author: Sarah Mak
homepage: https://github.com/ThePlasmak/whisperx
tags: ["audio", "transcription", "whisperx", "speech-to-text", "diarization", "alignment", "subtitles", "ml", "cuda", "gpu"]
platforms: ["linux", "macos", "wsl2"]
metadata: {"openclaw":{"emoji":"🎙️","requires":{"bins":["ffmpeg","python3"]}}}
---

# WhisperX

Speech-to-text with **word-level timestamps**, **speaker diarization**, and **forced alignment** — built on faster-whisper with batched inference for up to **70x realtime** transcription speed.

WhisperX extends Whisper with three key capabilities that faster-whisper alone doesn't provide:
1. **Forced alignment** — precise word-level timestamps via phoneme ASR models (wav2vec2)
2. **Speaker diarization** — label who said what (via pyannote.audio)
3. **Batched inference** — process audio in parallel chunks for massive speedup

## When to Use

Use this skill when you need to:
- **Transcribe with word-level timing** — subtitles, captions, karaoke-style highlighting
- **Identify speakers** — meetings, interviews, podcasts, multi-speaker recordings
- **Generate subtitle files** — SRT, VTT with accurate word timestamps
- **Translate speech to English** — from any supported language
- **Batch transcribe** — efficient processing of multiple files

**Trigger phrases:** "transcribe with speakers", "who said what", "diarize", "make subtitles", "word timestamps", "speaker identification", "meeting transcript"

**When NOT to use:**
- Simple transcription without speaker/timing needs → use **faster-whisper** (lighter, faster)
- Real-time/streaming transcription
- Files <10 seconds where setup overhead isn't worth it

**WhisperX vs faster-whisper:**
| Feature | faster-whisper | WhisperX |
|---------|---------------|----------|
| Basic transcription | ✅ | ✅ |
| Word timestamps | ✅ (approximate) | ✅ (precise, aligned) |
| Speaker diarization | ❌ | ✅ |
| Forced alignment | ❌ | ✅ |
| Batched inference | ❌ | ✅ |
| Subtitle generation | Manual | Built-in (SRT/VTT) |
| Setup complexity | Simple | Requires HF token for diarization |

## Quick Reference

| Task | Command | Notes |
|------|---------|-------|
| **Basic transcription** | `./scripts/transcribe audio.mp3` | Word-aligned by default |
| **With speakers** | `./scripts/transcribe audio.mp3 --diarize --hf-token TOKEN` | Labels each segment |
| **SRT subtitles** | `./scripts/transcribe audio.mp3 --srt -o subs.srt` | Ready for video players |
| **VTT subtitles** | `./scripts/transcribe audio.mp3 --vtt -o subs.vtt` | Web-compatible format |
| **JSON output** | `./scripts/transcribe audio.mp3 --json` | Full data with word timestamps |
| **All formats** | `./scripts/transcribe audio.mp3 --output-format all` | Generates SRT+VTT+TXT+JSON+TSV |
| **Translate to English** | `./scripts/transcribe audio.mp3 --translate` | Any language → English |
| **Fast, no alignment** | `./scripts/transcribe audio.mp3 --no-align` | Skip forced alignment |
| **Specific language** | `./scripts/transcribe audio.mp3 -l en` | Faster than auto-detect |

## Model Selection

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `tiny` | 39M | Fastest | Basic | Quick drafts, testing |
| `base` | 74M | Very fast | Good | General use |
| `small` | 244M | Fast | Better | Default for whisperx CLI |
| `medium` | 769M | Moderate | High | Quality transcription |
| `large-v2` | 1.5GB | Slower | Excellent | Best diarization compat |
| `large-v3` | 1.5GB | Slower | Best | Maximum accuracy |
| **`large-v3-turbo`** | 809M | Fast | Excellent | **Recommended (default)** |

**Note:** WhisperX defaults to `small` but this skill defaults to `large-v3-turbo` for the best speed/accuracy balance with GPU.

## Setup

### Prerequisites
- Python 3.10+
- ffmpeg
- NVIDIA GPU with CUDA (strongly recommended)

### Installation

```bash
# Run the setup script (detects existing install or creates venv)
./setup.sh
```

Or install manually:

```bash
pip install whisperx
```

### Speaker Diarization Setup

Diarization requires a free Hugging Face account and access to two gated models:

1. Create account at [huggingface.co](https://huggingface.co) (if you don't have one)
2. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a **read** access token
3. Accept **both** model agreements (click "Agree and access repository" on each):
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Save the token (pick one method):
   - **Recommended:** `mkdir -p ~/.cache/huggingface && echo -n "hf_YOUR_TOKEN" > ~/.cache/huggingface/token` (auto-detected by the script)
   - Or set env var: `export HF_TOKEN=hf_YOUR_TOKEN`
   - Or pass per-command: `--hf-token hf_YOUR_TOKEN`

**Note:** The token and model access are completely free. The models are just gated behind a click-to-agree license. Without step 3, you'll get a 403 error even with a valid token.

### Platform Support

| Platform | Acceleration | Speed |
|----------|-------------|-------|
| **Linux + NVIDIA GPU** | CUDA (batched) | ~70x realtime 🚀 |
| **WSL2 + NVIDIA GPU** | CUDA (batched) | ~70x realtime 🚀 |
| macOS Apple Silicon | CPU | ~3-5x realtime |
| macOS Intel | CPU | ~1-2x realtime |
| Linux (no GPU) | CPU | ~1x realtime |

## Usage

```bash
# Basic transcription (word-aligned)
./scripts/transcribe audio.mp3

# With speaker diarization
./scripts/transcribe audio.mp3 --diarize --hf-token YOUR_TOKEN

# Generate SRT subtitles with word highlighting
./scripts/transcribe audio.mp3 --srt --highlight-words -o subtitles.srt

# Maximum accuracy
./scripts/transcribe audio.mp3 --model large-v3 --beam-size 10

# Translate non-English audio to English
./scripts/transcribe audio.mp3 --translate --language ja

# Fast mode (skip alignment)
./scripts/transcribe audio.mp3 --no-align

# Generate all output formats
./scripts/transcribe audio.mp3 --output-format all --output-dir ./output

# JSON with full metadata
./scripts/transcribe audio.mp3 --json -o transcript.json

# Specify known speaker count for better diarization
./scripts/transcribe audio.mp3 --diarize --min-speakers 2 --max-speakers 4 --hf-token TOKEN
```

## Options

```
AUDIO_FILE               Path to audio/video file

Model options:
  -m, --model NAME       Whisper model (default: large-v3-turbo)
  --batch-size N         Batch size for inference (default: 8, lower if OOM)
  --beam-size N          Beam search size (higher = slower but more accurate)

Device options:
  --device               cpu, cuda, or auto (default: auto)
  --compute-type         int8, float16, float32, or auto (default: auto)

Language options:
  -l, --language CODE    Language code (auto-detects if omitted)
  --translate            Translate to English

Alignment options:
  --no-align             Skip forced alignment (no word timestamps)
  --align-model MODEL    Custom phoneme ASR model for alignment

Speaker diarization:
  --diarize              Enable speaker labels
  --hf-token TOKEN       Hugging Face access token
  --min-speakers N       Minimum speaker count hint
  --max-speakers N       Maximum speaker count hint

Output options:
  -j, --json             JSON output with segments and word timestamps
  --srt                  SRT subtitle format
  --vtt                  WebVTT subtitle format
  --output-format FMT    all, srt, vtt, txt, tsv, json, aud (default: txt)
  -o, --output FILE      Save to file instead of stdout
  --output-dir DIR       Directory for output files
  --highlight-words      Underline words as spoken in SRT/VTT

Miscellaneous:
  --suppress-numerals    Spell out numbers instead of digits
  --verbose              Show full whisperx output
  -q, --quiet            Suppress progress messages
```

## Examples

```bash
# Transcribe a meeting recording with speakers
./scripts/transcribe meeting.mp3 --diarize --hf-token TOKEN \
  --min-speakers 3 --max-speakers 5 --json -o meeting.json

# Generate YouTube-style subtitles
./scripts/transcribe video.mp4 --srt --highlight-words -o video.srt

# Batch transcribe a folder
for file in recordings/*.mp3; do
  ./scripts/transcribe "$file" --json -o "${file%.mp3}.json"
done

# Transcribe YouTube audio (with yt-dlp)
yt-dlp -x --audio-format mp3 <URL> -o audio.mp3
./scripts/transcribe audio.mp3 --diarize --hf-token TOKEN

# Quick draft (fast, no alignment)
./scripts/transcribe audio.mp3 --model base --no-align

# German audio with specific alignment model
./scripts/transcribe audio.mp3 -l de --model large-v3-turbo
```

## Common Mistakes

| Mistake | Problem | Solution |
|---------|---------|----------|
| **Using CPU when GPU available** | 10-70x slower | Check `nvidia-smi`; verify CUDA |
| **Missing HF token for diarize** | Diarization fails | Get token from huggingface.co/settings/tokens |
| **Not accepting model agreement** | 403 error on diarization model | Accept at huggingface.co/pyannote/speaker-diarization-community-1 |
| **batch_size too high** | CUDA OOM | Lower `--batch-size` (try 4 or 2) |
| **Using large-v3 when turbo works** | Unnecessary slowdown | `large-v3-turbo` is faster with near-identical accuracy |
| **Forgetting --language** | Wastes time auto-detecting | Specify `-l en` when you know the language |
| **Using WhisperX for simple transcription** | Heavier setup for no benefit | Use faster-whisper for basic transcription |

## Performance Notes

- **First run**: Downloads model + alignment model (one-time)
- **GPU batched inference**: Up to 70x realtime with large-v2
- **Diarization**: Adds ~30-60s overhead for model loading
- **Memory** (GPU VRAM):
  - `large-v3-turbo`: ~2-3GB
  - `large-v3` + diarization: ~4-5GB
  - Reduce `--batch-size` if OOM

## Troubleshooting

**"CUDA not available"**: Install PyTorch with CUDA (`pip install torch --index-url https://download.pytorch.org/whl/cu121`)

**"No module named whisperx"**: Run `./setup.sh` or `pip install whisperx`

**Diarization 403 error**: Accept model agreement at huggingface.co/pyannote/speaker-diarization-community-1

**OOM on GPU**: Lower `--batch-size` to 4 or 2

**Alignment fails for language X**: Check supported languages in [whisperx alignment.py](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py)

**Slow on CPU**: Expected — use GPU for practical transcription

## References

- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [WhisperX Paper (INTERSPEECH 2023)](https://arxiv.org/abs/2303.00747)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — speaker diarization
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 backend
