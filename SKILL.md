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

All commands use `./scripts/transcribe` (the skill wrapper), **not** the `whisperx` CLI directly. The wrapper applies a required PyTorch compatibility patch — see "PyTorch 2.6+ Compatibility" below.

| Task | Command | Notes |
|------|---------|-------|
| **Basic transcription** | `./scripts/transcribe audio.mp3` | Word-aligned by default |
| **With speakers** | `./scripts/transcribe audio.mp3 --diarize` | Auto-reads `~/.cache/huggingface/token` |
| **SRT subtitles** | `./scripts/transcribe audio.mp3 --srt -o subs.srt` | Ready for video players |
| **VTT subtitles** | `./scripts/transcribe audio.mp3 --vtt -o subs.vtt` | Web-compatible format |
| **JSON output** | `./scripts/transcribe audio.mp3 --json` | Full data with word timestamps |
| **Translate to English** | `./scripts/transcribe audio.mp3 --translate` | Any language → English |
| **Fast, no alignment** | `./scripts/transcribe audio.mp3 --no-align` | Skip forced alignment |
| **Specific language** | `./scripts/transcribe audio.mp3 -l en` | Faster than auto-detect |

**⚠️ Do NOT run `whisperx` CLI directly** — it will crash on PyTorch 2.6+ with pyannote models. Always use this skill's `./scripts/transcribe` wrapper.

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

### First-Time Setup

**Prerequisites:** Python 3.10+, ffmpeg, NVIDIA GPU with CUDA (strongly recommended)

**Step 1: Install whisperx**

```bash
# Option A: Run the setup script (auto-detects GPU, creates venv if needed)
./setup.sh

# Option B: Install globally (if you prefer)
pip install whisperx
```

**Step 2 (optional, for diarization): Set up Hugging Face token**

Speaker diarization requires a free Hugging Face account and access to two gated models. Skip this if you only need transcription/alignment.

1. Create account at [huggingface.co](https://huggingface.co) (if you don't have one)
2. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a **read** access token
3. Accept **both** model agreements (click "Agree and access repository" on each page):
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Save the token so the script auto-detects it:
   ```bash
   mkdir -p ~/.cache/huggingface && echo -n "hf_YOUR_TOKEN" > ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token
   ```
   Alternatively: set `HF_TOKEN` env var, or pass `--hf-token` per-command.

**Note:** The token and model access are completely free. The models are just gated behind a click-to-agree license. Without step 3, you'll get a 403 error even with a valid token.

### Subsequent Runs

No setup needed — just run `./scripts/transcribe`. The wrapper script:
- Auto-detects GPU/CPU and picks optimal compute type
- Auto-reads the HF token from `~/.cache/huggingface/token` for diarization
- Applies the PyTorch 2.6+ compatibility patch automatically (see below)
- First run for a new model downloads it to `~/.cache/huggingface/` (one-time per model)

### Checking If It's Working

```bash
# Quick test — should print transcript to stdout
./scripts/transcribe some_audio.mp3

# Test diarization — should show [SPEAKER_00], [SPEAKER_01], etc.
./scripts/transcribe some_audio.mp3 --diarize

# If diarization fails with 403: model agreements not accepted (see step 3 above)
# If it crashes with pickle/weights_only error: you're running `whisperx` CLI directly instead of the wrapper
```

### Platform Support

| Platform | Acceleration | Speed |
|----------|-------------|-------|
| **Linux + NVIDIA GPU** | CUDA (batched) | ~70x realtime 🚀 |
| **WSL2 + NVIDIA GPU** | CUDA (batched) | ~70x realtime 🚀 |
| macOS Apple Silicon | CPU | ~3-5x realtime |
| macOS Intel | CPU | ~1-2x realtime |
| Linux (no GPU) | CPU | ~1x realtime |

## Usage

All commands use `./scripts/transcribe` — resolve the path relative to this skill's directory.

```bash
# Basic transcription (word-aligned)
./scripts/transcribe audio.mp3

# With speaker diarization (auto-reads ~/.cache/huggingface/token)
./scripts/transcribe audio.mp3 --diarize

# With explicit HF token
./scripts/transcribe audio.mp3 --diarize --hf-token hf_YOUR_TOKEN

# Generate SRT subtitles
./scripts/transcribe audio.mp3 --srt -o subtitles.srt

# Maximum accuracy
./scripts/transcribe audio.mp3 --model large-v3

# Translate non-English audio to English
./scripts/transcribe audio.mp3 --translate -l ja

# Fast mode (skip alignment)
./scripts/transcribe audio.mp3 --no-align

# JSON with full metadata and word timestamps
./scripts/transcribe audio.mp3 --json -o transcript.json

# Specify known speaker count for better diarization
./scripts/transcribe audio.mp3 --diarize --min-speakers 2 --max-speakers 4
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
  --hf-token TOKEN       Hugging Face access token (also reads ~/.cache/huggingface/token or HF_TOKEN env)
  --min-speakers N       Minimum speaker count hint
  --max-speakers N       Maximum speaker count hint

Output options:
  -j, --json             JSON output with segments and word timestamps
  --srt                  SRT subtitle format
  --vtt                  WebVTT subtitle format
  -o, --output FILE      Save to file instead of stdout

Miscellaneous:
  -q, --quiet            Suppress progress messages
```

## Examples

```bash
# Transcribe a meeting recording with speakers
./scripts/transcribe meeting.mp3 --diarize \
  --min-speakers 3 --max-speakers 5 --json -o meeting.json

# Generate subtitles
./scripts/transcribe video.mp4 --srt -o video.srt

# Batch transcribe a folder
for file in recordings/*.mp3; do
  ./scripts/transcribe "$file" --json -o "${file%.mp3}.json"
done

# Transcribe YouTube audio (with yt-dlp)
yt-dlp -x --audio-format mp3 <URL> -o audio.mp3
./scripts/transcribe audio.mp3 --diarize

# Quick draft (fast, no alignment)
./scripts/transcribe audio.mp3 --model base --no-align

# German audio
./scripts/transcribe audio.mp3 -l de --model large-v3-turbo
```

## Common Mistakes

| Mistake | Problem | Solution |
|---------|---------|----------|
| **Using CPU when GPU available** | 10-70x slower | Check `nvidia-smi`; verify CUDA |
| **Missing HF token for diarize** | Diarization fails | Get token from huggingface.co/settings/tokens |
| **Not accepting model agreements** | 403 error on diarization model | Accept **both** pyannote/speaker-diarization-3.1 AND pyannote/segmentation-3.0 (see Setup) |
| **Running `whisperx` CLI directly** | Crashes on PyTorch 2.6+ | Always use `./scripts/transcribe` wrapper (applies torch.load patch) |
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

## PyTorch 2.6+ Compatibility (CRITICAL)

**⚠️ PyTorch 2.6 changed `torch.load()` to default to `weights_only=True`.** This breaks pyannote.audio's model loading (used by both VAD and diarization), because the model checkpoints contain globals like `omegaconf.listconfig.ListConfig` and `torch.torch_version.TorchVersion` that aren't allowlisted.

**Symptoms:**
- `_pickle.UnpicklingError: Weights only load failed` when loading VAD or diarization models
- `'NoneType' object has no attribute 'to'` (pipeline silently returns None)

**How this skill handles it:**

The `scripts/transcribe.py` uses the **whisperx Python API directly** (not as a subprocess) so it can monkey-patch `torch.load` before any model loading happens:

```python
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False  # Must FORCE, not setdefault — lightning_fabric passes True explicitly
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
```

**Key details:**
- The patch **must** use `kwargs['weights_only'] = False` (forced override), NOT `kwargs.setdefault('weights_only', False)` — because `lightning_fabric` explicitly passes `weights_only=True`, which `setdefault` won't override
- The patch **must** be applied before importing `whisperx`, `pyannote`, or any model loading code
- This is why the skill uses the Python API instead of shelling out to `whisperx` CLI — a subprocess can't inherit the monkey-patch
- VAD defaults to **silero** (not pyannote) to avoid a second torch.load issue in the VAD pipeline. Silero loads fine without the patch, but diarization still needs it

**If whisperx CLI is updated to fix this upstream**, the monkey-patch can be removed and the script could switch back to subprocess mode. Track: [whisperX#972](https://github.com/m-bain/whisperX/issues/972)

## Troubleshooting

**`_pickle.UnpicklingError: Weights only load failed`**: PyTorch 2.6+ compat issue. If running via CLI (`whisperx` command directly), this can't be fixed without patching the installed library. Use this skill's `scripts/transcribe` wrapper instead, which applies the patch automatically. See "PyTorch 2.6+ Compatibility" section above.

**"CUDA not available"**: Install PyTorch with CUDA (`pip install torch --index-url https://download.pytorch.org/whl/cu121`)

**"No module named whisperx"**: Run `./setup.sh` or `pip install whisperx`

**Diarization 403 error**: You must accept **both** model agreements — see "Speaker Diarization Setup" above

**`'NoneType' object has no attribute 'to'`**: Either the HF token is invalid, the model agreements haven't been accepted, or the torch.load patch isn't applied. Check all three.

**OOM on GPU**: Lower `--batch-size` to 4 or 2

**Alignment fails for language X**: Check supported languages in [whisperx alignment.py](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py)

**Slow on CPU**: Expected — use GPU for practical transcription

## References

- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [WhisperX Paper (INTERSPEECH 2023)](https://arxiv.org/abs/2303.00747)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — speaker diarization
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 backend
