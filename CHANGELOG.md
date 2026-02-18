# Changelog

## 1.1.0 (2026-02-17)

### Bug Fixes
- **CRITICAL**: `--beam-size` and `--initial-prompt` were passed to `model.transcribe()` which doesn't accept them — they're now correctly passed via `asr_options` to `load_model()` (previously these flags would crash if used)

### New Features
- `--hotwords` — boost recognition of specific terms via CTranslate2 hotword biasing (requires whisperx ≥3.7.5)
- `--merge-speakers` — merge consecutive segments from the same speaker for cleaner diarized output
- `--speaker-names "Alice,Bob,..."` — rename SPEAKER_00, SPEAKER_01, etc. to real names
- `--max-line-width N` — wrap subtitle text at word boundaries (standard TV: 42 chars)
- `--threads N` — control CTranslate2 CPU inference threads
- Stdin pipe support: use `-` as filename to read audio from stdin (e.g., `ffmpeg ... | ./scripts/transcribe -`)

### Improvements
- Diarization wrapped in try/except with diagnostic error messages (403 → model agreements, auth → token check) — falls back gracefully instead of crashing
- Empty results (no speech) now produce a warning and valid empty output instead of crashing
- Speaker summary stats printed after diarization (speaker count + talk time per speaker)
- VTT output now uses proper `<v Speaker>` voice tags when diarizing (WebVTT standard)
- Signal handling (SIGINT/SIGTERM) for graceful shutdown and temp file cleanup
- Validation warnings when flags require other flags (e.g., `--merge-speakers` without `--diarize`)

### Documentation
- Added hotwords vs initial prompt comparison guide
- Updated diarization setup for whisperx ≥3.8.0 (pyannote v4, community-1 model)
- Documented all new features with examples
- Added upstream changes section

## 1.0.0 (2026-02-17)

### Features
- Word-level timestamps via forced alignment (wav2vec2)
- Speaker diarization support (pyannote)
- Batched inference for 70x realtime speed
- Multiple output formats: TXT, SRT, VTT, JSON, TSV
- Word-level subtitles (`--word-level`) for karaoke-style SRT/VTT
- Time range trimming (`--start`/`--end`) for partial transcription
- Initial prompt (`--initial-prompt`) to improve domain-specific accuracy
- Auto-detect output format from `-o` file extension
- Auto GPU/CUDA detection
- Translation mode (any language → English)
- PyTorch 2.6+ compatibility patch (monkey-patches torch.load for pyannote)
- CLI wrapper with sensible defaults (large-v3-turbo, silero VAD)
