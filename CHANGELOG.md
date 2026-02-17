# Changelog

## 1.1.0 (2026-02-17)

### Features
- **Word-level subtitles**: `--word-level` flag for SRT/VTT — karaoke-style, one word per cue with precise timing
- **Time range trimming**: `--start` and `--end` to transcribe a section (supports seconds, MM:SS, HH:MM:SS)
- **Initial prompt**: `--initial-prompt` to condition the model with domain-specific terms, names, acronyms
- **TSV output**: `--tsv` for tab-separated values, spreadsheet-friendly
- **Auto-detect format**: `-o output.srt` auto-detects format from file extension (SRT, VTT, JSON, TSV, TXT)
- **Version flag**: `-V` / `--version`
- **Cleaner JSON output**: Structured segments with word-level confidence scores, properly serializable

### Fixes
- Fixed setup.sh pointing to wrong diarization model link (was community-1, now correctly references speaker-diarization-3.1 and segmentation-3.0)
- Better error handling: validates file existence, empty files, unsupported extensions, CUDA availability
- JSON output is now clean and serializable (no raw whisperx internal objects)
- Timestamps correctly offset when using `--start` trim

### Documentation
- Added output format examples (plain text, diarized, SRT, word-level SRT, JSON, TSV)
- Added supported languages list with alignment availability
- Added troubleshooting for empty output and timestamp trimming
- Expanded comparison table (WhisperX vs faster-whisper)
- Improved CLI help with more examples

## 1.0.0 (2026-02-17)

### Features
- Initial release
- Word-level timestamps via forced alignment
- Speaker diarization support (pyannote)
- Batched inference for 70x realtime speed
- Multiple output formats (TXT, SRT, VTT, JSON)
- Auto GPU/CUDA detection
- Translation mode (any language → English)
- CLI wrapper with sensible defaults
