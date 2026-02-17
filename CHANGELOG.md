# Changelog

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
