#!/usr/bin/env python3
"""
WhisperX transcription CLI wrapper
Speech-to-text with word-level alignment, speaker diarization, and batched inference.

Uses the whisperx Python API directly (not subprocess) to apply PyTorch 2.6+
compatibility patches for pyannote model loading.
"""

__version__ = "1.1.0"

import sys
import os
import json
import argparse
import time
from pathlib import Path

# --- PyTorch 2.6+ compatibility patch ---
# pyannote.audio and lightning_fabric use torch.load with weights_only=True (new default),
# but the model checkpoints contain globals that aren't allowlisted.
# Force weights_only=False for these trusted HuggingFace models.
try:
    import torch
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
except ImportError:
    pass
# --- End patch ---

SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus",
    ".webm", ".mp4", ".mkv", ".avi", ".mov", ".wma", ".aac",
}

# Auto-detect output format from file extension
FORMAT_FROM_EXT = {
    ".srt": "srt",
    ".vtt": "vtt",
    ".json": "json",
    ".tsv": "tsv",
    ".txt": "txt",
}


def check_cuda_available():
    """Check if CUDA is available and return device info."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
        return False, None
    except ImportError:
        return False, None


def get_hf_token(args_token):
    """Resolve HF token from args, env, or cache."""
    if args_token:
        return args_token
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    return None


def trim_audio(audio, sr, start=None, end=None):
    """Trim audio array to start/end times (in seconds). sr is sample rate."""
    if start is not None:
        start_sample = int(start * sr)
        audio = audio[start_sample:]
    if end is not None:
        end_sample = int(end * sr)
        if start is not None:
            # end is relative to original, adjust for already-trimmed start
            end_sample = end_sample - int(start * sr)
        audio = audio[:end_sample]
    return audio


def parse_time(time_str):
    """Parse a time string like '1:30', '90', '1:02:30' into seconds."""
    if time_str is None:
        return None
    parts = time_str.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        else:
            raise ValueError
    except (ValueError, IndexError):
        print(f"❌ Invalid time format: '{time_str}'. Use seconds (90), MM:SS (1:30), or HH:MM:SS (1:02:30)", file=sys.stderr)
        sys.exit(1)


def run_whisperx(args):
    """Run whisperx using the Python API."""
    try:
        import whisperx
    except ImportError:
        print("❌ whisperx not installed", file=sys.stderr)
        print("   Install: pip install whisperx", file=sys.stderr)
        print("   Or run: ./setup.sh", file=sys.stderr)
        sys.exit(1)

    import gc

    audio_path = Path(args.audio)

    # Validate file extension
    ext = audio_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        print(f"⚠️  Unrecognized audio format '{ext}'. Trying anyway (ffmpeg may handle it).", file=sys.stderr)

    # Resolve device
    cuda_available, gpu_name = check_cuda_available()
    if args.device == "auto":
        device = "cuda" if cuda_available else "cpu"
    else:
        device = args.device

    if device == "cuda" and not cuda_available:
        print("❌ CUDA requested but not available. Use --device cpu or install CUDA.", file=sys.stderr)
        sys.exit(1)

    if device == "cpu" and not args.quiet:
        print("⚠️  CUDA not available — using CPU (this will be slow!)", file=sys.stderr)

    # Resolve compute type
    if args.compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    else:
        compute_type = args.compute_type

    if not args.quiet:
        if device == "cuda" and gpu_name:
            print(f"🎙️  WhisperX | model: {args.model} | {device} ({compute_type}) on {gpu_name}", file=sys.stderr)
        else:
            print(f"🎙️  WhisperX | model: {args.model} | {device} ({compute_type})", file=sys.stderr)

        features = []
        if args.diarize:
            features.append("diarization")
        if not args.no_align:
            features.append("alignment")
        if args.word_level:
            features.append("word-level subtitles")
        if features:
            print(f"   Features: {', '.join(features)}", file=sys.stderr)

        trim_info = []
        if args.start:
            trim_info.append(f"from {args.start}")
        if args.end:
            trim_info.append(f"to {args.end}")
        if trim_info:
            print(f"   Trimming: {' '.join(trim_info)}", file=sys.stderr)

        print(f"   Transcribing: {audio_path.name}", file=sys.stderr)
        print("", file=sys.stderr)

    start_time = time.time()

    # 1. Load audio
    try:
        audio = whisperx.load_audio(str(audio_path))
    except Exception as e:
        print(f"❌ Failed to load audio: {e}", file=sys.stderr)
        print("   Make sure ffmpeg is installed and the file is a valid audio/video.", file=sys.stderr)
        sys.exit(1)

    audio_sr = 16000  # whisperx always loads at 16kHz
    original_duration = len(audio) / audio_sr

    # Trim audio if start/end specified
    start_seconds = parse_time(args.start)
    end_seconds = parse_time(args.end)

    if start_seconds is not None or end_seconds is not None:
        if start_seconds and start_seconds >= original_duration:
            print(f"❌ Start time ({start_seconds:.1f}s) exceeds audio duration ({original_duration:.1f}s)", file=sys.stderr)
            sys.exit(1)
        if end_seconds and end_seconds > original_duration:
            print(f"⚠️  End time ({end_seconds:.1f}s) exceeds audio duration ({original_duration:.1f}s), clamping.", file=sys.stderr)
            end_seconds = original_duration

        audio = trim_audio(audio, audio_sr, start_seconds, end_seconds)

        if not args.quiet:
            trimmed_duration = len(audio) / audio_sr
            print(f"   Trimmed: {trimmed_duration:.1f}s (from {original_duration:.1f}s original)", file=sys.stderr)

    # 2. Transcribe with batched inference
    task = "translate" if args.translate else "transcribe"
    model = whisperx.load_model(
        args.model,
        device,
        compute_type=compute_type,
        language=args.language,
        task=task,
        vad_method="silero",  # silero avoids pyannote VAD torch.load issue
    )

    transcribe_kwargs = {
        "batch_size": args.batch_size,
        "language": args.language,
        "print_progress": not args.quiet,
    }
    if args.beam_size:
        transcribe_kwargs["beam_size"] = args.beam_size
    if args.initial_prompt:
        transcribe_kwargs["initial_prompt"] = args.initial_prompt

    result = model.transcribe(audio, **transcribe_kwargs)

    detected_language = result.get("language", args.language or "en")
    if not args.quiet and not args.language:
        print(f"   Detected language: {detected_language}", file=sys.stderr)

    # Free model memory
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # 3. Forced alignment (unless --no-align)
    if not args.no_align:
        if not args.quiet:
            print("   Aligning...", file=sys.stderr)

        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=detected_language,
                device=device,
                model_name=args.align_model,
            )
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                device,
                return_char_alignments=False,
            )

            del align_model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            if not args.quiet:
                print(f"   ⚠️  Alignment failed for language '{detected_language}': {e}", file=sys.stderr)
                print("   Continuing without word-level timestamps. Use --no-align to suppress this.", file=sys.stderr)

    # 4. Speaker diarization (if requested)
    if args.diarize:
        hf_token = get_hf_token(args.hf_token)
        if not hf_token:
            print("⚠️  Diarization requires a Hugging Face token.", file=sys.stderr)
            print("   Get one at: https://huggingface.co/settings/tokens", file=sys.stderr)
            print("   Then pass --hf-token TOKEN or set HF_TOKEN env var", file=sys.stderr)
            sys.exit(1)

        if not args.quiet:
            print("   Diarizing speakers...", file=sys.stderr)

        from whisperx.diarize import DiarizationPipeline
        diarize_model = DiarizationPipeline(
            use_auth_token=hf_token,
            device=device,
        )

        diarize_kwargs = {}
        if args.min_speakers:
            diarize_kwargs["min_speakers"] = args.min_speakers
        if args.max_speakers:
            diarize_kwargs["max_speakers"] = args.max_speakers

        diarize_segments = diarize_model(audio, **diarize_kwargs)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        del diarize_model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Adjust timestamps if audio was trimmed
    if start_seconds and start_seconds > 0:
        _offset_timestamps(result, start_seconds)

    # 5. Output results
    segments = result.get("segments", [])

    output_format = args.output_format
    output = _format_output(segments, output_format, args.diarize, args.word_level)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        if not args.quiet:
            print(f"\n✅ Saved to: {args.output}", file=sys.stderr)
    else:
        print(output)

    if not args.quiet:
        elapsed = time.time() - start_time
        audio_duration = len(audio) / audio_sr
        speed = audio_duration / elapsed if elapsed > 0 else 0
        seg_count = len(segments)
        word_count = sum(len(seg.get("words", [])) for seg in segments)
        print(f"\n   Done in {elapsed:.1f}s ({speed:.1f}x realtime) — {seg_count} segments, {word_count} words", file=sys.stderr)


def _offset_timestamps(result, offset):
    """Add offset to all timestamps (for trimmed audio)."""
    segments = result.get("segments", [])
    for seg in segments:
        if "start" in seg:
            seg["start"] = seg["start"] + offset
        if "end" in seg:
            seg["end"] = seg["end"] + offset
        for word in seg.get("words", []):
            if "start" in word:
                word["start"] = word["start"] + offset
            if "end" in word:
                word["end"] = word["end"] + offset


def _format_output(segments, output_format, include_speaker, word_level):
    """Route to the appropriate output formatter."""
    if output_format == "json":
        return _segments_to_json(segments)
    elif output_format == "srt":
        if word_level:
            return _segments_to_word_srt(segments, include_speaker)
        return _segments_to_srt(segments, include_speaker)
    elif output_format == "vtt":
        if word_level:
            return _segments_to_word_vtt(segments, include_speaker)
        return _segments_to_vtt(segments, include_speaker)
    elif output_format == "tsv":
        return _segments_to_tsv(segments, include_speaker)
    else:
        # Plain text
        lines = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if include_speaker and "speaker" in seg:
                lines.append(f"[{seg['speaker']}] {text}")
            else:
                lines.append(text)
        return "\n".join(lines)


def _format_timestamp_srt(seconds):
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    if seconds is None or seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds):
    """Format seconds as VTT timestamp: HH:MM:SS.mmm"""
    if seconds is None or seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _segments_to_json(segments):
    """Convert segments to clean JSON output."""
    # Build a clean, serializable structure
    clean_segments = []
    for seg in segments:
        clean_seg = {
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": seg.get("text", "").strip(),
        }
        if "speaker" in seg:
            clean_seg["speaker"] = seg["speaker"]
        if "words" in seg:
            clean_words = []
            for w in seg["words"]:
                clean_word = {"word": w.get("word", "")}
                if "start" in w:
                    clean_word["start"] = w["start"]
                if "end" in w:
                    clean_word["end"] = w["end"]
                if "score" in w:
                    clean_word["confidence"] = round(w["score"], 3)
                if "speaker" in w:
                    clean_word["speaker"] = w["speaker"]
                clean_words.append(clean_word)
            clean_seg["words"] = clean_words
        clean_segments.append(clean_seg)

    output = {"segments": clean_segments}
    return json.dumps(output, indent=2, ensure_ascii=False)


def _segments_to_srt(segments, include_speaker=False):
    """Convert segments to SRT subtitle format."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_timestamp_srt(seg.get("start", 0))
        end = _format_timestamp_srt(seg.get("end", 0))
        text = seg.get("text", "").strip()
        if include_speaker and "speaker" in seg:
            text = f"[{seg['speaker']}] {text}"
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def _segments_to_vtt(segments, include_speaker=False):
    """Convert segments to WebVTT subtitle format."""
    lines = ["WEBVTT\n"]
    for seg in segments:
        start = _format_timestamp_vtt(seg.get("start", 0))
        end = _format_timestamp_vtt(seg.get("end", 0))
        text = seg.get("text", "").strip()
        if include_speaker and "speaker" in seg:
            text = f"[{seg['speaker']}] {text}"
        lines.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def _segments_to_word_srt(segments, include_speaker=False):
    """Convert to word-level SRT — one word per cue for karaoke-style subtitles."""
    lines = []
    i = 1
    for seg in segments:
        words = seg.get("words", [])
        if not words:
            # Fallback to segment-level if no word timestamps
            start = _format_timestamp_srt(seg.get("start", 0))
            end = _format_timestamp_srt(seg.get("end", 0))
            text = seg.get("text", "").strip()
            if include_speaker and "speaker" in seg:
                text = f"[{seg['speaker']}] {text}"
            lines.append(f"{i}\n{start} --> {end}\n{text}\n")
            i += 1
            continue

        speaker_prefix = ""
        if include_speaker and "speaker" in seg:
            speaker_prefix = f"[{seg['speaker']}] "

        for w in words:
            if "start" not in w or "end" not in w:
                continue
            start = _format_timestamp_srt(w["start"])
            end = _format_timestamp_srt(w["end"])
            word_text = w.get("word", "").strip()
            if not word_text:
                continue
            lines.append(f"{i}\n{start} --> {end}\n{speaker_prefix}{word_text}\n")
            i += 1

    return "\n".join(lines)


def _segments_to_word_vtt(segments, include_speaker=False):
    """Convert to word-level WebVTT — one word per cue for karaoke-style subtitles."""
    lines = ["WEBVTT\n"]
    for seg in segments:
        words = seg.get("words", [])
        if not words:
            start = _format_timestamp_vtt(seg.get("start", 0))
            end = _format_timestamp_vtt(seg.get("end", 0))
            text = seg.get("text", "").strip()
            if include_speaker and "speaker" in seg:
                text = f"[{seg['speaker']}] {text}"
            lines.append(f"{start} --> {end}\n{text}\n")
            continue

        speaker_prefix = ""
        if include_speaker and "speaker" in seg:
            speaker_prefix = f"[{seg['speaker']}] "

        for w in words:
            if "start" not in w or "end" not in w:
                continue
            start = _format_timestamp_vtt(w["start"])
            end = _format_timestamp_vtt(w["end"])
            word_text = w.get("word", "").strip()
            if not word_text:
                continue
            lines.append(f"{start} --> {end}\n{speaker_prefix}{word_text}\n")

    return "\n".join(lines)


def _segments_to_tsv(segments, include_speaker=False):
    """Convert segments to TSV format (tab-separated values)."""
    lines = ["start\tend\tspeaker\ttext" if include_speaker else "start\tend\ttext"]
    for seg in segments:
        start = f"{seg.get('start', 0):.3f}"
        end = f"{seg.get('end', 0):.3f}"
        text = seg.get("text", "").strip().replace("\t", " ")
        if include_speaker:
            speaker = seg.get("speaker", "")
            lines.append(f"{start}\t{end}\t{speaker}\t{text}")
        else:
            lines.append(f"{start}\t{end}\t{text}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with WhisperX — word-level timestamps, speaker diarization, and forced alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.mp3                            # Basic transcription
  %(prog)s audio.mp3 --diarize                  # With speaker labels
  %(prog)s audio.mp3 --srt -o subtitles.srt     # Generate subtitles
  %(prog)s audio.mp3 --srt --word-level         # Word-level (karaoke) subtitles
  %(prog)s audio.mp3 --json                     # JSON with word timestamps
  %(prog)s audio.mp3 --translate                # Translate to English
  %(prog)s audio.mp3 --start 1:30 --end 5:00   # Transcribe a section
  %(prog)s audio.mp3 --initial-prompt "OpenAI"  # Improve accuracy for terms
"""
    )

    parser.add_argument(
        "audio",
        metavar="AUDIO_FILE",
        help="Path to audio/video file (mp3, wav, m4a, flac, ogg, opus, webm, mp4, mkv, avi, mov, wma, aac)"
    )
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {__version__} (WhisperX skill)"
    )

    # Model options
    model_group = parser.add_argument_group("Model options")
    model_group.add_argument(
        "-m", "--model",
        default="large-v3-turbo",
        metavar="NAME",
        help="Whisper model (default: large-v3-turbo). Options: tiny, base, small, medium, large-v2, large-v3, large-v3-turbo"
    )
    model_group.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="Batch size for inference (default: 8, lower if OOM)"
    )
    model_group.add_argument(
        "--beam-size",
        type=int,
        default=None,
        metavar="N",
        help="Beam search size (higher = more accurate but slower)"
    )
    model_group.add_argument(
        "--initial-prompt",
        default=None,
        metavar="TEXT",
        help="Initial text prompt to condition the model (improves accuracy for domain-specific terms, names, acronyms)"
    )

    # Device options
    device_group = parser.add_argument_group("Device options")
    device_group.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device (default: auto-detect)"
    )
    device_group.add_argument(
        "--compute-type",
        default="auto",
        choices=["auto", "int8", "float16", "float32"],
        help="Quantization (default: auto — float16 on GPU, int8 on CPU)"
    )

    # Language options
    lang_group = parser.add_argument_group("Language options")
    lang_group.add_argument(
        "-l", "--language",
        default=None,
        metavar="CODE",
        help="Language code (e.g., en, es, fr, zh, ja, ko, de). Auto-detects if omitted"
    )
    lang_group.add_argument(
        "--translate",
        action="store_true",
        help="Translate audio to English"
    )

    # Time range
    range_group = parser.add_argument_group("Time range (partial transcription)")
    range_group.add_argument(
        "--start",
        default=None,
        metavar="TIME",
        help="Start time — seconds (90), MM:SS (1:30), or HH:MM:SS (1:02:30)"
    )
    range_group.add_argument(
        "--end",
        default=None,
        metavar="TIME",
        help="End time — seconds (300), MM:SS (5:00), or HH:MM:SS"
    )

    # Alignment options
    align_group = parser.add_argument_group("Alignment options")
    align_group.add_argument(
        "--no-align",
        action="store_true",
        help="Skip forced alignment (faster, but no word timestamps)"
    )
    align_group.add_argument(
        "--align-model",
        default=None,
        metavar="MODEL",
        help="Custom phoneme-level ASR model for alignment"
    )

    # Diarization options
    diarize_group = parser.add_argument_group("Speaker diarization")
    diarize_group.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization (requires HF token)"
    )
    diarize_group.add_argument(
        "--hf-token",
        default=None,
        metavar="TOKEN",
        help="Hugging Face access token for diarization models"
    )
    diarize_group.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        metavar="N",
        help="Minimum number of speakers (helps diarization accuracy)"
    )
    diarize_group.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of speakers (helps diarization accuracy)"
    )

    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "-j", "--json",
        action="store_true",
        help="Output as JSON with segments and word timestamps"
    )
    output_group.add_argument(
        "--srt",
        action="store_true",
        help="Output as SRT subtitle format"
    )
    output_group.add_argument(
        "--vtt",
        action="store_true",
        help="Output as WebVTT subtitle format"
    )
    output_group.add_argument(
        "--tsv",
        action="store_true",
        help="Output as TSV (tab-separated values)"
    )
    output_group.add_argument(
        "--word-level",
        action="store_true",
        help="Word-level subtitles (one word per cue) for SRT/VTT — karaoke-style timing"
    )
    output_group.add_argument(
        "--output-format",
        default=None,
        choices=["srt", "vtt", "txt", "json", "tsv"],
        help="Output format (default: auto-detect from -o extension, or txt)"
    )
    output_group.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Save output to FILE instead of stdout (format auto-detected from extension)"
    )

    # Misc options
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    # Validate audio file
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"❌ Audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)
    if audio_path.stat().st_size == 0:
        print(f"❌ Audio file is empty: {args.audio}", file=sys.stderr)
        sys.exit(1)

    # Resolve output format: explicit flags > --output-format > auto-detect from -o extension > txt
    if args.json:
        args.output_format = "json"
    elif args.srt:
        args.output_format = "srt"
    elif args.vtt:
        args.output_format = "vtt"
    elif args.tsv:
        args.output_format = "tsv"
    elif args.output_format is None:
        # Auto-detect from output file extension
        if args.output:
            ext = Path(args.output).suffix.lower()
            args.output_format = FORMAT_FROM_EXT.get(ext, "txt")
        else:
            args.output_format = "txt"

    # Warn if --word-level used without subtitle format
    if args.word_level and args.output_format not in ("srt", "vtt"):
        print("⚠️  --word-level only applies to SRT/VTT output. Ignoring.", file=sys.stderr)
        args.word_level = False

    # Warn if --word-level used with --no-align
    if args.word_level and args.no_align:
        print("⚠️  --word-level requires alignment. Ignoring --no-align.", file=sys.stderr)
        args.no_align = False

    run_whisperx(args)


if __name__ == "__main__":
    main()
