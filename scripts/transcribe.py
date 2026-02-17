#!/usr/bin/env python3
"""
WhisperX transcription CLI wrapper
Speech-to-text with word-level alignment, speaker diarization, and batched inference.

Uses the whisperx Python API directly (not subprocess) to apply PyTorch 2.6+
compatibility patches for pyannote model loading.
"""

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


def run_whisperx(args):
    """Run whisperx using the Python API."""
    import whisperx
    import gc

    audio_path = Path(args.audio)

    # Resolve device
    cuda_available, gpu_name = check_cuda_available()
    if args.device == "auto":
        device = "cuda" if cuda_available else "cpu"
    else:
        device = args.device

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
        if features:
            print(f"   Features: {', '.join(features)}", file=sys.stderr)
        print(f"   Transcribing: {audio_path.name}", file=sys.stderr)
        print("", file=sys.stderr)

    start_time = time.time()

    # 1. Load audio
    audio = whisperx.load_audio(str(audio_path))

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

    # 5. Output results
    segments = result.get("segments", [])

    if args.json or args.output_format == "json":
        output = json.dumps(result, indent=2, ensure_ascii=False)
    elif args.srt or args.output_format == "srt":
        output = _segments_to_srt(segments, args.diarize)
    elif args.vtt or args.output_format == "vtt":
        output = _segments_to_vtt(segments, args.diarize)
    else:
        # Plain text
        lines = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if args.diarize and "speaker" in seg:
                lines.append(f"[{seg['speaker']}] {text}")
            else:
                lines.append(text)
        output = "\n".join(lines)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        if not args.quiet:
            print(f"\n✅ Saved to: {args.output}", file=sys.stderr)
    else:
        print(output)

    if not args.quiet:
        elapsed = time.time() - start_time
        audio_duration = len(audio) / 16000  # whisperx loads at 16kHz
        speed = audio_duration / elapsed if elapsed > 0 else 0
        print(f"\n   Done in {elapsed:.1f}s ({speed:.1f}x realtime)", file=sys.stderr)


def _format_timestamp_srt(seconds):
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds):
    """Format seconds as VTT timestamp: HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


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


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with WhisperX — word-level timestamps, speaker diarization, and forced alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.mp3                          # Basic transcription
  %(prog)s audio.mp3 --diarize                # With speaker labels (needs HF token)
  %(prog)s audio.mp3 --model large-v3-turbo   # Maximum accuracy
  %(prog)s audio.mp3 --srt -o subtitles.srt   # Generate subtitles
  %(prog)s audio.mp3 --json                   # JSON with word timestamps
  %(prog)s audio.mp3 --translate              # Translate to English
"""
    )

    parser.add_argument(
        "audio",
        metavar="AUDIO_FILE",
        help="Path to audio/video file (mp3, wav, m4a, flac, ogg, webm, mp4)"
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
        help="Language code (e.g., en, es, fr, zh). Auto-detects if omitted"
    )
    lang_group.add_argument(
        "--translate",
        action="store_true",
        help="Translate audio to English"
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
        "--output-format",
        default="txt",
        choices=["srt", "vtt", "txt", "json"],
        help="Output format (default: txt)"
    )
    output_group.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Save output to FILE instead of stdout"
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

    # Override output_format based on shorthand flags
    if args.json:
        args.output_format = "json"
    elif args.srt:
        args.output_format = "srt"
    elif args.vtt:
        args.output_format = "vtt"

    # Check whisperx
    try:
        import whisperx
    except ImportError:
        print("❌ whisperx not installed", file=sys.stderr)
        print("   Install: pip install whisperx", file=sys.stderr)
        print("   Or run: ./setup.sh", file=sys.stderr)
        sys.exit(1)

    run_whisperx(args)


if __name__ == "__main__":
    main()
