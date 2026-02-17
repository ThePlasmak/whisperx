#!/usr/bin/env python3
"""
WhisperX transcription CLI wrapper
Speech-to-text with word-level alignment, speaker diarization, and batched inference.
"""

import sys
import os
import json
import argparse
import subprocess
import tempfile
from pathlib import Path


def check_cuda_available():
    """Check if CUDA is available and return device info."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
        return False, None
    except ImportError:
        return False, None


def check_whisperx():
    """Check if whisperx is installed."""
    try:
        import whisperx
        return True
    except ImportError:
        return False


def run_whisperx_cli(args):
    """Run whisperx as a subprocess using the CLI."""
    cmd = ["whisperx", str(args.audio)]

    # Model
    cmd.extend(["--model", args.model])

    # Device
    cuda_available, gpu_name = check_cuda_available()
    if args.device == "auto":
        device = "cuda" if cuda_available else "cpu"
    else:
        device = args.device
    cmd.extend(["--device", device])

    # Compute type
    if args.compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    else:
        compute_type = args.compute_type
    cmd.extend(["--compute_type", compute_type])

    # Batch size
    cmd.extend(["--batch_size", str(args.batch_size)])

    # Language
    if args.language:
        cmd.extend(["--language", args.language])

    # Output
    output_dir = tempfile.mkdtemp(prefix="whisperx_") if not args.output_dir else args.output_dir
    cmd.extend(["--output_dir", output_dir])
    cmd.extend(["--output_format", args.output_format])

    # Task (transcribe or translate)
    if args.translate:
        cmd.extend(["--task", "translate"])

    # Alignment
    if args.no_align:
        cmd.append("--no_align")
    if args.align_model:
        cmd.extend(["--align_model", args.align_model])

    # Diarization
    if args.diarize:
        cmd.append("--diarize")
        if args.hf_token:
            cmd.extend(["--hf_token", args.hf_token])
        elif os.environ.get("HF_TOKEN"):
            cmd.extend(["--hf_token", os.environ["HF_TOKEN"]])
        else:
            # Check cached token
            token_path = Path.home() / ".cache" / "huggingface" / "token"
            if token_path.exists():
                cmd.extend(["--hf_token", token_path.read_text().strip()])
            else:
                print("⚠️  Diarization requires a Hugging Face token.", file=sys.stderr)
                print("   Get one at: https://huggingface.co/settings/tokens", file=sys.stderr)
                print("   Then pass --hf-token YOUR_TOKEN or set HF_TOKEN env var", file=sys.stderr)
                print("   You must also accept the model agreement at:", file=sys.stderr)
                print("   https://huggingface.co/pyannote/speaker-diarization-community-1", file=sys.stderr)
                sys.exit(1)

        if args.min_speakers:
            cmd.extend(["--min_speakers", str(args.min_speakers)])
        if args.max_speakers:
            cmd.extend(["--max_speakers", str(args.max_speakers)])

    # VAD
    if args.vad_method:
        cmd.extend(["--vad_method", args.vad_method])

    # Beam size
    if args.beam_size:
        cmd.extend(["--beam_size", str(args.beam_size)])

    # Highlight words in SRT/VTT
    if args.highlight_words:
        cmd.extend(["--highlight_words", "True"])

    # Segment resolution
    if args.segment_resolution:
        cmd.extend(["--segment_resolution", args.segment_resolution])

    # Suppress numerals
    if args.suppress_numerals:
        cmd.append("--suppress_numerals")

    # Print progress
    cmd.extend(["--print_progress", "True"])

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

        print(f"   Transcribing: {Path(args.audio).name}", file=sys.stderr)
        print("", file=sys.stderr)

    # Run whisperx
    result = subprocess.run(cmd, capture_output=not args.verbose, text=True)

    if result.returncode != 0:
        if not args.verbose and result.stderr:
            print(result.stderr, file=sys.stderr)
        print(f"❌ WhisperX exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    # Read and output results
    audio_stem = Path(args.audio).stem

    if args.json or args.output_format == "json":
        # Return JSON output
        json_file = Path(output_dir) / f"{audio_stem}.json"
        if json_file.exists():
            content = json_file.read_text(encoding="utf-8")
            if args.output:
                Path(args.output).write_text(content, encoding="utf-8")
                if not args.quiet:
                    print(f"✅ Saved JSON to: {args.output}", file=sys.stderr)
            else:
                print(content)
        else:
            print(f"⚠️  JSON output not found at {json_file}", file=sys.stderr)

    elif args.srt or args.output_format == "srt":
        srt_file = Path(output_dir) / f"{audio_stem}.srt"
        if srt_file.exists():
            content = srt_file.read_text(encoding="utf-8")
            if args.output:
                Path(args.output).write_text(content, encoding="utf-8")
                if not args.quiet:
                    print(f"✅ Saved SRT to: {args.output}", file=sys.stderr)
            else:
                print(content)
        else:
            print(f"⚠️  SRT output not found at {srt_file}", file=sys.stderr)

    elif args.vtt or args.output_format == "vtt":
        vtt_file = Path(output_dir) / f"{audio_stem}.vtt"
        if vtt_file.exists():
            content = vtt_file.read_text(encoding="utf-8")
            if args.output:
                Path(args.output).write_text(content, encoding="utf-8")
                if not args.quiet:
                    print(f"✅ Saved VTT to: {args.output}", file=sys.stderr)
            else:
                print(content)
        else:
            print(f"⚠️  VTT output not found at {vtt_file}", file=sys.stderr)

    else:
        # Default: txt output (plain transcript)
        txt_file = Path(output_dir) / f"{audio_stem}.txt"
        if txt_file.exists():
            content = txt_file.read_text(encoding="utf-8")
            if args.output:
                Path(args.output).write_text(content, encoding="utf-8")
                if not args.quiet:
                    print(f"✅ Saved transcript to: {args.output}", file=sys.stderr)
            else:
                print(content)
        else:
            # Try JSON fallback
            json_file = Path(output_dir) / f"{audio_stem}.json"
            if json_file.exists():
                data = json.loads(json_file.read_text(encoding="utf-8"))
                segments = data.get("segments", [])
                text = " ".join(s.get("text", "").strip() for s in segments)
                print(text)
            else:
                print(f"⚠️  No output found in {output_dir}", file=sys.stderr)

    # List all generated files
    if not args.quiet and args.output_format == "all":
        generated = list(Path(output_dir).glob(f"{audio_stem}.*"))
        if generated:
            print("", file=sys.stderr)
            print("📁 Generated files:", file=sys.stderr)
            for f in sorted(generated):
                size = f.stat().st_size
                if size > 1024:
                    print(f"   {f.name} ({size / 1024:.1f} KB)", file=sys.stderr)
                else:
                    print(f"   {f.name} ({size} B)", file=sys.stderr)

    # Clean up temp dir if we created one and user didn't ask for all formats
    if not args.output_dir and args.output_format != "all":
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
    elif not args.output_dir:
        if not args.quiet:
            print(f"   Location: {output_dir}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with WhisperX — word-level timestamps, speaker diarization, and forced alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.mp3                          # Basic transcription
  %(prog)s audio.mp3 --diarize --hf-token X   # With speaker labels
  %(prog)s audio.mp3 --model large-v3-turbo   # Maximum accuracy
  %(prog)s audio.mp3 --srt -o subtitles.srt   # Generate subtitles
  %(prog)s audio.mp3 --json                    # JSON with word timestamps
  %(prog)s audio.mp3 --translate               # Translate to English
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

    # VAD options
    vad_group = parser.add_argument_group("Voice Activity Detection")
    vad_group.add_argument(
        "--vad-method",
        default=None,
        choices=["pyannote", "silero"],
        help="VAD method (default: pyannote)"
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
        choices=["all", "srt", "vtt", "txt", "tsv", "json", "aud"],
        help="Output format (default: txt). Use 'all' to generate all formats"
    )
    output_group.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Save output to FILE instead of stdout"
    )
    output_group.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Directory for output files (for 'all' format)"
    )
    output_group.add_argument(
        "--highlight-words",
        action="store_true",
        help="Underline each word as spoken in SRT/VTT"
    )
    output_group.add_argument(
        "--segment-resolution",
        default=None,
        choices=["sentence", "chunk"],
        help="Segment resolution (default: sentence)"
    )

    # Misc options
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "--suppress-numerals",
        action="store_true",
        help="Suppress numeric output (spell out numbers)"
    )
    misc_group.add_argument(
        "--verbose",
        action="store_true",
        help="Show full whisperx output"
    )
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

    # Check whisperx is available
    if not check_whisperx():
        print("❌ whisperx not installed", file=sys.stderr)
        print("   Install: pip install whisperx", file=sys.stderr)
        print("   Or run: ./setup.sh", file=sys.stderr)
        sys.exit(1)

    run_whisperx_cli(args)


if __name__ == "__main__":
    main()
