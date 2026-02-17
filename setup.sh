#!/usr/bin/env bash
# WhisperX skill setup
# Creates venv and installs whisperx with dependencies (GPU support auto-detected)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "🎙️ Setting up WhisperX skill..."

# Detect OS
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux*)  OS_TYPE="linux" ;;
    Darwin*) OS_TYPE="macos" ;;
    *)       OS_TYPE="unknown" ;;
esac

echo "✓ Platform: $OS_TYPE ($ARCH)"

# Check for Python 3.10+
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "❌ Python 3.10+ required (found $PYTHON_VERSION)"
    exit 1
fi

echo "✓ Python $PYTHON_VERSION"

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ ffmpeg not found (required for audio processing)"
    echo ""
    echo "Install ffmpeg:"
    if [ "$OS_TYPE" = "macos" ]; then
        echo "   brew install ffmpeg"
    else
        echo "   Ubuntu/Debian: sudo apt install ffmpeg"
        echo "   Fedora: sudo dnf install ffmpeg"
        echo "   Arch: sudo pacman -S ffmpeg"
    fi
    echo ""
    exit 1
fi

echo "✓ ffmpeg found"

# Detect GPU
HAS_CUDA=false
GPU_NAME=""
NVIDIA_SMI=""

if [ "$OS_TYPE" = "linux" ]; then
    if command -v nvidia-smi &> /dev/null; then
        NVIDIA_SMI="nvidia-smi"
    else
        # WSL2
        if grep -qi microsoft /proc/version 2>/dev/null; then
            for wsl_smi in /usr/lib/wsl/lib/nvidia-smi /usr/lib/wsl/drivers/*/nvidia-smi; do
                if [ -f "$wsl_smi" ]; then
                    NVIDIA_SMI="$wsl_smi"
                    echo "✓ WSL2 detected"
                    break
                fi
            done
        fi
    fi

    if [ -n "$NVIDIA_SMI" ]; then
        GPU_NAME=$($NVIDIA_SMI --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        if [ -n "$GPU_NAME" ]; then
            HAS_CUDA=true
        fi
    fi
fi

if [ "$HAS_CUDA" = true ]; then
    echo "✓ GPU detected: $GPU_NAME"
fi

# Check if whisperx is already installed system-wide
if command -v whisperx &> /dev/null; then
    WX_VERSION=$(whisperx --version 2>&1 | head -1 || echo "unknown")
    echo "✓ WhisperX already installed system-wide ($WX_VERSION)"
    echo ""
    echo "✅ Setup complete! WhisperX is ready to use."
    echo ""
    echo "Usage:"
    echo "  $SCRIPT_DIR/scripts/transcribe audio.mp3"
    echo "  $SCRIPT_DIR/scripts/transcribe audio.mp3 --diarize --hf-token YOUR_TOKEN"
    echo ""
    chmod +x "$SCRIPT_DIR/scripts/"* 2>/dev/null || true
    exit 0
fi

# Create venv if needed
if [ -d "$VENV_DIR" ]; then
    echo "✓ Virtual environment exists"
else
    echo "Creating virtual environment..."
    if command -v uv &> /dev/null; then
        uv venv "$VENV_DIR" --python python3
    else
        python3 -m venv "$VENV_DIR"
    fi
    echo "✓ Virtual environment created"
fi

# Install PyTorch first (GPU-aware)
if [ "$HAS_CUDA" = true ]; then
    echo ""
    echo "🚀 Installing PyTorch with CUDA support..."
    if command -v uv &> /dev/null; then
        uv pip install --python "$VENV_DIR/bin/python" torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        "$VENV_DIR/bin/pip" install --upgrade pip
        "$VENV_DIR/bin/pip" install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    fi
    echo "✓ PyTorch with CUDA installed"
else
    echo ""
    echo "Installing PyTorch (CPU)..."
    if command -v uv &> /dev/null; then
        uv pip install --python "$VENV_DIR/bin/python" torch torchaudio
    else
        "$VENV_DIR/bin/pip" install --upgrade pip
        "$VENV_DIR/bin/pip" install torch torchaudio
    fi
    echo "✓ PyTorch installed"
fi

# Install whisperx
echo ""
echo "Installing WhisperX..."
if command -v uv &> /dev/null; then
    uv pip install --python "$VENV_DIR/bin/python" whisperx
else
    "$VENV_DIR/bin/pip" install whisperx
fi
echo "✓ WhisperX installed"

# Make scripts executable
chmod +x "$SCRIPT_DIR/scripts/"*

echo ""
echo "✅ Setup complete!"
echo ""
if [ "$HAS_CUDA" = true ]; then
    echo "🚀 GPU acceleration enabled — expect ~70x realtime speed with batched inference"
else
    echo "💻 CPU mode — transcription will work but GPU is strongly recommended"
fi
echo ""
echo "Usage:"
echo "  $SCRIPT_DIR/scripts/transcribe audio.mp3"
echo "  $SCRIPT_DIR/scripts/transcribe audio.mp3 --diarize --hf-token YOUR_TOKEN"
echo ""
echo "First run will download the model (~809MB for large-v3-turbo)."
echo ""
echo "For speaker diarization, you need a Hugging Face token:"
echo "  1. Get token: https://huggingface.co/settings/tokens"
echo "  2. Accept model: https://huggingface.co/pyannote/speaker-diarization-community-1"
