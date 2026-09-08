#!/usr/bin/env bash
# =============================================================================
# Jetson Orin NX installer.
#
#   bash scripts/install_jetson.sh
#
# Idempotent: safe to re-run. Every privileged step is announced before it runs.
#
# The one thing this script will not do is install PyTorch from PyPI. On a Jetson the
# PyPI wheel is either x86-only or a CPU build without CUDA, and installing it is the
# most common way to end up with a broken environment. The correct wheel comes from
# NVIDIA's index for the detected JetPack version.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="${VENV:-$REPO_ROOT/.venv}"
PY="${PY:-python3}"
SKIP_APT="${SKIP_APT:-0}"
SKIP_TORCH="${SKIP_TORCH:-0}"

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

run_sudo() {
  printf '\033[1;35m[sudo]\033[0m %s\n' "$*"
  sudo "$@"
}

# ---------------------------------------------------------------- 1. platform
log "Detecting the platform"

ARCH="$(uname -m)"
[ "$ARCH" = "aarch64" ] || warn "architecture is $ARCH, not aarch64: this script targets a Jetson"

if [ -f /etc/nv_tegra_release ]; then
  TEGRA_LINE="$(head -1 /etc/nv_tegra_release)"
  echo "  $TEGRA_LINE"
  L4T_MAJOR="$(sed -n 's/.*# R\([0-9]\+\).*/\1/p' /etc/nv_tegra_release)"
  L4T_MINOR="$(sed -n 's/.*REVISION: \([0-9]\+\).*/\1/p' /etc/nv_tegra_release)"
else
  warn "/etc/nv_tegra_release not found: this does not look like Jetson hardware"
  L4T_MAJOR=""
  L4T_MINOR=""
fi

[ -f /proc/device-tree/model ] && echo "  model: $(tr -d '\0' < /proc/device-tree/model)"
echo "  python: $($PY --version 2>&1)"
echo "  memory: $(free -g | awk '/^Mem:/ {print $2" GB"}')"

# Map L4T to JetPack and to the matching NVIDIA wheel index.
JP=""
TORCH_INDEX=""
case "$L4T_MAJOR" in
  32) JP="4.x" ;;
  35) JP="5.1.x"; TORCH_INDEX="https://pypi.jetson-ai-lab.dev/jp5/cu114" ;;
  36) JP="6.x";   TORCH_INDEX="https://pypi.jetson-ai-lab.dev/jp6/cu126" ;;
  38) JP="7.x";   TORCH_INDEX="https://pypi.jetson-ai-lab.dev/jp7/cu130" ;;
  *)  JP="unknown" ;;
esac
echo "  L4T R${L4T_MAJOR:-?}.${L4T_MINOR:-?}  ->  JetPack ${JP}"

case "$JP" in
  5.1.x|6.x|7.x) : ;;
  4.x) die "JetPack 4.x is too old: it ships Python 3.6 and deepfilternet needs >= 3.8. Reflash with JetPack 5.1.x or newer." ;;
  *)   warn "Unrecognised L4T release. The install will continue but the PyTorch wheel index must be set manually with TORCH_INDEX=..." ;;
esac

PY_VER="$($PY -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PY_OK="$($PY -c 'import sys; print(1 if (3,8) <= sys.version_info[:2] < (3,12) else 0)')"
[ "$PY_OK" = "1" ] || die "Python $PY_VER is unsupported. deepfilternet 0.5.6 needs 3.8-3.11 (its numpy<2 pin rules out 3.12+). Install python3.10 and re-run with PY=python3.10."

# ------------------------------------------------------------ 2. system packages
if [ "$SKIP_APT" = "1" ]; then
  log "Skipping system packages (SKIP_APT=1)"
else
  log "Installing system packages"
  echo "  needed for: audio I/O (portaudio, libsndfile), decoding (ffmpeg),"
  echo "  and building the pesq extension (python3-dev, build-essential)"
  run_sudo apt-get update
  run_sudo apt-get install -y --no-install-recommends \
    build-essential python3-dev python3-venv python3-pip \
    libsndfile1 libportaudio2 portaudio19-dev libasound2-dev \
    ffmpeg libopenblas-dev git
fi

# -------------------------------------------------------------- 3. environment
log "Creating the virtual environment at $VENV"
if [ ! -x "$VENV/bin/python" ]; then
  # --system-site-packages matters: on a Jetson, torch is often already installed
  # system-wide by JetPack, and reusing it avoids a large download.
  $PY -m venv --system-site-packages "$VENV"
fi
VPY="$VENV/bin/python"
"$VPY" -m pip install --upgrade pip setuptools wheel

# numpy first and pinned: deepfilternet requires <2, and letting pip resolve it later
# can pull numpy 2.x and break the model at import time.
log "Pinning numpy < 2 (required by deepfilternet 0.5.6)"
"$VPY" -m pip install "numpy>=1.22,<2"

# -------------------------------------------------------------------- 4. torch
if [ "$SKIP_TORCH" = "1" ]; then
  log "Skipping PyTorch (SKIP_TORCH=1)"
elif "$VPY" -c 'import torch' 2>/dev/null; then
  log "PyTorch is already importable"
  "$VPY" - <<'PYEOF'
import torch
print(f"  torch {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}")
PYEOF
else
  log "Installing PyTorch for JetPack ${JP}"
  if [ -z "$TORCH_INDEX" ]; then
    die "No wheel index known for this JetPack. Find the wheel for your release at
  https://developer.nvidia.com/embedded/downloads  or  https://pypi.jetson-ai-lab.dev
then re-run with:  TORCH_INDEX=<url> bash scripts/install_jetson.sh
Do NOT 'pip install torch' from PyPI on a Jetson: you will get a wheel without CUDA."
  fi
  echo "  index: $TORCH_INDEX"
  if ! "$VPY" -m pip install --index-url "$TORCH_INDEX" torch torchaudio; then
    die "PyTorch install failed from $TORCH_INDEX.
Alternatives, in order of preference:
  1. Use the NGC container: nvcr.io/nvidia/l4t-pytorch matching your L4T release.
  2. Download the wheel from https://developer.nvidia.com/embedded/downloads and
     install it with: $VPY -m pip install ./torch-*.whl
  3. Install the JetPack system package and re-run this script: the venv was created
     with --system-site-packages so a system torch will be picked up.
CPU-only operation is also viable: this workload measured RTF 0.05-0.06 on a single
x86 CPU thread. Set neural.device=cpu and continue."
  fi
fi

# --------------------------------------------------------------- 5. the package
log "Installing anc_defence and its dependencies"
"$VPY" -m pip install -e .

log "Installing the PESQ metric (compiles from source on aarch64)"
if ! "$VPY" -m pip install -e '.[metrics]'; then
  warn "pesq failed to build. Everything else works; PESQ will be reported as unavailable."
  warn "Set metrics.pesq=false in your config to silence it."
fi

log "Installing the dashboard"
if ! "$VPY" -m pip install -e '.[dashboard]'; then
  warn "streamlit failed to install; the CLI and PDF report still work."
fi

# ---------------------------------------------------------------- 6. model cache
log "Downloading and caching the DeepFilterNet3 weights"
"$VPY" - <<'PYEOF'
import warnings
warnings.filterwarnings("ignore")
from df.enhance import init_df
model, state, _ = init_df(log_level="warning")
n = sum(p.numel() for p in model.parameters())
print(f"  DeepFilterNet3 cached: {n/1e6:.2f}M parameters, {state.sr()} Hz")
PYEOF

# ------------------------------------------------------------------- 7. runtime
log "Runtime recommendations (not run automatically)"
cat <<'EOF'
  Put the board in its maximum performance mode before benchmarking or demoing:

      sudo nvpmodel -m 0        # maximum power mode
      sudo jetson_clocks        # lock clocks to maximum

  Both need root and both change the thermal behaviour of the board, so they are
  printed rather than executed. Check the current mode with:

      sudo nvpmodel -q

  Audio: confirm the capture device is visible with `arecord -l`, then
  `.venv/bin/anc list-devices`. The pipeline runs at 48 kHz; if the device refuses
  that rate, pick another one rather than resampling the capture path.
EOF

# ------------------------------------------------------------------ 8. selftest
log "Running the self-test"
"$VENV/bin/anc" selftest || warn "self-test reported problems: read the actions listed above"

log "Done"
cat <<EOF

  Activate the environment:   source $VENV/bin/activate

  Quick checks:
    anc selftest
    anc corpus
    anc benchmark --seconds 5 --threads 1,0

  Evaluate with the Jetson overlay:
    anc -c configs/default.yaml -c configs/jetson.yaml evaluate --methods delivered

  Live microphone:
    anc -c configs/default.yaml -c configs/jetson.yaml run --mode live_mic --duration 30

  Dashboard:
    anc dashboard --port 8501 --headless
EOF
