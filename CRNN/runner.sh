#!/usr/bin/env bash
set -euo pipefail

# ── Default values ─────────────────────────────────────────────────────────
CONDA_ENV="venv"
REQ_FILE="requirements.txt"
CONFIG_FILE="config.yaml"

# ── Parse flags ────────────────────────────────────────────────────────────
print_usage() {
  echo "Usage: $0 [-v|--venv VENV_DIR] [-r|--req REQUIREMENTS_FILE] [-c|--config CONFIG_FILE]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -v|--venv)
      CONDA_ENV="$2"; shift 2 ;;
    -r|--req|--requirements)
      REQ_FILE="$2"; shift 2 ;;
    -c|--config)
      CONFIG_FILE="$2"; shift 2 ;;
    -h|--help)
      print_usage ;;
    *)
      echo "Unknown option: $1"; print_usage ;;
  esac
done

# ── 1) Load .env ─────────────────────────────────────────────────────────────
if [ -f .env ]; then
  echo "Loading environment from .env…"
  export $(grep -v '^\s*#' .env | xargs)
else
  echo "⚠️  No .env file found"
fi



# ── 3) Init & activate Conda ───────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
  echo "❌ 'conda' not in PATH. Please install Anaconda/Miniconda."
  exit 1
fi

# Ensure conda commands are available
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "⏳ Activating Conda environment '$CONDA_ENV'..."
conda activate "$CONDA_ENV" || {
  echo "❌ Failed to activate Conda env '$CONDA_ENV'. Make sure it exists."
  exit 1
}


# ── 3) Install requirements ─────────────────────────────────────────────────
if [ -f "$REQ_FILE" ]; then
  echo "Installing dependencies from '$REQ_FILE'…"
  pip install --upgrade pip
  pip install -r "$REQ_FILE"
else
  echo "❌ Requirements file '$REQ_FILE' not found!"
  exit 1
fi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/lib
# ── 4) Run training ─────────────────────────────────────────────────────────
if [ ! -f "$CONFIG_FILE" ]; then
  echo "❌ Config file '$CONFIG_FILE' not found!"
  exit 1
fi

echo "Starting training with config '$CONFIG_FILE'…"
python3 train.py --config "$CONFIG_FILE"

echo "✅ Training complete!"
