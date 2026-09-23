#!/bin/bash
# Bring a bare CUDA box up to a working Obsessed Encoder checkout.
#
# Tested on vast.ai, image nvidia/cuda:12.4.1-devel-ubuntu22.04, A100-SXM4-80GB.
# The box needs >=80 GB VRAM (peak is 44.1 GB, see results/phase0_throughput.json)
# and >=100 GB disk (ImageNet-1k is ~47 GB plus a ~15 GB venv).
#
#   bash setup_vast.sh            # clone + install only
#   bash setup_vast.sh --data     # also fetch and verify ImageNet-1k (~47 GB)
set -euo pipefail

REPO=${REPO:-https://github.com/Enigma-Incorporated/The-Obsessed-Encoder.git}
ROOT=${ROOT:-/root/oe}

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git curl ca-certificates tmux

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

[ -d "$ROOT" ] || git clone "$REPO" "$ROOT"
cd "$ROOT"
echo "checkout: $(git log --oneline -1)"
echo "dirty files: $(git status --porcelain | wc -l)   # 0 = pristine upstream"

uv sync --frozen
uv run python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"

if [ "${1:-}" = "--data" ]; then
  export DATA_DIR="$ROOT/data" HF_HOME="$ROOT/data/hf_cache"
  uv run python lejepa/additional_files/prepare_data.py --dataset imagenet1k
fi
