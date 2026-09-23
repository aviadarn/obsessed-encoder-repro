#!/bin/bash
# Measure seconds/step for one LeJEPA arm at the shipped operating point.
#
# Runs the same arm at two step counts and reports the slope, so process
# startup (model init, dataset open, cuDNN warmup) cancels out instead of
# inflating a single short run.
#
#   bash throughput.sh clean
#   bash throughput.sh watermarked
#   bash throughput.sh random_control
#
# Env: ROOT (checkout, default /root/oe), SHORT/LONG step counts.
set -euo pipefail

ARM=${1:?usage: throughput.sh <clean|watermarked|random_control>}
ROOT=${ROOT:-/root/oe}
SHORT=${SHORT:-40}
LONG=${LONG:-160}

export PATH="$HOME/.local/bin:$PATH"
cd "$ROOT"
export DATA_DIR="$ROOT/data" HF_HOME="$ROOT/data/hf_cache"
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WM="+watermark_opacity=0.05 +watermark_modulus=4096 +watermark_bits=12
    +watermark_tile=32 +origin_anchor=gabor +random_anchor=true"
case "$ARM" in
  clean)          EXTRA="" ;;
  watermarked)    EXTRA="$WM +watermark_repeat=true" ;;
  random_control) EXTRA="$WM +watermark_repeat=false" ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

time_run () {
  local steps=$1 rd="$ROOT/bench_${ARM}_${steps}"
  rm -rf "$rd"; mkdir -p "$rd"
  local s=$SECONDS
  # shellcheck disable=SC2086
  uv run python lejepa/lejepa_minimal.py +dataset=imagenet1k +bs=256 +lr=2e-3 \
    +lamb=0.02 +proj_dim=16 +V=4 +epochs=1000 +num_workers=16 \
    +max_steps="$steps" +eval_every_steps=100000 +lr_horizon=200000 $EXTRA \
    +seed=0 +run_dir="$rd" +data_dir="$ROOT/data" hydra.run.dir="$rd/hydra" \
    > "$rd/out.log" 2>&1 || { echo "run failed:"; tail -20 "$rd/out.log"; exit 1; }
  echo $((SECONDS - s))
}

a=$(time_run "$SHORT")
b=$(time_run "$LONG")
echo "$ARM: ${SHORT} steps ${a}s | ${LONG} steps ${b}s"
python3 -c "print(f'  slope: {($b - $a) / ($LONG - $SHORT):.3f} s/step   (startup ~{$a - $SHORT * ($b - $a) / ($LONG - $SHORT):.0f}s)')"
