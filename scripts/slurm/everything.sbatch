#!/bin/bash -l
#SBATCH --job-name=samed-everything
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/everything-%A_%a.out
#
# Strategy S1: automatic mode, the one strategy with no prompt.
#
# By far the most expensive stage. It samples a points_per_side^2 grid over
# every image, decodes a mask per grid point and runs non-maximum suppression
# over the result: ~2 s per image for ViT-B and ~3 s for ViT-H, against ~0.01 s
# for a prompted decode (paper Table 4). It also cannot reuse the embedding
# cache, because the generator crops and tiles the image itself.
#
# Hence the a100 partition and 16 shards for what the prompted stage does in one
# minute. The grid ablation of the paper's Table 5 multiplies this again, so it
# is run separately and only on a subset:
#   POINTS_PER_SIDE=64 sbatch.tinygpu scripts/slurm/everything.sbatch
#
# Submit from the repository root:
#   cd <repo> && sbatch.tinygpu scripts/slurm/everything.sbatch

# Slurm runs a *copy* of this script from its spool directory, so "$0" points
# there rather than into the repository and `dirname "$0"` cannot find the
# shared setup. SLURM_SUBMIT_DIR is the directory sbatch was invoked from.
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
COMMON="$PROJECT_DIR/scripts/slurm/common.sh"
if [[ ! -r "$COMMON" ]]; then
    echo "cannot read $COMMON - submit from the repository root:" >&2
    echo "  cd <repo> && sbatch.tinygpu scripts/slurm/$(basename "$0")" >&2
    exit 1
fi
source "$COMMON"

MODEL="${MODEL:-sam_vit_b}"
CHECKPOINT="${CHECKPOINT:-$CKPT_ROOT/sam_vit_b_01ec64.pth}"
MANIFEST="${MANIFEST:-$DATA_ROOT/manifest.csv}"
POINTS_PER_SIDE="${POINTS_PER_SIDE:-32}"
SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

OUT_DIR="$RESULT_ROOT/raw/$MODEL/jitter-none/seed-0"
mkdir -p "$OUT_DIR"

log "model=$MODEL grid=${POINTS_PER_SIDE}x${POINTS_PER_SIDE} shard=$SHARD/$SHARDS"

IMAGES=$(stage_to_tmpdir "$DATA_ROOT/images" images)

srun python -m samed.cli.everything \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --images "$IMAGES" \
    --labels "$DATA_ROOT/labels" \
    --out "$OUT_DIR" \
    --points-per-side "$POINTS_PER_SIDE" \
    --shard "$SHARD" \
    --num-shards "$SHARDS" \
    --skip-existing

log "shard $SHARD done"
