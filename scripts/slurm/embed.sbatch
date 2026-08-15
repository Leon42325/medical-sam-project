#!/bin/bash -l
#SBATCH --job-name=samed-embed
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/embed-%A_%a.out
#
# Stage 1: run each image encoder once and cache the result.
#
# This is the stage that makes the whole study fit in a shared GPU quota.  The
# encoder costs ~0.13 s per image for ViT-B and ~0.47 s for ViT-H, while prompt
# encoding and mask decoding cost ~0.01 s (paper Table 4).  Six strategies, three
# jitter levels and three seeds over the same images would re-run the encoder
# roughly fifty times for no reason; caching turns that into one pass.
#
# Partition choice: ViT-H and Hiera backbones want the 40 GB A100s.  ViT-B fits
# comfortably on the RTX 2080 Ti nodes in the default `work` partition, which is
# usually the least contended - override with:
#   sbatch.tinygpu -p work --gres=gpu:1 scripts/slurm/embed.sbatch
#
# Submit from the frontend as:
#   sbatch.tinygpu scripts/slurm/embed.sbatch

source "$(dirname "$0")/common.sh"

MODEL="${MODEL:-sam_vit_b}"
CHECKPOINT="${CHECKPOINT:-$CKPT_ROOT/sam_vit_b_01ec64.pth}"
MANIFEST="${MANIFEST:-$DATA_ROOT/manifest.csv}"
SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

OUT_DIR="$EMBED_ROOT/$MODEL"
mkdir -p "$OUT_DIR"

log "model=$MODEL shard=$SHARD/$SHARDS"

IMAGES=$(stage_to_tmpdir "$DATA_ROOT/images" images)

srun python -m samed.cli.embed \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --images "$IMAGES" \
    --out "$OUT_DIR" \
    --shard "$SHARD" \
    --num-shards "$SHARDS" \
    --skip-existing

log "shard $SHARD done"
