#!/bin/bash -l
#SBATCH --job-name=samed-predict
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/predict-%A_%a.out
#
# Stage 2: decode masks for the prompt strategies S2-S6 against cached embeddings.
#
# Cheap by construction - the encoder does not run here at all - so this stage
# goes to the default `work` partition, where the RTX 2080 Ti nodes are usually
# free. The everything mode (S1) is NOT part of this job: it samples a grid over
# the whole image and runs heavy non-maximum suppression, costing ~2 s (ViT-B) to
# ~3 s (ViT-H) per image, and it cannot reuse a cached embedding the same way.
# It has its own script.
#
# Every candidate mask is written out, not just the selected one, so that the
# oracle rule of the paper and the deployable selection rules can be compared
# afterwards without re-running inference (see samed.selection).

source "$(dirname "$0")/common.sh"

MODEL="${MODEL:-sam_vit_b}"
CHECKPOINT="${CHECKPOINT:-$CKPT_ROOT/sam_vit_b_01ec64.pth}"
MANIFEST="${MANIFEST:-$DATA_ROOT/manifest.csv}"
STRATEGIES="${STRATEGIES:-S2,S3,S4,S5,S6}"
JITTER="${JITTER:-none}"          # none | 0-10 | 10-20 | 20-30
SEED="${SEED:-0}"
SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

OUT_DIR="$RESULT_ROOT/raw/$MODEL/jitter-$JITTER/seed-$SEED"
mkdir -p "$OUT_DIR"

log "model=$MODEL strategies=$STRATEGIES jitter=$JITTER seed=$SEED shard=$SHARD/$SHARDS"

EMBEDDINGS=$(stage_to_tmpdir "$EMBED_ROOT/$MODEL" "embeddings-$MODEL")

srun python -m samed.cli.predict \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --embeddings "$EMBEDDINGS" \
    --labels "$DATA_ROOT/labels" \
    --strategies "$STRATEGIES" \
    --jitter "$JITTER" \
    --seed "$SEED" \
    --out "$OUT_DIR" \
    --shard "$SHARD" \
    --num-shards "$SHARDS" \
    --skip-existing

log "shard $SHARD done"
