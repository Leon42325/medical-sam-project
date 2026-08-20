#!/bin/bash -l
#SBATCH --job-name=samed-finetune
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/finetune-%j.out
#
# Fine-tune SAM under one arm of the representation-versus-readout experiment.
#
#   ARM=decoder              sbatch.tinygpu scripts/slurm/finetune.sbatch
#   ARM=lora_encoder         sbatch.tinygpu scripts/slurm/finetune.sbatch
#   ARM=lora_encoder_decoder sbatch.tinygpu scripts/slurm/finetune.sbatch
#
# The arms differ by two orders of magnitude in cost, and inherently so. The
# decoder arm never runs the image encoder - its embeddings are cached, so a
# step is a decode from a stored 256x64x64 tensor and an epoch takes seconds.
# The LoRA arms change the encoder every step, so every image is re-encoded at
# 1024x1024: roughly 0.3 s per example on an A100, some two hours for 20 epochs
# over the CHAOS training split.
#
# That gap is the answer to why the paper froze the encoder, and it is worth
# stating in the report rather than hiding behind a wall-clock number.
#
# Submit from the repository root.

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
COMMON="$PROJECT_DIR/scripts/slurm/common.sh"
if [[ ! -r "$COMMON" ]]; then
    echo "cannot read $COMMON - submit from the repository root:" >&2
    echo "  cd <repo> && sbatch.tinygpu scripts/slurm/$(basename "$0")" >&2
    exit 1
fi
source "$COMMON"

ARM="${ARM:-decoder}"
MODEL="${MODEL:-sam_vit_b}"
VARIANT="${VARIANT:-vit_b}"
CHECKPOINT="${CHECKPOINT:-$CKPT_ROOT/sam_vit_b_01ec64.pth}"
PREPARED="${PREPARED:-$DATA_ROOT}"
MANIFEST_STEM="${MANIFEST_STEM:-manifest-chaos}"
EPOCHS="${EPOCHS:-20}"
LORA_RANK="${LORA_RANK:-8}"
SEED="${SEED:-0}"

OUT_DIR="$PROJECT/runs/$MODEL/$ARM-r$LORA_RANK-seed$SEED"
mkdir -p "$OUT_DIR"

log "arm=$ARM model=$MODEL rank=$LORA_RANK epochs=$EPOCHS -> $OUT_DIR"

srun python -m samed.cli.finetune \
    --arm "$ARM" \
    --variant "$VARIANT" \
    --checkpoint "$CHECKPOINT" \
    --train "$PREPARED/$MANIFEST_STEM-train.csv" \
    --val "$PREPARED/$MANIFEST_STEM-val.csv" \
    --images "$PREPARED/images" \
    --labels "$PREPARED/labels" \
    --embeddings "$EMBED_ROOT/$MODEL" \
    --out "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --lora-rank "$LORA_RANK" \
    --seed "$SEED"

log "done"
