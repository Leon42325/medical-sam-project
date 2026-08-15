#!/bin/bash
# One-time environment setup on NHR@FAU TinyGPU.  Run once from the frontend
# (tinyx.nhr.fau.de); everything afterwards is batch jobs.
#
# The conda prefix deliberately lives in $WORK.  $HOME allows only 500k inodes
# and a single conda environment can hold well over a hundred thousand small
# files, so building there is a reliable way to exhaust the file quota while
# using a fraction of the 100 GB of space.  $WORK allows 5M inodes.
#
# Note $WORK has neither snapshots nor backup: code stays in $HOME (or in git),
# data and environments live in $WORK, and nothing irreplaceable lives only there.

set -euo pipefail

PROJECT="${PROJECT:-$WORK/medical-sam-project}"
ENV_PREFIX="$PROJECT/env"
DATA_ROOT="$PROJECT/data"
EMBED_ROOT="$PROJECT/embeddings"
CKPT_ROOT="$PROJECT/checkpoints"

mkdir -p "$DATA_ROOT" "$EMBED_ROOT" "$CKPT_ROOT" "$PROJECT/logs"

module load python

# `conda activate` is a shell function, and a non-interactive script has not
# sourced the profile that defines it. Without this hook the script dies with
# "CommandNotFoundError: Your shell has not been properly configured".
eval "$(conda shell.bash hook)"

# Keep conda's own caches off $HOME as well - the package cache is another
# large collection of small files.
export CONDA_PKGS_DIRS="$PROJECT/conda/pkgs"

# Clone by absolute path rather than by name: the base environment is the one
# the `python` module provides, and resolving it by name depends on envs_dirs,
# which we are deliberately not relying on here.
BASE_ENV="${BASE_ENV:-$(dirname "$(dirname "$(command -v python)")")}"

if [[ ! -d "$ENV_PREFIX" ]]; then
    # Cloning the cluster's maintained PyTorch environment is faster and better
    # matched to the local CUDA stack than resolving torch from scratch.
    echo "cloning $BASE_ENV -> $ENV_PREFIX (this takes a while)"
    conda create --yes --prefix "$ENV_PREFIX" --clone "$BASE_ENV"
fi

conda activate "$ENV_PREFIX"
echo "python: $(command -v python)"

python -m pip install --no-cache-dir -r "$(dirname "$0")/../requirements-gpu.txt"
python -m pip install --no-cache-dir -e "$(dirname "$0")/.."

cat <<EOF

Environment ready.
  project   $PROJECT
  env       $ENV_PREFIX
  data      $DATA_ROOT
  embeddings$EMBED_ROOT
  weights   $CKPT_ROOT

Checkpoints still to download into \$CKPT_ROOT:
  sam_vit_b_01ec64.pth, sam_vit_h_4b8939.pth   (facebookresearch/segment-anything)
  medsam_vit_b.pth                             (bowang-lab/MedSAM)

Submit work with the .tinygpu wrappers from the frontend, e.g.
  sbatch.tinygpu scripts/slurm/embed.sbatch
EOF
