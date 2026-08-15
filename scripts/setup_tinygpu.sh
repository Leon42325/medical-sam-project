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
#
# `set -u` has to be lifted around anything conda: its own activate/deactivate
# hooks read variables without defaults - the MKL ones in particular - so under
# `set -u` re-activating an already-active environment aborts with
# "CONDA_MKL_INTERFACE_LAYER_BACKUP: unbound variable". Our own code stays
# strict; only the conda calls are exempted.
set +u
eval "$(conda shell.bash hook)"

# Keep conda's own caches off $HOME as well - the package cache is another
# large collection of small files.
export CONDA_PKGS_DIRS="$PROJECT/conda/pkgs"

# Clone the cluster's maintained PyTorch environment: it is faster than
# resolving torch from scratch and is already matched to the local CUDA stack.
#
# Name it explicitly. Deriving it from `command -v python` looks tidier and is
# wrong - after `module load python` that resolves to the conda *base*
# environment, which has no torch, and the resulting env fails only much later
# at "No module named 'torch'".
BASE_ENV="${BASE_ENV:-$(conda info --base)/envs/pytorch2.6-py3.12}"

if [[ ! -d "$BASE_ENV" ]]; then
    echo "no base environment at $BASE_ENV" >&2
    echo "available environments:" >&2
    conda env list >&2
    echo "set BASE_ENV=<path> to pick one with PyTorch in it" >&2
    exit 1
fi

if [[ ! -d "$ENV_PREFIX" ]]; then
    echo "cloning $BASE_ENV -> $ENV_PREFIX (this takes a while)"
    conda create --yes --prefix "$ENV_PREFIX" --clone "$BASE_ENV"
fi

conda activate "$ENV_PREFIX"
set -u
echo "python: $(command -v python)"

# The cluster's PyTorch environment ships torch without torchvision, and
# `segment_anything` needs it (torchvision.ops supplies the NMS its automatic
# mask generator runs). Installing it naively is how you break the environment:
# pip would resolve torchvision from PyPI, decide the installed torch does not
# satisfy it, and replace a CUDA build with a CPU one - silently, and only
# visible later as "CUDA available: False" on a GPU node.
#
# So: pull from PyTorch's own index for the CUDA tag torch was built against,
# pin the version that pairs with it, and --no-deps so torch is never touched.
if ! python -c "import torchvision" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
    CUDA_TAG=$(python -c "import torch; v = torch.version.cuda; print('cu' + v.replace('.', '') if v else 'cpu')")

    case "$TORCH_VERSION" in
        2.4.*) VISION_VERSION=0.19.* ;;
        2.5.*) VISION_VERSION=0.20.* ;;
        2.6.*) VISION_VERSION=0.21.* ;;
        2.7.*) VISION_VERSION=0.22.* ;;
        *)
            echo "no known torchvision pairing for torch $TORCH_VERSION" >&2
            echo "add it to the table in $0, or set VISION_VERSION" >&2
            exit 1
            ;;
    esac

    echo "installing torchvision $VISION_VERSION for torch $TORCH_VERSION ($CUDA_TAG)"
    python -m pip install --no-cache-dir --no-deps \
        --index-url "https://download.pytorch.org/whl/$CUDA_TAG" \
        "torchvision==$VISION_VERSION"
fi

python -m pip install --no-cache-dir -r "$(dirname "$0")/../requirements-gpu.txt"
python -m pip install --no-cache-dir -e "$(dirname "$0")/.."

# Verify what was built, rather than announcing success and finding out later.
# The failure this catches - cloning an environment without PyTorch - stayed
# hidden until the first GPU job, several steps downstream.
echo
echo "checking the environment:"
python - <<'CHECK'
import importlib, sys

missing = []
for module in ("torch", "cv2", "pydicom", "nibabel", "segment_anything", "samed"):
    try:
        importlib.import_module(module)
        print(f"  ok    {module}")
    except ImportError as error:
        print(f"  MISSING {module}: {error}")
        missing.append(module)

try:
    import torch

    print(f"  torch {torch.__version__} (CUDA build: {torch.version.cuda or 'none'})")
    print(f"  CUDA available here: {torch.cuda.is_available()}  "
          "(False on the frontend is expected - it has no GPU)")
    if torch.version.cuda is None:
        print("  WARNING: this is a CPU-only torch build; a dependency install "
              "has replaced the cluster's CUDA build")
        missing.append("torch (CUDA build)")
except ImportError:
    pass

if missing:
    print(f"\nenvironment is incomplete: {', '.join(missing)}")
    sys.exit(1)
CHECK

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
