# Shared setup sourced by every batch job.  Not executable on its own.
#
# Two cluster facts drive the design of these jobs:
#
#   * Every TinyGPU partition caps walltime at 24 h, with no exception, so no
#     stage may be a single long job.  Everything is an array of shards, each
#     sized to finish comfortably inside the limit.
#   * Jobs are preemptible in practice by queue pressure, so every shard writes
#     its own output file and skips work already on disk.  Losing a job costs one
#     shard, never the run.

set -euo pipefail

PROJECT="${PROJECT:-$WORK/medical-sam-project}"
ENV_PREFIX="${ENV_PREFIX:-$PROJECT/env}"
DATA_ROOT="${DATA_ROOT:-$PROJECT/data}"
EMBED_ROOT="${EMBED_ROOT:-$PROJECT/embeddings}"
CKPT_ROOT="${CKPT_ROOT:-$PROJECT/checkpoints}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT/results}"

module load python
# See scripts/setup_tinygpu.sh: batch jobs run a non-interactive shell, where
# `conda activate` is undefined until the hook is evaluated.
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"

# Read data from node-local SSD rather than the shared filesystem.  Embeddings
# are many small files and the parallel filesystem is the wrong tool for that
# access pattern; $TMPDIR gives at least 1.8 TB and is cleared at job end.
stage_to_tmpdir() {
    local src="$1" name="$2"
    if [[ -d "$src" ]]; then
        mkdir -p "$TMPDIR/$name"
        cp -r "$src/." "$TMPDIR/$name/"
        echo "$TMPDIR/$name"
    else
        echo "$src"
    fi
}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

log "job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
