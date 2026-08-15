# Put the project environment on PATH in the CURRENT shell.
#
#   source $HOME/medical-sam-project/scripts/activate.sh
#
# Must be sourced, not executed: a child process cannot change its parent's
# PATH, which is exactly why running setup_tinygpu.sh leaves the calling shell
# without `python`. Batch jobs do the same three steps through
# scripts/slurm/common.sh and do not need this file.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This script must be sourced, not executed:" >&2
    echo "  source ${BASH_SOURCE[0]}" >&2
    exit 1
fi

_samed_env="${ENV_PREFIX:-${PROJECT:-$WORK/medical-sam-project}/env}"

if [[ ! -d "$_samed_env" ]]; then
    echo "no environment at $_samed_env" >&2
    echo "run scripts/setup_tinygpu.sh first" >&2
else
    module load python
    eval "$(conda shell.bash hook)"
    conda activate "$_samed_env"
    echo "activated $_samed_env"
    echo "python: $(command -v python)"
fi

unset _samed_env
