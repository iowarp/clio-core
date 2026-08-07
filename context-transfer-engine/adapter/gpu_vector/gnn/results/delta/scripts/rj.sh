#!/bin/bash
# Run a command on the held A100 node, inside the CUDA deps container.
#   ./rj.sh <command...>
# Env:
#   JOB   - slurm jobid of the holder allocation (default 20900276)
#   CPUS  - cpus for the step (default 32)
#   NOGPU - set to 1 to skip --nv (pure CPU steps, e.g. data prep)
set -uo pipefail
JOB="${JOB:-20900276}"
CPUS="${CPUS:-32}"
SIF=/u/rpawar/containers/deps-nvidia.sif
NV="--nv"
[ "${NOGPU:-0}" = "1" ] && NV=""

exec srun --jobid="$JOB" --overlap -n1 -c"$CPUS" \
     apptainer exec $NV -B /work,/tmp,/u,/projects "$SIF" "$@"
