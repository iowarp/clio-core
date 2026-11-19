#!/usr/bin/env bash
#PBS -l select=4
#PBS -l place=scatter
#PBS -l walltime=00:15:00
#PBS -l filesystems=home:flare:daos_user_fs
#PBS -l daos=daos_user
#PBS -q debug-scaling
#PBS -A IOWarp
#PBS -j oe
#
# FOUR-NODE tiered run: the two-node tier script under the four-node
# headers (debug-scaling, since debug admits at most two nodes; fifteen
# minutes, the plan's per-test ceiling). pbs_newcoro_aurora_2n_tier.sh
# takes its node count from the allocation, so nothing else changes; every
# BENCH_* knob documented there applies. The per-rank cap defaults to 780 s
# here, leaving the dfuse launch and the tier listing their minute.
#
# PBS runs a copy of this file from its spool directory, so the tier script
# is found through ROOT (which submit_2n_aurora.sh always passes), not
# through this file's own location.
set -u
export BENCH_CAP="${BENCH_CAP:-780}"
ROOT=${ROOT:-/home/llogan/clio-core/.claude/worktrees/gpu-coro}
exec bash "${ROOT}/context-transfer-engine/adapter/gpu_vector/benchmark/pbs_newcoro_aurora_2n_tier.sh"
