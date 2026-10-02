#!/usr/bin/env bash
#SBATCH --job-name=ior-tiers
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1 --ntasks=16 --cpus-per-task=1 --gpus-per-node=1 --mem=64g
#SBATCH --time=00:10:00
#SBATCH --output=ior-tiers-%j.out
#===============================================================================
# ior_tiers.sh -- IOR bandwidth of each storage tier, 1 MiB transfers:
#   dram          /dev/shm (tmpfs, host DRAM)
#   nvme          /tmp, the node's local NVMe (xfs)
#   burst_buffer  /work/nvme, Delta's flash Lustre ("SSD")
#   lustre        /work/hdd, the HDD Lustre PFS ("HDD")
# For each tier and each process count in RANKS: a durable write (POSIX, file
# per process, fsync at the end of the phase, -e) then a read of the same files
# with O_DIRECT (buffered on /dev/shm, which has no O_DIRECT), REPS repetitions
# each. Every run moves TOTAL_GB in all, split evenly over the processes. The
# IOR logs and JSON summaries stay under OUT/raw; ior_to_csv.py turns them into
# OUT/ior_bw.csv (same columns as probe_tiers.sh's tier_bw.csv). Test files are
# removed after each tier.
#
#   sbatch ior_tiers.sh
#
# Environment: TOTAL_GB (4), REPS (3), RANKS ("1 16"), XFER (1m),
#              OUT (np-codec-sweep/tier-probe/ior-<stamp>-<job>)
#===============================================================================
set -uo pipefail
ulimit -c 0
if [ -n "${SLURM_JOB_ID:-}" ]; then
  SELF=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/ Command=/{print $2; exit}')
else
  SELF=${BASH_SOURCE[0]}
fi
HERE=$(cd "$(dirname "$SELF")" && pwd)
JOB=${SLURM_JOB_ID:-$$}
TOTAL_GB=${TOTAL_GB:-4}
REPS=${REPS:-3}
RANKS=${RANKS:-"1 16"}
XFER=${XFER:-1m}
OUT=${OUT:-/projects/bekn/$USER/np-codec-sweep/tier-probe/ior-$(date +%m%d%H%M)-$JOB}
IOR=${IOR:-/sw/rh9.4/spack/v1.0.0/sw/linux-x86_64_v2/ior-3.3.0-oznn4mz/bin/ior}
TIERS="dram nvme burst_buffer lustre"
declare -A DIRS=([dram]=/dev/shm/ior-$JOB [nvme]=/tmp/ior-$JOB
                 [burst_buffer]=/work/nvme/bekn/$USER/ior-$JOB
                 [lustre]=/work/hdd/bekn/$USER/ior-$JOB)
trap 'rm -rf "${DIRS[@]}"' EXIT

# run_ior TIER N OP -- one IOR run (OP = write or read) with N processes;
# logs to OUT/raw/<tier>-n<N>-<op>.{log,json}
run_ior() {
  local tier=$1 n=$2 op=$3 dir=${DIRS[$1]} block flags
  block=$(( TOTAL_GB * 1024 / n ))m
  if [ "$op" = write ]; then flags=(-w -e -k)
  else flags=(-r -k); [ "$tier" != dram ] && flags+=(--posix.odirect); fi
  srun -n "$n" --cpu-bind=cores "$IOR" -a POSIX -F "${flags[@]}" -t "$XFER" \
    -b "$block" -i "$REPS" -o "$dir/ior.dat" -O summaryFormat=JSON \
    -O summaryFile="$OUT/raw/$tier-n$n-$op.json" > "$OUT/raw/$tier-n$n-$op.log" 2>&1
  echo "$tier n=$n $op rc=$? $(grep -E '^(write|read) ' "$OUT/raw/$tier-n$n-$op.log" | tail -1)"
}

mkdir -p "$OUT/raw"
{ hostname; for t in $TIERS; do mkdir -p "${DIRS[$t]}"
    echo "$t ${DIRS[$t]} $(df -T "${DIRS[$t]}" | awk 'NR==2{print $2}')"; done; } | tee "$OUT/node.txt"
for t in $TIERS; do
  for n in $RANKS; do
    run_ior "$t" "$n" write
    run_ior "$t" "$n" read
    rm -f "${DIRS[$t]}"/ior.dat*
  done
done
python3 "$HERE/ior_to_csv.py" "$OUT/raw" > "$OUT/ior_bw.csv" && column -s, -t "$OUT/ior_bw.csv"
echo "csv: $OUT/ior_bw.csv"
