#!/usr/bin/env bash
#SBATCH --job-name=phase1
#SBATCH --account=CHANGEME   # override on the command line: sbatch -A <account>
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=64g
#SBATCH --time=00:10:00
#SBATCH --output=$WORK/phase1/%x-%j.out
#
# PHASE 1 -- the in-action-space per-chunk oracle.
#
# best_mode sweeps the FULL remaining action space on every chunk and measures
# each candidate, so the explore log becomes the oracle table: one row per
# (chunk, candidate) with measured ratio, compress time, decompress time,
# quality and cost, plus `adopted` marking which one was stored.
#
# NO ORACLE LEAKAGE, by construction rather than by discipline: best mode
# suppresses BOTH SGD phases (upstream's `&& !g_best_mode.load()`), so nothing
# measured here can reach the weights. The model still predicts every
# candidate, and those predictions are logged, so per-codec prediction error
# becomes measurable for the first time -- normally only the SELECTED
# candidate is ever measured.
#
# What this cannot answer: cuSZ, ndzip and cuSZp are excluded from NeuroPress's
# candidate set by kNeuroPressTrainedGpuBaseIds, so they are outside this
# oracle. That is Phase 3, and it is a question about the ACTION SPACE, not
# about the selector.
#
# Ratio cost model, per the evaluation spec.
set -uo pipefail
. "$(dirname "$0")/env.sh"
H=$(dirname "$0")
R=$WORK/phase1/${WL}; mkdir -p "$R"
CHUNKS=${CHUNKS:-300}
export W_CT=0 W_DT=0 W_IO=1

case "$WL" in
  nyx)    FIELDS=$NYX_FIELDS ;;
  vpic)   FIELDS=$VPIC_FIELDS ;;
  warpx)  FIELDS=$WARPX_FIELDS ;;
  lammps) FIELDS=insitu ;;
  *) echo "unknown WL=$WL" >&2; exit 2 ;;
esac

echo "=== $(hostname) / $(nvidia-smi -L | head -1) ==="
echo "PHASE 1 oracle: WL=$WL chunks=$CHUNKS  best_mode=on K=31 threshold=0.0  ratio cost"
echo "libso: $(stat -c %y $BUILD/bin/libclio_cte_compressor_runtime.so)"
s=$(date +%s.%N)
# THRESH=0.0 so every chunk is swept; K=31 is the whole remaining space.
# MEASURE_DT makes the sweep decompress each candidate, which is the only
# place a measured decompression time exists at all.
COMMON=(NP_EXPLORE=true BEST=true EXPLORE_K=31 THRESH=0.0
        CLIO_NEUROPRESS_EXPLORE_MEASURE_DT=1 CTP_LOG_LEVEL=info)
if [ "$WL" = lammps ]; then
  env "${COMMON[@]}" "$H/run_learn_lammps.sh" 100 $((CHUNKS+200)) 20 8388608 1e-3 0.10 "$R/oracle" 2>&1 | tail -10
else
  env "${COMMON[@]}" "$H/run_learn.sh" "$FIELDS" "$CHUNKS" 8388608 1e-3 0.10 "$R/oracle" 2>&1 | tail -10
fi
rc=${PIPESTATUS[0]}
echo "rc=$rc wall=$(awk -v a=$s -v b=$(date +%s.%N) 'BEGIN{printf "%.1f",b-a}')s"
rm -f "$R/oracle/chi_bdev.dat" "$R/oracle"/cte_tier.dat*

echo
echo "=== oracle table ==="
python3 - "$R/oracle" "$WL" <<'PYEOF'
import csv, json, os, sys, collections
d, wl = sys.argv[1], sys.argv[2]
ep = f"{d}/explore.csv"
if not os.path.exists(ep) or os.path.getsize(ep) == 0:
    print("  explore.csv EMPTY -- the sweep did not run or did not log"); sys.exit(1)
ex = list(csv.DictReader(open(ep)))
per = collections.defaultdict(list)
for r in ex: per[r["blob"]].append(r)
cand = collections.Counter(len(v) for v in per.values())
print(f"  chunks swept        {len(per)}")
print(f"  rows                {len(ex)}   candidates/chunk {dict(sorted(cand.items()))}")
prim = sum(1 for r in ex if r["role"] == "primary")
print(f"  primary rows        {prim}   alt rows {len(ex)-prim}")
qv = collections.Counter(r.get("quality_measured","?") for r in ex)
print(f"  quality_measured    {dict(qv)}")
dt = [float(r["dt_ms"]) for r in ex if r["dt_ms"] not in ("", "NA") and float(r["dt_ms"]) > 0]
print(f"  decompress timed on {len(dt)}/{len(ex)} rows")
libs = collections.Counter(r["lib_name"] for r in ex)
print(f"  distinct codecs     {len(libs)}: {dict(libs.most_common(10))}")
PYEOF
