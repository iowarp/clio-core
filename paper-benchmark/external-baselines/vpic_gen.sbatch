#!/usr/bin/env bash
#SBATCH --job-name=vpic-gen
#SBATCH --account=CHANGEME   # override: sbatch -A <account>
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=64g
#SBATCH --time=00:35:00
#SBATCH --output=vpicgen-%j.out
#
# Phase 1: regenerate VPIC Weibel field dumps. np-vpic-30g holds only the
# RESULTS of the earlier 30 GB run -- the dumps themselves were cleaned up --
# so there is nothing to replay until this runs.
#
# 3000 steps, dumping every 24 -> 125 frames x 16 field variables = 2000 files
# of 8 MiB = 15.6 GiB, matching the Nyx learning run chunk for chunk. The dump
# cadence is the published run's (6000/25); the step count is half of it.
#
# This is a simulation, not a benchmark arm -- it does not fit the 5-10 minute
# rule. The previous 6000-step job took 38 minutes WITH an in-situ K=31 sweep
# that the summary attributes 37.8 min of wall to, so pure generation at half
# the steps should be well inside 35 min.
#
# --clean-div 10 is the harness default and matters: at 0, four of the sixteen
# variables (div_e_err, div_b_err, rhob, rhof) are never recomputed and dump
# bit-identical every frame -- 25% of the payload that does not evolve, which
# is exactly what a learning study must not be fed.
set -uo pipefail
. "$(dirname "$0")/env.sh"
PB=$REPO/paper-benchmark
OUT=${VPIC_FIELDS}
# LD_LIBRARY_PATH comes from env.sh
echo "=== $(hostname) / $(nvidia-smi -L | head -1) ==="
echo "target: $OUT"
df -h /projects | tail -1 | awk '{print "disk before: "$4" avail"}'

s=$(date +%s.%N)
OUT="$OUT" "$PB/vpic/gen_fields.sh" --steps 3000 --dump-int 24 --clean-div 10 --out "$OUT"
rc=$?
echo "gen rc=$rc wall=$(awk -v a=$s -v b=$(date +%s.%N) 'BEGIN{printf "%.1f",b-a}')s"

echo; echo "=== what landed ==="
echo "files: $(find $OUT -name '*.f32' 2>/dev/null | wc -l)   frames: $(ls -d $OUT/plt* 2>/dev/null | wc -l)"
echo "size:  $(du -sh $OUT 2>/dev/null | cut -f1)"
cat $OUT/gen.json 2>/dev/null
echo "vars:  $(find $OUT -name '*.f32' -printf '%f\n' 2>/dev/null | sed -E 's/fab[0-9]+_comp[0-9]+_//; s/\.f32//' | sort -u | tr '\n' ' ')"

echo; echo "=== do the frames evolve? (the check Nyx had to pass) ==="
python3 - "$OUT" <<'PY'
import sys, os, glob, hashlib, collections
out = sys.argv[1]
fs = sorted(glob.glob(os.path.join(out, "plt*", "*.f32")))
print("  files:", len(fs))
if not fs: sys.exit(0)
h = collections.Counter()
for f in fs:
    with open(f, "rb") as fh:
        h[hashlib.blake2b(fh.read(), digest_size=8).hexdigest()] += 1
dup = sum(v - 1 for v in h.values() if v > 1)
print("  distinct payloads: %d / %d  (%d duplicate files)" % (len(h), len(fs), dup))
if dup:
    for k, v in h.most_common(3):
        if v > 1: print("    a payload repeated %d times" % v)
PY
df -h /projects | tail -1 | awk '{print "disk after:  "$4" avail"}'
echo "===== done ====="
