#!/usr/bin/env bash
#SBATCH --job-name=np-nyx-validate
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=64g
#SBATCH --time=00:20:00
#SBATCH --output=/projects/bekn/imuradli/np-nyx-validate/slurm-%j.out
#
# GO / NO-GO for the 30 GiB campaign. One Nyx cell at the quick profile, run
# end to end under the new defaults (K=31, threshold -1, lossy, mandatory error
# bound, quality measured), then CHECKED rather than eyeballed.
#
# Every check below prints PASS or FAIL and the job exits non-zero if any of
# them fails, so a green exit code means the pipeline is safe to scale up. The
# checks exist because each has actually broken during this work:
#
#   * gen_fields silently producing nothing         (deleted dumps)
#   * the run exiting 0 with no blobs on the tier   (the WarpX VOL trap)
#   * explore.csv missing entirely                  (bench_compose read
#                                                    NP_EXPLORE too late)
#   * chunks skipped by the trigger gate            (104 of 126 at threshold 0)
#   * quality columns empty                         (MEASURE_QUALITY unset)
#   * meta.json malformed, taking collect.py down   (grep -c || echo 0)
set -uo pipefail
REPO=/u/imuradli/clio-core; PB=$REPO/paper-benchmark
export BUILD=/u/imuradli/np-build/clio
export NYX_BIN=/u/imuradli/np-build/nyx/Exec/HydroTests/nyx_HydroTests
export LD_LIBRARY_PATH=/u/imuradli/np-env/np/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

EB=${EB:-0.01}
ROOT=${ROOT:-/projects/bekn/imuradli/np-nyx-validate}
FIELDS=$ROOT/fields; RESULTS=$ROOT/results
mkdir -p "$RESULTS"
tag=nyx_lossy_balance_nvme_5GBs
S=$RESULTS/$tag

FAILED=0
chk(){ if [ "$1" = 0 ]; then printf "   PASS  %s\n" "$2"; else printf "   FAIL  %s\n" "$2"; FAILED=$((FAILED+1)); fi; }
say(){ echo; echo "[$(date +%H:%M:%S)] $*"; }

say "node=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^/       gpu: /'

# ---- 1. generate ----------------------------------------------------------
say "1. gen_fields (128^3, 200 steps, dump every 10)"
if [ "$(find "$FIELDS" -name '*.f32' 2>/dev/null | wc -l)" -gt 0 ]; then
  echo "       dumps already present, reusing (Nyx is deterministic for a fixed deck)"
else
  "$PB/nyx/gen_fields.sh" --ncell 128 --steps 200 --plot-int 10 --out "$FIELDS" 2>&1 | sed 's/^/       /'
fi
NF=$(find "$FIELDS" -name '*.f32' 2>/dev/null | wc -l)
chk $([ "$NF" -gt 0 ] && echo 0 || echo 1) "field dumps produced ($NF files, $(du -sh "$FIELDS" 2>/dev/null|cut -f1))"
[ "$NF" -eq 0 ] && { echo "cannot continue"; exit 1; }

# ---- 2. run ---------------------------------------------------------------
say "2. exploration run (K=31, thresh -1, eb=$EB, balance, quality on)"
env CLIO_NEUROPRESS_STAGE_H2D=1 \
    "$PB/nyx/run_config.sh" explore-balance --fields "$FIELDS" \
      --bw 5e6 --eb "$EB" --explore-k 31 --explore-thresh -1 --chunk 8388608 \
      --results "$RESULTS" --tag "$tag" 2>&1 | sed 's/^/       /'
rc=${PIPESTATUS[0]}
chk $([ "$rc" -eq 0 ] && echo 0 || echo 1) "run exited 0 (rc=$rc)"

# ---- 3. artefacts ---------------------------------------------------------
say "3. artefacts"
for f in blobs.csv selection.csv explore.csv meta.json stdout.log; do
  chk $([ -s "$S/$f" ] && echo 0 || echo 1) "$f exists and is non-empty"
done
NB=0; [ -f "$S/blobs.csv" ] && NB=$(( $(wc -l < "$S/blobs.csv") - 1 ))
chk $([ "$NB" -gt 0 ] && echo 0 || echo 1) "blobs actually reached the tier ($NB)"
python3 -c "import json;json.load(open('$S/meta.json'))" 2>/dev/null
chk $? "meta.json is valid JSON"

# ---- 4. coverage and measurement -----------------------------------------
say "4. exploration coverage and measured columns"
python3 - "$S/explore.csv" "$S/selection.csv" <<'PY'
import csv, sys
from collections import Counter
rows=list(csv.DictReader(open(sys.argv[1])))
sel=[r for r in csv.DictReader(open(sys.argv[2])) if r.get('role')=='primary']
per=Counter(r['blob'] for r in rows)
acts={(r['lib_name'],r['shuffle'],r['quantize']) for r in rows}
def ok(c,m): print(f"   {'PASS' if c else 'FAIL'}  {m}"); return 0 if c else 1
bad=0
bad+=ok(set(per.values())=={32}, f"32 rows per chunk (got {sorted(set(per.values()))})")
bad+=ok(len(acts)==32, f"32/32 distinct actions ({len(acts)})")
bad+=ok(len(per)==len(sel), f"EVERY chunk explored ({len(per)} of {len(sel)})")
q=Counter(r['quantize'] for r in rows)
bad+=ok(q.get('0')==q.get('1'), f"16 lossless + 16 quantize per chunk ({q.get('0')}/{q.get('1')})")
for c in ('ratio','ct_ms','dt_ms'):
    v=[r[c] for r in rows if r.get(c) not in (None,'')]
    bad+=ok(len(v)==len(rows) and all(float(x)>0 for x in v), f"{c} measured on all {len(rows)} rows")
same=sum(1 for r in rows if r.get('dt_ms')==r.get('pred_dt_ms'))
bad+=ok(same < len(rows)*0.5, f"dt_ms is measured, not predicted ({same} identical)")
qz=[r for r in rows if r['quantize']=='1']
m=sum(1 for r in qz if r.get('quality_measured')=='1')
bad+=ok(m > len(qz)*0.9, f"quality measured on the quantized half ({m} of {len(qz)})")
sys.exit(1 if bad else 0)
PY
chk $? "coverage + measurement block"

# ---- 5. errors in the log -------------------------------------------------
say "5. runtime log"
# Match the LOGGER'S OWN level tokens, not the word "error" anywhere. The
# expected WarnIfBoundUnachievable warning contains the phrase "the requested
# error bound ... is NOT achievable", which a bare /error/ flags as a failure.
E=$(grep -cE "\bERROR\b|\bFATAL\b|terminate called|Segmentation fault|core dumped" "$S/runtime.log" 2>/dev/null); E=${E:-0}
chk $([ "$E" -eq 0 ] && echo 0 || echo 1) "no errors in runtime.log ($E)"
W=$(grep -c "WARNING" "$S/runtime.log" 2>/dev/null); W=${W:-0}
echo "   note  $W warning(s) -- the unachievable-bound one on rho_E/rho_e is expected"

# ---- 6. collect -----------------------------------------------------------
say "6. collect.py"
"$PB/collect.py" "$RESULTS" > "$ROOT/collect.out" 2>&1
chk $? "collect.py ran clean"
grep -E "^\| lossy" "$ROOT/collect.out" | head -2 | sed 's/^/       /'

rm -f "$S/chi_bdev.dat" "$S"/cte_tier.dat*
say "VERDICT"
if [ "$FAILED" -eq 0 ]; then echo "   ALL CHECKS PASSED -- safe to submit the 30 GiB campaign"; else echo "   $FAILED CHECK(S) FAILED -- do not scale up"; fi
exit $FAILED
