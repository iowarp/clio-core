#!/usr/bin/env bash
#SBATCH --job-name=lossy-sweep
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=64g
#SBATCH --time=00:10:00
#SBATCH --output=lossy-sweep-%j.out
#===============================================================================
# run_lossy_sweep.sh -- run every GPU lossy codec (one lossy_sweep_<codec>
# binary each, see build_lossy.sh) over whole fields of each workload at
# value-range-relative error bounds, one CSV per (workload, codec).
#
#   FILES=2 sbatch run_lossy_sweep.sh     # smoke: 2 evenly spaced fields each
#   sbatch --time=... run_lossy_sweep.sh  # every field
#
# Environment:
#   CODECS     space list (fsz cuszp3 cuszp2 cuszi cuszhi pfpl fzgpu cuszx cuzfp)
#   WORKLOADS  space list (nyx vpic hurricane cesm-atm cesm-atm-3d scale-letkf qmcpack)
#   EBS        comma list of relative bounds (0.01,0.02,0.03)
#   FILES      fields per workload, evenly spaced over the sorted list; 0 = all
#   OUT        results directory (np-codec-sweep/{smoke,full}/lossy-<stamp>-<job>;
#              smoke when FILES > 0)
#   DUMPS      simulation dumps (/work/hdd/bekn/$USER/np-dumps)
#   SDR        SDRBench root (/projects/bekn/$USER/sdrbench)
#   BUILD_DIR  binaries ($HOME/np-build/lossy-sweep)
#   STAGE      1 = copy the chosen fields to node-local /tmp first (1)
#   VARIANTS   comma list passed to --variants (default: every variant of the
#              codec); the CSV is then named <codec>-<variants>.csv
#   LIST       explicit field list (paths relative to the workload's source
#              directory) instead of picking from it; for re-running only the
#              fields a timed-out job never reached. One workload per job.
#   CHUNK_FIELDS  > 0: restart the codec in a fresh process every N fields and
#              append to the CSV after each batch. Bounds a codec's per-call
#              device leak (cuSZ-Hi's LC stages never free their buffers). (0)
#===============================================================================
set -uo pipefail
# No core dumps: a crashing codec on a 40 GB GPU writes ~37 GB of core into the
# results folder (four did on 2026-10-02).
ulimit -c 0
if [ -n "${SLURM_JOB_ID:-}" ]; then
  SELF=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/ Command=/{print $2; exit}')
else
  SELF=${BASH_SOURCE[0]}
fi
HERE=$(cd "$(dirname "$SELF")" && pwd)

CODECS=${CODECS:-fsz cuszp3 cuszp2 cuszi cuszhi pfpl fzgpu cuszx cuzfp}
WORKLOADS=${WORKLOADS:-nyx vpic hurricane cesm-atm cesm-atm-3d scale-letkf qmcpack}
EBS=${EBS:-0.01,0.02,0.03}
FILES=${FILES:-0}
DUMPS=${DUMPS:-/work/hdd/bekn/$USER/np-dumps}
SDR=${SDR:-/projects/bekn/$USER/sdrbench}
# Smoke runs (a field subset) and full runs land in separate trees.
KIND=$([ "$FILES" -gt 0 ] && echo smoke || echo full)
OUT=${OUT:-/projects/bekn/$USER/np-codec-sweep/$KIND/lossy-$(date +%m%d%H%M)-${SLURM_JOB_ID:-local}}
BUILD_DIR=${BUILD_DIR:-$HOME/np-build/lossy-sweep}
STAGE=${STAGE:-1}
VARIANTS=${VARIANTS:-}
CHUNK_FIELDS=${CHUNK_FIELDS:-0}
LIST=${LIST:-}

# wl_info WL -- print "source_dir dims" for a workload (dims: x,y,z, x fastest)
wl_info() {
  case "$1" in
    nyx)       echo "$DUMPS/nyx/fields 128,128,128" ;;
    vpic)      echo "$DUMPS/vpic/fields 128,128,128" ;;
    hurricane) echo "$SDR/Hurricane-ISABEL-series 500,500,100" ;;
    cesm-atm)  echo "$SDR/CESM-ATM 3600,1800" ;;
    cesm-atm-3d) echo "$SDR/CESM-ATM-3D 3600,1800,26" ;;
    scale-letkf) echo "$SDR/SCALE-LETKF 1200,1200,98" ;;
    qmcpack)   echo "$SDR/QMCPACK 69,69,33120" ;;
    *) return 1 ;;
  esac
}

# pick_files SRC LIST -- the sorted .f32 files under SRC, FILES of them
# evenly spaced when FILES > 0, written to LIST (relative paths)
pick_files() {
  (cd "$1" && find . -name '*.f32' -type f -printf '%P\n') | LC_ALL=C sort |
    awk -v k="$FILES" '{ f[NR] = $0 } END {
      if (k <= 0 || k >= NR) { for (i = 1; i <= NR; i++) print f[i]; exit }
      for (j = 0; j < k; j++) print f[1 + int(j * (NR - 1) / (k > 1 ? k - 1 : 1))] }' > "$2"
}

# stage SRC LIST DST -- copy the listed files to DST with parallel streams
stage() {
  mkdir -p "$3"
  # Create every directory first: parallel `cp --parents` race on a shared
  # parent ("cannot make directory ... File exists") and drop files.
  (cd "$3" && awk -F/ 'NF > 1 { NF--; print }' OFS=/ "$2" | sort -u | xargs -r mkdir -p)
  (cd "$1" && xargs -a "$2" -P 16 -n 8 cp --parents -t "$3")
  local want got; want=$(wc -l < "$2"); got=$(cd "$3" && xargs -a "$2" ls 2>/dev/null | wc -l)
  [ "$got" -eq "$want" ] || { echo "staged $got of $want files" >&2; return 1; }
}

# run_codec BIN DIR DIMS WL LIST CSV LOG -- one codec over LIST, in one process or
# in batches of CHUNK_FIELDS fields; the CSV grows after every batch.
# @return the last non-zero exit code of any batch, else 0
run_codec() {
  local bin=$1 dir=$2 dims=$3 wl=$4 list=$5 csv=$6 log=$7 rc=0 r part
  local args=(--dir "$dir" --dims "$dims" --ebs "$EBS" --workload "$wl")
  [ -n "$VARIANTS" ] && args+=(--variants "$VARIANTS")
  if [ "$CHUNK_FIELDS" -le 0 ]; then
    "$bin" "${args[@]}" --list "$list" --out "$csv" > "$log" 2>&1
    return $?
  fi
  local parts=$STAGED/parts-$wl-$(basename "$csv" .csv)
  mkdir -p "$parts" && split -l "$CHUNK_FIELDS" -d -a 5 "$list" "$parts/p."
  : > "$log"
  for part in "$parts"/p.*; do
    "$bin" "${args[@]}" --list "$part" --out "$part.csv" >> "$log" 2>&1
    r=$?; [ $r -ne 0 ] && rc=$r
    if [ -s "$csv" ]; then tail -n +2 "$part.csv" >> "$csv"; else cat "$part.csv" > "$csv"; fi
  done
  return $rc
}

mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
echo "codecs: $CODECS | workloads: $WORKLOADS | ebs: $EBS | files: $FILES | variants: ${VARIANTS:-all} | chunk: $CHUNK_FIELDS -> $OUT"
STAGED=/tmp/lossy-sweep-${SLURM_JOB_ID:-$$}
trap 'rm -rf "$STAGED"' EXIT
for wl in $WORKLOADS; do
  read -r src dims < <(wl_info "$wl") || { echo "unknown workload $wl"; continue; }
  mkdir -p "$OUT/$wl"
  if [ -n "$LIST" ]; then cp "$LIST" "$OUT/$wl/files.txt"
  else pick_files "$src" "$OUT/$wl/files.txt"; fi
  dir=$src
  if [ "$STAGE" = 1 ]; then
    t0=$(date +%s)
    stage "$src" "$OUT/$wl/files.txt" "$STAGED/$wl" && dir=$STAGED/$wl
    echo "$wl: $(wc -l < "$OUT/$wl/files.txt") fields, dims $dims, staged in $(( $(date +%s) - t0 )) s"
  fi
  for c in $CODECS; do
    bin=$BUILD_DIR/lossy_sweep_$c
    [ -x "$bin" ] || { echo "  $c: missing $bin"; echo "$wl $c missing" >> "$OUT/status.txt"; continue; }
    t0=$(date +%s)
    name=$c${VARIANTS:+-${VARIANTS//,/-}}
    run_codec "$bin" "$dir" "$dims" "$wl" "$OUT/$wl/files.txt" \
      "$OUT/$wl/$name.csv" "$OUT/$wl/$name.log"
    rc=$?
    echo "$wl $name rc=$rc $(( $(date +%s) - t0 ))s" | tee -a "$OUT/status.txt"
  done
  rm -rf "$STAGED/$wl"
done
echo "results: $OUT"
