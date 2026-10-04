#!/usr/bin/env bash
#===============================================================================
# run_float32_sweep.sh -- every lossless GPU codec setting of the float32 grid
# (corpus_grid.py --profile float32: nvcomp at 3 internal chunk sizes with
# gdeflate/deflate at level 0 and 5 only, ndzip 1-D/2-D/3-D, GPULZ, SPspeed,
# SPratio), each with shuffle none / byte / bit, on K evenly spaced 4 MiB
# chunks of every workload. Runs locally on this host (no SLURM), one
# workload after another. One corpus_sweep row per (chunk, setting): bytes,
# median compress / decompress ms over REPS, bit-exact round trip. The
# shuffle and its inverse are outside the timed region.
#
#   ./run_float32_sweep.sh                     # every workload
#   WLS="nyx vpic" K=4 REPS=1 ./run_float32_sweep.sh
#   OUT=<finished run> ADD="store" ./run_float32_sweep.sh
#             # measure extra settings on that run's chunks and append them
#   ENGINE=cpu CHUNKS_FROM=<gpu run> ./run_float32_sweep.sh
#             # the CPU codecs (cpu_corpus_sweep) on the same chunks
#
# Settings include "store" (uncompressed, ratio 1) once, as the baseline.
#
# Environment:
#   WLS       workloads, space separated                  (every one below)
#             simulations: lammps nyx vpic
#             SDRBench:    cesm exaalt hacc hurricane miranda sdr-nyx
#                          qmcpack s3d scale
#   K         chunks to sample per workload, evenly spaced            (32)
#   REPS      timed round trips per (chunk, setting)                   (3)
#   SHUFFLES  comma list for corpus_grid.py --shuffle      (none,byte,bit)
#   SIMDATA   simulation dumps root                       ($HOME/np-data)
#   SDR       SDRBench root                   ($HOME/externals/datasets)
#   OUT       results dir        ($HOME/np-codec-sweep/float32-<stamp>)
#   NPENV     codec libraries root                         ($HOME/np-env)
#   BUILD_DIR where corpus_sweep is built     ($HOME/np-build/codec-sweep)
#   ADD       ";"-separated spec lines to add to an existing OUT     (unset)
#   ENGINE    gpu (corpus_sweep) or cpu (cpu_corpus_sweep)            (gpu)
#   THREADS   cpu: comma list of thread counts       (1,<physical cores>)
#   CHUNKS_FROM  reuse <dir>/<wl>/chunks.txt of an earlier run      (unset)
#   BATCH     gpu: time each chunk as BATCH copies in one call and divide,
#             so nvcomp gets enough work to fill the GPU   (64 = 256 MiB)
#===============================================================================
set -euo pipefail
ulimit -c 0
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE=${ENGINE:-gpu}
WLS=${WLS:-lammps nyx vpic cesm exaalt hacc hurricane miranda sdr-nyx qmcpack s3d scale}
K=${K:-32}
REPS=${REPS:-3}
SHUFFLES=${SHUFFLES:-none,byte,bit}
CHUNK=4194304
SIMDATA=${SIMDATA:-$HOME/np-data}
SDR=${SDR:-$HOME/externals/datasets}
NPENV=${NPENV:-$HOME/np-env}
BUILD_DIR=${BUILD_DIR:-$HOME/np-build/codec-sweep}
if [ "$ENGINE" = cpu ]; then
  OUT=${OUT:-$HOME/np-codec-sweep/float32-cpu-$(date +%m%d%H%M)}
  BIN=$BUILD_DIR/cpu_corpus_sweep
  SRC=$HERE/cpu_corpus_sweep.cc
  HDR=$HERE/cpu_codecs.h
  CORES=$(lscpu -p=core,socket | grep -v '^#' | sort -u | wc -l)
  THREADS=${THREADS:-1,$CORES}
  export OMP_PLACES=cores OMP_PROC_BIND=spread
else
  OUT=${OUT:-$HOME/np-codec-sweep/float32-$(date +%m%d%H%M)}
  BIN=$BUILD_DIR/corpus_sweep
  SRC=$HERE/corpus_sweep.cu
  HDR=$HERE/gpu_codecs.cuh
  BATCH=${BATCH:-64}
fi

# wl_info WL -- "source_dir file_glob ndzip_shapes dtype": shapes are the
# fastest dims, slowest first (2-D = one row, 3-D = one plane), comma
# separated, "-" for 1-D only; a 3-D plane must fit in one 4 MiB chunk (1 Mi
# floats). dtype is the element type: the byte shuffle's element size, and
# the CPU float codecs' type.
wl_info() {
  case "$1" in
    lammps)    echo "$SIMDATA/lammps-b70-2000/fields *.bin - double" ;;
    nyx)       echo "$SIMDATA/nyx-256-2000/fields *.f32 256,256x256 float" ;;
    vpic)      echo "$SIMDATA/vpic-126-2000/fields *.f32 128,128x128 float" ;;
    cesm)      echo "$SDR/cesm/SDRBENCH-CESM-ATM-cleared-1800x3600 *.dat 3600 float" ;;
    exaalt)    echo "$SDR/exaalt/SDRBENCH-EXAALT-2869440 *.f32 - float" ;;
    hacc)      echo "$SDR/hacc/EXASKY-HACC-data-medium-size *.f32 - float" ;;
    hurricane) echo "$SDR/hurricane/100x500x500 *.f32 500,500x500 float" ;;
    miranda)   echo "$SDR/miranda/SDRBENCH-Miranda-256x384x384 *.f32 384,384x384 float" ;;
    sdr-nyx)   echo "$SDR/nyx/SDRBENCH-EXASKY-NYX-512x512x512 *.f32 512,512x512 float" ;;
    qmcpack)   echo "$SDR/qmcpack/SDRBENCH-QMCPack *.f32 69,69x69 float" ;;
    s3d)       echo "$SDR/s3d/SDRBENCH-S3D *.f32 500,500x500 float" ;;
    scale)     echo "$SDR/scale/SDRBENCH-SCALE_98x1200x1200 *.f32 1200 float" ;;
    *) return 1 ;;
  esac
}

# build_cpu -- compile cpu_corpus_sweep (g++ + OpenMP) when stale
build_cpu() {
  local tmp=$BIN.tmp.$$
  g++ -O2 -std=c++17 -fopenmp -I"$NPENV/ndzip/include" "$SRC" -o "$tmp" \
    -L"$NPENV/ndzip/lib" -L/usr/local/lib -lndzip -lfpzip -lzfp -lblosc2 \
    -lzstd -llz4 -lz -lbz2 -llzma -lbrotlienc -lbrotlidec -lsnappy -llzo2 \
    -Wl,-rpath,"$NPENV/ndzip/lib:/usr/local/lib"
  mv -f "$tmp" "$BIN"
}

# build -- compile the sweep binary when missing or older than its sources;
# link to a private name and rename so a concurrent run never sees half.
build() {
  mkdir -p "$BUILD_DIR"
  [ -x "$BIN" ] && [ "$BIN" -nt "$SRC" ] && [ "$BIN" -nt "$HDR" ] && return 0
  if [ "$ENGINE" = cpu ]; then build_cpu; return; fi
  local tmp=$BIN.tmp.$$
  nvcc -O2 -std=c++17 -arch=sm_80 --expt-relaxed-constexpr \
    -I"$NPENV/np/include" -I"$NPENV/ndzip/include" -I"$NPENV/gpulz/include" \
    -I"$NPENV/fpcompress/include" "$SRC" -o "$tmp" \
    -L"$NPENV/np/lib" -L"$NPENV/ndzip/lib" -L"$NPENV/gpulz/lib" \
    -L"$NPENV/fpcompress/lib" -lnvcomp -lndzip-cuda -lndzip -lgpulz -lfpcompress \
    -Xlinker -rpath="$NPENV/np/lib:$NPENV/ndzip/lib:$NPENV/gpulz/lib:$NPENV/fpcompress/lib"
  mv -f "$tmp" "$BIN"
}

# sample_chunks SRC GLOB K -- K "path k" lines evenly spaced over every 4 MiB
# chunk of the files under SRC matching GLOB (sorted by path)
sample_chunks() {
  python3 - "$1" "$2" "$3" "$CHUNK" <<'EOF'
import fnmatch, os, sys
src, glob, k, chunk = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
files = sorted(os.path.relpath(os.path.join(d, f), src)
               for d, _, fs in os.walk(src) for f in fs if fnmatch.fnmatch(f, glob))
pairs = []
for f in files:
    n = -(-os.path.getsize(os.path.join(src, f)) // chunk)
    pairs += [(f, i) for i in range(n)]
k = min(k, len(pairs))
for j in range(k):
    f, i = pairs[j * (len(pairs) - 1) // max(k - 1, 1)]
    print(f, i)
EOF
}

# run_wl WL -- sample WL's chunks, write its grid, and run corpus_sweep
run_wl() {
  local wl=$1 src glob shapes dtype
  read -r src glob shapes dtype < <(wl_info "$wl") || { echo "unknown workload $wl" >&2; return 2; }
  [ -d "$src" ] || { echo "$wl: missing $src" >&2; return 2; }
  local dir=$OUT/$wl
  mkdir -p "$dir"
  if [ -n "${CHUNKS_FROM:-}" ]; then
    cp "$CHUNKS_FROM/$wl/chunks.txt" "$dir/chunks.txt"
  else
    sample_chunks "$src" "$glob" "$K" > "$dir/chunks.txt"
  fi
  local args=(--profile float32 --shuffle "$SHUFFLES" --dtype "$dtype")
  if [ "$ENGINE" = cpu ]; then
    args=(--profile cpu --shuffle "$SHUFFLES" --dtype "$dtype" --threads "$THREADS")
  fi
  if [ "$shapes" != - ]; then
    for s in ${shapes//,/ }; do args+=(--ndzip-shape "$s"); done
  fi
  python3 "$HERE/corpus_grid.py" settings "${args[@]}" > "$dir/settings.txt"
  echo "== $wl: $(wc -l < "$dir/chunks.txt") chunks x $(wc -l < "$dir/settings.txt") settings, reps $REPS"
  local rc=0
  "$BIN" --dir "$src" --list "$dir/chunks.txt" --configs "$dir/settings.txt" \
    --out "$dir/results.csv" --reps "$REPS" --chunk "$CHUNK" ${BATCH:+--batch "$BATCH"} \
    2> "$dir/progress.log" || rc=$?
  tail -1 "$dir/progress.log"
  echo "$wl exit $rc"
}

# add_wl WL -- run the ADD spec lines missing from WL's settings.txt on its
# chunks.txt and append them to its settings.txt and results.csv
add_wl() {
  local wl=$1 src glob shapes dtype dir=$OUT/$1
  read -r src glob shapes dtype < <(wl_info "$wl") || { echo "unknown workload $wl" >&2; return 2; }
  [ -f "$dir/results.csv" ] || { echo "$wl: no results in $dir, skipped"; return 0; }
  local new=$dir/settings.add.txt
  tr ';' '\n' <<< "$ADD" | sed 's/^ *//; s/ *$//; /^$/d' \
    | grep -vxF -f "$dir/settings.txt" > "$new" || true
  if [ ! -s "$new" ]; then echo "$wl: nothing to add"; rm -f "$new"; return 0; fi
  echo "== $wl: add $(wc -l < "$new") settings x $(wc -l < "$dir/chunks.txt") chunks, reps $REPS"
  local rc=0
  "$BIN" --dir "$src" --list "$dir/chunks.txt" --configs "$new" \
    --out "$dir/results.add.csv" --reps "$REPS" --chunk "$CHUNK" ${BATCH:+--batch "$BATCH"} \
    2> "$dir/progress.add.log" || rc=$?
  if [ $rc -eq 0 ]; then
    tail -n +2 "$dir/results.add.csv" >> "$dir/results.csv"
    cat "$new" >> "$dir/settings.txt"
    rm -f "$new" "$dir/results.add.csv"
  fi
  echo "$wl exit $rc"
  return $rc
}

mkdir -p "$OUT"
build
echo "out: $OUT"
fail=0
for wl in $WLS; do
  if [ -n "${ADD:-}" ]; then add_wl "$wl" || fail=1; else run_wl "$wl" || fail=1; fi
done
exit $fail
