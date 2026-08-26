#!/usr/bin/env bash
# ============================================================================
# Clio-NeuroPress benchmark: four simulation workloads, exploration mode, the
# two cost models, lossless and lossy, across a bandwidth ladder.
#
#   ./run_benchmark.sh [--smoke] [--profile quick|mid|full] [--workloads "a b c"]
#                      [--bw-policy reduced|full|one] [--repeats N] [--results DIR]
#                      [--skip-gen] [--keep-native] [--check-bound] [--dry-run]
#
# ALL FOUR WORKLOADS RUN THEIR PREPROCESSING ON THE GPU. Nyx and VPIC hand the
# compressor device memory in situ, LAMMPS gathers on the device, and WarpX --
# whose stock binary gives HDF5 host memory -- stages each chunk H2D and runs
# the same CUDA kernels. There is no CPU quantize or byte shuffle left to fall
# back to; a chunk that cannot reach the GPU is refused, not degraded.
#
# --smoke is the "does every path still work" run: the quick profile, one
# repeat, one bandwidth, so each workload contributes exactly the four cells
# that exercise both modes and both cost models. 4 workloads x 4 cells = 16
# runs. Use it after touching any of this, before starting a real campaign.
#
# Every run is one cell of this matrix:
#
#     workload  x  {lossless, lossy}  x  {balance, ratio}  x  bandwidth
#
# and lands in its own directory named for the cell, e.g.
#
#     warpx_lossless_balance_nvme_1GBs
#     warpx_lossless_ratio_dram
#     nyx_lossy_balance_nvme_10GBs
#
# Raw per-chunk measurements are kept per run and never aggregated away:
#   blobs.csv      one row per chunk: bytes, codec, ratio, stored,
#                  compress_ms, decompress_ms
#   selection.csv  NeuroPress's own per-chunk record: the three input
#                  statistics, predicted vs actual ratio/time, PSNR
#   explore.csv    one row per MEASURED candidate, with the adopted flag
# ../collect.py turns the set into summary.csv / summary.md.
#
# ---------------------------------------------------------------------------
# EXPLORATION MODE is the only selection policy used here, via the
# explore-balance / explore-ratio configs each workload's run_config.sh
# defines. K=3 (NeuroPress's own ranked window) and threshold 0 -- see the note
# in any run_config.sh for why the threshold is not upstream's 0.5 for a
# comparison matrix.
#
# THRESHOLD 0 IS NOT FULL COVERAGE. The gate is strict (`error_pct >
# threshold`, upstream's own), so a chunk the model priced exactly right never
# explores. Under the RATIO cost model that is most of them -- cost collapses
# to bytes/(min(ratio,100)*bw), and two ratios past the 100x cap price
# identically -- so those runs are part exploration and part plain inference.
# `explored` in summary.csv reports the split per run; do not read a
# `explore-ratio` row as if every chunk had been measured.
#
# LOSSLESS vs LOSSY is CLIO_NEUROPRESS_ERROR_BOUND, passed as --eb. It is an
# ABSOLUTE bound: |original - decoded| <= eb. 0 masks NeuroPress's 16 quantize
# actions, so selection covers the 16 lossless configurations only; positive
# makes all 32 reachable. LOSSY_EB below is 1e-3, which is upstream
# NeuroPress's own benchmark value (VPIC_ERROR_BOUND:-0.001) and the default
# Clio's neuropress_explore_sweep.cc already used.
#
# Bit-exact verification is turned off automatically for lossy runs -- the
# decoded bytes are not the input bytes by design, so a digest check would
# report FAILED on correct behaviour. Quality for a lossy run is the PSNR
# column in selection.csv.
#
# BANDWIDTH is CLIO_NEUROPRESS_COST_BW, passed as --bw, in BYTES PER
# MILLISECOND -- the unit of RankingWeights::bandwidth_bytes_per_ms in
# context-transport-primitives/include/clio_ctp/compress/model/predictor.h,
# NOT bytes/s and NOT GB/s. It enters the cost model as
#
#     cost = w_ct*compress_ms + w_dt*decompress_ms + bytes/(ratio*bw)
#
# so 1 GB/s = 1e6 B/ms. NeuroPress's shipped default is 5e6 = 5 GB/s.
# ============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --- Bandwidth ladder ------------------------------------------------------
# NVMe tier: 1-10 GB/s spans a SATA-era device through a current PCIe 4.0 x4
# drive, which is the range a real storage tier sits in.
#
# DRAM: 2.26e8 B/ms = 226 GB/s, MEASURED on this machine rather than taken
# from a spec sheet -- STREAM triad, 128 threads, dual EPYC 7763 / DDR4-3200
# (16 channels, 409.6 GB/s theoretical, so 55% of peak, which is normal for
# triad on dual-socket EPYC). Re-measure before quoting it on other hardware.
BW_NAMES=(nvme_1GBs nvme_2GBs nvme_5GBs nvme_10GBs dram)
BW_VALUES=(1e6      2e6       5e6       1e7        2.26e8)
# NeuroPress's shipped default, and the two-point control set for the ratio
# cost model (see --bw-policy below).
BW_REDUCED=(nvme_5GBs dram)

LOSSY_EB=1e-3

# REPEATS DEFAULTS TO 3, and that is not caution -- it is required to read the
# results at all. Exploration measures each candidate's real compress and
# decompress time on the device and adopts on the resulting cost, so when two
# candidates are close the winner flips with the measurement. Repeated runs on
# BYTE-IDENTICAL input therefore disagree: measured on the Nyx replay at a
# FIXED 5 GB/s, three runs gave 1078.96x, 879.38x and 1082.71x -- a 23% spread
# with nothing changed between them.
#
# The spread also grows with bandwidth, for a reason worth knowing: at low
# bandwidth the bytes/(ratio*bw) term dominates the cost and selection is
# stable (1 GB/s repeated to 38.27/37.89/38.10, ~1%), while at high bandwidth
# that term shrinks and the noisy MEASURED times dominate (10 GB/s gave
# 43.95/40.49/62.42). So a single run per cell cannot distinguish a bandwidth
# effect from run-to-run noise, and the high-bandwidth cells are exactly where
# it is least able to.
REPEATS=3
PROFILE=full
WORKLOADS="warpx vpic nyx lammps"
BW_POLICY=reduced
RESULTS="$HERE/results/benchmark"
SKIP_GEN=0 KEEP_NATIVE=0 DRY=0 CHECK_BOUND=0
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE=$2; shift 2;;
    --workloads) WORKLOADS=$2; shift 2;;
    --bw-policy) BW_POLICY=$2; shift 2;;
    --repeats) REPEATS=$2; shift 2;;
    --smoke) PROFILE=quick; REPEATS=1; BW_POLICY=one; shift;;
    --results) RESULTS=$2; shift 2;;
    --skip-gen) SKIP_GEN=1; shift;;
    --keep-native) KEEP_NATIVE=1; shift;;
    --check-bound) CHECK_BOUND=1; shift;;
    --dry-run) DRY=1; shift;;
    -h|--help) sed -n '2,60p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# --- Per-workload sizing ---------------------------------------------------
# THE THREE TARGETS IN THE BRIEF -- 1000+ timesteps, ~30 GB, 5-10 minutes --
# are not simultaneously reachable on this machine, and that is a measured
# statement rather than a guess: exploration mode sustains ~30 MiB/s here
# (1008 MiB of Nyx in 32.8 s at a 16 MiB chunk; neither more runtime threads
# nor a larger chunk moves it, and only ~3 s of that is codec time -- the rest
# is per-chunk task dispatch). 30 GB therefore costs ~17-25 min per run, not
# 5-10. Pick which target to hold:
#
#   quick  ~1 GB      ~1-2 min/run   pipeline validation, and what the tracked
#                                    nyx/ and vpic/ field dumps already are
#   mid    ~8-10 GB   ~5-10 min/run  holds the RUNTIME target; 1000+ steps
#   full   ~30 GB     ~17-25 min/run holds the VOLUME target; 1000+ steps
#
# Every profile runs >= 1000 timesteps except `quick`, which is not meant to.
case "$PROFILE" in
  quick)
    LMP_BOX=40   LMP_STEPS=200  LMP_GAP=50  LMP_CHUNK=4194304
    NYX_NCELL=128 NYX_STEPS=200  NYX_PLOTINT=10 NYX_CHUNK=8388608
    VPIC_NCELL=126 VPIC_STEPS=200 VPIC_DUMPINT=25 VPIC_CHUNK=8388608
    WARPX_NCELL="64 64 512" WARPX_STEPS=40 WARPX_INTERVAL=10 WARPX_CHUNK=1048576
    ;;
  mid)
    LMP_BOX=64   LMP_STEPS=1000 LMP_GAP=25  LMP_CHUNK=33554432
    NYX_NCELL=128 NYX_STEPS=1000 NYX_PLOTINT=6  NYX_CHUNK=8388608
    VPIC_NCELL=126 VPIC_STEPS=1000 VPIC_DUMPINT=12 VPIC_CHUNK=8388608
    WARPX_NCELL="64 64 512" WARPX_STEPS=1000 WARPX_INTERVAL=10 WARPX_CHUNK=1048576
    ;;
  full)
    # LAMMPS: 4*100^3 = 4,000,000 atoms; 3 fields * 3 comps * 8 B = 72 B per
    # atom per frame; 101 frames -> 29.1 GB, none of it ever on disk.
    LMP_BOX=100  LMP_STEPS=1000 LMP_GAP=10  LMP_CHUNK=33554432
    # Nyx: 256^3 float32 = 64 MiB per component, 6 components = 384 MiB per
    # frame; 84 frames -> 31.5 GB, ON DISK (this is a replay workload).
    NYX_NCELL=256 NYX_STEPS=1000 NYX_PLOTINT=12 NYX_CHUNK=67108864
    # VPIC: (254+2)^3 float32 = 64 MiB per variable, 16 variables = 1 GiB per
    # frame; 30 frames -> 30 GB, ON DISK.
    VPIC_NCELL=254 VPIC_STEPS=1000 VPIC_DUMPINT=33 VPIC_CHUNK=67108864
    # WarpX: 128*128*512 float32 = 32 MiB per field; ~10 fields -> 320 MiB per
    # dump, 101 dumps -> ~32 GB. NOTE this workload also writes that much
    # NATIVE openPMD per run (deleted after each run unless --keep-native).
    WARPX_NCELL="128 128 512" WARPX_STEPS=1000 WARPX_INTERVAL=10 WARPX_CHUNK=1048576
    ;;
  *) echo "unknown profile: $PROFILE (quick|mid|full)" >&2; exit 2;;
esac

case "$BW_POLICY" in
  reduced|full|one) ;;
  *) echo "unknown --bw-policy: $BW_POLICY (reduced|full|one)" >&2; exit 2;;
esac

# Bandwidths for one cost model.
#
# WHY `reduced` IS THE DEFAULT FOR THE RATIO MODEL, and why that is not a
# corner cut: under the ratio-only weights the cost is bytes/(ratio*bw) and
# nothing else, so bandwidth is a positive scalar on the SOLE term. It cannot
# reorder candidates, and a bandwidth sweep there is arithmetically a no-op.
# Measured on the bit-reproducible Nyx replay: selections at 1e6, 5e6 and 1e7
# are byte-identical (same md5 over blob,codec,stored), while the same ladder
# under `balance` moved the run ratio 37x -> 80x -> 120x. `dram` is kept as a
# control so the invariance is visible in the results rather than asserted.
# --bw-policy full runs the whole ladder on both models anyway.
bw_list_for() {
  local model=$1
  # `one` is the smoke setting: a single bandwidth for both cost models, so
  # the matrix collapses to the four cells that still cover every code path.
  if [ "$BW_POLICY" = one ]; then
    echo "nvme_5GBs"
  elif [ "$model" = balance ] || [ "$BW_POLICY" = full ]; then
    printf '%s\n' "${BW_NAMES[@]}"
  else
    printf '%s\n' "${BW_REDUCED[@]}"
  fi
}
bw_value_for() {
  local name=$1 i
  for i in "${!BW_NAMES[@]}"; do
    [ "${BW_NAMES[$i]}" = "$name" ] && { echo "${BW_VALUES[$i]}"; return; }
  done
  echo "5e6"
}

mkdir -p "$RESULTS"

# --- Preflight -------------------------------------------------------------
# Refuse to start a campaign that cannot fit rather than filling the disk
# halfway through. The replay workloads stage their whole payload on disk and
# WarpX writes its native openPMD alongside the compressed tier, so the
# requirement is roughly one workload's payload plus headroom.
need_gb=0
case "$PROFILE" in quick) need_gb=6;; mid) need_gb=16;; full) need_gb=40;; esac
free_gb=$(df -BG --output=avail "$RESULTS" | tail -1 | tr -dc '0-9')
echo "== Clio-NeuroPress benchmark"
echo "   profile=$PROFILE  bw-policy=$BW_POLICY  repeats=$REPEATS  results=$RESULTS"
echo "   disk: ${free_gb} GB free, profile needs ~${need_gb} GB"
if [ "$free_gb" -lt "$need_gb" ]; then
  echo
  echo "REFUSING TO START: ~${need_gb} GB needed, ${free_gb} GB free." >&2
  echo "Free space, use a smaller --profile, or point --results at another" >&2
  echo "filesystem. The replay workloads stage their entire payload on disk." >&2
  exit 1
fi
echo

# --- Field generation for the two replay workloads -------------------------
# Nyx and VPIC replay dumps, so the simulation runs ONCE and every cell of the
# matrix replays the identical bytes -- which is what makes the cost-model and
# bandwidth comparison exact rather than confounded by a re-run of a
# non-bit-reproducible simulation.
gen_fields() {
  local w=$1; shift
  local dir="$HERE/$w/fields"
  if [ "$SKIP_GEN" = 1 ] && [ -d "$dir" ]; then
    echo "-- $w: --skip-gen, reusing $(du -sh "$dir" 2>/dev/null | cut -f1) in $dir"
    return 0
  fi
  echo "-- $w: generating fields ($*)"
  [ "$DRY" = 1 ] && return 0
  "$HERE/$w/gen_fields.sh" "$@" || return 1
}

run_cell() {
  local w=$1 mode=$2 model=$3 bwname=$4 rep=$5; shift 5
  local bw; bw=$(bw_value_for "$bwname")
  # `__r<n>` is the separator the per-workload run_sweep.sh scripts already
  # use for a repeat. collect.py groups repeats by the CELL they occupy
  # (mode, cost model, bandwidth) rather than by parsing this name, so the
  # suffix is for humans reading the directory listing.
  local tag="${w}_${mode}_${model}_${bwname}"
  [ "$REPEATS" -gt 1 ] && tag="${tag}__r${rep}"
  local eb=()
  [ "$mode" = lossy ] && eb=(--eb "$LOSSY_EB")
  # WarpX is the one workload whose round-trip check is OPT-IN (--verify): its
  # application is a stock binary, so verification is a separate read-back
  # process rather than something the driver does inline, and run_config.sh
  # defaults it off. Ask for it on lossless runs so all four workloads are
  # checked the same way -- without this WarpX reported `verified: n/a` while
  # the other three reported `pass`. Lossy runs are left alone: the guard in
  # run_config.sh turns any digest check off there anyway.
  local vfy=()
  [ "$w" = warpx ] && [ "$mode" = lossless ] && vfy=(--verify)
  # Nyx and VPIC verify in situ, against the bytes the simulation submitted:
  # the adapter digests each chunk as it stages it and reads every blob back
  # through the decompressor at the end. Without this they reported
  # `verified: n/a` on every cell -- a whole campaign with nothing checking
  # that what came back is what went in.
  case "$w" in nyx|vpic) [ "$mode" = lossless ] && vfy=(--verify) ;; esac
  # --check-bound: on a LOSSY run, verify |original - decoded| <= eb instead of
  # a digest, which lossy data must fail. Nyx and VPIC do it IN SITU -- there
  # is no source file to re-read, so the adapter holds each frame's submitted
  # bytes for one frame and compares the read-back against those, which bounds
  # the extra memory at a frame rather than the run. LAMMPS generates its
  # arrays in memory and WarpX's originals are the native openPMD output, and
  # neither adapter retains them, so those cells stay unchecked rather than
  # silently reporting a pass.
  if [ "$CHECK_BOUND" = 1 ] && [ "$mode" = lossy ]; then
    case "$w" in nyx|vpic) vfy+=(--check-bound) ;; esac
  fi

  if [ "$DRY" = 1 ]; then
    echo "   [dry-run] $tag  (bw=$bw${eb[*]:+ ${eb[*]}})"
    return 0
  fi

  # Nyx and VPIC run IN SITU: the simulation hands the compressor DEVICE
  # memory, so quantization, byte shuffle and codec selection all take their
  # GPU paths, and the wrapper sets CLIO_NEUROPRESS_REQUIRE_DEVICE=1 so a
  # host-resident chunk is refused rather than quietly computed on the CPU.
  # The file-replay route (run_config.sh) cannot do that -- it reads .f32
  # files into host shm, and measured 0 of 254 chunks device-resident.
  local cfg="$HERE/$w/run_config.sh"
  [ -x "$HERE/$w/run_config_insitu.sh" ] && cfg="$HERE/$w/run_config_insitu.sh"
  "$cfg" "explore-$model" \
      --bw "$bw" "${eb[@]}" "${vfy[@]}" \
      --chunk "$CHUNK" "${SIZE_ARGS[@]}" \
      --results "$RESULTS" --tag "$tag"
  local rc=$?
  # WarpX writes a full native openPMD set per run on top of the compressed
  # tier; at the larger profiles that is tens of GB of duplicate output nobody
  # reads. Deleted AFTER the run (verification, where enabled, needs it).
  if [ "$w" = warpx ] && [ "$KEEP_NATIVE" = 0 ]; then
    rm -rf "$RESULTS/$tag/run/diags"
  fi
  # The tier and block-device files are sparse but still hold every compressed
  # byte the run produced. Keeping one per cell would multiply the campaign's
  # footprint by the number of cells; the measurements are in the CSVs.
  rm -f "$RESULTS/$tag/chi_bdev.dat" "$RESULTS/$tag"/cte_tier.dat*
  return $rc
}

FAILED=0 RAN=0
for w in $WORKLOADS; do
  [ -d "$HERE/$w" ] || { echo "!! no such workload: $w" >&2; FAILED=$((FAILED+1)); continue; }

  case "$w" in
    # --require-device switches the driver to --order device (chunks gathered
    # ON the GPU into CUDA-IPC backends) and sets CLIO_NEUROPRESS_REQUIRE_DEVICE
    # so a host-resident chunk is refused instead of quietly preprocessed on
    # the CPU. Without it the driver's default --order id gathers through
    # lammps_gather_atoms into host memory, and every chunk took the host
    # quantize/shuffle path -- measured 0 of 30 device-resident.
    lammps) SIZE_ARGS=(--box "$LMP_BOX" --steps "$LMP_STEPS" --gap "$LMP_GAP"
                       --require-device)
            CHUNK=$LMP_CHUNK ;;
    # No dump phase in situ: the simulation runs inside the benchmark process
    # and hands its device memory straight over, so there are no .f32 files to
    # generate or replay. The size knobs go to the simulation itself.
    nyx)    SIZE_ARGS=(--ncell "$NYX_NCELL" --steps "$NYX_STEPS"
                       --int "$NYX_PLOTINT")
            CHUNK=$NYX_CHUNK ;;
    vpic)   SIZE_ARGS=(--ncell "$VPIC_NCELL" --steps "$VPIC_STEPS"
                       --int "$VPIC_DUMPINT")
            CHUNK=$VPIC_CHUNK ;;
    # --stage-h2d, not --require-device alone: a stock WarpX hands HDF5 HOST
    # memory and there is no device pointer to be had, so the chunk is copied
    # up and the CUDA kernels run on it -- upstream's own route for a
    # host-resident caller (gpucompress_compress: "Transfers data to GPU,
    # compresses, and returns result to host"). All preprocessing is still on
    # the GPU; what differs is one H2D per chunk.
    #
    # So WarpX RATIOS compare with the other three; its TIMINGS do not, and
    # `residency` in meta.json says device-staged-h2d rather than device to
    # keep that visible. --stage-h2d implies --require-device, so a staging
    # failure refuses rather than reverting to a host path.
    warpx)  SIZE_ARGS=(--ncell "$WARPX_NCELL" --steps "$WARPX_STEPS"
                       --interval "$WARPX_INTERVAL" --stage-h2d)
            CHUNK=$WARPX_CHUNK ;;
    *)      SIZE_ARGS=(); CHUNK=4194304 ;;
  esac

  for mode in lossless lossy; do
    for model in balance ratio; do
      while read -r bwname; do
        [ -z "$bwname" ] && continue
        for rep in $(seq 1 "$REPEATS"); do
          RAN=$((RAN+1))
          run_cell "$w" "$mode" "$model" "$bwname" "$rep" \
            || { FAILED=$((FAILED+1))
                 echo "   ^ ${w}_${mode}_${model}_${bwname} r$rep FAILED"; }
        done
      done < <(bw_list_for "$model")
    done
  done
  echo
done

echo "== $RAN cell(s) attempted, $FAILED failed"
if [ "$DRY" = 0 ] && [ -x "$HERE/collect.py" ]; then
  "$HERE/collect.py" "$RESULTS"
fi
exit $(( FAILED > 0 ))
