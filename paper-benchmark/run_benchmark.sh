#!/usr/bin/env bash
# Clio-NeuroPress benchmark: four simulation workloads, exploration mode, the
# two cost models, lossless and lossy, across a bandwidth ladder.
#
#   ./run_benchmark.sh [--smoke] [--profile quick|mid|full] [--workloads "a b c"]
#                      [--cost-models "balance ratio speed"] [--explore-k N]
#                      [--eb X] [--bw-policy reduced|full|one] [--repeats N]
#                      [--results DIR] [--keep-native] [--check-bound] [--dry-run]
#
# Every run is one cell of
#
#     workload  x  {lossless, lossy}  x  cost model  x  bandwidth
#
# named for the cell it occupies (nyx_lossy_balance_nvme_10GBs), keeping its
# raw per-chunk measurements -- blobs.csv, selection.csv, explore.csv. Nothing
# is aggregated away; collect.py turns the set into summary.csv / summary.md,
# and audit_run.py reconciles one run's CSVs against what reached the tier.
#
# --smoke is the "does every path still work" pass: quick profile, one repeat,
# one bandwidth, so each workload contributes the four cells that cover both
# modes and both cost models. 16 runs. Use it before starting a campaign.
#
# WHY THE KNOBS ARE SET THE WAY THEY ARE -- the bandwidth ladder, why repeats
# are mandatory, why the exploration threshold is not upstream's 0.5, what an
# error bound means per workload, and how device residency is enforced -- is
# in BENCHMARK.md. Section pointers appear beside the code they explain.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# BENCHMARK.md "Bandwidth moves the balance model" and "Bandwidth is close to
# inert for the ratio model". Values are BYTES PER MILLISECOND (1 GB/s = 1e6),
# the unit of RankingWeights::bandwidth_bytes_per_ms. dram is MEASURED on this
# machine (STREAM triad), not a spec sheet -- re-measure on other hardware.
BW_NAMES=(nvme_1GBs nvme_2GBs nvme_5GBs nvme_10GBs dram)
BW_VALUES=(1e6      2e6       5e6       1e7        2.26e8)
BW_REDUCED=(nvme_5GBs dram)     # control set for the ratio model

# The three corners of the cost model. balance weights compress time,
# decompress time and I/O equally; ratio zeroes the two latency weights so cost
# collapses to bytes/(ratio*bw); speed zeroes the I/O weight so only latency
# counts. Override with --cost-models to run a subset or add speed.
COST_MODELS="balance ratio"

# Alternatives MEASURED per chunk. Passed through to each workload's
# --explore-k; 3 is NeuroPress's own ranked window, 31 the exhaustive action
# space minus the primary. Wider is not automatically better: on the three
# workloads whose data barely compresses, a wider sweep measures more
# candidates under more GPU contention and adopts worse ones. BENCHMARK.md.
EXPLORE_K=3

LOSSY_EB=1e-3                   # upstream NeuroPress's own benchmark value

# 3, and not out of caution: exploration adopts on MEASURED cost, so repeated
# runs on byte-identical input disagree -- 23% spread at 5 GB/s, and it widens
# with bandwidth. BENCHMARK.md "Exploration-mode selection is nondeterministic".
REPEATS=3
PROFILE=full
WORKLOADS="warpx vpic nyx lammps"
BW_POLICY=reduced
RESULTS="$HERE/results/benchmark"
KEEP_NATIVE=0 DRY=0 CHECK_BOUND=0

while [ $# -gt 0 ]; do
  case "$1" in
    --profile)     PROFILE=$2; shift 2;;
    --workloads)   WORKLOADS=$2; shift 2;;
    --cost-models) COST_MODELS=$2; shift 2;;
    --explore-k)   EXPLORE_K=$2; shift 2;;
    --eb)          LOSSY_EB=$2; shift 2;;
    --bw-policy)   BW_POLICY=$2; shift 2;;
    --repeats)     REPEATS=$2; shift 2;;
    --results)     RESULTS=$2; shift 2;;
    --smoke)       PROFILE=quick; REPEATS=1; BW_POLICY=one; shift;;
    --keep-native) KEEP_NATIVE=1; shift;;
    --check-bound) CHECK_BOUND=1; shift;;
    --dry-run)     DRY=1; shift;;
    -h|--help)     sed -n '2,23p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

case "$BW_POLICY" in reduced|full|one) ;;
  *) echo "unknown --bw-policy: $BW_POLICY (reduced|full|one)" >&2; exit 2;; esac
for m in $COST_MODELS; do
  case "$m" in balance|ratio|speed) ;;
    *) echo "unknown cost model: $m (balance|ratio|speed)" >&2; exit 2;; esac
done

# ---------------------------------------------------------------------------
# Profiles -- the three targets in the brief are not simultaneously reachable
# on this machine; each profile holds a different one. BENCHMARK.md section 2.
#   quick ~1 GB, 1-2 min/run     mid ~8-10 GB, 5-10 min/run (runtime target)
#   full  ~30 GB, 17-25 min/run (volume target).  All but quick run >=1000 steps.
# ---------------------------------------------------------------------------
case "$PROFILE" in
  quick)
    LMP_BOX=40    LMP_STEPS=200   LMP_GAP=50   LMP_CHUNK=4194304
    NYX_NCELL=128 NYX_STEPS=200   NYX_PLOTINT=10  NYX_CHUNK=8388608
    VPIC_NCELL=126 VPIC_STEPS=200 VPIC_DUMPINT=25 VPIC_CHUNK=8388608
    WARPX_NCELL="64 64 512" WARPX_STEPS=40 WARPX_INTERVAL=10 WARPX_CHUNK=1048576
    NEED_GB=6 ;;
  mid)
    LMP_BOX=64    LMP_STEPS=1000  LMP_GAP=25   LMP_CHUNK=33554432
    NYX_NCELL=128 NYX_STEPS=1000  NYX_PLOTINT=6   NYX_CHUNK=8388608
    VPIC_NCELL=126 VPIC_STEPS=1000 VPIC_DUMPINT=12 VPIC_CHUNK=8388608
    WARPX_NCELL="64 64 512" WARPX_STEPS=1000 WARPX_INTERVAL=10 WARPX_CHUNK=1048576
    NEED_GB=16 ;;
  full)
    LMP_BOX=100   LMP_STEPS=1000  LMP_GAP=10   LMP_CHUNK=33554432
    NYX_NCELL=256 NYX_STEPS=1000  NYX_PLOTINT=12  NYX_CHUNK=67108864
    VPIC_NCELL=254 VPIC_STEPS=1000 VPIC_DUMPINT=33 VPIC_CHUNK=67108864
    WARPX_NCELL="128 128 512" WARPX_STEPS=1000 WARPX_INTERVAL=10 WARPX_CHUNK=1048576
    NEED_GB=40 ;;
  *) echo "unknown profile: $PROFILE (quick|mid|full)" >&2; exit 2;;
esac

# ---------------------------------------------------------------------------
# Workloads -- ONE place per workload. Sets W_SIZE, W_CHUNK, W_VERIFY.
#
# W_VERIFY names how a workload can check itself, which is a property of how
# its data reaches Clio rather than a preference:
#   insitu    the adapter digests each chunk as it stages it and reads every
#             blob back at the end, so it can also bound-check a lossy run
#             against the bytes the simulation submitted (no source file
#             exists in situ). --verify lossless, --check-bound lossy.
#   readback  a stock application, so verification is a separate read-back
#             process rather than inline. --verify lossless only.
#   inline    the driver always verifies and takes no flag.
#
# The residency flags live here too: LAMMPS gathers on the device rather than
# through lammps_gather_atoms, and a stock WarpX has no device pointer to give
# so each chunk is staged H2D. BENCHMARK.md section 3b.
# ---------------------------------------------------------------------------
workload_spec() {
  case "$1" in
    nyx)    W_SIZE=(--ncell "$NYX_NCELL" --steps "$NYX_STEPS" --int "$NYX_PLOTINT")
            W_CHUNK=$NYX_CHUNK   W_VERIFY=insitu ;;
    vpic)   W_SIZE=(--ncell "$VPIC_NCELL" --steps "$VPIC_STEPS" --int "$VPIC_DUMPINT")
            W_CHUNK=$VPIC_CHUNK  W_VERIFY=insitu ;;
    lammps) W_SIZE=(--box "$LMP_BOX" --steps "$LMP_STEPS" --gap "$LMP_GAP"
                    --require-device)
            W_CHUNK=$LMP_CHUNK   W_VERIFY=inline ;;
    warpx)  W_SIZE=(--ncell "$WARPX_NCELL" --steps "$WARPX_STEPS"
                    --interval "$WARPX_INTERVAL" --stage-h2d)
            W_CHUNK=$WARPX_CHUNK W_VERIFY=readback ;;
    *)      W_SIZE=() W_CHUNK=4194304 W_VERIFY=inline ;;
  esac
}

# Bandwidths for one cost model. `one` is the smoke setting.
# Only `balance` has a bandwidth-sensitive cost. Under `ratio` bandwidth is a
# positive scalar on the sole term, so it cannot reorder candidates; under
# `speed` there is no bandwidth term at all. Both get the two-point control set
# so the invariance is visible in the results rather than asserted.
bw_list_for() {
  if   [ "$BW_POLICY" = one ];                            then echo "nvme_5GBs"
  elif [ "$1" = balance ] || [ "$BW_POLICY" = full ];     then printf '%s\n' "${BW_NAMES[@]}"
  else                                                         printf '%s\n' "${BW_REDUCED[@]}"; fi
}
bw_value_for() {
  local i
  for i in "${!BW_NAMES[@]}"; do
    [ "${BW_NAMES[$i]}" = "$1" ] && { echo "${BW_VALUES[$i]}"; return; }
  done
  echo "5e6"
}

# Verification flags for one cell, from the workload's W_VERIFY policy. A lossy
# digest check is never requested: decoded bytes are not input bytes by design.
verify_args_for() {
  local mode=$1
  W_VFY=()
  case "$W_VERIFY" in
    insitu)
      [ "$mode" = lossless ] && W_VFY=(--verify)
      [ "$CHECK_BOUND" = 1 ] && [ "$mode" = lossy ] && W_VFY+=(--check-bound) ;;
    readback)
      [ "$mode" = lossless ] && W_VFY=(--verify) ;;
  esac
}

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
mkdir -p "$RESULTS"
free_gb=$(df -BG --output=avail "$RESULTS" | tail -1 | tr -dc '0-9')
echo "== Clio-NeuroPress benchmark"
echo "   profile=$PROFILE  models='$COST_MODELS'  K=$EXPLORE_K  eb=$LOSSY_EB"
echo "   bw-policy=$BW_POLICY  repeats=$REPEATS  results=$RESULTS"
echo "   disk: ${free_gb} GB free, profile needs ~${NEED_GB} GB"
if [ "$free_gb" -lt "$NEED_GB" ]; then
  echo
  echo "REFUSING TO START: ~${NEED_GB} GB needed, ${free_gb} GB free." >&2
  echo "Free space, use a smaller --profile, or point --results at another" >&2
  echo "filesystem. Most of it is WarpX's native openPMD output, which it" >&2
  echo "writes per run alongside the compressed tier (see --keep-native)." >&2
  exit 1
fi
echo

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
# Nyx and VPIC prefer their in-situ wrapper: the simulation hands the
# compressor DEVICE memory and the wrapper sets CLIO_NEUROPRESS_REQUIRE_DEVICE,
# so a host-resident chunk is refused rather than quietly computed on the CPU.
config_for() {
  local w=$1 cfg="$HERE/$w/run_config.sh"
  [ -x "$HERE/$w/run_config_insitu.sh" ] && cfg="$HERE/$w/run_config_insitu.sh"
  echo "$cfg"
}

run_cell() {
  local w=$1 mode=$2 model=$3 bwname=$4 rep=$5
  local bw; bw=$(bw_value_for "$bwname")
  # `__r<n>` is for humans reading the directory listing; collect.py groups
  # repeats by the cell they occupy, not by parsing this name.
  local tag="${w}_${mode}_${model}_${bwname}"
  [ "$REPEATS" -gt 1 ] && tag="${tag}__r${rep}"

  local eb=()
  [ "$mode" = lossy ] && eb=(--eb "$LOSSY_EB")
  verify_args_for "$mode"

  local cfg; cfg=$(config_for "$w")
  if [ "$DRY" = 1 ]; then
    echo "[dry] $(basename "$cfg") explore-$model --bw $bw ${eb[*]} ${W_VFY[*]} --explore-k $EXPLORE_K --chunk $W_CHUNK ${W_SIZE[*]} --tag $tag"
    return 0
  fi

  "$cfg" "explore-$model" \
      --bw "$bw" "${eb[@]}" "${W_VFY[@]}" \
      --explore-k "$EXPLORE_K" \
      --chunk "$W_CHUNK" "${W_SIZE[@]}" \
      --results "$RESULTS" --tag "$tag"
  local rc=$?

  # WarpX writes a full native openPMD set per run on top of the compressed
  # tier -- tens of GB of duplicate output at the larger profiles. Deleted
  # AFTER the run, since verification reads it.
  [ "$w" = warpx ] && [ "$KEEP_NATIVE" = 0 ] && rm -rf "$RESULTS/$tag/run/diags"
  # The tier and bdev files hold every compressed byte; one per cell would
  # multiply the campaign's footprint. The measurements are in the CSVs.
  rm -f "$RESULTS/$tag/chi_bdev.dat" "$RESULTS/$tag"/cte_tier.dat*
  return $rc
}

FAILED=0 RAN=0
for w in $WORKLOADS; do
  [ -d "$HERE/$w" ] || { echo "!! no such workload: $w" >&2; FAILED=$((FAILED+1)); continue; }
  workload_spec "$w"
  for mode in lossless lossy; do
    for model in $COST_MODELS; do
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
[ "$DRY" = 0 ] && [ -x "$HERE/collect.py" ] && "$HERE/collect.py" "$RESULTS"
exit $(( FAILED > 0 ))
