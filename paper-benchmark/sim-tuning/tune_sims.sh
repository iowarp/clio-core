#!/usr/bin/env bash
#===============================================================================
# tune_sims.sh -- which simulation parameters make one codec NOT the best?
# Small probe runs of LAMMPS and Nyx, one per parameter set; for each, an
# exhaustive search through Clio (all 45 settings on every chunk) and
# tune_report.py's per-chunk opportunity over the best single codec, per field
# and per quarter of the run.
#
#   tune_sims.sh [PROBE ...]     (default: every probe below, in order)
#
# Probes (name | simulation | parameters):
#   LAMMPS, box 40 (256,000 atoms), 2000 steps, a frame every 20 steps
#   (101 frames x force/position/velocity, float64, ~1.9 GB):
#     lammps-tune-melt6    in.melt, T = 6.0 (the current full workload's deck)
#     lammps-tune-ramp     in.melt_ramp, crystal -> hot liquid (T 0.05 -> 6.0)
#     lammps-tune-ramp15   in.melt_ramp, crystal -> liquid near melting (0.05 -> 1.5)
#     lammps-tune-quench   in.melt_ramp, hot liquid -> cold (6.0 -> 0.05)
#   Nyx Sedov blast, 128^3, 1000 steps (the same physical time as the full
#   256^3 / 2000-step run), a frame every 20 steps (51 frames x 6 fields,
#   float32, ~2.4 GB):
#     nyx-tune-e1          blast energy 1.0 (the deck's, as the full workload)
#     nyx-tune-e01         blast energy 0.1  (shell ~0.63x the radius)
#     nyx-tune-e001        blast energy 0.01 (shell ~0.40x the radius)
#
# Data: /mnt/nvme0/tune/<probe>/fields (linked from ~/np-data/<probe>), staged
# as workload ref-<probe>; exhaustive baselines in the baseline store.
# Progress: /mnt/nvme0/v2-work/runs/tune_sims.log, ending TUNE_SIMS_DONE.
#===============================================================================
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
MA=$HERE/../model-accuracy
LMP=$HERE/../../context-transfer-engine/compressor/example/neuropress_lammps_lib
LOG=/mnt/nvme0/v2-work/runs/tune_sims.log
TUNE=/mnt/nvme0/tune
PROBES=("$@")
if [ ${#PROBES[@]} -eq 0 ]; then
  PROBES=(lammps-tune-melt6 lammps-tune-ramp lammps-tune-ramp15 lammps-tune-quench
          nyx-tune-e1 nyx-tune-e01 nyx-tune-e001)
fi

# LAMMPS probe: deck and its -var settings.
gen_lammps() {
  local name=$1 deck=$2; shift 2
  local vars=()
  for v in "$@" SKIN=0.8 EVERY=5 GAP=20 NSTEPS=2000; do vars+=(--var "$v"); done
  # The driver links cuSZ, whose library is not on its rpath on this host.
  (cd "$LMP" && LD_LIBRARY_PATH="$HOME/np-env/cusz/lib:${LD_LIBRARY_PATH:-}" ./run.sh --box 40 --steps 2000 --gap 20 --chunk 0 --kokkos --order id \
      --store "$TUNE/$name/store" --port 9601 --deck "$HERE/../lammps/$deck" \
      "${vars[@]}" --no-compress --raw "$TUNE/$name/fields") > "$TUNE/$name/gen.log" 2>&1
  # The driver's known teardown segfault (rc 139) comes after every frame is
  # written, so the frame count is the check, not the exit code.
  rm -rf "$TUNE/$name/store"
}

# Nyx probe: blast energy.
gen_nyx() {
  local name=$1 energy=$2
  "$HERE/../nyx/gen_fields.sh" --ncell 128 --steps 1000 --plot-int 20 \
      --exp-energy "$energy" --out "$TUNE/$name/fields" > "$TUNE/$name/gen.log" 2>&1
}

echo "$(date +%F_%T) start ${#PROBES[@]} probes" >> "$LOG"
for p in "${PROBES[@]}"; do
  rm -rf "$TUNE/$p" "/mnt/nvme0/v2-work/ref-$p"
  mkdir -p "$TUNE/$p/fields"
  ln -sfn "$TUNE/$p" "$HOME/np-data/$p"
  t0=$(date +%s)
  case $p in
    lammps-tune-melt6)  gen_lammps "$p" in.melt TEMP=6.0 ;;
    lammps-tune-ramp)   gen_lammps "$p" in.melt_ramp T0=0.05 T1=6.0 ;;
    lammps-tune-ramp15) gen_lammps "$p" in.melt_ramp T0=0.05 T1=1.5 ;;
    lammps-tune-quench) gen_lammps "$p" in.melt_ramp T0=6.0 T1=0.05 ;;
    nyx-tune-e1)        gen_nyx "$p" 1.0 ;;
    nyx-tune-e01)       gen_nyx "$p" 0.1 ;;
    nyx-tune-e001)      gen_nyx "$p" 0.01 ;;
    *) echo "unknown probe $p" >> "$LOG"; continue ;;
  esac
  n=$(find "$TUNE/$p/fields" -type f ! -name '*.json' ! -name '*.log' | wc -l)
  echo "$(date +%T) $p generated: $n files, $(du -sh "$TUNE/$p/fields" | cut -f1), $(( $(date +%s) - t0 )) s" >> "$LOG"
  [ "$n" -gt 0 ] || { echo "$(date +%T) $p: no output, see $TUNE/$p/gen.log" >> "$LOG"; continue; }
  (cd "$MA" && python3 stage_full_workloads.py "ref-$p" >> "$LOG" 2>&1 &&
   NO_SELECTION_LOG=1 RUN_TAG=nolog MODES=exhaustive ./run_v2_all_workloads.sh "ref-$p" &&
   python3 baseline_store.py save "ref-$p" exhaustive >> "$LOG" 2>&1)
  echo "$(date +%T) $p exhaustive done" >> "$LOG"
  (cd "$HERE" && python3 tune_report.py >> "$LOG" 2>&1)
done
echo "$(date +%T) TUNE_SIMS_DONE" >> "$LOG"
