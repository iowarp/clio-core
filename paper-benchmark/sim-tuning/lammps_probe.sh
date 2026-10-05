#!/usr/bin/env bash
#===============================================================================
# lammps_probe.sh -- one LAMMPS tuning probe: run a deck through the LAMMPS
# library driver (float32 output, atom-ID order), stage it, search it
# exhaustively through Clio and score it (probe_eval.py).
#
#   [BOX=64] [STEPS=400] [GAP=10] [FIELDS=position] [DECK=in.melt_slabs] \
#     lammps_probe.sh NAME [NOTE] [VAR=value ...]
#
# VAR=value pairs go to the deck (-var). Data: /mnt/nvme0/tune/lmpf32-tune-NAME
# (linked from ~/np-data), workload ref-lmpf32-tune-NAME (float32: the
# ref-lammps names mean the float64 dumps).
# Log: /mnt/nvme0/v2-work/runs/lammps_tuning.log
#===============================================================================
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
MA=$HERE/../model-accuracy
LMP=$HERE/../../context-transfer-engine/compressor/example/neuropress_lammps_lib
LOG=/mnt/nvme0/v2-work/runs/lammps_tuning.log
NAME=lmpf32-tune-$1; NOTE=${2:-}; shift; [ $# -gt 0 ] && shift
BOX=${BOX:-64} STEPS=${STEPS:-400} GAP=${GAP:-10} FIELDS=${FIELDS:-position}
DECK=${DECK:-in.melt_slabs}
D=/mnt/nvme0/tune/$NAME
rm -rf "$D" "/mnt/nvme0/v2-work/ref-$NAME" "/mnt/nvme0/v2-work/baselines/ref-$NAME"
mkdir -p "$D/fields"
ln -sfn "$D" "$HOME/np-data/$NAME"
vars=(--var BOX="$BOX" --var GAP="$GAP")
for v in "$@"; do vars+=(--var "$v"); done
t0=$(date +%s)
# The driver links cuSZ, whose library is not on its rpath on this host; no
# cuSZ codec is used. Its known teardown segfault (rc 139) comes after every
# frame is written, so the frame count is the check.
(cd "$LMP" && LD_LIBRARY_PATH="$HOME/np-env/cusz/lib:${LD_LIBRARY_PATH:-}" ./run.sh \
    --box "$BOX" --steps "$STEPS" --gap "$GAP" --chunk 0 --kokkos --order id \
    --store "$D/store" --port 9621 --deck "$HERE/../lammps/$DECK" "${vars[@]}" \
    --f32 --fields "$FIELDS" --no-compress --raw "$D/fields") > "$D/gen.log" 2>&1
rm -rf "$D/store"
n=$(find "$D/fields" -type f | wc -l)
echo "$(date +%T) $NAME generated $n files, $(du -sh "$D/fields" | cut -f1), $(( $(date +%s) - t0 )) s | $NOTE | $DECK box $BOX steps $STEPS gap $GAP $FIELDS $*" >> "$LOG"
[ "$n" -gt 0 ] || { echo "$NAME: no output, see $D/gen.log" | tee -a "$LOG"; exit 1; }
cd "$MA"
python3 stage_full_workloads.py "ref-$NAME" >> "$LOG" 2>&1
NO_SELECTION_LOG=1 RUN_TAG=nolog MODES=exhaustive ./run_v2_all_workloads.sh "ref-$NAME"
python3 baseline_store.py save "ref-$NAME" exhaustive >> "$LOG" 2>&1
OPENBLAS_NUM_THREADS=4 ~/np-venv/bin/python "$HERE/probe_eval.py" "ref-$NAME" --note "$NOTE" 2>/dev/null | tee -a "$LOG"
