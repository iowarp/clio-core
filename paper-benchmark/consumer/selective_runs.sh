#!/bin/bash
# Measured in-memory runs for the selective-lossy study. Run after
# selective_traces.sh and `selective_analyze.py --write-replays`.
#
#   1. per workload and bound: NeuroPress in learning mode against the
#      trace's fastest uniform-lossy configuration (balanced cost). eb 1e-3 is
#      inmem.sh's own campaign (results/inmem/<wl>), so it is not repeated.
#   2. per workload and bound: the ideal per-chunk policies replayed with no
#      selection or learning (CLIO_NEUROPRESS_REPLAY_CHOICES) -- `oracle`,
#      each chunk's best action, and `protect`, uniform lossy with the chunks
#      quantizing damages kept lossless.
#
#   ./selective_runs.sh [-w "nyx vpic"] [--ebs "1e-2 1e-4"] [--replay-ebs "1e-3 1e-2 1e-4"]
#                       [--policies "oracle protect"]
#
# Environment: REPS (3), and everything inmem.sh reads.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BENCH=$(cd "$HERE/.." && pwd)
TRACES=$BENCH/results/selective INMEM=$BENCH/results/inmem
WLS="nyx vpic" EBS="1e-2 1e-4" REPLAY_EBS="1e-3 1e-2 1e-4" POLICIES="oracle protect"

while [ $# -gt 0 ]; do
  case "$1" in
    -w) WLS=$2; shift 2 ;;
    --ebs) EBS=$2; shift 2 ;;
    --replay-ebs) REPLAY_EBS=$2; shift 2 ;;
    --policies) POLICIES=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
exec > >(tee -a "$TRACES/runs.log") 2>&1

# uniform_cfg <wl> <eb> -- the trace's fastest fixed quantized action as an
# inmem.sh config name, e.g. "bitcomp|q1|s0" -> static-bitcomp-q.
uniform_cfg() {
  python3 - "$TRACES/selective.json" "$1-eb$2" <<'PY'
import json, sys
lib, _, s = json.load(open(sys.argv[1]))["traces"][sys.argv[2]]["fixed_lossy"].split("|")
print(f"static-{lib}-q" + ("-s4" if s == "s1" else ""))
PY
}

for eb in $EBS; do
  for wl in $WLS; do
    cfg=$(uniform_cfg "$wl" "$eb") || { echo "no trace for $wl eb $eb" >&2; continue; }
    echo "==== $wl eb $eb: NeuroPress vs uniform lossy $cfg ($(date -u +%H:%M) UTC)"
    EB=$eb EXTRA_ARMS="Uniform_lossy|$cfg|$eb" "$HERE/inmem.sh" -w "$wl" \
      --only "NeuroPress,Uniform_lossy" --out "$INMEM/$wl-eb$eb"
  done
done

for eb in $REPLAY_EBS; do
  for wl in $WLS; do
    for pol in $POLICIES; do
      f=$TRACES/$wl-eb$eb/replay_$pol.csv
      [ -f "$f" ] || continue
      echo "==== $wl eb $eb: replay $pol ($(date -u +%H:%M) UTC)"
      EB=$eb XENV="CLIO_NEUROPRESS_REPLAY_CHOICES=$f" "$HERE/inmem.sh" -w "$wl" \
        --only NeuroPress --out "$INMEM/$wl-eb$eb-replay-$pol"
    done
  done
done
