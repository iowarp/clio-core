#!/bin/bash
# Selective-lossy study, RATIO cost model (learn-ratio: compress and decompress
# weights zeroed, the objective is bytes saved). Per bound, in memory:
#   <wl>-eb<eb>-ratio             NeuroPress (learn-ratio) and uniform lossy with
#                                 the trace's highest-ratio quantized fixed config
#                                 (arm "uniform_zstd": zstd on VPIC at every bound)
#   <wl>-eb<eb>-np-ratiofloor40   NeuroPress (learn-ratio) with the 40 dB quality floor
#
#   ./selective_ratio.sh [-w vpic] [--ebs "1e-3 1e-4 1e-2"]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
INMEM=$(cd "$HERE/.." && pwd)/results/inmem
WL=vpic EBS="1e-3 1e-4 1e-2"
while [ $# -gt 0 ]; do
  case "$1" in
    -w) WL=$2; shift 2 ;;
    --ebs) EBS=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
# max_ratio_cfg <wl> <eb> -- the trace's highest-ratio quantized fixed action as
# an inmem.sh config, e.g. "zstd|q1|s1" -> static-zstd-q-s4.
max_ratio_cfg() {
  ~/np-venv/bin/python - "$HERE" "$1" "$2" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import selective_analyze as A
d = A.load_trace(f"{sys.argv[1]}/../results/selective/{sys.argv[2]}-eb{sys.argv[3]}/explore.csv")
B = d["B"]
cr = B.sum() / (B[:, None] / d["ratio"]).sum(0)
cr[~d["quant"]] = 0
lib, _, sh = d["cfgs"][int(cr.argmax())].split("|")
print(f"static-{lib}-q" + ("-s4" if sh == "s1" else ""))
PY
}

for eb in $EBS; do
  cfg=$(max_ratio_cfg "$WL" "$eb") || { echo "no trace for $WL eb $eb" >&2; continue; }
  echo "==== $WL eb $eb: ratio model, NeuroPress vs uniform $cfg ($(date -u +%H:%M) UTC)"
  EB=$eb EXTRA_ARMS="NP_ratio|learn-ratio|$eb Uniform_zstd|$cfg|$eb" \
    "$HERE/inmem.sh" -w "$WL" --only "NP_ratio,Uniform_zstd" --out "$INMEM/$WL-eb$eb-ratio"
  echo "==== $WL eb $eb: ratio model + 40 dB quality floor ($(date -u +%H:%M) UTC)"
  EB=$eb EXTRA_ARMS="NP_ratio|learn-ratio|$eb" XENV="CLIO_NEUROPRESS_TARGET_PSNR=40" \
    "$HERE/inmem.sh" -w "$WL" --only "NP_ratio" --out "$INMEM/$WL-eb$eb-np-ratiofloor40"
done
