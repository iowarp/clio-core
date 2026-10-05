#!/usr/bin/env bash
#===============================================================================
# gen_camels_series.sh -- the astro-camels-6snap workload: CAMELS IllustrisTNG
# CV_0 as a time series, snapshots 040, 050, ..., 090 (particles + group
# catalog of each), every HDF5 dataset as one native array file.
#
#   gen_camels_series.sh download   fetch the snapshots and group catalogs
#   gen_camels_series.sh extract    write the arrays
#
# Raw files: RAW (default /mnt/nvme0/np-data-new/raw/camels); arrays: OUT
# (default /mnt/nvme0/np-data-new/astro-camels-6snap), named
# <snap|grp>_<dataset path>_s<NNN>.<type>, so stage_full_workloads.py replays
# them snapshot by snapshot (the _sNNN tag is the time step).
# Total about 15 GB (each snapshot 2.3-2.7 GB).
#===============================================================================
set -euo pipefail
RAW=${RAW:-/mnt/nvme0/np-data-new/raw/camels}
OUT=${OUT:-/mnt/nvme0/np-data-new/astro-camels-6snap}
B=https://users.flatironinstitute.org/~camels/Sims/IllustrisTNG/CV/CV_0
SNAPS=(040 050 060 070 080 090)

# Download one file in 8 parallel byte ranges, then check its size.
get() {
  local f=$1 total n=8 step i a b
  [ -f "$RAW/$f" ] && return 0
  total=$(curl -sI "$B/$f" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
  step=$(( (total + n - 1) / n ))
  for i in $(seq 0 $((n - 1))); do
    a=$(( i * step )); b=$(( a + step - 1 )); [ $b -ge "$total" ] && b=$((total - 1))
    curl -fsSL --retry 5 -r $a-$b -o "$RAW/$f.part$i" "$B/$f" &
  done
  wait
  cat $(for i in $(seq 0 $((n - 1))); do echo "$RAW/$f.part$i"; done) > "$RAW/$f.tmp"
  rm -f "$RAW/$f".part*
  [ "$(stat -c %s "$RAW/$f.tmp")" = "$total" ] || { echo "SIZE MISMATCH $f" >&2; exit 1; }
  mv "$RAW/$f.tmp" "$RAW/$f"
  echo "$(date +%T) $f $(( total / 1048576 )) MiB"
}

case "${1:-}" in
  download)
    mkdir -p "$RAW"
    for s in "${SNAPS[@]}"; do get "groups_$s.hdf5"; get "snapshot_$s.hdf5"; done
    echo CAMELS_DOWNLOAD_DONE ;;
  extract)
    mkdir -p "$OUT"
    RAW="$RAW" OUT="$OUT" SNAPS="${SNAPS[*]}" ~/np-venv/bin/python - <<'PY'
import os
import h5py
import numpy as np
raw, out = os.environ["RAW"], os.environ["OUT"]
tag = {"f4": "f32", "f8": "f64", "i4": "i32", "i8": "i64", "u4": "u32", "u8": "u64",
       "i2": "i16", "u2": "u16", "u1": "u8", "i1": "i8"}
tot = 0
for s in os.environ["SNAPS"].split():
    for fn, pre in ((f"groups_{s}.hdf5", "grp"), (f"snapshot_{s}.hdf5", "snap")):
        with h5py.File(os.path.join(raw, fn), "r") as f:
            def visit(name, obj):
                global tot
                if not isinstance(obj, h5py.Dataset) or obj.size == 0:
                    return
                a = np.ascontiguousarray(obj[...])
                a.tofile(os.path.join(out, f"{pre}_{name.replace('/', '_')}_s{s}."
                                           f"{tag[a.dtype.str[1:]]}"))
                tot += a.nbytes
            f.visititems(visit)
    print(f"snapshot {s}: {tot / 2**30:.2f} GiB so far", flush=True)
PY
    echo CAMELS_EXTRACT_DONE ;;
  *) echo "usage: $0 download|extract" >&2; exit 2 ;;
esac
