#!/usr/bin/env bash
# Prove the driver stores the SAME bytes `dump h5md` would have written.
#
#   ./run.sh --box 10 --steps 100 --gap 50            # CPU, --order id
#   ./crosscheck_h5md.sh --box 10 --steps 100 --gap 50
#
# Runs the stock `lmp` binary on the sibling deck (../neuropress_lammps_h5/
# in.melt_clio, which DOES dump h5md) with plain HDF5 -- no VOL, no Clio --
# then digests each frame of /particles/all/{position,velocity,force}/value
# and compares against the digests the driver recorded in $STORE/blobs.csv.
#
# Only meaningful for a CPU run with --order id and --chunk 0 (or a chunk no
# smaller than one field): the GPU run is not bit-reproducible, and
# --order local is LAMMPS' internal atom order rather than h5md's ID order.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LMP=${LMP:-$HOME/src/lammps/build/lmp}
BOX=10 STEPS=100 GAP=50
STORE=${STORE:-$HERE/store}
while [ $# -gt 0 ]; do
  case "$1" in
    --box) BOX=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --gap) GAP=$2; shift 2;;
    --store) STORE=$2; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -x "$LMP" ] || { echo "missing $LMP (set LMP=)" >&2; exit 1; }
[ -f "$STORE/blobs.csv" ] || { echo "no $STORE/blobs.csv -- run ./run.sh first" >&2; exit 1; }
command -v h5dump >/dev/null || { echo "h5dump not found" >&2; exit 1; }

REF=$STORE/ref_h5md
rm -rf "$REF"; mkdir -p "$REF"
# Plain HDF5: make sure no VOL connector is inherited from the environment.
env -u HDF5_VOL_CONNECTOR -u HDF5_PLUGIN_PATH \
  "$LMP" -log "$REF/log.lammps" -screen none \
  -in "$HERE/../neuropress_lammps_h5/in.melt_clio" \
  -var BOX "$BOX" -var GAP "$GAP" -var STEPS "$STEPS" -var OUT "$REF/ref.h5"

NATOMS=$(( 4 * BOX * BOX * BOX ))
for f in position velocity force; do
  h5dump -d /particles/all/$f/value -b LE -o "$REF/$f.bin" "$REF/ref.h5" > /dev/null
done

python3 - "$STORE/blobs.csv" "$REF" "$NATOMS" "$GAP" <<'PY'
import sys, csv
csv_path, ref, natoms, gap = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
def fnv1a(b):
    h = 0xcbf29ce484222325
    for x in b:
        h ^= x; h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h
frame_bytes = natoms * 3 * 8
recorded = {}
for row in csv.DictReader(open(csv_path)):
    recorded[row['blob']] = (int(row['bytes']), int(row['fnv1a64'], 16))
ok = bad = 0
for f in ('position', 'velocity', 'force'):
    data = open(f'{ref}/{f}.bin', 'rb').read()
    nframes = len(data) // frame_bytes
    for i in range(nframes):
        name = f'{f}/step_{i*gap}/chunk_0'
        if name not in recorded:
            print(f'  {name:28s} not in blobs.csv (chunked run?)'); bad += 1; continue
        nbytes, digest = recorded[name]
        h = fnv1a(data[i*frame_bytes:(i+1)*frame_bytes])
        same = (nbytes == frame_bytes and h == digest)
        print(f'  {name:28s} h5md={h:016x} driver={digest:016x} {"same" if same else "DIFFERENT"}')
        ok += same; bad += (not same)
print(f'{ok} frame(s) byte-identical to dump h5md, {bad} differ')
sys.exit(1 if bad else 0)
PY
