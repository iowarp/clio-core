#!/usr/bin/env bash
# Are the blobs Clio stored the same bytes the deck would have written?
#
#   ./crosscheck.sh [--ncell 30] [--steps 50] [--int 25] [--nppc 8]
#
# Runs the deck ONCE with both paths on: VPIC_DUMP_FIELDS=1 writes the flat
# .f32 files the offline benchmark replays, and VPIC_INSITU=1 hands the same
# device arrays to Clio at the same instant in begin_diagnostics. Afterwards
# it compares an FNV-1a-64 of each dumped file with the digest the adapter
# recorded for the corresponding blob.
#
# One run, not two: VPIC's current deposition accumulates through atomics, so
# two runs of the same deck are not bit-reproducible and a cross-RUN
# comparison would fail for reasons that have nothing to do with Clio.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NCELL=30 STEPS=50 INT=25 NPPC=8
while [ $# -gt 0 ]; do
  case "$1" in
    --ncell) NCELL=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --int) INT=$2; shift 2;;
    --nppc) NPPC=$2; shift 2;;
    -h|--help) sed -n '2,14p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/dump"
echo "== VPIC ${NCELL}^3, $STEPS steps: file dump AND in-situ hand-over from the same state"
VPIC_DUMP_FIELDS=1 VPIC_DUMP_INT="$INT" VPIC_DUMP_DIR="$TMP/dump" \
  "$HERE/run.sh" --ncell "$NCELL" --steps "$STEPS" --int "$INT" --nppc "$NPPC" \
                 --verify --store "$TMP/store" "$@" | grep -E "stored|VERIFIED|FAILED" || true
python3 - "$TMP" <<'PY'
import csv, os, re, sys
X = sys.argv[1]
def fnv(b):
    h = 14695981039346656037
    for x in b: h = ((h ^ x) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h
off = {}
for frame in sorted(os.listdir(os.path.join(X, "dump"))):
    d = os.path.join(X, "dump", frame)
    if not os.path.isdir(d): continue
    for fn in sorted(os.listdir(d)):
        m = re.match(r"fab0000_comp(\d+)_(.+)\.f32", fn)
        if not m: continue
        p = os.path.join(d, fn)
        off[(frame, m.group(2))] = (fnv(open(p, "rb").read()), os.path.getsize(p))
ins = {}
for r in csv.DictReader(open(os.path.join(X, "store", "blobs.csv"))):
    var, step, _ = r["blob"].split("/")
    ins[(step, var)] = (int(r["fnv1a64"], 16), int(r["bytes"]))
frames = sorted({k[0] for k in off}); steps = sorted({k[0] for k in ins})
same = diff = missing = 0
for i, fr in enumerate(frames):
    if i >= len(steps): break
    for var in sorted({k[1] for k in off if k[0] == fr}):
        a, b = off[(fr, var)], ins.get((steps[i], var))
        if b is None:
            missing += 1; print(f"   no in-situ blob for {fr}/{var}"); continue
        if a == b: same += 1
        else:
            diff += 1
            print(f"   DIFFER {fr}/{var}: file={a[0]:016x}/{a[1]}B insitu={b[0]:016x}/{b[1]}B")
print(f"\n{same} blob(s) byte-identical to the deck's own file dump, {diff} differ, {missing} missing")
sys.exit(1 if (diff or missing or not same) else 0)
PY
