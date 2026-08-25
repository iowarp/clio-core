#!/usr/bin/env bash
# Are the blobs Clio stored the same bytes AMReX would have written?
#
#   ./crosscheck_plotfile.sh [--ncell 32] [--steps 10] [--int 10]
#                            [--max-grid N]      # N < ncell => several boxes
#                            [--mpi] [--ranks N] # under mpirun; N runtimes
#
# Runs Nyx once with --hook plotfile, so the in-situ hand-over fires from
# Nyx::writePlotFile, on the same MultiFab, at the same instant as the native
# plotfile Nyx then writes. Afterwards it parses the plotfile's FAB directly
# -- AMReX's own VisMF format, written host-side down a completely different
# code path -- and compares an FNV-1a-64 of each component's valid box with
# the digest the in-situ path recorded for the corresponding blob.
#
# This is the independent check. --verify and read.sh both read back through
# the same Clio decompressor that wrote the data, so a codec that corrupted
# symmetrically would pass them; this one has no Clio in it at all on the
# reference side. It is also what establishes that the valid-box extraction
# (a strided cudaMemcpy3D out of the ghost-inclusive FAB) picks exactly the
# region a plotfile stores.
#
# It also reports the same comparison against the .f32 files of
# paper-benchmark/nyx's raw dump, which fires in the same run -- see the
# README's "What the raw dump actually dumps".
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NCELL=32 STEPS=10 INT=10 MAXGRID= MPIARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --ncell) NCELL=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --int) INT=$2; shift 2;;
    --max-grid) MAXGRID=$2; shift 2;;
    --mpi) MPIARGS+=(--mpi); shift;;
    --ranks) MPIARGS+=(--ranks "$2"); shift 2;;
    -h|--help) sed -n '2,23p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
EXTRA=()
[ -n "$MAXGRID" ] && EXTRA+=(--max-grid "$MAXGRID")

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
echo "== Nyx, --hook plotfile: Clio in situ AND a native plotfile from the same state"
STORE=$TMP/store WORK=$TMP/work NYX_DUMP_DIR=$TMP/dump \
  "$HERE/run.sh" --hook plotfile --ncell "$NCELL" --steps "$STEPS" --int "$INT" \
  --chunk 0 --store "$TMP/store" ${MPIARGS+"${MPIARGS[@]}"} ${EXTRA+"${EXTRA[@]}"} | tail -3

python3 - "$TMP" "$INT" <<'XPY'
import sys, os, re
TMP, INT = sys.argv[1], int(sys.argv[2])

def fnv(b):
    h = 14695981039346656037
    for x in b:
        h ^= x; h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h

def read_multifab(level_dir):
    """AMReX VisMF. Cell_H lists the BoxArray and, per FAB, the file and byte
       offset it starts at, in MFIter index order. Each FAB is an ASCII
       'FAB ((n,(..)),(m,(..)))<box> <ncomp>' line followed by ncomp * numPts
       raw values, component-major, over the box in that header -- a plotfile
       MultiFab carries no ghost cells, so that box IS the valid box."""
    hdr = open(os.path.join(level_dir, "Cell_H")).read().splitlines()
    fabs = [l.split()[1:] for l in hdr if l.startswith("FabOnDisk:")]
    out = []
    for fname, off in fabs:
        data = open(os.path.join(level_dir, fname), 'rb').read()
        nl = data.index(b'\n', int(off))
        line = data[int(off):nl].decode()
        m = re.match(r'FAB \(\((\d+), \([^)]*\)\),\((\d+), \([^)]*\)\)\)'
                     r'\(\((\d+),(\d+),(\d+)\) \((\d+),(\d+),(\d+)\) \([^)]*\)\) (\d+)',
                     line)
        if not m:
            raise SystemExit("unrecognised FAB header: " + line)
        rsize = int(m.group(2))
        lo = tuple(int(m.group(i)) for i in (3, 4, 5))
        hi = tuple(int(m.group(i)) for i in (6, 7, 8))
        npts = (hi[0]-lo[0]+1) * (hi[1]-lo[1]+1) * (hi[2]-lo[2]+1)
        ncomp = int(m.group(9))
        out.append((rsize, npts, ncomp, data[nl+1:nl+1+rsize*npts*ncomp]))
    return out

names = ["density", "xmom", "ymom", "zmom", "rho_E", "rho_e"]
# Under MPI every rank writes its own CSV, in its own store. The blob NAMES
# carry the global box index, so the union across ranks is exactly the set a
# single-rank run would have written -- and a duplicate name would mean two
# ranks published the same box, which is the thing the naming has to rule out.
csv = {}
dups = []
import glob
paths = sorted(glob.glob(os.path.join(TMP, "store", "blobs*.csv")) +
               glob.glob(os.path.join(TMP, "store", "rank*", "blobs*.csv")))
for path in paths:
    for line in open(path).read().splitlines()[1:]:
        p = line.split(',')
        if p[0] in csv:
            dups.append(p[0])
        csv[p[0]] = (int(p[1]), int(p[2], 16))
print("-- %d CSV(s), %d blob name(s), %d duplicate name(s) across ranks"
      % (len(paths), len(csv), len(dups)))

plts = sorted(d for d in os.listdir(os.path.join(TMP, "work")) if d.startswith("plt"))
same = differ = 0
rb_same = rb_differ = 0
for k, plt in enumerate(plts):
    step = int(plt[3:])
    fabs = read_multifab(os.path.join(TMP, "work", plt, "Level_0"))
    print("-- %s: %d FAB(s), real=%dB npts=%d ncomp=%d"
          % (plt, len(fabs), fabs[0][0], fabs[0][1], fabs[0][2]))
    for fi, (rsize, npts, ncomp, body) in enumerate(fabs):
        for c, nm in enumerate(names):
            seg = body[c*npts*rsize:(c+1)*npts*rsize]
            key = "%s/step_%05d/fab%04d/chunk_0" % (nm, step, fi)
            if key not in csv:
                print("   %s: no such blob" % key); differ += 1; continue
            nbytes, digest = csv[key]
            ok = (len(seg) == nbytes) and (fnv(seg) == digest)
            if not ok or len(fabs) <= 2:
                print("   %-44s plotfile=%016x blob=%016x %s"
                      % (key, fnv(seg), digest, "same" if ok else "DIFFER"))
            same, differ = (same+1, differ) if ok else (same, differ+1)
            # And the raw dump of paper-benchmark/nyx, which fired in this run too.
            rb = os.path.join(TMP, "dump", "plt%05d" % k,
                              "fab%04d_comp%02d_%s.f32" % (fi, c, nm))
            if os.path.exists(rb):
                rbb = open(rb, 'rb').read()
                if len(rbb) == nbytes and fnv(rbb) == digest: rb_same += 1
                else: rb_differ += 1

print("\n%d blob(s) byte-identical to the native plotfile, %d differ" % (same, differ))
if dups:
    print("DUPLICATE blob names across ranks: %s" % sorted(set(dups))[:5])
if rb_same + rb_differ:
    print("%d identical to the raw .f32 dump, %d differ -- see README, "
          "'What the raw dump actually dumps'" % (rb_same, rb_differ))
raise SystemExit(1 if (differ or dups) else 0)
XPY
