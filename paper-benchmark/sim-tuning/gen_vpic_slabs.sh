#!/usr/bin/env bash
#===============================================================================
# gen_vpic_slabs.sh -- make the vpic-slabs workload: the 25 GiB VPIC Weibel
# run with 64 layered slabs (vpic_probe.sh final25g, 2026-10-05), with the
# exact settings, and optionally cut it into the 4 MiB chunks the benchmark
# replays.
#
#   gen_vpic_slabs.sh FIELDS_DIR [STAGE_ROOT]
#
# FIELDS_DIR  where VPIC writes its dump (plt00000 ... plt00100, one
#             fab0000_comp02_ez.f32 each; ~26 GB). Any file system.
# STAGE_ROOT  optional: also write STAGE_ROOT/vpic-slabs/fields/*.chunk
#             (stage_full_workloads.py; 6464 chunks of 4 MiB, ~26 GB more).
#
# Needs the deck built first (../vpic/build_deck.sh -> ../vpic/weibel_clio.Linux,
# VPIC-Kokkos f01d295; BIN= overrides the binary) and one GPU. ~400 s on an
# A100. Expected result (Chameleon, 2026-10-05):
#   101 files of 268435456 B ((254+2) x (254+2) x (1022+2) float32, ghosts)
#   sha256 plt00000/fab0000_comp02_ez.f32
#     a1f6ece6fbf27e728831643294a303acdc096ca43cbb4e94e539c5d9dc4ea57a
#   sha256 plt00100/fab0000_comp02_ez.f32
#     d8ce835436a60c638f7a2d031cd01345a97969b34e39578a15394e68bb3d5668
# The settings are environment variables the deck reads (LAYERED SLABS in
# ../vpic/weibel_clio.cxx); gen.json does not record them, this script does.
#===============================================================================
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
FIELDS=$(realpath -m "${1:?usage: gen_vpic_slabs.sh FIELDS_DIR [STAGE_ROOT]}")
STAGE=${2:-}

# 64 slabs of 16 cells along z (one 4 MiB chunk of ez lies in one slab):
# C = vacuum with a frozen "clumpy" field (a few exact values),
# S = vacuum with a frozen smooth field, P = plasma (two slabs
# in the middle, streaming out over time); cells of 2^-5 so the frozen fields
# stay bit-exact. See LAYERED SLABS in ../vpic/weibel_clio.cxx.
export VPIC_SLABS=CSCSCSCSCSCSCSCSCSCSCSCSCSCSCSCPPSCSCSCSCSCSCSCSCSCSCSCSCSCSCSCC
export VPIC_CELL=0.03125      # cell size on every axis (box = cells x cell)
export VPIC_DT_DX=0.5         # dt = 0.5 x cell / c
export VPIC_VTHE=0.03         # cold electrons
export VPIC_VTHEX=0.006       # small anisotropy
export VPIC_DUMP_VARS=ez      # dump ez only (one 4 MiB-aligned array per frame)

mkdir -p "$FIELDS"
"$HERE/../vpic/gen_fields.sh" --ncell 254 --nz 1022 --nppc 8 --steps 303 \
  --dump-int 3 --clean-div 0 --out "$FIELDS"
# record what gen.json leaves out
env | grep '^VPIC_' | sort > "$FIELDS/vpic_env.txt"
echo "ncell=254 nz=1022 nppc=8 steps=303 dump_int=3 clean_div=0" >> "$FIELDS/vpic_env.txt"

n=$(find "$FIELDS" -name '*.f32' | wc -l)
echo "VPIC wrote $n .f32 files (expected 101)"
sha256sum "$FIELDS/plt00000/fab0000_comp02_ez.f32" "$FIELDS/plt00100/fab0000_comp02_ez.f32" || true

if [ -n "$STAGE" ]; then
  python3 "$HERE/../model-accuracy/stage_full_workloads.py" vpic-slabs \
    --src "$FIELDS" --out "$STAGE"
fi
