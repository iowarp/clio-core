#!/usr/bin/env bash
# Feed each per-kernel SPIR-V image from -fsycl-dump-device-code to IGC on its
# own, and report which ones crash it. Runs on the login node: no allocation.
#
#   bisect_igc.sh <dump-dir> <out-dir>
#
# lammps_md's device code crashes IGC (Internal Compiler Error: Segmentation
# violation) 13 times out of ~20 images under per-kernel split. This is how
# those 13 get names. The mangled kernel type name is pulled out of the SPIR-V
# with strings(1), which is crude but needs no disassembler.
set -u
DUMP=${1:?dump dir}
OUT=${2:?out dir}
mkdir -p "$OUT"
pass=0; crash=0
for f in "$DUMP"/*.spv; do
  b=$(basename "$f" .spv)
  if /usr/bin/ocloc compile -file "$f" -spirv_input -device pvc \
       -output "$OUT/$b" -output_no_suffix > "$OUT/$b.log" 2>&1; then
    r=PASS; pass=$((pass+1))
  else
    r=CRASH; crash=$((crash+1))
  fi
  k=$(strings "$f" | grep -oE "_ZTS[A-Za-z0-9_]{8,80}" | head -1)
  printf '%-5s %s %s\n' "$r" "$b" "${k:-?}"
done
echo "BISECT-DONE pass=$pass crash=$crash"
