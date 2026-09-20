#!/usr/bin/env bash
# Transpile and SYCL device-compile every ported paged benchmark, on Aurora.
# See build_newcoro_aurora.sh for what this does and does not prove.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BENCHES=(gmx:gmx lbann:lbann weights:weights grayscott:grayscott
         kmeans:kmeans lammps_md:lammps_md)

pass=0
fail=0
declare -a failed=()
for b in "${BENCHES[@]}"; do
  dir=${b%%:*}
  name=${b##*:}
  if "${HERE}/build_newcoro_aurora.sh" "$dir" "$name" > "/tmp/aurora_$name.log" 2>&1; then
    size=$(grep -o 'SYCL OK: [0-9]*' "/tmp/aurora_$name.log" | grep -o '[0-9]*')
    fns=$(grep -o '[0-9]* suspending' "/tmp/aurora_$name.log" | grep -o '[0-9]*')
    printf 'PASS  %-12s %2s suspending fn(s), %s byte object\n' "$name" "$fns" "$size"
    pass=$((pass + 1))
  else
    printf 'FAIL  %-12s\n' "$name"
    grep -E "error|FAILED" "/tmp/aurora_$name.log" | head -4 | sed 's/^/        /'
    failed+=("$name")
    fail=$((fail + 1))
  fi
done

echo
echo "${pass} passed, ${fail} failed"
[ "$fail" -eq 0 ] || { echo "failed: ${failed[*]}"; exit 1; }
