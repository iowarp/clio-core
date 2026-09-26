#!/usr/bin/env bash
# Build clio_cte_vector_stress on Aurora against an existing build tree,
# without reconfiguring it (build-fresh has CLIO_CORE_ENABLE_BENCHMARKS=OFF).
# Host-only: no SYCL device pass. Flags mirror the tree's own core_client.cc
# compile command; the link line mirrors build_newcoro_aurora.sh.
#
#   build_stress_aurora.sh [build dir]   -> <build dir>/bin/clio_cte_vector_stress
set -eu
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
W=$(cd "$HERE/../.." && pwd)
BUILD=${1:-$W/build-fresh}
SPACK_ROOT=${SPACK_ROOT:-$(ls -d /opt/aurora/*/spack/unified/*/install/linux-x86_64 2>/dev/null | head -1)}
ICPX=${ICPX:-icpx}
python3 - "$BUILD" > "$HERE/.stress_flags" <<'PY'
import json, shlex, sys
cc = json.load(open(sys.argv[1] + '/compile_commands.json'))
c = [c for c in cc if c['file'].endswith('context-transfer-engine/core/src/core_client.cc')][0]
args = shlex.split(c['command']) if 'command' in c else c['arguments']
out = []; i = 0
while i < len(args):
    a = args[i]
    if a in ('-isystem', '-I', '-D'):
        out += [a, args[i + 1]]; i += 2; continue
    if a.startswith(('-I', '-isystem', '-D')) and 'EXPORTS' not in a and 'BUILDING_DLL' not in a:
        out.append(a)
    i += 1
print(' '.join(shlex.quote(x) for x in out))
PY
FLAGS=$(cat "$HERE/.stress_flags"); rm -f "$HERE/.stress_flags"
LIBDIRS=(-L"$BUILD/bin" -Wl,-rpath,"$BUILD/bin")
for pkg in yaml-cpp libzmq zeromq; do
  d=$(ls -d "$SPACK_ROOT"/${pkg}-*/lib64 "$SPACK_ROOT"/${pkg}-*/lib 2>/dev/null | head -1)
  [ -n "$d" ] && LIBDIRS+=(-L"$d" -Wl,-rpath,"$d")
done
for d in "$HOME"/spack/opt/spack/linux-*/libzmq-*/lib "$HOME"/spack/opt/spack/linux-*/yaml-cpp-*/lib "$HOME"/spack/opt/spack/linux-*/yaml-cpp-*/lib64; do
  [ -d "$d" ] && LIBDIRS+=(-L"$d" -Wl,-rpath,"$d")
done
set -x
eval "$ICPX" -std=c++20 -O2 -g $FLAGS -I"$HERE/../adapter/gpu_vector/benchmark" -I"$HERE/../checkpoint/include" -I"$HERE/../../build-fresh/context-transfer-engine/checkpoint/include" \
  "$HERE/clio_cte_vector_stress.cc" -o "$BUILD/bin/clio_cte_vector_stress" \
  "${LIBDIRS[@]}" -lclio_cte_checkpoint_client -lclio_cte_core_client -lclio_cte_core_runtime -lclio_admin_client \
  -lclio_bdev_client -lclio_run_cxx -lclio_ctp_host -lzmq -lyaml-cpp -lsycl -lpthread -ldl -lrt
set +x
echo "built $BUILD/bin/clio_cte_vector_stress"
