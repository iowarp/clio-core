#!/usr/bin/env bash
# Transpile, compile with NVCC, link and run one ported paged benchmark.
#   bench.sh <dir> <name> [run args...]
set -u
DIR=$1; NAME=$2; shift 2
W=/workspace
B=/home/iowarp/bnv2
G=/home/iowarp/gen_$NAME
O=/home/iowarp/ov_$NAME
GVI=context-transfer-engine/adapter/gpu_vector/include
BD=context-transfer-engine/adapter/gpu_vector/benchmark/$DIR
SRC=$W/$BD/clio_${NAME}_paged_newcoro.cc

python3 - <<'PY' > /tmp/nvflags.txt
import json, shlex
d = json.load(open("/home/iowarp/bnv2/compile_commands.json"))
pick = next(e for e in d if "core_runtime.cc" in e["file"]
            and "context-transfer-engine" in e["file"])
a = shlex.split(pick["command"])[1:]
keep, i = [], 0
while i < len(a):
    x = a[i]
    if x == "-isystem" and i + 1 < len(a):
        keep += [x, a[i + 1]]; i += 2; continue
    if x.startswith("-I") or x.startswith("-D"):
        keep.append(x)
    i += 1
keep = [x for x in keep if "CLIO_YIELD_CORO" not in x]
keep = [x for x in keep if not (x.startswith("-I") and "gpu_vector" in x)]
keep.append("-DCLIO_COROC")
print(" ".join(keep))
PY
FLAGS=$(cat /tmp/nvflags.txt)
EXTRA="-I$W/context-runtime/include -I$W/context-runtime/modules/admin/include
       -I$W/context-runtime/modules/bdev/include
       -I$W/context-transfer-engine/core/include
       -I$W/context-transport-primitives/include
       -I$B/context-transport-primitives/src/include
       -I$W/context-transfer-engine/checkpoint/include
       -I$W/context-transfer-engine/compressor/include"

# The TOOL parses with clang 18, which cannot read CUDA 13.2's headers, so
# it parses against 12.6 -- and every 13.2 include path has to move with it
# or crt/math_functions.hpp arrives from the wrong toolkit. Only the parse is
# affected; nvcc below still builds against 13.2.
TPFLAGS=$(echo "$FLAGS $EXTRA" | sed -e "s#cuda-13[.]2#cuda-12.6#g" -e "s#/usr/local/cuda/#/usr/local/cuda-12.6/#g")

echo "### [$NAME] 1. transpile"
rm -rf "$G"
/tmp/bcoroc/clio-coroc "$SRC" --verbose --rewrite-root=$W --mirror-to=$G \
  -- -x cuda --cuda-path=/usr/local/cuda-12.6 --cuda-gpu-arch=sm_80 \
     --cuda-device-only -Wno-unknown-cuda-version -std=c++20 -Wno-everything -ferror-limit=0 \
     -I$W/$BD -I$W/$GVI $TPFLAGS > /tmp/tp_$NAME.log 2>&1
grep -E "clio-coroc" /tmp/tp_$NAME.log | head -5
if ! test -f "$G/$BD/clio_${NAME}_paged_newcoro.cc"; then
  echo "TRANSPILE FAILED"; grep -E "error: clio-coroc" /tmp/tp_$NAME.log | head -10; exit 1
fi

echo "### [$NAME] 2. overlay + nvcc"
rm -rf "$O"; mkdir -p "$O"
cp -r "$W/$GVI" "$O/gv_include"
cp "$G/$GVI/clio_cte/gpu_vector/device_vector.h" \
   "$O/gv_include/clio_cte/gpu_vector/device_vector.h" 2>/dev/null || true
/usr/local/cuda-13.2/bin/nvcc -x cu "$G/$BD/clio_${NAME}_paged_newcoro.cc" \
  -c -o /tmp/${NAME}.o -arch=sm_120 -std=c++20 -O2 -rdc=true \
  -Xcompiler -fPIC --expt-relaxed-constexpr --extended-lambda \
  -I$W/$BD -I"$O/gv_include" -I"$O/gv_include/clio_cte/gpu_vector" \
  $EXTRA $FLAGS > /tmp/nv_$NAME.log 2>&1
if ! test -f /tmp/${NAME}.o; then
  echo "NVCC FAILED"; grep -E "error" /tmp/nv_$NAME.log | head -15; exit 1
fi
echo "  object $(stat -c%s /tmp/${NAME}.o) bytes"

echo "### [$NAME] 3. link"
/usr/local/cuda-13.2/bin/nvcc -arch=sm_120 -rdc=true /tmp/${NAME}.o \
  -o /tmp/clio_${NAME}_paged_newcoro \
  -L$B/bin -Xlinker -rpath -Xlinker $B/bin \
  -lclio_cte_core_client -lclio_cte_core_runtime \
  -lclio_admin_client -lclio_bdev_client \
  -lclio_run_cxx -lclio_run_cxx_gpu -lclio_ctp_host -lclio_ctp_cuda \
  -L/usr/local/cuda-13.2/lib64 -lcudart -lzmq -lyaml-cpp -lpthread -ldl -lrt \
  > /tmp/ln_$NAME.log 2>&1
test -x /tmp/clio_${NAME}_paged_newcoro || { echo "LINK FAILED";
  grep -E "error|undefined" /tmp/ln_$NAME.log | head -10; exit 1; }

echo "### [$NAME] 4. run: $*"
mkdir -p /home/iowarp/rundir && cd /home/iowarp/rundir && rm -f *.yaml
timeout 1200 /tmp/clio_${NAME}_paged_newcoro "$@" 2>&1 \
  | grep -viE "INFO|DEBUG" | grep -E "GATE|PASS|FAIL|paging:|faults|ERROR|error" | head -20
echo "### [$NAME] exit=${PIPESTATUS[0]}"
