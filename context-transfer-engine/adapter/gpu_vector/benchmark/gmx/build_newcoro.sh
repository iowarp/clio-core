set -u
W=/workspace
B=/home/iowarp/bnv2
G=/home/iowarp/gen2
O=/home/iowarp/ov2
GVI=context-transfer-engine/adapter/gpu_vector/include
GMX=context-transfer-engine/adapter/gpu_vector/benchmark/gmx
SRC=$W/$GMX/clio_gmx_paged_newcoro.cc

# Flags: take a real CTE TU's -I/-D set from this build, drop the ones that
# belong to that TU's own target, add the gmx include set.
python3 - <<'PY' > /tmp/nvflags.txt
import json, shlex
d = json.load(open("/home/iowarp/bnv2/compile_commands.json"))
pick = None
for e in d:
    if "core_runtime.cc" in e["file"] and "context-transfer-engine" in e["file"]:
        pick = e
        break
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

echo "### 1. transpile"
rm -rf "$G"
/tmp/bcoroc/clio-coroc "$SRC" --verbose \
  --rewrite-root=$W --mirror-to=$G \
  -- -x cuda --cuda-path=/usr/local/cuda-12.6 --cuda-gpu-arch=sm_80 \
     --cuda-device-only -Wno-unknown-cuda-version -std=c++20 -Wno-everything \
     -I$W/$GMX -I$W/$GVI $EXTRA $FLAGS 2>&1 | grep -E "clio-coroc"
test -f "$G/$GMX/clio_gmx_paged_newcoro.cc" || { echo "TRANSPILE PRODUCED NOTHING"; exit 1; }

echo "### 2. overlay"
rm -rf "$O"; mkdir -p "$O"
cp -r "$W/$GVI" "$O/gv_include"
cp "$G/$GVI/clio_cte/gpu_vector/device_vector.h" \
   "$O/gv_include/clio_cte/gpu_vector/device_vector.h"

echo "### 3. compile with NVCC (the compiler co_await cannot use)"
/usr/local/cuda-13.2/bin/nvcc -x cu "$G/$GMX/clio_gmx_paged_newcoro.cc" \
  -c -o /tmp/newcoro_nvcc.o \
  -arch=sm_120 -std=c++20 -O2 -rdc=true -Xcompiler -fPIC -Xptxas -v \
  --expt-relaxed-constexpr --extended-lambda \
  -I$W/$GMX -I"$O/gv_include" -I"$O/gv_include/clio_cte/gpu_vector" \
  $EXTRA $FLAGS 2>&1 > /tmp/nv.log 2>&1; grep -E "error" /tmp/nv.log | head -25
test -f /tmp/newcoro_nvcc.o || { echo "NVCC COMPILE FAILED"; exit 1; }
echo "compiled: $(ls -la /tmp/newcoro_nvcc.o | awk '{print $5}') bytes"

echo "### 4. link"
/usr/local/cuda-13.2/bin/nvcc -arch=sm_120 -rdc=true /tmp/newcoro_nvcc.o -o /tmp/clio_gmx_paged_newcoro \
  -L$B/bin -Xlinker -rpath -Xlinker $B/bin \
  -lclio_cte_core_client -lclio_cte_core_runtime \
  -lclio_admin_client -lclio_bdev_client \
  -lclio_run_cxx -lclio_run_cxx_gpu -lclio_ctp_host -lclio_ctp_cuda \
  -L/usr/local/cuda-13.2/lib64 -lcudart -lzmq -lyaml-cpp -lpthread -ldl -lrt 2>&1 | head -25
test -x /tmp/clio_gmx_paged_newcoro && echo "LINKED OK" || { echo "LINK FAILED"; exit 1; }

echo "### 5. run -- resident, then out-of-core with eviction"
mkdir -p /home/iowarp/rundir && cd /home/iowarp/rundir
rm -f gv_gmx_bench.yaml
timeout 900 /tmp/clio_gmx_paged_newcoro --page-kb 128 --blocks 8 --atoms 200000 \
  2>&1 | grep -E "mesh=|atoms=|paging:|spread |GATE|ALL GATES"
rm -f gv_gmx_bench.yaml
timeout 900 /tmp/clio_gmx_paged_newcoro --page-kb 128 --blocks 8 --atoms 200000 --cap 40 \
  2>&1 | grep -E "mesh=|atoms=|paging:|spread |GATE|ALL GATES"
