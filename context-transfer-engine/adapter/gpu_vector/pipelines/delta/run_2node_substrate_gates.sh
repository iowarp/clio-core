#!/bin/bash
#SBATCH --job-name=gv_2node_gates
#SBATCH --partition=gpuA100x4
#SBATCH --account=bekn-delta-gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# EVERY WORKLOAD x EVERY SUBSTRATE, TWO REAL NODES, SMALL AND FAST.
#
# This is a CORRECTNESS gate, not a measurement: smoke-sized problems, ~15
# minutes for all 24 cells, no jarvis. Its job is to answer "does the whole
# matrix still run distributed and still agree?" before a sweep spends an
# allocation. gv_sweep_substrates_a100.yaml is the measurement counterpart.
#
# EVERY CELL ASSERTS ITS OWN PARALLELISM, and that is the point of the file.
# The science gates cannot see whether a run was distributed -- during this
# work three separate configurations ran as independent single-rank jobs and
# passed every gate:
#
#   - cray-mpich launched under `srun --mpi=pmi2` gets no PMI info and does
#     a SINGLETON MPI_Init, so each task becomes its own 1-rank job. Printed
#     "ranks=1" twice and passed everything. Use --mpi=cray_shasta.
#   - `module load aws-ofi-nccl | sed ...` runs module in a SUBSHELL and
#     discards every setenv it performs, so NCCL found no plugin and used
#     TCP sockets with GPU Direct RDMA disabled. Still 2-node, still correct,
#     not a Slingshot number. Never pipe module load.
#   - clio_lammps_md_nccl_bench compared the GLOBAL rank count against
#     node-local visible devices and returned 77 (skip) on 2 nodes x 1 GPU.
#
# THE RANK COUNT IS PRINTED FIVE DIFFERENT WAYS across the six benches --
# "ranks=2", "2 ranks", "PEs=2", "2 PEs", "2 ranks (model-parallel)" -- so
# the assertion regex covers all of them. A narrower pattern reports working
# cells as failures, which is its own way of wasting a day.
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo /u/llogan/clio-core)
source "$ROOT/context-transfer-engine/adapter/gpu_vector/pipelines/delta/env.sh"

BIN="$ROOT/${GV_BUILD:-build-clang}/bin"          # paged + nvshmem (HPC-X ok)
BINC="$ROOT/${GV_BUILD:-build-clang}/bin-cray"    # mpi + nccl (cray-mpich)
LOGD="${GV_LOGDIR:-$PWD/gv_2node_gates_$SLURM_JOB_ID}"; mkdir -p "$LOGD"
SUMMARY="$LOGD/summary.txt"; : > "$SUMMARY"
echo "### nodes: ${SLURM_JOB_NODELIST:-?}   logs: $LOGD"

[ -x "$BINC/clio_kmeans_mpi_bench" ] || {
  echo "no cray-mpich baselines in $BINC -- run build_baselines_cray.sh first"
  exit 1; }

# Covers every format the six benches use. Anchored on non-digits so "12
# ranks" cannot satisfy a request for 2.
DISTPAT='(ranks|PEs)=2([^0-9]|$)|(^|[^0-9])2 (ranks|PEs)([^0-9]|$)'

verdict() {  # verdict <label> <distpat> <log> <exit>
  local label="$1" want="$2" log="$3" ec="$4" v=PASS why=""
  [ "$ec" -ne 0 ] && { v=FAIL; why="exit=$ec"; }
  grep -qaE "$want" "$log" || { v=FAIL; why="${why:+$why }NOT-DISTRIBUTED"; }
  if grep -qaE "GATE: FAIL|GATE FAILURE" "$log"; then v=FAIL; why="${why:+$why }gate"
  elif grep -qa "one GPU per rank" "$log"; then v=FAIL; why="${why:+$why }skipped-77"
  elif ! grep -qaE "ALL GATES PASS|GATE: PASS|checksum=OK" "$log"; then
    v=FAIL; why="${why:+$why }no-gate-output"; fi
  printf '%-24s %-5s %s\n' "$label" "$v" "$why" | tee -a "$SUMMARY"
}

cell() {  # cell <label> <cmd...>
  local label="$1"; shift
  local log="$LOGD/${label// /_}.log"
  timeout -k 30 600 "$@" > "$log" 2>&1
  verdict "$label" "$DISTPAT" "$log" $?
}

S_MPI="srun --mpi=$CLIO_DELTA_SRUN_MPI -n 2 --ntasks-per-node=1"
S_NVS="srun --mpi=$CLIO_DELTA_SRUN_NVSHMEM -n 2 --ntasks-per-node=1"

echo; echo "== MPI (cray-mpich -> CXI) =="
cell "MPI kmeans"    $S_MPI $BINC/clio_kmeans_mpi_bench    --blocks 8 --iters 3 --data-mb 64
cell "MPI weights"   $S_MPI $BINC/clio_weights_mpi_bench   --blocks 8 --data-mb 2
cell "MPI grayscott" $S_MPI $BINC/clio_grayscott_mpi_bench --blocks 8 --steps 4 --data-mb 64
cell "MPI gmx"       $S_MPI $BINC/clio_gmx_mpi_bench       --atoms 20000
cell "MPI lbann"     $S_MPI $BINC/clio_lbann_mpi_bench     --blocks 4
cell "MPI lammps_md" $S_MPI $BINC/clio_lammps_md_mpi_bench --lattice 28 --steps 20 --md

echo; echo "== NVSHMEM (libfabric -> CXI) =="
cell "NVSHMEM kmeans"    $S_NVS $BIN/clio_kmeans_nvshmem_bench    --blocks 8 --iters 3 --data-mb 64
cell "NVSHMEM weights"   $S_NVS $BIN/clio_weights_nvshmem_bench   --blocks 8 --data-mb 2
cell "NVSHMEM grayscott" $S_NVS $BIN/clio_grayscott_nvshmem_bench --blocks 8 --steps 4 --data-mb 64
cell "NVSHMEM gmx"       $S_NVS $BIN/clio_gmx_nvshmem_bench       --atoms 20000
# --lr 0.0001: the default 0.01 diverges to NaN at this size.
cell "NVSHMEM lbann"     $S_NVS $BIN/clio_lbann_nvshmem_bench     --blocks 4 --lr 0.0001
cell "NVSHMEM lammps_md" $S_NVS $BIN/clio_lammps_md_nvshmem_bench --lattice 28 --steps 20 --md

echo; echo "== NCCL (aws-ofi-nccl -> CXI) =="
# env.sh exported NCCL_NET_PLUGIN/FI_PROVIDER and the plugin lib dirs WITHOUT
# a subshell. Confirm rather than assume: the plugin path prints "Using
# network Libfabric"; the fallback prints "Using network Socket".
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET LD_PRELOAD="${CLIO_DELTA_NCCL_PRELOAD:-}" \
  $S_MPI $BINC/clio_kmeans_nccl_bench --blocks 8 --iters 3 --data-mb 64 \
  > "$LOGD/nccl_transport_probe.log" 2>&1
if grep -qa "Using network Socket" "$LOGD/nccl_transport_probe.log"; then
  echo "!! NCCL fell back to TCP SOCKETS -- the numbers below are not Slingshot"
elif grep -qa "Using network Libfabric" "$LOGD/nccl_transport_probe.log"; then
  echo "   NCCL transport: Libfabric/CXI (aws-ofi-nccl)"
else
  echo "!! NCCL transport undetermined; see $LOGD/nccl_transport_probe.log"
fi
export LD_PRELOAD="${CLIO_DELTA_NCCL_PRELOAD:-}"
cell "NCCL kmeans"    $S_MPI $BINC/clio_kmeans_nccl_bench    --blocks 8 --iters 3 --data-mb 64
cell "NCCL weights"   $S_MPI $BINC/clio_weights_nccl_bench   --blocks 8 --data-mb 2
cell "NCCL grayscott" $S_MPI $BINC/clio_grayscott_nccl_bench --blocks 8 --steps 4 --data-mb 64
cell "NCCL gmx"       $S_MPI $BINC/clio_gmx_nccl_bench       --atoms 20000
cell "NCCL lbann"     $S_MPI $BINC/clio_lbann_nccl_bench     --blocks 4 --lr 0.0001
cell "NCCL lammps_md" $S_MPI $BINC/clio_lammps_md_nccl_bench --lattice 28 --steps 20 --md
unset LD_PRELOAD

echo; echo "== PAGED (CTE cluster, one process per node) =="
# Not MPI programs: one process per node, each hosting the runtime, joined
# through a generated CLIO_SERVER_CONF. The parallelism assertion here is
# DISTINCT HOSTNAMES -- two processes on one node would satisfy "nodes=2".
D="$LOGD/paged"; mkdir -p "$D"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$D/hostfile.txt"
head -1 "$D/hostfile.txt" > "$D/hostfile1.txt"
mk_conf() {
cat > "$1" <<YAML
networking:
  port: 9425
  hostfile: "$2"
runtime:
  num_threads: 8
  queue_depth: 8192
  first_busy_wait: 2000000000
gpu:
  queue_depth: 8192
compose:
  - mod_name: clio_bdev
    pool_name: "ram::chi_default_bdev"
    pool_query: local
    pool_id: "301.0"
    bdev_type: ram
    capacity: "1GB"
  - mod_name: clio_cte_core
    pool_name: cte_core
    pool_query: local
    pool_id: "512.0"
    storage:
      - path: "hbm::gvw_hbm"
        bdev_type: "hbm"
        capacity_limit: "512MB"
        score: 1.0
      - path: "ram::gvw_ram"
        bdev_type: "ram"
        capacity_limit: "4GB"
        score: 0.2
    dpe:
      dpe_type: "max_bw"
YAML
}
mk_conf "$D/conf2.yaml" "$D/hostfile.txt"
mk_conf "$D/conf1.yaml" "$D/hostfile1.txt"
cat > "$D/gvw_node.sh" <<'NODE'
#!/bin/sh
i="${SLURM_PROCID:-0}"
log="${GVW_LOGBASE}.node${i}.log"
echo "host=$(hostname) node=$i nodes=${GVW_NODES}" > "$log"
"$@" --node "$i" >> "$log" 2>&1
rc=$?
touch "${GVW_DONEDIR}/done_${i}"
w=0
while [ "$(ls "${GVW_DONEDIR}"/done_* 2>/dev/null | wc -l)" -lt "${GVW_NODES}" ] && [ "$w" -lt "${GVW_BARRIER}" ]; do
  sleep 1; w=$((w+1))
done
exit $rc
NODE
chmod +x "$D/gvw_node.sh"
export GVW_BARRIER=300 GVW_DONEDIR="$D"

paged_run() {  # paged_run <tag> <nodes> <bench> <args...>
  local tag="$1" n="$2" bench="$3"; shift 3
  export GVW_NODES="$n" GVW_LOGBASE="$D/$tag" CLIO_SERVER_CONF="$D/conf${n}.yaml"
  rm -f "$D"/done_*
  timeout -k 30 600 srun -N "$n" --ntasks-per-node=1 --gpus-per-node=1 \
      "$D/gvw_node.sh" "$bench" --nodes "$n" "$@" > "$LOGD/$tag.log" 2>&1
  local ec=$?
  cat "$D/$tag".node*.log >> "$LOGD/$tag.log" 2>/dev/null
  local hosts; hosts=$(grep -ho 'host=[^ ]*' "$D/$tag".node*.log 2>/dev/null | sort -u | wc -l)
  [ "$hosts" -ge 2 ] && echo "distinct_hosts=2" >> "$LOGD/$tag.log"
  return $ec
}

paged_cell() {  # paged_cell <label> <bench> <args...>
  local label="$1"; shift
  local tag="${label// /_}"
  paged_run "$tag" 2 "$@"
  verdict "$label" 'distinct_hosts=2' "$LOGD/$tag.log" $?
}

# gmx, lbann and lammps_md carry their own gates; weights self-checks with
# checksum=OK. kmeans and grayscott print ONLY a checksum, so a 2-node run
# alone cannot tell a right answer from a wrong one -- they get a 1-node
# reference and a comparison, which is what the docker harness does.
paged_cell "PAGED weights"   $BIN/clio_weights_paged_bench   --blocks 8 --pages 4
paged_cell "PAGED gmx"       $BIN/clio_gmx_paged_bench       --atoms 20000 --page-kb 32 --nvme-path /tmp/gv_tier.dat
paged_cell "PAGED lbann"     $BIN/clio_lbann_paged_bench     --blocks 4
paged_cell "PAGED lammps_md" $BIN/clio_lammps_md_paged_bench --lattice 28 --steps 20 --md

ref_cell() {  # ref_cell <label> <key> <reltol> <bench> <args...>
  local label="$1" key="$2" tol="$3"; shift 3
  local tag="${label// /_}"
  paged_run "${tag}_1n" 1 "$@"
  paged_run "${tag}_2n" 2 "$@"
  local a b
  a=$(grep -ahoE "$key=[-0-9.]+" "$LOGD/${tag}_1n.log" | head -1 | cut -d= -f2)
  b=$(grep -ahoE "$key=[-0-9.]+" "$LOGD/${tag}_2n.log" | head -1 | cut -d= -f2)
  # RELATIVE, not bitwise. Both benches reduce with atomicAdd, whose ordering
  # changes with the shard layout AND run to run at a fixed node count -- the
  # observed run-to-run spread (1.4e-08 on kmeans) exceeds the 1-vs-2-node
  # difference, so demanding bit equality reports a healthy cell as broken.
  python3 - "$label" "$a" "$b" "$tol" <<'PY' | tee -a "$SUMMARY"
import sys
label, a, b, tol = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
if not a or not b:
    print('%-24s %-5s missing %s' % (label, 'FAIL', 'value')); raise SystemExit
a, b = float(a), float(b)
rel = abs(b - a) / (abs(a) or 1.0)
ok = rel <= tol
print('%-24s %-5s 1n=%.6f 2n=%.6f rel=%.2e (tol %.0e)'
      % (label, 'PASS' if ok else 'FAIL', a, b, rel, tol))
PY
}
ref_cell "PAGED kmeans"    centroid_checksum 1e-4 $BIN/clio_kmeans_paged_bench    --blocks 8 --iters 3 --data-mb 64 --nvme-path /tmp/gv_tier.dat
ref_cell "PAGED grayscott" v_checksum        1e-6 $BIN/clio_grayscott_paged_bench --blocks 8 --steps 4 --data-mb 64 --nvme-path /tmp/gv_tier.dat

echo; echo "======================= SUMMARY ======================="
cat "$SUMMARY"
printf 'PASS %d / %d\n' "$(grep -c ' PASS' "$SUMMARY")" "$(wc -l < "$SUMMARY")"
grep -q ' FAIL' "$SUMMARY" && exit 1 || exit 0
