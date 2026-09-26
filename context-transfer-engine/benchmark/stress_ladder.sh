#!/usr/bin/env bash
# The CTE stress ladder: clio_cte_vector_stress on ONE node with co-located
# runtimes, the rung set that found defects 8-11 (RELIABILITY.md in
# adapter/gpu_vector/benchmark). No GPU needed. Every rung must exit 0 on
# every rank with zero mismatches; the exit status is the number of failed
# rungs, so this is a gate, not a report.
#
#   stress_ladder.sh <workdir> [rung ...]
#
# Rungs (default: all, ~4 minutes):
#   a_2r_light       2 ranks, 512 x 1 MB pages, batches, checkpoints      (torn read, RPC path)
#   b_4r_2gb         4 ranks, 2 GB per rank                                (barrier after checkpoint)
#   c_4r_64t         4 ranks, 64 threads, halo 2, stream 4, 4 GB per rank  (concurrency)
#   d_4r_scalar      4 ranks, scalar put/get path                          (torn read, scalar)
#   e_4r_nogen       4 ranks, non-generational gets, batches of 64
#   f_4r_small       4 ranks, 32768 x 64 KB pages                          (metadata pressure)
#   fast_1r_nogen    1 rank, 64 threads, non-generational LOCAL reads      (torn read, zero-IPC path)
#   bar_4r           4 ranks, 300 barrier-only rounds                      (gate exchange)
#   soak_4r          4 ranks, 12 steps, checkpoint every step, 2000 rounds
#
# Environment: BENCH_TIERS / CLIO_MAIN_SEGMENT_SIZE as run_colocated.sh reads
# them (defaults below suit a 512 GB node), STRESS_EXE to override the binary.
set -u
workdir=${1:?usage: stress_ladder.sh <workdir> [rung ...]}
shift
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "${here}/../.." && pwd)
bench=${root}/context-transfer-engine/adapter/gpu_vector/benchmark
exe=${STRESS_EXE:-${root}/build-fresh/bin/clio_cte_vector_stress}
export CLIO_MAIN_SEGMENT_SIZE=${CLIO_MAIN_SEGMENT_SIZE:-8G}
export BENCH_TIERS=${BENCH_TIERS:-"ram::cte_stress_dram|ram|60000MB|0.5"}
rungs=("$@")
[ ${#rungs[@]} -eq 0 ] && rungs=(a_2r_light b_4r_2gb c_4r_64t d_4r_scalar e_4r_nogen f_4r_small fast_1r_nogen bar_4r soak_4r)
failed=0

# args <rung>: rank count, per-rank cap in seconds, then the benchmark arguments.
args() {
  case "$1" in
    a_2r_light)    echo 2 300 --pages-per-node 512 --page-kb 1024 --threads 8 --steps 3 --halo 1 --stream 2 --batch 16 --ckpt-every 2 ;;
    b_4r_2gb)      echo 4 400 --pages-per-node 2048 --page-kb 1024 --threads 16 --steps 4 --halo 1 --stream 2 --batch 16 --ckpt-every 2 ;;
    c_4r_64t)      echo 4 600 --pages-per-node 4096 --page-kb 1024 --threads 64 --steps 4 --halo 2 --stream 4 --batch 16 --ckpt-every 2 ;;
    d_4r_scalar)   echo 4 400 --pages-per-node 1024 --page-kb 1024 --threads 32 --steps 3 --halo 1 --stream 2 --batch 1 --ckpt-every 3 ;;
    e_4r_nogen)    echo 4 400 --pages-per-node 2048 --page-kb 1024 --threads 32 --steps 3 --halo 1 --stream 2 --batch 64 --no-gen ;;
    f_4r_small)    echo 4 600 --pages-per-node 32768 --page-kb 64 --threads 32 --steps 3 --halo 1 --stream 2 --batch 64 --ckpt-every 3 ;;
    fast_1r_nogen) echo 1 300 --pages-per-node 2048 --page-kb 1024 --threads 64 --steps 6 --halo 2 --stream 4 --batch 1 --no-gen ;;
    bar_4r)        echo 4 300 --pages-per-node 64 --page-kb 64 --threads 4 --steps 1 --halo 1 --stream 0 --batch 1 --barriers 300 ;;
    soak_4r)       echo 4 500 --pages-per-node 512 --page-kb 1024 --threads 16 --steps 12 --halo 1 --stream 2 --batch 16 --ckpt-every 1 --barriers 2000 ;;
    *) return 1 ;;
  esac
}

rung() {
  local name=$1 n cap r rc t0=$SECONDS
  local dir="${workdir}/${name}"
  local spec
  spec=$(args "${name}") || { echo "FAIL ${name}: unknown rung"; failed=$((failed + 1)); return; }
  set -- ${spec}; n=$1; cap=$2; shift 2
  echo "=== ${name}: ${n} rank(s): $*"
  BENCH_CAP=${cap} bash "${bench}/run_colocated.sh" "${n}" "${dir}" "${exe}" "$@" --barrier-timeout 60 > "${dir}.out" 2>&1
  rc=$?
  for (( r = 0; r < n; ++r )); do
    grep -a '^STRESS \|^  barriers' "${dir}/rank${r}.log" | cut -c1-300
    grep -a '^STRESS ERROR\|timed out' "${dir}/rank${r}.log" | head -4
  done
  grep -a 'STALE' "${dir}.out" | head -1
  if [ "${rc}" -eq 0 ]; then
    echo "PASS ${name} ($((SECONDS - t0)) s)"
  else
    echo "FAIL ${name}: rc=${rc} ($((SECONDS - t0)) s; logs in ${dir})"; failed=$((failed + 1))
  fi
  pkill -KILL -f 'clio_cte_vector_stres[s]' 2>/dev/null; sleep 1
}

mkdir -p "${workdir}"
pkill -KILL -f 'clio_cte_vector_stres[s]' 2>/dev/null
for r in "${rungs[@]}"; do rung "${r}"; done
echo "stress ladder: ${failed} failed rung(s)"
exit "${failed}"
