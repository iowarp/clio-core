#!/usr/bin/env bash
# The distributed-failure reproducers as a test suite, on ONE node with
# co-located runtimes. Each case states what must be in the logs; a fix that
# regresses turns a case red, and the "control" case keeps the reproducer
# honest by showing the defect still appears when the fix is disabled.
#
#   bench_colocated_tests.sh <workdir> [case ...]
#
# Cases (default: all): smoke, range_split, targets_race, targets_race_control,
# peer_death, probe_ooc. Needs the kmeans benchmark binary
# (BENCH_KMEANS, default build-spike/clio_kmeans_paged_newcoro_aot_x_ckpt2)
# and a GPU node with at least 4 tiles. Exit status: number of failed cases.
set -u
workdir=${1:?usage: bench_colocated_tests.sh <workdir> [case ...]}
shift
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "${here}/../../../.." && pwd)
km=${BENCH_KMEANS:-${root}/build-spike/clio_kmeans_paged_newcoro_aot_x_ckpt2}
export IGC_FunctionControl=3
cases=("$@"); [ ${#cases[@]} -eq 0 ] && cases=(smoke range_split targets_race targets_race_control peer_death probe_ooc)
failed=0

# strip <log>: the log without colour codes and source prefixes.
strip() { sed -E 's/\x1b\[[0-9;]*m//g; s/^.*src\///' "$1"; }
# ranks <n> <dir> <extra env...>: run kmeans on n co-located ranks.
run() {  # run <n> <dir> <kmeans args...>
  local n=$1 dir=$2; shift 2
  bash "${here}/run_colocated.sh" "${n}" "${dir}" "${km}" "$@" > "${dir}.out" 2>&1
}
# expect <case> <condition text> <shell test...>
expect() {
  local name=$1 what=$2; shift 2
  if "$@"; then echo "ok   ${name}: ${what}"; else echo "FAIL ${name}: ${what}"; failed=$(( failed + 1 )); fi
}
checksums_agree() {  # checksums_agree <dir> <n>
  local dir=$1 n=$2 r c0 c
  c0=$(grep -aoE 'centroid_checksum=[-0-9.]+' "${dir}/rank0.log" | tail -1)
  [ -n "${c0}" ] || return 1
  for (( r = 1; r < n; ++r )); do
    c=$(grep -aoE 'centroid_checksum=[-0-9.]+' "${dir}/rank${r}.log" | tail -1)
    [ "${c}" = "${c0}" ] || return 1
  done
}
small="--data-mb 4096 --iters 2 --page-kb 1024 --blocks 256 --threads 256 --slots 16 --publish-seed --repeat 1"

case_smoke() {
  local d="${workdir}/smoke"
  BENCH_CAP=200 run 2 "${d}" ${small}
  expect smoke "both ranks exit 0" grep -q 'worst exit=0' "${d}.out"
  expect smoke "checksums agree" checksums_agree "${d}" 2
}
case_range_split() {
  local d="${workdir}/range_split" r
  BENCH_NEIGHBORHOOD=2 BENCH_CAP=200 run 4 "${d}" --data-mb 8192 --iters 2 --page-kb 1024 --blocks 256 --threads 256 --slots 16 --publish-seed --repeat 1
  expect range_split "4 ranks exit 0 with neighborhood_size 2" grep -q 'worst exit=0' "${d}.out"
  for r in 0 1 2 3; do
    expect range_split "rank ${r} registered its target" bash -c "strip() { sed -E 's/\x1b\[[0-9;]*m//g' \"\$1\"; }; strip '${d}/rank${r}.log' | grep -q '[1-9][0-9]* storage target(s) registered'"
  done
}
case_targets_race() {
  local d="${workdir}/targets_race"
  BENCH_RANK_ENV_1="CLIO_CTE_REGISTER_DELAY_MS=6000" BENCH_CAP=200 run 2 "${d}" ${small}
  expect targets_race "peer's puts waited for Create" grep -q 'waiting for Create to finish' "${d}/rank1.log"
  expect targets_race "both ranks exit 0" grep -q 'worst exit=0' "${d}.out"
  expect targets_race "checksums agree" checksums_agree "${d}" 2
}
case_targets_race_control() {
  local d="${workdir}/targets_race_control"
  CLIO_CTE_TARGETS_READY_WAIT_MS=0 BENCH_RANK_ENV_1="CLIO_CTE_REGISTER_DELAY_MS=6000" BENCH_CAP=200 run 2 "${d}" ${small}
  # The owner's ERROR line is the deterministic evidence; the GPU-side FATAL
  # print in rank 0 can be cut off by the abort that follows it.
  expect targets_race_control "without the wait the owner refuses the put (rc 11)" grep -aq 'is refused (clients see rc 11)' "${d}/rank1.log"
  expect targets_race_control "and the writer aborts" grep -q '^rank 0 on .* exit=134' "${d}/rank0.log"
}
case_peer_death() {
  local d="${workdir}/peer_death"
  BENCH_NO_FAILFAST=1 BENCH_PROGRESS_MS=5000 BENCH_RANK_KILL_1=10 BENCH_CAP=150 run 2 "${d}" --data-mb 4096 --iters 800 --page-kb 1024 --blocks 256 --threads 256 --slots 16 --publish-seed --repeat 1
  expect peer_death "survivor marked the dead peer dead" bash -c "grep -aq 'marking it dead' '${d}/rank0.log'"
  expect peer_death "survivor failed before its cap (no hang)" bash -c "! grep -q '^rank 0 on .* exit=124' '${d}/rank0.log'"
}
case_probe_ooc() {
  local d="${workdir}/probe_ooc"
  BENCH_PROGRESS_MS=1000 BENCH_CAP=240 run 2 "${d}" --data-mb 8192 --iters 3 --page-kb 1024 --blocks 256 --threads 256 --slots 2 --publish-seed --repeat 1
  expect probe_ooc "both ranks exit 0" grep -q 'worst exit=0' "${d}.out"
  expect probe_ooc "evictions happened (really out of core)" bash -c "grep -aoE 'evicts=[1-9][0-9]*' '${d}/rank0.log' | grep -q ."
  expect probe_ooc "no replica declared Gone" bash -c "! grep -aq 'Gone on its node' '${d}/rank0.log' '${d}/rank1.log'"
}

mkdir -p "${workdir}"
for c in "${cases[@]}"; do
  echo "=== ${c}"
  "case_${c}"
done
echo "colocated tests: ${failed} failure(s)"
exit "${failed}"
