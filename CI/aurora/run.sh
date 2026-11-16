#!/usr/bin/env bash
# Aurora clio-fs validation: the clio_fs_suite (the same tests CI/ares runs)
# on a PBS allocation, plus IO500 against DAOS.
#
#   CI/aurora/run.sh --build <build dir> [--suite NAME[,NAME...]] [--nodes 32]
#                    [--out DIR] [--baseline DIR] [--queue debug-scaling]
#                    [--time 01:00:00] [--account IOWarp] [--wait] [--resume]
#   CI/aurora/run.sh --report <out dir> [--baseline DIR]
#
# Suites: smoke, cache, posix, faults, stress, safe, perf, io500, all (= every one
# but smoke). Outside a PBS job it submits ONE JOB PER SUITE (Aurora's debug
# queues cap a job at one hour and a user at one running job, so the jobs
# run one after another) and prints the job ids; --wait blocks until they
# end and prints the report. Inside a job it runs the suite named by the
# submission. Results land in --out:
#   <out>/<leg>/{results.jsonl,summary.md,suite.log,logs/}
#   <out>/io500/{cliofs,daos}-<run>.out, <out>/io500/io500.md (--repeat N
#   runs of each, alternating; the table reports mean, range and spread)
#   <out>/report.md   (CI/ares/compare.py: failures, regressions vs --baseline)
# A leg that runs out of walltime records the tests it did not start as
# NOTRUN; submit the same command with --out <same dir> --resume to go on.
# See CI/aurora/README.md.
set -uo pipefail

# PBS runs a COPY of this script from its spool directory, so the job learns
# where the checkout is from CLIO_AURORA_HERE (set at submission).
HERE="${CLIO_AURORA_HERE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO="$(cd "${HERE}/../.." && pwd)"
SUITE_DIR="${REPO}/context-transfer-engine/test/integration/clio_fs_suite"
IO500_DIR="${REPO}/benchmarks/io500"

BUILD="" SUITE="all" NODES=32 OUT="" BASELINE="" QUEUE="debug-scaling"
REPEAT="${IO500_REPEAT:-1}"
TIME="01:00:00" ACCOUNT="${CLIO_AURORA_ACCOUNT:-IOWarp}" THRESHOLD="0.20"
WAIT=0 RESUME=0 REPORT="" REL_DISK_GB=13
FILESYSTEMS="${CLIO_AURORA_FILESYSTEMS:-home:flare:daos_user}"
LOCAL_ROOT="/tmp/${USER}/clio_fs_suite"
while [ $# -gt 0 ]; do
  case "$1" in
    --build) BUILD="$2"; shift 2 ;;
    --suite) SUITE="$2"; shift 2 ;;
    --nodes) NODES="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --baseline) BASELINE="$2"; shift 2 ;;
    --queue) QUEUE="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --rel-disk-gb) REL_DISK_GB="$2"; shift 2 ;;
    --local-root) LOCAL_ROOT="$2"; shift 2 ;;
    --wait) WAIT=1; shift ;;
    --resume) RESUME=1; shift ;;
    --report) REPORT="$2"; shift 2 ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# Write <out>/report.md (and the IO500 table when both runs are there).
report() {  # <out dir>
  local out="$1" rc=0
  local cmp_args=("${out}" --threshold "${THRESHOLD}" --title Aurora)
  [ -n "${BASELINE}" ] && cmp_args+=(--baseline "${BASELINE}")
  if ls "${out}"/*/results.jsonl >/dev/null 2>&1; then
    python3 -B "${HERE}/../ares/compare.py" "${cmp_args[@]}" || rc=1
  fi
  if ls "${out}"/io500/*.out >/dev/null 2>&1; then
    python3 -B "${IO500_DIR}/compare_io500.py" "${out}/io500" \
        --out "${out}/io500/io500.md" || rc=1
  fi
  return ${rc}
}

# Inside a job the settings come from the submission (qsub -v), not argv.
BUILD="${CLIO_AURORA_BUILD:-${BUILD}}"; SUITE="${CLIO_AURORA_SUITE:-${SUITE}}"
NODES="${CLIO_AURORA_NODES:-${NODES}}"; OUT="${CLIO_AURORA_OUT:-${OUT}}"
TIME="${CLIO_AURORA_TIME:-${TIME}}"; RESUME="${CLIO_AURORA_RESUME:-${RESUME}}"
REL_DISK_GB="${CLIO_AURORA_REL_DISK_GB:-${REL_DISK_GB}}"
LOCAL_ROOT="${CLIO_AURORA_LOCAL_ROOT:-${LOCAL_ROOT}}"
if [ -n "${REPORT}" ]; then
  report "$(cd "${REPORT}" && pwd)"; exit $?
fi

[ -n "${BUILD}" ] || { echo "--build <build dir> is required" >&2; exit 2; }
BUILD="$(cd "${BUILD}" && pwd)" || exit 2
[ -x "${BUILD}/bin/clio_run" ] || { echo "${BUILD}/bin/clio_run missing: build first" >&2; exit 2; }
[ -x "${BUILD}/bin/clio_cte_fuse" ] || { echo "${BUILD}/bin/clio_cte_fuse missing (CLIO_CTE_ENABLE_FUSE_ADAPTER)" >&2; exit 2; }
SHA="$(git -C "${REPO}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
[ -n "${OUT}" ] || OUT="${HOME}/clio_aurora_runs/$(date +%Y%m%d-%H%M%S)-${SHA}-${NODES}n"
mkdir -p "${OUT}"
OUT="$(cd "${OUT}" && pwd)"
[ "${SUITE}" = all ] && SUITE="posix,faults,stress,safe,perf,io500"

# Seconds in an HH:MM:SS walltime.
secs_of() { IFS=: read -r h m s <<< "$1"; echo $(( 10#$h * 3600 + 10#$m * 60 + 10#$s )); }

# ---- not inside a PBS job: submit one job per suite ------------------------
if [ -z "${PBS_JOBID:-}" ]; then
  # agent.py and the driver need Python 3.7+; the compute image's
  # /usr/bin/python3 is 3.6. Use the python3 on PATH here (a module, on a
  # filesystem the compute nodes mount) unless CLIO_SUITE_PYTHON names one.
  PY="${CLIO_SUITE_PYTHON:-$(command -v python3)}"
  "${PY}" -c 'import sys; sys.exit(sys.version_info < (3, 7))' \
    || { echo "${PY} is older than 3.7: set CLIO_SUITE_PYTHON" >&2; exit 2; }
  # patchelf lets the suite re-point its binary snapshot at itself; without
  # it the snapshot still loads libraries from the build directory.
  TOOLS="$(dirname "$(command -v patchelf 2>/dev/null || echo /nonexistent/x)")"
  # qsub, retried while the queue refuses another queued job from this user
  # (debug-scaling allows one): the next suite goes in once the previous
  # one has started. Prints the job id.
  submit() {
    local err
    while :; do
      err="$(qsub "$@" 2>&1)" && { echo "${err}"; return 0; }
      case "${err}" in
        *"per-user limit"*) sleep 60 ;;
        *) echo "${err}" >&2; return 1 ;;
      esac
    done
  }
  jobs=()
  for s in ${SUITE//,/ }; do
    if [ "${s}" = io500 ]; then
      mkdir -p "${OUT}/io500"
      # Same work per rank for both systems; see benchmarks/io500. mdtest is
      # 125 files per rank because DAOS removes ~0.5 kIOPS through this path:
      # more would spend the hour in the two delete phases.
      common="BUILD_DIR=${BUILD},HERE=${IO500_DIR},SYNC_FS_ONLY=1,RUN_TAG=ci"
      common+=",EASY_MB=${IO500_EASY_MB:-768},HARD_SEG=${IO500_HARD_SEG:-400},MD_N=${IO500_MD_N:-125}"
      # The measured clio-fs configuration (needs a build with
      # CLIO_CORE_ENABLE_THALLIUM=ON and CLIO_ENABLE_DAOS=ON): DRAM + a
      # native DAOS tier, peers over Thallium on the Slingshot fabric.
      # IO500_CLIOFS_VARS appends to (and so overrides) these.
      tuned="NO_FLARE=1,DAOS_NATIVE=1,DAOS_NATIVE_OPTS=prezero=0,META_RAM_MB=512"
      tuned+=",CLIO_FUSE_MAX_BACKGROUND=256,CLIO_FUSE_READAHEAD_PAGES=4"
      tuned+=",CLIO_FUSE_ATTR_CACHE_S=1,CLIO_FUSE_DIR_ATTR_CACHE_S=1,CLIO_PREFAULT=0"
      tuned+=",CLIO_NET_SEND_THREADS=4,CLIO_NET_TIER_LANES=4,CLIO_PEER_TRANSPORT=thallium"
      tuned+=",CLIO_LBM_THALLIUM_PROTOCOL=ofi+cxi,CLIO_LBM_THALLIUM_LISTEN=cxi0"
      tuned+=",SETTLE=30,UP_WAIT=600,FUSE_PROFILE=0,HANG_DUMP_S=1200"
      # --repeat N: N runs of each system, alternating, so that both see the
      # same spread of nodes and of load on the shared DAOS pool over time.
      for r in $(seq 1 "${REPEAT}"); do
        for mode in cliofs daos; do
          extra="MODE=${mode}"
          [ "${mode}" = cliofs ] && extra+=",${tuned}${IO500_CLIOFS_VARS:+,${IO500_CLIOFS_VARS}}"
          [ "${mode}" = daos ] && extra+="${IO500_DAOS_VARS:+,${IO500_DAOS_VARS}}"
          jid="$(submit -q "${QUEUE}" -A "${ACCOUNT}" -l select="${NODES}" \
                   -l walltime="${TIME}" -l filesystems="${FILESYSTEMS}" \
                   -N "clio-io500-${mode}" -o "${OUT}/io500/${mode}-${r}.out" \
                   -v "${common},${extra}" "${IO500_DIR}/pbs_io500_aurora.sh")" || exit 3
          echo "submitted io500/${mode}-${r}: ${jid}"
          jobs+=("${jid}")
        done
      done
      continue
    fi
    jid="$(submit -q "${QUEUE}" -A "${ACCOUNT}" -l select="${NODES}" -l place=scatter \
             -l walltime="${TIME}" -l filesystems="${FILESYSTEMS}" -j oe \
             -N "clio-ci-${s}" -o "${OUT}/pbs-${s}.out" \
             -v "CLIO_AURORA_HERE=${HERE},CLIO_AURORA_BUILD=${BUILD},CLIO_AURORA_SUITE=${s},CLIO_AURORA_NODES=${NODES},CLIO_AURORA_OUT=${OUT},CLIO_AURORA_TIME=${TIME},CLIO_AURORA_RESUME=${RESUME},CLIO_AURORA_REL_DISK_GB=${REL_DISK_GB},CLIO_AURORA_LOCAL_ROOT=${LOCAL_ROOT},CLIO_SUITE_PYTHON=${PY},CLIO_AURORA_TOOLS=${TOOLS}" \
             "${HERE}/run.sh")" || exit 3
    echo "submitted ${s}: ${jid}"
    jobs+=("${jid}")
  done
  printf '%s\n' "${jobs[@]}" >> "${OUT}/jobs.txt"
  echo "results: ${OUT}"
  if [ "${WAIT}" != 1 ]; then
    echo "when the jobs have ended:  ${HERE}/run.sh --report ${OUT}"
    exit 0
  fi
  while :; do
    left=0
    for j in "${jobs[@]}"; do
      st="$(qstat -f "${j}" 2>/dev/null | sed -n 's/^ *job_state = //p')"
      [ -n "${st}" ] && [ "${st}" != F ] && left=$(( left + 1 ))
    done
    [ "${left}" = 0 ] && break
    sleep 60
  done
  report "${OUT}"; rc=$?
  [ -f "${OUT}/report.md" ] && sed -n '1,40p' "${OUT}/report.md"
  [ -f "${OUT}/io500/io500.md" ] && cat "${OUT}/io500/io500.md"
  exit ${rc}
fi

# ---- inside a PBS job ------------------------------------------------------
T_START=${SECONDS}
HOSTS="$(awk '!seen[$0]++' "${PBS_NODEFILE}" | head -n "${NODES}" | paste -sd,)"
NHOSTS="$(echo "${HOSTS}" | tr ',' '\n' | wc -l)"
echo "nodes (${NHOSTS}): ${HOSTS}"
echo "commit $(git -C "${REPO}" rev-parse --short HEAD) build ${BUILD}"

[ -d "${CLIO_AURORA_TOOLS:-/nonexistent}" ] && export PATH="${CLIO_AURORA_TOOLS}:${PATH}"
PY="${CLIO_SUITE_PYTHON:-$(command -v python3)}"
export CLIO_SUITE_PYTHON="${PY}"
# The daemons need the compiler's runtime libraries, which come from a module
# here (no RUNPATH entry): the build's compiler lib dir, then the job's own.
CXX="$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' "${BUILD}/CMakeCache.txt")"
CXX_LIB="$(cd "$(dirname "${CXX}")/../lib" 2>/dev/null && pwd)"
export CLIO_SUITE_LD_LIBRARY_PATH="${CXX_LIB:+${CXX_LIB}:}${LD_LIBRARY_PATH:-}"
# bdev perf statistics go to the shared home from a runtime worker; under
# load that write stalled workers for seconds and peers declared the node
# dead (seen in the IO500 runs here).
export CLIO_SUITE_PASS_CLIO_BDEV_PERSIST_STATS=0

# Aurora compute nodes have no local drive: every tier, log and mountpoint
# lives in /tmp (tmpfs, node-local, counted against the node's memory). The
# tiers are separate CTE targets with different scores, so tiering,
# flushing and restart recovery run as on real devices; only the device
# speed is not representative. One leg = "name|profile|groups|suite args".
REL="--disk-gb ${REL_DISK_GB}"
case "${SUITE}" in
  smoke)  LEGS=("smoke|persistent|posix,dist|${REL} --only rw_sizes,rename_file,append_mode,xnode_create_visible,xnode_shared_file_disjoint,xnode_unlink_rename_visible") ;;
  posix)  LEGS=("posix_dist|persistent|posix,dist,dirs,apps,cache|${REL}") ;;
  cache)  LEGS=("cache|persistent|cache|${REL}") ;;
  faults) LEGS=("node_faults|persistent|fault,repl|${REL}") ;;
  stress) LEGS=("stress_tiers|tiered|stress,cache|--ram-mb 512 --fast-mb 1024 ${REL} --probe-silence-s 30") ;;
  safe)   LEGS=("safe_bdev|safe|safe|--ram-mb 512 --fast-mb 1024 ${REL}") ;;
  perf)   LEGS=("perf|tiered|tier,perf|--ram-mb 2048 --fast-mb 4096 --disk-gb 32 --probe-silence-s 30") ;;
  *) echo "unknown suite ${SUITE}" >&2; exit 2 ;;
esac

"${PY}" - "${OUT}/meta-${SUITE}.json" "${SHA}" "${HOSTS}" "${SUITE}" "${PBS_JOBID}" <<'EOF'
import json, sys, time
json.dump({'sha': sys.argv[2], 'hosts': sys.argv[3].split(','),
           'suite': sys.argv[4], 'job': sys.argv[5],
           'start': time.strftime('%F %T')}, open(sys.argv[1], 'w'), indent=1)
EOF

# Leave MARGIN_S of the walltime for teardown and the report.
MARGIN_S=420
WALL_S="$(secs_of "${TIME}")"
overall=0
for leg in "${LEGS[@]}"; do
  IFS='|' read -r name profile groups extra <<< "${leg}"
  budget=$(( WALL_S - MARGIN_S - (SECONDS - T_START) ))
  [ "${budget}" -gt 60 ] || { echo "== leg ${name}: no walltime left"; overall=1; continue; }
  echo "== leg ${name}: profile=${profile} groups=${groups} ${extra} (budget ${budget} s)"
  resume=()
  [ "${RESUME}" = 1 ] && resume=(--resume)
  # shellcheck disable=SC2086
  "${PY}" -B "${SUITE_DIR}/suite.py" --bin "${BUILD}/bin" --out "${OUT}/${name}" \
      --hosts "${HOSTS}" --net-suffix "" --local-root "${LOCAL_ROOT}" \
      --profile "${profile}" --groups "${groups}" --budget-s "${budget}" \
      "${resume[@]}" ${extra} || overall=1
done
PATH="$(dirname "${PY}"):${PATH}" report "${OUT}" || overall=1
echo "results: ${OUT}  (report: ${OUT}/report.md)"
exit ${overall}
