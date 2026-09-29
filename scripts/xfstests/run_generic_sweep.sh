#!/usr/bin/env bash
# Hang-robust xfstests sweep runner for the clio FUSE filesystem.
#
# Used to discover the maximum passing set across the whole generic/* group.
# Unlike run_clio_xfstests.sh, on a per-test timeout it FORCE-unmounts (lazy)
# to release the D-state processes wedged on a stuck FUSE daemon, records the
# test as HANG, and keeps going -- some clio FUSE tests (e.g. generic/438,
# mmap+fallocate) wedge the daemon so hard that a normal `timeout` cannot reap
# the uninterruptible children and the whole run stalls. See issue #680.
#
# NOTE: this runner (like run_clio_xfstests.sh) reports a 'notrun' test as pass
# because xfstests prints "Passed all 0 tests" for it; reclassify notrun via the
# results/generic/NNN.notrun markers when computing the true pass set.
#
# Usage:
#   scripts/xfstests/run_generic_sweep.sh <testlist-file> <results-file>
# Env: CLIO_BUILD_DIR, PERTEST_TIMEOUT (default 75s).
set -u

LISTFILE="${1:?usage: run_generic_sweep.sh <testlist> <results>}"
RESULTS="${2:?usage: run_generic_sweep.sh <testlist> <results>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_BIN="${CLIO_BUILD_DIR:-${REPO_ROOT}/build}/bin"
FUSE_BIN="${BUILD_BIN}/clio_cte_fuse"
XFSTESTS_DIR="${REPO_ROOT}/external/xfstests"
TEST_DIR=/tmp/clio_xfs_test
TIMEOUT="${PERTEST_TIMEOUT:-75}"

# namespaced root so ./check runs unprivileged
if [ "$(id -u)" -ne 0 ] && [ -z "${INNS:-}" ]; then
  exec env INNS=1 unshare -rm bash "$0" "$@"
fi

mkdir -p "${TEST_DIR}"
cat > "${XFSTESTS_DIR}/local.config" <<EOF
export FSTYP=fuse
export TEST_DEV=clio_test
export TEST_DIR=${TEST_DIR}
EOF

teardown() {
  fusermount3 -uz "${TEST_DIR}" 2>/dev/null
  umount -l "${TEST_DIR}" 2>/dev/null
  pkill -9 -x clio_cte_fuse 2>/dev/null; pkill -9 -x clio_run 2>/dev/null
  # vfstest re-parents itself out of check's process group, so a hung test's
  # vfstest outlives the kill below and lingers for hours; one left on a
  # node coincided with that node's clio runtime wedging at startup.
  pkill -9 -f "${XFSTESTS_DIR}/src/vfs/vfstest" 2>/dev/null
  rm -rf "/tmp/clio_$(id -un)" /dev/shm/clio_run* 2>/dev/null
}
trap 'teardown; exit 0' EXIT

mount_fresh() {
  teardown
  # The previous daemon must be GONE before remounting: a lazily-detached
  # old mount still answered `mountpoint`, check then found no live mount
  # and failed the test on "could not mount clio_test" (a harness race that
  # failed generic/014, 075, 114, 134, 248 only inside a full sweep).
  for _ in $(seq 1 50); do
    pgrep -x clio_cte_fuse >/dev/null 2>&1 || break; sleep 0.1
  done
  sleep 0.2
  # Load the same 2g DRAM-tier config the real runner uses; otherwise the daemon
  # falls back to a ~100 MB default tier that starves large-write/O_DIRECT tests
  # with ENOSPC (a sweep-only false failure). Matches run_clio_xfstests.sh.
  CLIO_SERVER_CONF="${CLIO_SERVER_CONF:-${SCRIPT_DIR}/clio_xfstests_config.yaml}" \
    CLIO_REPO_PATH="${BUILD_BIN}" LD_LIBRARY_PATH="${BUILD_BIN}:${HOME}/.local/lib:${LD_LIBRARY_PATH:-}" \
    CLIO_WITH_RUNTIME=1 CLIO_BIND_ADDR=127.0.0.1 \
    "${FUSE_BIN}" "${TEST_DIR}" -o fsname=clio_test -f >/dev/null 2>&1 &
  for _ in $(seq 1 50); do
    if mountpoint -q "${TEST_DIR}" && touch "${TEST_DIR}/.clio_probe" 2>/dev/null &&
       rm -f "${TEST_DIR}/.clio_probe"; then
      return 0
    fi
    sleep 0.2
  done
  return 1
}

cd "${XFSTESTS_DIR}" || exit 1
n=0; total=$(grep -cE '^generic/' "${LISTFILE}")
while read -r t; do
  case "$t" in generic/*) ;; *) continue;; esac
  n=$((n+1))
  if ! mount_fresh; then echo "${t} : MOUNTFAIL" | tee -a "${RESULTS}"; continue; fi
  OUT=$(mktemp)
  # Own session/process group, so a hang can kill the WHOLE test tree: the
  # old `pkill -P` reached only check's direct children, and grandchildren
  # (e.g. vfstest) kept running against later tests' mounts, failing them.
  setsid ./check "${t}" >"${OUT}" 2>&1 &
  cpid=$!
  waited=0; hung=1
  while [ "${waited}" -lt "${TIMEOUT}" ]; do
    kill -0 "${cpid}" 2>/dev/null || { hung=0; break; }
    sleep 1; waited=$((waited+1))
  done
  if [ "${hung}" -eq 1 ]; then
    # release D-state waiters first (abort the FUSE connection so blocked
    # callers get ENOTCONN), THEN kill the check tree
    for c in $(awk -v m="${TEST_DIR}" '$5==m {split($3,a,":"); print a[2]}' /proc/self/mountinfo); do
      echo 1 > "/sys/fs/fuse/connections/${c}/abort" 2>/dev/null
    done
    fusermount3 -uz "${TEST_DIR}" 2>/dev/null; umount -l "${TEST_DIR}" 2>/dev/null
    pkill -9 -x clio_cte_fuse 2>/dev/null
    kill -9 -- "-${cpid}" 2>/dev/null; kill -9 "${cpid}" 2>/dev/null
    echo "${t} : HANG  (${n}/${total})" | tee -a "${RESULTS}"
  elif grep -q '^Passed all' "${OUT}"; then echo "${t} : pass  (${n}/${total})" | tee -a "${RESULTS}"
  elif grep -q '^Not run:' "${OUT}"; then echo "${t} : notrun  (${n}/${total})" | tee -a "${RESULTS}"
  else echo "${t} : FAIL  (${n}/${total})" | tee -a "${RESULTS}"
  fi
  rm -f "${OUT}"
done < "${LISTFILE}"
echo "[robust] done ${n} tests"
