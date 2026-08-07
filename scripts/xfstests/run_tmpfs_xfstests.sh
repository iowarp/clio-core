#!/usr/bin/env bash
#
# Native (tmpfs) xfstests baseline, for comparison against the clio FUSE run
# (scripts/xfstests/run_clio_xfstests.sh). Mounts a fresh tmpfs at
# $HOME/xfstests/{test,scratch} inside a user+mount namespace (no sudo, nothing
# on real disks is touched), then runs a test list and records per-test timing
# in results/check.time -- the same format the fuse run produced.
#
# Usage:
#   scripts/xfstests/run_tmpfs_xfstests.sh [test-list...]
#   TESTS="generic/001 generic/002" scripts/xfstests/run_tmpfs_xfstests.sh
#
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -z "${XFSTESTS_DIR:-}" ]; then
  for cand in /opt/xfstests "${REPO_ROOT}/external/xfstests"; do
    [ -x "${cand}/check" ] && XFSTESTS_DIR="${cand}" && break
  done
fi
[ -x "${XFSTESTS_DIR:-/nonexistent}/check" ] || { echo "ERROR: xfstests not found"; exit 1; }

# --- mountpoints (under $HOME/xfstests, as requested) -----------------------
BASE="${TMPFS_XFS_BASE:-${HOME}/xfstests}"
TEST_DIR="${BASE}/test"
SCRATCH_MNT="${BASE}/scratch"
mkdir -p "${TEST_DIR}" "${SCRATCH_MNT}"

# --- namespaced root (so we can mount tmpfs without sudo) -------------------
if [ "$(id -u)" -ne 0 ] && [ -z "${TMPFS_XFS_INNS:-}" ]; then
  echo "[tmpfs-xfs] not root; re-executing under 'unshare -rm' (namespaced root)"
  exec env TMPFS_XFS_INNS=1 unshare -rm "$0" "$@"
fi

# Inside the namespace: mount tmpfs for TEST_DIR and SCRATCH_MNT. tmpfs ignores
# the mount source, so distinct TEST_DEV/SCRATCH_DEV labels are just for ./check
# bookkeeping. ./check (re)mounts/unmounts these itself across tests.
mount -t tmpfs tmpfs_test    "${TEST_DIR}"
mount -t tmpfs tmpfs_scratch "${SCRATCH_MNT}"

cat > "${XFSTESTS_DIR}/local.config" <<EOF
export FSTYP=tmpfs
export TEST_DEV=tmpfs_test
export TEST_DIR=${TEST_DIR}
export SCRATCH_DEV=tmpfs_scratch
export SCRATCH_MNT=${SCRATCH_MNT}
EOF

cd "${XFSTESTS_DIR}" || exit 1

# --- test list: default to exactly what the fuse run executed ---------------
if [ "$#" -gt 0 ]; then RAW=("$@")
elif [ -n "${TESTS:-}" ]; then RAW=(${TESTS})  # shellcheck disable=SC2206
else
  RAW=(generic/001 generic/002 generic/006 generic/007 generic/011 generic/013
       generic/014 generic/028 generic/075 generic/091 generic/112 generic/113
       generic/131 generic/198 generic/207 generic/210 generic/212 generic/236
       generic/245 generic/246 generic/247 generic/248 generic/249 generic/257
       generic/263 generic/308 generic/362 generic/364 generic/428 generic/430
       generic/431 generic/432 generic/433 generic/437 generic/443 generic/448
       generic/451 generic/465 generic/471 generic/478 generic/490 generic/504
       generic/532 generic/571 generic/591 generic/609 generic/632 generic/637
       generic/638 generic/676 generic/706 generic/708 generic/736 generic/755
       generic/763)
fi

echo "[tmpfs-xfs] running ${#RAW[@]} test(s) on tmpfs at ${TEST_DIR}"
./check "${RAW[@]}"
echo "[tmpfs-xfs] done. timings in ${XFSTESTS_DIR}/results/check.time"
