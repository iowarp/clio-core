#!/usr/bin/env bash
#
# Drive xfstests against the clio filesystem via the libfuse adapter
# (clio_cte_fuse), which delegates to the context-filesystem chimod.
#
# This is meant as the guiding conformance driver for the clio filesystem:
# it mounts the FUSE adapter, points xfstests' TEST_DIR at the mount, and runs
# a (configurable) set of generic tests, reporting where they pass/fail.
#
# Prereqs (see scripts/xfstests/README.md):
#   - xfstests built (XFSTESTS_DIR, default /opt/xfstests or
#     <repo>/external/xfstests). Build needs: uuid-dev libattr1-dev
#     libacl1-dev libaio-dev (+ xfsprogs e2fsprogs attr acl quota at runtime).
#   - clio built (clio_cte_fuse in <build>/bin); fusermount3 available.
#
# Usage:
#   scripts/xfstests/run_clio_xfstests.sh [test-list...]
#   TESTS="generic/001 generic/002" scripts/xfstests/run_clio_xfstests.sh
#   scripts/xfstests/run_clio_xfstests.sh -g quick      # pass-through to ./check
#
set -u

# --- locate the repo + build ------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_BIN="${CLIO_BUILD_DIR:-${REPO_ROOT}/build}/bin"
FUSE_BIN="${BUILD_BIN}/clio_cte_fuse"

# --- locate xfstests --------------------------------------------------------
if [ -z "${XFSTESTS_DIR:-}" ]; then
  for cand in /opt/xfstests "${REPO_ROOT}/external/xfstests"; do
    [ -x "${cand}/check" ] && XFSTESTS_DIR="${cand}" && break
  done
fi
if [ -z "${XFSTESTS_DIR:-}" ] || [ ! -x "${XFSTESTS_DIR}/check" ]; then
  echo "ERROR: xfstests not found/built. Set XFSTESTS_DIR or build it:" >&2
  echo "  sudo apt-get install -y uuid-dev libattr1-dev libacl1-dev libaio-dev \\" >&2
  echo "       xfsprogs e2fsprogs attr acl quota" >&2
  echo "  git clone --depth 1 https://git.kernel.org/pub/scm/fs/xfs/xfstests-dev.git \\" >&2
  echo "       ${REPO_ROOT}/external/xfstests && make -C ${REPO_ROOT}/external/xfstests" >&2
  exit 1
fi

if [ ! -x "${FUSE_BIN}" ]; then
  echo "ERROR: clio_cte_fuse not found at ${FUSE_BIN}" >&2
  echo "  build it: cmake --build <build> --target clio_cte_fuse" >&2
  exit 1
fi
command -v fusermount3 >/dev/null || { echo "ERROR: fusermount3 not installed (fuse3)"; exit 1; }

# --- mountpoints ------------------------------------------------------------
TEST_DIR="${CLIO_XFS_TEST_DIR:-/tmp/clio_xfs_test}"
SCRATCH_MNT="${CLIO_XFS_SCRATCH_MNT:-/tmp/clio_xfs_scratch}"
mkdir -p "${TEST_DIR}" "${SCRATCH_MNT}"

FUSE_PIDS=()
cleanup() {
  echo "[xfstests] cleaning up mounts..."
  fusermount3 -u "${TEST_DIR}" 2>/dev/null
  fusermount3 -u "${SCRATCH_MNT}" 2>/dev/null
  for p in "${FUSE_PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done
  pkill -x clio_run 2>/dev/null
}
trap cleanup EXIT

mount_fuse() {  # $1 = mountpoint, $2 = with_runtime(1/0)
  local mnt="$1" with_rt="$2"
  echo "[xfstests] mounting clio FUSE at ${mnt} (runtime=${with_rt})"
  CHI_REPO_PATH="${BUILD_BIN}" LD_LIBRARY_PATH="${BUILD_BIN}:${HOME}/.local/lib:${LD_LIBRARY_PATH:-}" \
    CLIO_WITH_RUNTIME="${with_rt}" CLIO_BIND_ADDR=127.0.0.1 \
    "${FUSE_BIN}" "${mnt}" -f &
  FUSE_PIDS+=("$!")
  # wait for the mount to appear
  for _ in $(seq 1 50); do
    mountpoint -q "${mnt}" && return 0
    sleep 0.2
  done
  echo "ERROR: clio FUSE did not mount at ${mnt}" >&2
  return 1
}

# TEST mount brings up the co-located runtime; SCRATCH mount joins as a client.
mount_fuse "${TEST_DIR}" 1 || exit 1
mount_fuse "${SCRATCH_MNT}" 0 || echo "[xfstests] warning: scratch mount failed (SCRATCH tests will notrun)"

# --- xfstests config --------------------------------------------------------
# A from-scratch FUSE fs has no mkfs/mount-by-device, so we present the already
# mounted dirs and treat FSTYP generically. SCRATCH-reformatting tests will
# 'notrun'; TEST_DIR-only tests execute.
export FSTYP="${FSTYP:-fuse}"
export TEST_DEV="${TEST_DEV:-clio_fuse_test}"
export TEST_DIR
export SCRATCH_DEV="${SCRATCH_DEV:-clio_fuse_scratch}"
export SCRATCH_MNT
cat > "${XFSTESTS_DIR}/local.config" <<EOF
export FSTYP=${FSTYP}
export TEST_DEV=${TEST_DEV}
export TEST_DIR=${TEST_DIR}
export SCRATCH_DEV=${SCRATCH_DEV}
export SCRATCH_MNT=${SCRATCH_MNT}
EOF
echo "[xfstests] local.config:"; cat "${XFSTESTS_DIR}/local.config"

# --- run --------------------------------------------------------------------
cd "${XFSTESTS_DIR}" || exit 1
if [ "$#" -gt 0 ]; then
  ARGS=("$@")
elif [ -n "${TESTS:-}" ]; then
  # shellcheck disable=SC2206
  ARGS=(${TESTS})
else
  # A small, file-data-focused default set: open/write/read/seek/truncate,
  # the exact surface the chimod implements. Expand as the fs matures.
  ARGS=(generic/001 generic/002 generic/005 generic/007 generic/011
        generic/013 generic/124 generic/130)
fi

echo "[xfstests] running: ./check ${ARGS[*]}"
./check "${ARGS[@]}"
rc=$?
echo "[xfstests] ./check exit=${rc}"
exit "${rc}"
