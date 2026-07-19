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

# Give the embedded runtime a larger DRAM tier than its ~100 MB default so
# large-write tests (fsx/fstest, generic/074, generic/091, ...) aren't starved
# by ENOSPC. The config is compose-only (no networking), so CLIO_PORT and other
# env still win. Override by exporting CLIO_SERVER_CONF before invoking.
export CLIO_SERVER_CONF="${CLIO_SERVER_CONF:-${SCRIPT_DIR}/clio_xfstests_config.yaml}"

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

mount_fuse() {  # $1 = mountpoint, $2 = with_runtime(1/0), $3 = fsname (mount source)
  local mnt="$1" with_rt="$2" fsname="$3"
  echo "[xfstests] mounting clio FUSE at ${mnt} (runtime=${with_rt}, fsname=${fsname})"
  # Distinct -o fsname per mount so xfstests can tell TEST_DEV and SCRATCH_DEV
  # apart: _check_mounted_on uses `findmnt -S <dev>` and aborts if a device
  # appears mounted on more than one target. (Both mounts otherwise share the
  # source "clio_cte_fuse".)
  CLIO_REPO_PATH="${BUILD_BIN}" LD_LIBRARY_PATH="${BUILD_BIN}:${HOME}/.local/lib:${LD_LIBRARY_PATH:-}" \
    CLIO_WITH_RUNTIME="${with_rt}" CLIO_BIND_ADDR=127.0.0.1 \
    "${FUSE_BIN}" "${mnt}" -o "fsname=${fsname}" -f &
  FUSE_PIDS+=("$!")
  # wait for the mount to appear
  for _ in $(seq 1 50); do
    mountpoint -q "${mnt}" && return 0
    sleep 0.2
  done
  echo "ERROR: clio FUSE did not mount at ${mnt}" >&2
  return 1
}

# --- root requirement -------------------------------------------------------
# xfstests' ./check refuses to run unless euid==0. Re-exec under a user+mount
# namespace ('unshare -r' maps the caller to root) so this works unprivileged;
# a real-root invocation skips this.
if [ "$(id -u)" -ne 0 ] && [ -z "${CLIO_XFS_INNS:-}" ]; then
  echo "[xfstests] not root; re-executing under 'unshare -rm' (namespaced root)"
  exec env CLIO_XFS_INNS=1 unshare -rm "$0" "$@"
fi

# --- xfstests config --------------------------------------------------------
# SCRATCH_DEV is intentionally left UNSET: the clio fs can't be reformatted
# (no mkfs/mount-by-device), and if SCRATCH_DEV is set ./check tries to mkfs it
# during setup and aborts. With it unset, TEST_DIR tests run and
# scratch-reformatting tests 'notrun'. A distinct -o fsname lets ./check's
# device->mount detection (df / findmnt -S) recognize the already-mounted fs.
export FSTYP="${FSTYP:-fuse}"
TEST_DEV="${TEST_DEV:-clio_test}"
write_config() {
  cat > "${XFSTESTS_DIR}/local.config" <<EOF
export FSTYP=${FSTYP}
export TEST_DEV=${TEST_DEV}
export TEST_DIR=${TEST_DIR}
EOF
}
write_config

# (Re)mount the clio FUSE test fs fresh. ./check unmounts TEST_DIR after each
# test and our FUSE fs can't be remounted by ./check itself, so every test gets
# its own fresh mount + runtime.
remount() {
  pkill -9 -x clio_cte_fuse 2>/dev/null; pkill -9 -x clio_run 2>/dev/null
  fusermount3 -u "${TEST_DIR}" 2>/dev/null
  rm -rf "/tmp/clio_$(id -un)" /dev/shm/clio_run* 2>/dev/null
  sleep 0.5
  mount_fuse "${TEST_DIR}" 1 "${TEST_DEV}"
}

# --- resolve test list (expands groups like '-g quick' via ./check -n) ------
cd "${XFSTESTS_DIR}" || exit 1
if [ "$#" -gt 0 ]; then RAW=("$@")
elif [ -n "${TESTS:-}" ]; then RAW=(${TESTS})  # shellcheck disable=SC2206
else RAW=(generic/001 generic/002 generic/005 generic/007 generic/011
          generic/013 generic/124 generic/130); fi

remount >/dev/null 2>&1 || { echo "ERROR: initial clio FUSE mount failed" >&2; exit 1; }
mapfile -t LIST < <(./check -n "${RAW[@]}" 2>/dev/null | grep -E '^[a-z_]+/[0-9]+')
[ "${#LIST[@]}" -eq 0 ] && LIST=("${RAW[@]}")
echo "[xfstests] running ${#LIST[@]} test(s) (mount-per-test)"

# --- run, one fresh mount per test ------------------------------------------
pass=0; fail=0; notrun=0; hang=0; failed_list=""
for t in "${LIST[@]}"; do
  remount >/dev/null 2>&1 || { echo "${t}: MOUNTFAIL"; fail=$((fail+1)); failed_list="${failed_list} ${t}"; continue; }
  out=$(timeout "${CLIO_XFS_PERTEST_TIMEOUT:-90}" ./check "${t}" 2>/dev/null)
  rc=$?
  # NOTE: check "Not run:" BEFORE "Passed all". ./check prints BOTH
  # "Not run: <t>" AND "Passed all 0 tests" when the only test notruns, so
  # matching "^Passed all" first miscounts a notrun test as a pass -- which
  # would silently admit a scratch/hardware-requiring test into the CI gate
  # where it tests nothing. "Not run:" is the authoritative signal.
  if [ "${rc}" -eq 124 ]; then echo "${t}: HANG"; hang=$((hang+1)); failed_list="${failed_list} ${t}(hang)"
  elif echo "${out}" | grep -q "^Not run:"; then echo "${t}: notrun"; notrun=$((notrun+1))
  elif echo "${out}" | grep -q "^Passed all"; then echo "${t}: pass"; pass=$((pass+1))
  else echo "${t}: FAIL"; fail=$((fail+1)); failed_list="${failed_list} ${t}"; fi
done

ran=$((pass+fail))
echo "===================================================================="
echo "[xfstests] TALLY: pass=${pass} fail=${fail} notrun=${notrun} hang=${hang} (list=${#LIST[@]})"
[ "${ran}" -gt 0 ] && echo "[xfstests] pass rate (of run, excl. notrun): $((100*pass/ran))%"
[ -n "${failed_list}" ] && echo "[xfstests] not passing:${failed_list}"
exit 0
