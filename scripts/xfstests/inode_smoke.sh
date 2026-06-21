#!/usr/bin/env bash
#
# Fast inode smoke test for the clio filesystem (libfuse adapter).
#
# Checks the properties generic/637 (and POSIX) expect once inodes are derived
# from the TagId:
#   1. stat st_ino is non-zero and stable across repeated stats.
#   2. readdir d_ino matches stat st_ino for every directory entry (the
#      t_dir_offset2 consistency check).
#   3. hard-link aliases share the same inode (they share a TagId).
#   4. distinct files have distinct inodes.
#
# Usage:  [CLIO_BUILD_DIR=<dir>] bash scripts/xfstests/inode_smoke.sh
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_BIN="${CLIO_BUILD_DIR:-${REPO_ROOT}/build}/bin"
FUSE_BIN="${BUILD_BIN}/clio_cte_fuse"
MNT="${CLIO_INODE_MNT:-/tmp/clio_inode_smoke}"

command -v fusermount3 >/dev/null || { echo "ERROR: fusermount3 (fuse3) not installed"; exit 1; }
[ -x "${FUSE_BIN}" ] || { echo "ERROR: clio_cte_fuse not built at ${FUSE_BIN}"; exit 1; }
command -v python3 >/dev/null || { echo "ERROR: python3 required"; exit 1; }

FPID=""
cleanup() {
  fusermount3 -u "${MNT}" 2>/dev/null
  [ -n "${FPID}" ] && kill "${FPID}" 2>/dev/null
  pkill -x clio_run 2>/dev/null
}
trap cleanup EXIT

pkill -9 -x clio_run 2>/dev/null
fusermount3 -u "${MNT}" 2>/dev/null
sleep 1
rm -rf "${MNT}"; mkdir -p "${MNT}"

echo "[inode-smoke] mounting clio FUSE at ${MNT}"
CLIO_REPO_PATH="${BUILD_BIN}" \
  LD_LIBRARY_PATH="${BUILD_BIN}:${HOME}/.local/lib:${LD_LIBRARY_PATH:-}" \
  CLIO_WITH_RUNTIME=1 CLIO_BIND_ADDR=127.0.0.1 \
  "${FUSE_BIN}" "${MNT}" -o fsname=clio_inode_smoke -f > /tmp/clio_inode_fuse.log 2>&1 &
FPID=$!
for _ in $(seq 1 60); do mountpoint -q "${MNT}" && break; sleep 0.3; done
mountpoint -q "${MNT}" || { echo "[inode-smoke] ERROR: mount failed"; exit 1; }
echo "[inode-smoke] mounted."

CLIO_INODE_DIR="${MNT}" python3 - <<'PY'
import os, sys

d = os.path.join(os.environ["CLIO_INODE_DIR"], "idir")
os.mkdir(d)
failures = []

def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        failures.append(name)

# Build a small tree: files a, b and a subdirectory sub.
for n in ("a", "b"):
    with open(os.path.join(d, n), "w") as f:
        f.write(n * 100)
os.mkdir(os.path.join(d, "sub"))

# 1) st_ino non-zero and stable.
sa1 = os.stat(os.path.join(d, "a")).st_ino
sa2 = os.stat(os.path.join(d, "a")).st_ino
check("st_ino non-zero", sa1 != 0)
check("st_ino stable across stats", sa1 == sa2)

# 2) readdir d_ino == stat st_ino for every entry (the generic/637 check).
consistent = True
seen = {}
for entry in os.scandir(d):
    d_ino = entry.inode()                 # from getdents (readdir)
    st_ino = entry.stat().st_ino          # from stat
    if d_ino == 0 or d_ino != st_ino:
        print(f"    {entry.name}: d_ino={d_ino} st_ino={st_ino}")
        consistent = False
    seen[entry.name] = st_ino
check("readdir d_ino matches stat st_ino", consistent)

# 4) distinct files have distinct inodes.
check("distinct files distinct inodes", seen.get("a") != seen.get("b"))

# 3) hard-link aliases share an inode.
os.link(os.path.join(d, "a"), os.path.join(d, "c"))
ia = os.stat(os.path.join(d, "a")).st_ino
ic = os.stat(os.path.join(d, "c")).st_ino
check("hard link shares inode", ia == ic and ia != 0)

if failures:
    print(f"[inode-smoke] {len(failures)} FAILED: {failures}")
    sys.exit(1)
print("[inode-smoke] all inode checks passed")
PY
rc=$?
echo "[inode-smoke] result rc=${rc}"
exit "${rc}"
