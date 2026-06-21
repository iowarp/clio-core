#!/usr/bin/env bash
#
# Fast mmap smoke test for the clio filesystem via the libfuse adapter.
#
# mmap can only be exercised through a real FUSE mount (the kernel page cache
# faults pages through the adapter's read/write), not the unit-test client. This
# mounts clio_cte_fuse and checks the cases that issue #597 flagged as broken
# under direct_io (which returned ENODEV "No such device" for any map):
#   1. MAP_SHARED read  — pages faulted in through cte_fuse_read match the file.
#   2. MAP_SHARED write — stores through the mapping, after msync, are visible to
#      a plain read() (flushed through cte_fuse_write).
#   3. mmap that extends the file (write past the old size after ftruncate-grow).
#   4. MAP_PRIVATE read-only map.
#
# Usage:  [CLIO_BUILD_DIR=<dir>] bash scripts/xfstests/mmap_smoke.sh
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_BIN="${CLIO_BUILD_DIR:-${REPO_ROOT}/build}/bin"
FUSE_BIN="${BUILD_BIN}/clio_cte_fuse"
MNT="${CLIO_MMAP_MNT:-/tmp/clio_mmap_smoke}"

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

echo "[mmap-smoke] mounting clio FUSE at ${MNT}"
CLIO_REPO_PATH="${BUILD_BIN}" \
  LD_LIBRARY_PATH="${BUILD_BIN}:${HOME}/.local/lib:${LD_LIBRARY_PATH:-}" \
  CLIO_WITH_RUNTIME=1 CLIO_BIND_ADDR=127.0.0.1 \
  "${FUSE_BIN}" "${MNT}" -o fsname=clio_mmap_smoke -f > /tmp/clio_mmap_fuse.log 2>&1 &
FPID=$!
for _ in $(seq 1 60); do mountpoint -q "${MNT}" && break; sleep 0.3; done
mountpoint -q "${MNT}" || { echo "[mmap-smoke] ERROR: mount failed"; exit 1; }
echo "[mmap-smoke] mounted."

CLIO_MMAP_DIR="${MNT}" python3 - <<'PY'
import mmap, os, sys

d = os.environ["CLIO_MMAP_DIR"]
PS = mmap.PAGESIZE
failures = []

def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        failures.append(name)

# 1) MAP_SHARED read: file content must be visible through the mapping.
p = os.path.join(d, "m_read.bin")
data = bytes((i * 7 + 3) & 0xFF for i in range(PS * 3))  # 3 pages, non-trivial
with open(p, "wb") as f:
    f.write(data)
fd = os.open(p, os.O_RDWR)
try:
    m = mmap.mmap(fd, len(data), mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    check("MAP_SHARED read matches file", m[:] == data)

    # 2) MAP_SHARED write: stores + msync must be visible to a plain read().
    region = b"CLIO-MMAP-WRITE!" * 64  # 1024 bytes, crosses no page boundary issues
    m[PS:PS + len(region)] = region
    m.flush()  # msync
    m.close()
    os.close(fd)
    with open(p, "rb") as f:
        f.seek(PS)
        got = f.read(len(region))
    check("MAP_SHARED write visible via read()", got == region)
except OSError as e:
    check(f"mmap MAP_SHARED ({e})", False)
    os.close(fd)

# 3) mmap that extends into a freshly grown region (ftruncate then map+write).
p2 = os.path.join(d, "m_extend.bin")
with open(p2, "wb") as f:
    f.write(b"x" * 100)
fd2 = os.open(p2, os.O_RDWR)
try:
    newlen = PS * 2
    os.ftruncate(fd2, newlen)             # grow (sparse zeros)
    m2 = mmap.mmap(fd2, newlen, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    marker = b"ZZ"
    m2[newlen - 2:newlen] = marker        # write into the last page (was a hole)
    m2.flush()
    m2.close()
    os.close(fd2)
    with open(p2, "rb") as f:
        f.seek(newlen - 2)
        tail = f.read(2)
    sz = os.path.getsize(p2)
    check("mmap-extend write visible + size exact", tail == marker and sz == newlen)
except OSError as e:
    check(f"mmap extend ({e})", False)
    os.close(fd2)

# 4) MAP_PRIVATE read-only map.
p3 = os.path.join(d, "m_ro.bin")
ro = bytes(range(256)) * 8  # 2048 bytes
with open(p3, "wb") as f:
    f.write(ro)
fd3 = os.open(p3, os.O_RDONLY)
try:
    m3 = mmap.mmap(fd3, len(ro), mmap.MAP_PRIVATE, mmap.PROT_READ)
    check("MAP_PRIVATE read-only matches file", m3[:] == ro)
    m3.close()
except OSError as e:
    check(f"mmap MAP_PRIVATE ({e})", False)
finally:
    os.close(fd3)

if failures:
    print(f"[mmap-smoke] {len(failures)} FAILED: {failures}")
    sys.exit(1)
print("[mmap-smoke] all mmap checks passed")
PY
rc=$?
echo "[mmap-smoke] result rc=${rc}"
exit "${rc}"
