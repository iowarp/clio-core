#!/usr/bin/env bash
#
# Install everything needed to build + run xfstests against the clio
# filesystem (FUSE adapter), and build xfstests itself.
#
# Safe to re-run (idempotent). Uses sudo for the apt step; everything else
# runs as the current user.
#
#   bash scripts/xfstests/install_deps.sh
#
set -euo pipefail

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
XFSTESTS_DIR="${XFSTESTS_DIR:-${REPO_ROOT}/external/xfstests}"

echo "==> [1/4] Installing build + runtime dependencies (apt)"
# Build deps: xfstests' ./configure needs uuid/attr/acl/aio dev libs.
# Runtime deps: the tests invoke mkfs/xfs_io/attr/getfacl/quota.
# FUSE: required to mount the clio adapter under test.
export DEBIAN_FRONTEND=noninteractive
# Non-fatal: a broken/unsigned THIRD-PARTY apt source (e.g. a Cursor/Chrome
# PPA) makes `apt-get update` exit non-zero even though the Ubuntu archives
# refreshed fine. The packages we need all live in the Ubuntu repos, so don't
# let an unrelated repo abort the whole install.
$SUDO apt-get update || echo "WARNING: apt-get update reported errors (likely a third-party repo); continuing with the Ubuntu package lists."
$SUDO apt-get install -y --no-install-recommends \
  fuse3 libfuse3-dev \
  uuid-dev libattr1-dev libacl1-dev libaio-dev libgdbm-dev \
  xfslibs-dev \
  autoconf automake libtool-bin pkg-config gcc g++ make gawk \
  xfsprogs e2fsprogs attr acl quota \
  git

echo "==> [2/4] Cloning xfstests (if missing) into ${XFSTESTS_DIR}"
if [ ! -d "${XFSTESTS_DIR}/.git" ] && [ ! -f "${XFSTESTS_DIR}/configure.ac" ]; then
  git clone --depth 1 \
    https://git.kernel.org/pub/scm/fs/xfs/xfstests-dev.git "${XFSTESTS_DIR}"
else
  echo "    already present, skipping clone"
fi

echo "==> [3/4] Building xfstests"
make -C "${XFSTESTS_DIR}" -j"$(nproc)"
echo "    built: ${XFSTESTS_DIR}/check"

echo "==> [4/4] Creating xfstests test users (best-effort; some perm tests need them)"
# These are only needed by ownership/permission tests; data tests don't use
# them. Failures here are non-fatal.
$SUDO groupadd -f fsgqa 2>/dev/null || true
for u in fsgqa fsgqa2 123456-fsgqa; do
  id "$u" >/dev/null 2>&1 || $SUDO useradd -m "$u" 2>/dev/null || true
done

echo
echo "Done. Next:"
echo "  scripts/xfstests/run_clio_xfstests.sh            # run a curated subset"
echo "  TESTS='generic/091 generic/263' scripts/xfstests/run_clio_xfstests.sh"
