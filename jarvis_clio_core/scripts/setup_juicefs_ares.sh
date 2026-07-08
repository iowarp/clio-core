#!/usr/bin/env bash
#
# setup_juicefs_ares.sh -- install the toolchain needed by the JuiceFS fio
# sweep (jarvis_clio_core/pipelines/juicefs_fio_1n.yaml) on Ares.
#
# Installs (idempotently):
#   * fio            -- the I/O driver (conda-forge, else `module load`)
#   * juicefs        -- pinned release binary into ${HOME}/.local/bin
#   * checks redis-server / redis-cli are reachable (metadata engine)
#
# No sudo is required: juicefs lands in ${HOME}/.local/bin. Run this inside
# the `iowarp` conda env (so `conda install` targets it). Re-running is safe.
#
# Usage:  bash jarvis_clio_core/scripts/setup_juicefs_ares.sh
set -u

JUICEFS_VERSION="${JUICEFS_VERSION:-1.2.3}"   # pin a known-good release
LOCAL_BIN="${HOME}/.local/bin"
mkdir -p "${LOCAL_BIN}"
case ":${PATH}:" in
  *":${LOCAL_BIN}:"*) ;;
  *) export PATH="${LOCAL_BIN}:${PATH}" ;;
esac

log() { printf '\n=== %s ===\n' "$*"; }

# --- fio -------------------------------------------------------------------
log "fio"
if command -v fio >/dev/null 2>&1; then
  echo "fio already present: $(command -v fio)"
else
  if command -v conda >/dev/null 2>&1; then
    echo "Installing fio via conda-forge..."
    conda install -y -c conda-forge fio || true
  fi
  if ! command -v fio >/dev/null 2>&1; then
    echo "conda install failed/unavailable; trying 'module load fio'..."
    module load fio 2>/dev/null || true
  fi
fi
command -v fio >/dev/null 2>&1 || {
  echo "ERROR: fio is not installed. Install it (conda-forge or a module) and re-run." >&2
}

# --- juicefs ---------------------------------------------------------------
log "juicefs"
if command -v juicefs >/dev/null 2>&1; then
  echo "juicefs already present: $(command -v juicefs)"
else
  arch="$(uname -m)"
  case "${arch}" in
    x86_64|amd64) jarch="amd64" ;;
    aarch64|arm64) jarch="arm64" ;;
    *) jarch="amd64"; echo "WARN: unknown arch ${arch}, defaulting to amd64" ;;
  esac
  tarball="juicefs-${JUICEFS_VERSION}-linux-${jarch}.tar.gz"
  url="https://github.com/juicedata/juicefs/releases/download/v${JUICEFS_VERSION}/${tarball}"
  tmpd="$(mktemp -d)"
  echo "Downloading ${url}"
  if curl -fsSL "${url}" -o "${tmpd}/${tarball}"; then
    tar -xzf "${tmpd}/${tarball}" -C "${tmpd}" juicefs 2>/dev/null \
      || tar -xzf "${tmpd}/${tarball}" -C "${tmpd}"
    if [ -f "${tmpd}/juicefs" ]; then
      install -m 0755 "${tmpd}/juicefs" "${LOCAL_BIN}/juicefs"
      echo "Installed juicefs -> ${LOCAL_BIN}/juicefs"
    else
      echo "ERROR: juicefs binary not found in tarball ${tarball}" >&2
    fi
  else
    echo "ERROR: failed to download juicefs. If Ares has no outbound network," >&2
    echo "       fetch ${tarball} on the login node and place 'juicefs' in ${LOCAL_BIN}." >&2
  fi
  rm -rf "${tmpd}"
fi

# --- redis -----------------------------------------------------------------
log "redis"
if command -v redis-server >/dev/null 2>&1 && command -v redis-cli >/dev/null 2>&1; then
  echo "redis present: $(command -v redis-server)"
else
  echo "redis-server/redis-cli not on PATH. Provide via conda" >&2
  echo "('conda install -y -c conda-forge redis') or a module before running." >&2
fi

# --- summary ---------------------------------------------------------------
log "versions"
command -v fio        >/dev/null 2>&1 && fio --version
command -v juicefs    >/dev/null 2>&1 && juicefs version
command -v redis-server >/dev/null 2>&1 && redis-server --version

log "PATH note"
echo "Ensure ${LOCAL_BIN} is on PATH for the benchmark run:"
echo "  export PATH=\"${LOCAL_BIN}:\$PATH\""
