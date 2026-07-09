#!/bin/bash
# validate_apptainer.sh — runnable preflight + validation checklist for the #526
# APPTAINER pipelines on Ares. Run the cheap, fast-failing gates FIRST; each must
# pass before the next. Heavy pipeline runs (Gates 4-6) are driven via sbatch and
# are printed as instructions at the end.
#
# Usage (on the Ares LOGIN node, repo root or anywhere):
#   jarvis_clio_core/scripts/validate_apptainer.sh            # Phase 0 + Gates 0-3
#   APPEND_BASHRC=1 jarvis_clio_core/scripts/validate_apptainer.sh  # also wire PATH
#
# Knobs:
#   SPACK_SETUP   path to spack setup-env.sh (else spack must be on PATH)
#   SIF_BASENAME  default iowarp-regression-526 (must match the YAML container_image)
#   SIF_PATH      override the resolved SIF location
#   APPEND_BASHRC=1   append the apptainer bin to ~/.bashrc (needed for the
#                     distributed run's ssh fan-out; off by default).
set -uo pipefail

REPO="${REPO:-$HOME/clio-core-fork}"
SIF_BASENAME="${SIF_BASENAME:-iowarp-regression-526}"
JARVIS_DIR="${JARVIS_DIR:-$HOME/jarvis-cd}"
JARVIS_PIN="${JARVIS_PIN:-2925ee8}"

ok()   { echo "  [PASS] $*"; }
bad()  { echo "  [FAIL] $*" >&2; }
hdr()  { echo; echo "==================== $* ===================="; }

# Make spack available if a setup script was provided.
if [ -n "${SPACK_SETUP:-}" ] && [ -f "$SPACK_SETUP" ]; then
  # shellcheck disable=SC1090
  source "$SPACK_SETUP"
fi

# ---------------------------------------------------------------------------
hdr "Phase 0 — Install Apptainer via Spack (one-time, on the shared FS)"
# Apptainer is NOT preinstalled on Ares (mentor: install with spack; not on every
# node). Installing into the host spack on /mnt/common makes the binary visible
# on every compute node (shared FS). Ares has unprivileged user namespaces +
# subuid/subgid maps, so the non-setuid apptainer runs fine as an unprivileged
# user. If `spack install apptainer` tries to install setuid binaries and fails,
# retry with the non-setuid variant:  spack install apptainer ~suid
if ! command -v apptainer >/dev/null 2>&1; then
  if command -v spack >/dev/null 2>&1; then
    echo "apptainer not on PATH — installing via spack (this can take a while)..."
    # GOFLAGS=-buildvcs=false: apptainer's Go deps (go-md2man, gocryptfs) fail
    # when the spack build dir is not a git repo and Go tries to embed VCS info.
    GOFLAGS=-buildvcs=false spack install apptainer \
      || GOFLAGS=-buildvcs=false spack install apptainer ~suid || {
      bad "spack install apptainer failed — inspect the spack log"; }
    APPTAINER_PREFIX="$(spack location -i apptainer 2>/dev/null || true)"
    [ -n "$APPTAINER_PREFIX" ] && export PATH="$APPTAINER_PREFIX/bin:$PATH"
  else
    bad "spack not found. Pass SPACK_SETUP=.../share/spack/setup-env.sh"
  fi
fi

if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN_DIR="$(dirname "$(command -v apptainer)")"
  ok "apptainer present: $(command -v apptainer)"
  # PATH wiring for the distributed run: jarvis Pssh's `apptainer instance start`
  # to every node over ssh, so the bin must be on the remote NON-interactive
  # shell's PATH. ~/.bashrc is shared ($HOME on /mnt/common) and IS sourced by
  # the ssh fan-out on Ares (same mechanism that activates conda).
  BASHRC_LINE="export PATH=\"$APPTAINER_BIN_DIR:\$PATH\"  # apptainer (#526)"
  if [ "${APPEND_BASHRC:-0}" = "1" ]; then
    if ! grep -qF "$APPTAINER_BIN_DIR" "$HOME/.bashrc" 2>/dev/null; then
      echo "$BASHRC_LINE" >> "$HOME/.bashrc"
      ok "appended apptainer bin to ~/.bashrc"
    else
      ok "apptainer bin already in ~/.bashrc"
    fi
  else
    echo "  NOTE: for the DISTRIBUTED run, ensure remote shells see apptainer. Add:"
    echo "        $BASHRC_LINE"
    echo "        (re-run with APPEND_BASHRC=1 to do this automatically)"
  fi
else
  bad "apptainer still not available — fix Phase 0 before continuing"
fi

# ---------------------------------------------------------------------------
hdr "Gate 0 — Apptainer on login AND a compute node"
command -v apptainer >/dev/null 2>&1 && apptainer --version && ok "login apptainer OK" \
  || bad "apptainer missing on login"
echo "  checking a compute node via srun (login-shell PATH)..."
srun --partition=compute -N1 --time=00:02:00 bash -lc \
  'command -v apptainer >/dev/null && apptainer --version' \
  && ok "compute-node apptainer OK" \
  || bad "apptainer NOT reachable on compute (set APPEND_BASHRC=1 and retry Gate 0)"

# ---------------------------------------------------------------------------
hdr "Gate 1 — jarvis is dev @ $JARVIS_PIN with the apptainer code path"
HEAD="$(git -C "$JARVIS_DIR" rev-parse --short HEAD 2>/dev/null || echo '?')"
BR="$(git -C "$JARVIS_DIR" branch --show-current 2>/dev/null || echo '?')"
echo "  jarvis-cd: $BR @ $HEAD  (expect dev @ $JARVIS_PIN)"
[ "$HEAD" = "$JARVIS_PIN" ] && ok "jarvis pin matches" || bad "jarvis pin differs ($HEAD)"
if git -C "$JARVIS_DIR" grep -q "container_engine == 'apptainer'" -- jarvis_cd/core/installer.py 2>/dev/null; then
  ok "ContainerInstaller has the apptainer code path"
else
  bad "apptainer code path not found in installer.py"
fi

# ---------------------------------------------------------------------------
hdr "Gate 2 — Regression SIF present"
if [ -z "${SIF_PATH:-}" ]; then
  SHARED_DIR="$(grep -hE '^[[:space:]]*shared_dir:' "$HOME"/.ppi-jarvis/*.yaml 2>/dev/null \
                | head -1 | sed -E 's/^[[:space:]]*shared_dir:[[:space:]]*//; s/["'"'"']//g')"
  SIF_PATH="${SHARED_DIR:+$SHARED_DIR/containers/$SIF_BASENAME.sif}"
fi
echo "  SIF: ${SIF_PATH:-<unresolved>}"
if [ -n "${SIF_PATH:-}" ] && [ -f "$SIF_PATH" ]; then
  ok "SIF present"; ls -lh "$SIF_PATH"
else
  bad "SIF missing — build it on the login node:"
  echo "        cd $REPO && jarvis_clio_core/scripts/build_regression_image.sh"
fi

# ---------------------------------------------------------------------------
hdr "Gate 3 — FUSE smoke inside the SIF (the early kill-switch)"
# Confirms (a) juicefs + redis binaries present, (b) a FUSE mount works inside the
# container. KEY: the conda-forge apptainer is NON-SETUID, so it runs rootless and
# --add-caps SYS_ADMIN is a NO-OP (you can't gain host caps you lack). FUSE mount()
# needs CAP_SYS_ADMIN *effective*, which only --fakeroot provides (maps user->root
# in the userns). With --fakeroot apptainer also supplies a dev-enabled /dev/fuse,
# so NO --bind /dev/fuse is needed. If this fails, no pipeline will mount FUSE.
#
# GROUP GOTCHA: apptainer --fakeroot validates its GID mapping against our PERSONAL
# group (== $USER). Under Slurm the primary GID is the PROJECT group, which makes
# fakeroot fail on compute with "newgidmap: Target N is owned by a different user".
# So run the smoke under `sg $USER` whenever our current primary group != $USER
# (i.e. on a compute node / inside srun). The sbatch wrappers do the same around
# `jarvis ppl run`. NOTE: the login node may kill long FUSE jobs (watchdog), so
# prefer running this whole script on a compute node:  srun ... bash <this script>.
#
# redis/juicefs in-container gotchas baked in below:
#   * redis --daemonize yes silently fails in the userns -> start with `&`.
#   * a stale on-disk dump.rdb (newer RDB version) aborts load -> --dbfilename "".
#   * juicefs -d can't signal readiness in-container -> background `&` + mountpoint.
if [ -n "${SIF_PATH:-}" ] && [ -f "$SIF_PATH" ] && command -v apptainer >/dev/null 2>&1; then
  # Inner smoke written to a tempfile (on host /tmp, visible in-container via the
  # default /tmp bind) to sidestep sg/apptainer/bash nested-quoting.
  SMOKE_SH="$(mktemp /tmp/fuse_smoke.XXXXXX.sh)"
  cat > "$SMOKE_SH" <<'SMOKE'
set -e
SM="/tmp/jfs_smoke_$$"
cleanup() {
  fusermount -uz "$SM/mnt" 2>/dev/null || umount -l "$SM/mnt" 2>/dev/null || true
  redis-cli -p 6390 shutdown nosave 2>/dev/null || true
  rm -rf "$SM" 2>/dev/null || true
}
trap cleanup EXIT
pkill -f "redis-server.*6390" 2>/dev/null || true; sleep 1
mkdir -p "$SM/mnt" "$SM/data"
redis-server --port 6390 --save "" --dbfilename "" --loglevel warning &
sleep 2
redis-cli -p 6390 ping
juicefs format --storage file --bucket "$SM/data" redis://127.0.0.1:6390/9 smoketest
juicefs mount redis://127.0.0.1:6390/9 "$SM/mnt" --no-usage-report &
sleep 5
mountpoint -q "$SM/mnt" || { echo "FUSE mount not ready"; tail -20 "$HOME"/.juicefs/juicefs.log 2>/dev/null || true; exit 1; }
echo hello > "$SM/mnt/probe.txt" && grep -q hello "$SM/mnt/probe.txt"
echo "MOUNT OK"; echo "IO OK"
SMOKE
  if getent group "$USER" >/dev/null 2>&1 && [ "$(id -gn)" != "$USER" ]; then
    echo "  (running under personal group: sg $USER — current primary is $(id -gn))"
    SMOKE_RUN() { sg "$USER" -c "apptainer exec --fakeroot '$SIF_PATH' bash '$SMOKE_SH'"; }
  else
    SMOKE_RUN() { apptainer exec --fakeroot "$SIF_PATH" bash "$SMOKE_SH"; }
  fi
  if SMOKE_RUN; then ok "FUSE smoke passed (MOUNT OK / IO OK)"
  else bad "FUSE smoke FAILED — fix the fakeroot/userns/group story before touching the pipelines"; fi
  rm -f "$SMOKE_SH"
else
  echo "  [SKIP] need a present SIF + apptainer (fix Gates 0/2 first)"
fi

# ---------------------------------------------------------------------------
hdr "Gates 4-6 — pipeline runs (driven via sbatch; run after Gates 0-3 pass)"
cat <<EOF
  Gate 4 — single-backend smoke: temporarily trim single_node.yaml to ONLY the
           nfs_fio driver + 1 combo (comment out redis/cte/juicefs pkgs; set the
           vars/loop to a single value), then:
             sbatch $REPO/jarvis_clio_core/scripts/run_single_node_ares.sbatch
             grep -c success \$HOME/single_node_results/results.csv   # expect 1
  Gate 5 — full single_node sweep (restore all 4 backends, 24 combos):
             sbatch $REPO/jarvis_clio_core/scripts/run_single_node_ares.sbatch
             grep -c success \$HOME/single_node_results/results.csv   # expect 24
  Gate 6 — distributed: first set cte_bench.nprocs: [2] only, then restore [1,2,4]:
             sbatch $REPO/jarvis_clio_core/scripts/run_distributed_ares.sbatch
             grep -c success \$HOME/distributed_results/results.csv    # expect 3
  DONE when all 24 single-node + 3 distributed rows are status=success with
  non-zero agg_bw_mbps / agg_ops_per_sec.
EOF
echo
echo "=== preflight complete — review any [FAIL] lines above ==="
