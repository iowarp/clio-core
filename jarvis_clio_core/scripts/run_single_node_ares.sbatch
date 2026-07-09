#!/bin/bash
# Queue the #526 SINGLE-NODE container regression sweep on Ares (Pattern A:
# one allocation, in-process 24-combo sweep via `jarvis ppl run yaml`).
#
# Grabs one compute node, makes apptainer available, verifies the regression SIF
# is present, creates the host bind dirs (NFS target + juicefs object store) and
# the node-local /tmp bind dir, activates the iowarp conda env, (re)registers the
# repo, and runs single_node.yaml. Output lands in
# $HOME/single_node_results/results.csv (24 rows).
#
# PREREQUISITE: build the SIF ONCE on the login node (docker + apptainer):
#   cd ~/clio-core-fork && jarvis_clio_core/scripts/build_regression_image.sh
# It writes <jarvis shared_dir>/containers/iowarp-regression-526.sif on the shared
# FS; this job just consumes it (the YAML references it by basename).
#
# Quick queue (from anywhere):
#   sbatch jarvis_clio_core/scripts/run_single_node_ares.sbatch
#
# Override at submit time, e.g. (point at a Spack setup so `spack load apptainer`
# works, or pass an explicit apptainer bin dir):
#   sbatch --export=ALL,SPACK_SETUP=$HOME/spack/share/spack/setup-env.sh \
#          jarvis_clio_core/scripts/run_single_node_ares.sbatch
#   sbatch --export=ALL,APPTAINER_BIN=/path/to/apptainer/bin \
#          jarvis_clio_core/scripts/run_single_node_ares.sbatch
#
# Watch:   squeue -u $USER ; tail -f single_node-<jobid>.out
#
#SBATCH --job-name=single_node_526
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# ConstrainCores=yes on Ares hands an ntasks=1 job a single core unless we ask
# for more; chimaera workers + up to 32 bench threads + fio need real cores.
#SBATCH --cpus-per-task=32
# debug partition (mentor directive 2026-07-07): compute is ~fully allocated,
# debug has idle nodes (ares-comp-03..06) and a 4-day limit. Override for a
# one-off with `sbatch --partition=compute ...`.
#SBATCH --partition=debug
# Empirically MORE than ~11 min/combo: job 21064 estimated ~11 min/combo
# (~4.6 h/24), but the 6 h rerun still clipped at 22/24 -- actual cost is
# running >16 min/combo on average (probably rising with io_size/threads late
# in the grid, since the swept vars grow monotonically). Bumped to 12 h as a
# TIMING RUN: let it run to completion (or clip) and record wall-clock cost
# per combo from the .out timestamps before picking a final --time value.
#SBATCH --time=12:00:00
#SBATCH --output=single_node-%j.out
#SBATCH --error=single_node-%j.err

set -euo pipefail

# ---- knobs (override via --export=ALL,KEY=VAL) ----------------------------
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-iowarp}"
REPO="${REPO:-$HOME/clio-core-fork}"
PIPELINE="${PIPELINE:-$REPO/jarvis_clio_core/pipelines/ares/single_node.yaml}"
RESULTS="${RESULTS:-$HOME/single_node_results/results.csv}"
# Regression SIF: referenced by the YAML as container_image: <SIF_BASENAME>.
SIF_BASENAME="${SIF_BASENAME:-iowarp-regression-526}"
# How apptainer is made available (Spack-installed; not on PATH by default):
#   - APPTAINER_BIN : explicit dir containing the apptainer binary, OR
#   - SPACK_SETUP   : path to spack setup-env.sh -> `spack load apptainer`.
SPACK_SETUP="${SPACK_SETUP:-}"
APPTAINER_BIN="${APPTAINER_BIN:-}"
# Node-local /tmp bind root (MUST match tmp_bind_root in the YAML). jarvis binds
# <root>/single_node/tmp -> in-container /tmp; this job creates it.
TMP_BIND_ROOT="${TMP_BIND_ROOT:-/tmp}"
PIPELINE_NAME="single_node"

echo "=== #526 single-node sweep on $(hostname) — job ${SLURM_JOB_ID:-local} ==="
echo "conda: $CONDA_ENV   repo: $REPO"
echo "yaml:  $PIPELINE"

# ---- make apptainer available (Spack-installed on the shared FS) ----------
if ! command -v apptainer >/dev/null 2>&1; then
  if [ -n "$APPTAINER_BIN" ]; then
    export PATH="$APPTAINER_BIN:$PATH"
  elif [ -n "$SPACK_SETUP" ] && [ -f "$SPACK_SETUP" ]; then
    # shellcheck disable=SC1090
    source "$SPACK_SETUP"; spack load apptainer || true
  fi
fi
command -v apptainer >/dev/null 2>&1 || {
  echo "ERROR: apptainer not on PATH. Spack-install it (scripts/validate_apptainer.sh"
  echo "       Phase 0) and pass SPACK_SETUP=.../setup-env.sh or APPTAINER_BIN=.../bin." >&2
  exit 1
}
echo "apptainer: $(command -v apptainer) ($(apptainer --version 2>/dev/null))"

# ---- resolve + verify the regression SIF ----------------------------------
if [ -z "${SIF_PATH:-}" ]; then
  SHARED_DIR="$(grep -hE '^[[:space:]]*shared_dir:' "$HOME"/.ppi-jarvis/*.yaml 2>/dev/null \
                | head -1 | sed -E 's/^[[:space:]]*shared_dir:[[:space:]]*//; s/["'"'"']//g')"
  SIF_PATH="${SHARED_DIR:+$SHARED_DIR/containers/$SIF_BASENAME.sif}"
fi
echo "sif:   ${SIF_PATH:-<unresolved>}"
[ -n "${SIF_PATH:-}" ] && [ -f "$SIF_PATH" ] || {
  echo "ERROR: regression SIF not found: ${SIF_PATH:-<unresolved>}" >&2
  echo "Build it ONCE on the login node, then resubmit:" >&2
  echo "    cd $REPO && jarvis_clio_core/scripts/build_regression_image.sh" >&2
  exit 1
}

# ---- stale-state cleanup on THIS node (node hygiene, non-fatal) ----------
# single_node now PINS its hostfile (below) so it truly deploys on ONE node,
# which exposes it to the same node-carried ghosts that bit distributed. Both
# pipelines draw from the same compute pool, and a prior job (single_node OR
# distributed) that hung and was Slurm-killed can leave, on this node:
#   * a DAEMONIZED chimaera still bound to host port 9413 (apptainer instances
#     share the host network namespace) -> this run's runtime FATALs "Could not
#     start TCP server ... Port attempted: 9413" (distributed job 21473), then
#     shm_attach fails and the sweep produces no results.csv; and
#   * stale /tmp/<name>_run* scratch that accumulates until a later combo
#     ENOSPCs on `mkdir /tmp/..._run<N>` (distributed job 21294).
# Clear ONLY our own leftovers: stop orphaned instances (that kills the chimaera
# inside them), pkill any bare survivor, free 9413, drop stale chi_* shm, and
# reclaim stale per-run /tmp dirs older than 10 min (-mmin +10 protects a
# concurrent job's live dir; nodes are effectively exclusive at 32 cpus, and
# `instance stop --all` / pkill are per-user). No user data is touched -- this
# only reclaims this pipeline's own dead state. Bracketed pkill patterns
# ("[c]himaera") so they don't match a shell whose argv holds the pattern text.
echo "=== stale-state cleanup on $(hostname) ==="
echo "  /tmp free BEFORE: $(df -h "$TMP_BIND_ROOT" | awk 'NR==2{print $4" / "$2}')"
timeout 30 apptainer instance stop --all >/dev/null 2>&1 || true
pkill -9 -f '[c]himaera' >/dev/null 2>&1 || true
pkill -9 -f '[c]lio_run' >/dev/null 2>&1 || true
{ command -v fuser >/dev/null 2>&1 && fuser -k 9413/tcp >/dev/null 2>&1; } || true
rm -f /dev/shm/chi_* >/dev/null 2>&1 || true
find "$TMP_BIND_ROOT" -maxdepth 1 \
     \( -name "${PIPELINE_NAME}_run*" -o -name "$PIPELINE_NAME" \) \
     -mmin +10 -exec rm -rf {} + >/dev/null 2>&1 || true
rm -rf /tmp/iowarp_chimaera >/dev/null 2>&1 || true
echo "  /tmp free AFTER : $(df -h "$TMP_BIND_ROOT" | awk 'NR==2{print $4" / "$2}')"

# ---- host bind dirs (NFS experiment target + juicefs object store) +
#      node-local /tmp bind dir (tmp_bind_root). --------------------------
mkdir -p "$HOME/single_node_nfs" "$HOME/juicefs_data" \
         "$TMP_BIND_ROOT/$PIPELINE_NAME/tmp"

# ---- activate the IOWarp/Jarvis conda env (the OUTER jarvis) -------------
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"

# ---- register the repo with jarvis (idempotent) --------------------------
jarvis repo add "$REPO/jarvis_clio_core" 2>/dev/null || true

# ---- pin the jarvis hostfile to THIS ONE node ----------------------------
# `jarvis hostfile set` is PERSISTENT global state in $HOME/.jarvis, shared
# across all jobs. A prior DISTRIBUTED run leaves a 4-node hostfile behind, and
# this single-node job — which never set its own — then inherits it and deploys
# as if it had 4 nodes: jarvis fans `apptainer instance start` over ssh to the
# other 3 (NOT in this 1-node allocation) -> thousands of "Access denied by
# pam_slurm_adopt" lines, those instances never start, the chimaera runtime
# forms a 4-node cluster with 3 SWIM-dead members (30s retry timeouts), and
# juicefs (correctly NOT single_instance here) tries to reach redis on all 4 ->
# "mountpoint /tmp/jfs_mnt not ready" (job 21474: 1/24, was 24/24 in job 21061
# before any distributed run had poisoned the global hostfile). Pin to our own
# node so the deploy is single-node again, matching the YAML's intent.
HOSTFILE="$HOME/.jarvis/hostfile_single_${SLURM_JOB_ID:-local}.txt"
mkdir -p "$(dirname "$HOSTFILE")"
scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -1 > "$HOSTFILE"
echo "=== pinning jarvis hostfile to this node ==="; cat "$HOSTFILE"
jarvis hostfile set "$HOSTFILE"
trap 'rm -f "$HOSTFILE"' EXIT

# ---- clean slate so all 24 combos re-run (sweep-resume otherwise skips
#      rows already present, including failed ones). ------------------------
rm -f "$RESULTS"

# ---- run the 24-combo sweep (runner auto-detects vars:+loop:) ------------
# Run under our PERSONAL group ($USER), not the Slurm-assigned PROJECT group.
# The FUSE backends mount via `apptainer instance start --fakeroot`, whose GID
# mapping is validated against our personal group; under Slurm the primary GID
# is the project group, so fakeroot fails on compute with
#   newgidmap: Target N is owned by a different user
#   Error while waiting event for user namespace mappings: no event received
# `sg $USER` switches the primary group back for the jarvis process and every
# apptainer instance it spawns. conda is re-activated inside the new-group shell
# so `jarvis` + apptainer resolve. (Verified: `sg $USER -c 'apptainer exec
# --fakeroot ...'` starts the userns + mounts FUSE on a compute node; without it
# the userns setup fails outright.)
# NB: `sg $USER -c` runs its arg under /bin/sh (dash), which has no `source` —
# use the POSIX `.` so conda activation works in BOTH the sg and non-sg branches.
RUN_SWEEP=". '$CONDA_SH' && conda activate '$CONDA_ENV' && jarvis ppl run yaml '$PIPELINE'"
if getent group "$USER" >/dev/null 2>&1 && [ "$(id -gn)" != "$USER" ]; then
  echo "switching primary group $(id -gn) -> $USER for apptainer --fakeroot"
  sg "$USER" -c "$RUN_SWEEP"
else
  bash -c "$RUN_SWEEP"
fi

# ---- report ---------------------------------------------------------------
echo "=== done — results at $RESULTS ==="
if [ -f "$RESULTS" ]; then
  # `grep -c` on no-match exits 1 AND prints 0 already; `|| echo 0` then
  # appended a SECOND "0", so $n became "0\n0" -> `[: integer expression`.
  n=$(grep -c success "$RESULTS" 2>/dev/null || true); n=${n:-0}
  echo "successful rows: $n / 24"
  head -1 "$RESULTS"
  [ "$n" -ge 24 ] && echo "ALL 24 GREEN" || echo "INCOMPLETE — inspect $RESULTS / the .err log"
else
  echo "NO results.csv produced — check single_node-${SLURM_JOB_ID:-local}.err"
fi
