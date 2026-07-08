#!/bin/bash
# Queue the #526 DISTRIBUTED container regression sweep on Ares (Pattern A:
# ONE 4-node allocation, in-process node-count sweep via `jarvis ppl run yaml`).
#
# Grabs FOUR compute nodes. The regression SIF lives on the shared FS
# (/mnt/common), so it is ALREADY visible on every node — no per-node image
# distribution step (this is the big simplification over the old Docker version,
# which `srun docker pull`ed on every node). The job: makes apptainer available,
# verifies the SIF, prepares each node (node-local /tmp bind dir + stale-state
# cleanup + apptainer-reachability check), seeds the jarvis hostfile with all
# four nodes (Pssh fan-out for the per-node apptainer instances + runtime + CTE
# pool + clio_cte_bench), activates conda, (re)registers the repo, and runs
# distributed.yaml. The pipeline sweeps cte_bench.nprocs over [1,2,4] in-process
# => 3 rows in $HOME/distributed_results/results.csv.
#
# PREREQUISITES (one-time, on the login node):
#   1. Build the SIF:  cd ~/clio-core-fork &&
#        jarvis_clio_core/scripts/build_regression_image.sh
#   2. Make apptainer reachable on REMOTE non-interactive shells. jarvis Pssh's
#      `apptainer instance start` to every node over ssh, so the spack apptainer
#      bin MUST be on PATH there. Append it to ~/.bashrc (shared $HOME -> all
#      nodes), e.g.:  echo 'export PATH="$(spack location -i apptainer)/bin:$PATH"' >> ~/.bashrc
#      (see scripts/validate_apptainer.sh Phase 0). The per-node check below
#      fails fast if this was skipped.
#
# Quick queue (from anywhere):
#   sbatch jarvis_clio_core/scripts/run_distributed_ares.sbatch
#
# Watch:   squeue -u $USER ; tail -f distributed-<jobid>.out
#
#SBATCH --job-name=distributed_526
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
# debug partition (mentor directive 2026-07-07): compute is ~fully allocated,
# debug has 4 idle nodes (ares-comp-03..06) — exactly the 4 this run needs — and
# a 4-day limit. Override for a one-off with `sbatch --partition=compute ...`.
#SBATCH --partition=debug
# Empirically ~11-13 min WALL per combo (job 21073 run 0 measured 361 s bench
# runtime alone, plus 4-node apptainer instance start/stop + JuiceFS format +
# CTE fold each combo). 3 combos ~= 40-45 min; 1 h clipped job 21073 at the
# Slurm limit while it burned time on failure retries. 2 h gives real headroom.
#SBATCH --time=02:00:00
#SBATCH --output=distributed-%j.out
#SBATCH --error=distributed-%j.err

set -euo pipefail

# ---- knobs (override via --export=ALL,KEY=VAL) ----------------------------
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-iowarp}"
REPO="${REPO:-$HOME/clio-core-fork}"
PIPELINE="${PIPELINE:-$REPO/jarvis_clio_core/pipelines/ares/distributed.yaml}"
RESULTS="${RESULTS:-$HOME/distributed_results/results.csv}"
SIF_BASENAME="${SIF_BASENAME:-iowarp-regression-526}"
SPACK_SETUP="${SPACK_SETUP:-}"
APPTAINER_BIN="${APPTAINER_BIN:-}"
# Node-local /tmp bind root (MUST match tmp_bind_root in the YAML); jarvis binds
# <root>/distributed/tmp -> in-container /tmp on each node.
TMP_BIND_ROOT="${TMP_BIND_ROOT:-/tmp}"
PIPELINE_NAME="distributed"
# Optional node-count override for a fast smoke. NPROCS=2 runs a SINGLE 2-node
# point (Gate E de-risk: exercises remote-node --fakeroot + the multi-node MPI
# fan-out with one combo before the full [1,2,4] sweep). Comma-list allowed,
# e.g. NPROCS=1,2. Empty (default) => use the YAML's committed [1,2,4].
NPROCS="${NPROCS:-}"
RUN_PIPELINE=""   # set below iff NPROCS override is used (temp YAML)

echo "=== #526 distributed sweep — job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "conda: $CONDA_ENV   repo: $REPO"
echo "yaml:  $PIPELINE"

# ---- make apptainer available on the HEAD node ----------------------------
if ! command -v apptainer >/dev/null 2>&1; then
  if [ -n "$APPTAINER_BIN" ]; then
    export PATH="$APPTAINER_BIN:$PATH"
  elif [ -n "$SPACK_SETUP" ] && [ -f "$SPACK_SETUP" ]; then
    # shellcheck disable=SC1090
    source "$SPACK_SETUP"; spack load apptainer || true
  fi
fi
command -v apptainer >/dev/null 2>&1 || {
  echo "ERROR: apptainer not on PATH on the head node. See PREREQUISITES." >&2
  exit 1
}
echo "apptainer: $(command -v apptainer) ($(apptainer --version 2>/dev/null))"

# ---- resolve + verify the regression SIF (shared FS, seen by all nodes) ----
if [ -z "${SIF_PATH:-}" ]; then
  SHARED_DIR="$(grep -hE '^[[:space:]]*shared_dir:' "$HOME"/.ppi-jarvis/*.yaml 2>/dev/null \
                | head -1 | sed -E 's/^[[:space:]]*shared_dir:[[:space:]]*//; s/["'"'"']//g')"
  SIF_PATH="${SHARED_DIR:+$SHARED_DIR/containers/$SIF_BASENAME.sif}"
fi
echo "sif:   ${SIF_PATH:-<unresolved>}"
[ -n "${SIF_PATH:-}" ] && [ -f "$SIF_PATH" ] || {
  echo "ERROR: regression SIF not found: ${SIF_PATH:-<unresolved>}" >&2
  echo "Build it ONCE on the login node:  cd $REPO && \\" >&2
  echo "    jarvis_clio_core/scripts/build_regression_image.sh" >&2
  exit 1
}

# ---- optional single-combo override (NPROCS=... => temp YAML with just that
#      node count; keeps the committed cte_bench.nprocs: [1,2,4] intact). -----
EXPECT=3
if [ -n "$NPROCS" ]; then
  RUN_PIPELINE="$(mktemp --suffix=.yaml)"
  # Rewrite ONLY the sweep var line; config.name stays "distributed" so the
  # per-node /tmp prep + instance names still match tmp_bind_root.
  sed -E "s/^([[:space:]]*cte_bench\.nprocs:).*/\1  [$NPROCS]/" \
      "$PIPELINE" > "$RUN_PIPELINE"
  EXPECT=$(echo "$NPROCS" | tr ',' '\n' | grep -c .)
  echo "NPROCS override -> cte_bench.nprocs: [$NPROCS]  ($EXPECT point(s), temp: $RUN_PIPELINE)"
  grep -E 'cte_bench\.nprocs:' "$RUN_PIPELINE"
fi

# ---- per-node prep: apptainer reachable (login shell -> ~/.bashrc PATH),
#      node-local /tmp bind dir, and stale-state cleanup. A `bash -lc` login
#      shell exercises the SAME ~/.bashrc that jarvis's ssh fan-out relies on,
#      so a missing apptainer here fails the job before the sweep starts.
#
#      DIAGNOSTIC HARDENING (jobs 21065/21066 both self-terminated in this step
#      before the sweep ran): every task prints a "prep complete" sentinel and
#      the srun aggregate rc is reported. --kill-on-bad-exit=0 stops srun from
#      tearing down the WHOLE step if a single task exits non-zero (the likely
#      "srun: forcing job termination" trigger). `set +e` around srun keeps the
#      outer script from aborting so we still reach the R2 probe + sweep and
#      learn more in one run. Cleanup is non-fatal. -----------------------------
#
# STALE-STATE CLEANUP RATIONALE (kept OUT of the inline bash -lc below — see the
# CRITICAL note): a job that hangs and is Slurm-killed leaves its apptainer
# instances + runtime daemon ALIVE on every node it touched. Because apptainer
# instances share the HOST network namespace, that ghost runtime keeps port 9413
# bound, so THIS job's fresh runtime cannot bind -> "Could not start TCP server
# ... Port attempted: 9413 ... another process is already running" on ALL nodes,
# then shm_attach fails and the run hangs to the time limit with no results.csv
# (job 21473 died exactly there at combo 0). Separately, killed jobs never reach
# teardown, so their per-combo /tmp bind dirs (runtime scratch + juicefs cache +
# memfd) accumulate on frequently-drawn nodes until a fresh run ENOSPCs on
# `mkdir /tmp/..._run<N>` (job 21294 died there at combo 3). So the prep stops
# stale instances FIRST (that kills the runtime inside them), pkills any bare
# survivor, frees 9413, drops leftover shm + old /tmp scratch (-mmin +10 protects
# a concurrent job's live dir; node alloc is ~exclusive at 32 cpus), then remakes
# the bind dir.
#
# CRITICAL — why the inline block below carries NO comments and globs the memfd
# path: `pkill -f` matches a process's FULL argv, and this inline `bash -lc '...'`
# string IS the argv of both srun and each node's bash. Any un-bracketed token
# "chimaera"/"clio_run" ANYWHERE in that string (a comment word, or the literal
# path /tmp/iowarp_chimaera) makes `pkill -9 -f "[c]himaera"` match — and KILL —
# the prep shell itself. That is exactly what happened in job 21564: the prep
# self-immolated at the first pkill (line "…1816220 Killed srun…"), skipping the
# 9413-free + /tmp-reclaim + mkdir steps; the run only survived because the debug
# nodes were already clean. The `[c]` bracket protects the PATTERN token but not
# other occurrences, so this block stays token-free (all prose lives up here) and
# references the memfd dir as the glob /tmp/iowarp_chi* rather than by full name.
echo "=== per-node prep (apptainer + /tmp bind + stale cleanup) ==="
set +e
srun --kill-on-bad-exit=0 --ntasks-per-node=1 bash -lc '
  host=$(hostname)
  if ! command -v apptainer >/dev/null 2>&1; then
    echo "  [$host] ERROR: apptainer not on PATH (non-interactive ssh shell) — see PREREQUISITES."
    exit 1
  fi
  echo "  [$host] apptainer OK: $(command -v apptainer)"
  echo "  [$host] /tmp free BEFORE cleanup: $(df -h "'"$TMP_BIND_ROOT"'" | awk "NR==2{print \$4\" / \"\$2}")"
  # Stale-state cleanup + node-local /tmp reclaim (non-fatal). Full rationale is
  # in the outer comment above the srun; it is deliberately kept OUT of this
  # inline string because pkill -f scans full argv and this string IS the argv
  # (an un-bracketed chi/clio token here would make pkill kill the prep shell).
  timeout 30 apptainer instance stop --all >/dev/null 2>&1
  pkill -9 -f "[c]himaera" >/dev/null 2>&1
  pkill -9 -f "[c]lio_run" >/dev/null 2>&1
  command -v fuser >/dev/null 2>&1 && fuser -k 9413/tcp >/dev/null 2>&1
  rm -f /dev/shm/chi_* >/dev/null 2>&1
  find "'"$TMP_BIND_ROOT"'" -maxdepth 1 \( -name "'"$PIPELINE_NAME"'_run*" -o -name "'"$PIPELINE_NAME"'" \) \
       -mmin +10 -exec rm -rf {} + >/dev/null 2>&1
  rm -rf /tmp/iowarp_chi* >/dev/null 2>&1
  mkdir -p "'"$TMP_BIND_ROOT/$PIPELINE_NAME"'/tmp" || echo "  [$host] mkdir FAILED rc=$?"
  echo "  [$host] /tmp free AFTER  cleanup: $(df -h "'"$TMP_BIND_ROOT"'" | awk "NR==2{print \$4\" / \"\$2}")"
  echo "  [$host] prep complete"'
prep_rc=$?
set -e
echo "=== per-node prep srun rc=$prep_rc ==="
[ "$prep_rc" -eq 0 ] || echo "WARNING: per-node prep rc=$prep_rc — continuing to gather R2 + sweep diagnostics anyway."

# ---- host bind dirs (on the shared NFS home, so all nodes see them) ------
mkdir -p "$HOME/distributed_nfs" "$HOME/juicefs_data"

# ---- activate conda + register the repo ----------------------------------
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"
jarvis repo add "$REPO/jarvis_clio_core" 2>/dev/null || true

# ---- point the jarvis hostfile at ALL allocated nodes so the apptainer
#      instances, runtime, CTE compose, and clio_cte_bench PSSH-fan-out across
#      the reservation (validated pattern from run_cte_sweep_2n_ares.sbatch). --
HOSTFILE="$HOME/.jarvis/hostfile_${SLURM_JOB_ID:-local}.txt"
mkdir -p "$(dirname "$HOSTFILE")"
scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" > "$HOSTFILE"
echo "=== Allocated nodes (CTE runs on the first nprocs of them) ==="
cat "$HOSTFILE"
jarvis hostfile set "$HOSTFILE"
trap 'rm -f "$HOSTFILE" ${RUN_PIPELINE:+"$RUN_PIPELINE"}' EXIT

# ---- pre-flight over ssh (the channel jarvis uses to fan the container to
#      every node). Single-node never exercised this: it deploys LOCALLY with
#      the conda apptainer, whereas distributed PSSHes `apptainer instance start
#      --fakeroot` to each node, where the ssh-login PATH may resolve a DIFFERENT
#      apptainer. Two properties of that ssh-login apptainer must hold:
#        (1) R2 (group): the ssh-session primary group must be $USER, else the
#            remote --fakeroot GID map fails. A fresh ssh login uses initgroups
#            (personal group) unless pam_slurm_adopt forces the Slurm job group.
#            Verified empirically (job 21067): ssh group = $USER -> non-issue.
#        (2) SIF openability: a rootless apptainer mounts the squashfs SIF via
#            squashfuse, or falls back to EXTRACTING it with unsquashfs. If
#            NEITHER is on the ssh PATH, `apptainer instance start` FATALs
#            ("unsquashfs: not found") and every later `apptainer exec
#            instance://...` says "no instance found" -> redis/runtime/juicefs
#            all run into the void and the pipeline dies at the first fatal
#            checkpoint (juicefs mount timeout). Job 21067 died exactly here
#            because the ssh apptainer (~/.local/jarvis-apptainer) lacked both.
#      So test SIF-open HERE, loudly, before the sweep. NB `ssh -n`: ssh reads
#      stdin and would swallow the rest of the hostfile in a while-read loop
#      (job 21067 only probed the first host for that reason).
echo "=== pre-flight over ssh: group + SIF openability + REAL fakeroot instance start per node (want group=$USER, sif=OK, fakeroot=OK) ==="
# We test the ACTUAL operation jarvis fans out -- `apptainer instance start
# --fakeroot` -- not just `apptainer exec <sif> true`. `exec` never exercises
# the fakeroot userns/subuid mapping, so it passes even on a node whose
# `instance start --fakeroot` will FATAL with "newuidmap was not found in PATH"
# (job 21239: ares-comp-15 lacked the uidmap setuid helpers while its 3 peers
# had them -> that node's instance never started -> chimaera SWIM-marked it
# DEAD -> the whole distributed run hung on retry-to-dead-node timeouts and
# produced no results.csv). Catching it here turns a 30-min silent hang into an
# immediate, node-named abort. A throwaway instance is started and stopped;
# newuidmap/newgidmap are reported so the failure message is actionable.
PREFLIGHT_OK=1
BAD_NODES=""
while read -r h; do
  [ -n "$h" ] || continue
  probe=$(env -u LD_LIBRARY_PATH ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no "$h" \
    'grp=$(id -gn); app=$(command -v apptainer 2>/dev/null || echo none);
     sqf=$(command -v squashfuse 2>/dev/null || echo no);
     unsq=$(command -v unsquashfs 2>/dev/null || echo no);
     nuid=$(command -v newuidmap 2>/dev/null || echo no);
     ngid=$(command -v newgidmap 2>/dev/null || echo no);
     if apptainer exec "'"$SIF_PATH"'" true >/dev/null 2>&1; then sif=OK; else sif=FAIL; fi;
     pn=pf_probe_$$;
     apptainer instance stop "$pn" >/dev/null 2>&1;
     if apptainer instance start --fakeroot "'"$SIF_PATH"'" "$pn" >/dev/null 2>&1; then
       fr=OK; apptainer instance stop "$pn" >/dev/null 2>&1;
     else fr=FAIL; fi;
     echo "group=$grp apptainer=$app squashfuse=$sqf unsquashfs=$unsq newuidmap=$nuid newgidmap=$ngid sif=$sif fakeroot=$fr"' \
    2>/dev/null || echo '<ssh failed>')
  echo "  [$h] $probe"
  case "$probe" in
    *sif=OK*fakeroot=OK*) : ;;
    *) PREFLIGHT_OK=0; BAD_NODES="$BAD_NODES $h" ;;
  esac
done < "$HOSTFILE"
if [ "$PREFLIGHT_OK" -ne 1 ]; then
  # Comma-joined for `sbatch --exclude` (which rejects spaces).
  BAD_CSV=$(echo "${BAD_NODES# }" | tr ' ' ',')
  {
    echo "ERROR: pre-flight failed on:$BAD_NODES"
    echo "  * sif=FAIL  -> the ssh-login apptainer can't open the squashfs SIF."
    echo "     jarvis fans 'apptainer instance start' over ssh, so THAT apptainer"
    echo "     (not the conda one on the batch PATH) must mount the SIF via"
    echo "     squashfuse (or extract via unsquashfs). If squashfuse=no AND"
    echo "     unsquashfs=no, make a squashfuse-capable apptainer win on the"
    echo "     ssh-login PATH (e.g. prepend the conda-forge apptainer bin in"
    echo "     ~/.bashrc so non-interactive ssh resolves it first)."
    echo "  * fakeroot=FAIL (newuidmap=no / newgidmap=no) -> that node lacks the"
    echo "     uidmap setuid helpers rootless --fakeroot needs. This is per-node"
    echo "     heterogeneity on Ares -- the node simply can't run our containers."
    echo "     Re-submit to land on different nodes, or exclude the bad one(s):"
    echo "       sbatch --exclude=$BAD_CSV jarvis_clio_core/scripts/run_distributed_ares.sbatch"
    echo "  Aborting before the sweep rather than hanging on a SWIM-dead node."
  } >&2
  exit 1
fi

# ---- clean slate so all 3 node-count rows re-run (sweep-resume otherwise
#      skips rows already present, including failed ones). ------------------
rm -f "$RESULTS"

# ---- run the 3-point node-count sweep ------------------------------------
# Run under our PERSONAL group ($USER), not the Slurm-assigned PROJECT group, so
# apptainer --fakeroot (the FUSE backends) works on the HEAD node — Slurm makes
# the project group our primary GID, which breaks fakeroot's GID mapping. See
# run_single_node_ares.sbatch for the full rationale.
#
# CAVEAT (verify at Gate 6): jarvis PSSHes `apptainer instance start --fakeroot`
# to the OTHER nodes over ssh; that head-node `sg` does NOT carry into the remote
# ssh sessions. If those sessions land in the project group (Slurm/pam_slurm_adopt
# adoption) rather than the personal group, the remote fakeroot will fail the same
# way and we'll need a jarvis-side group switch (or a ~/.bashrc guard). Confirm a
# remote node's group first:
#   srun --nodes=$SLURM_NNODES --ntasks-per-node=1 bash -lc 'id -gn'
# NB: `sg $USER -c` runs its arg under /bin/sh (dash), which has no `source` —
# use the POSIX `.` so conda activation works in BOTH the sg and non-sg branches.
RUN_SWEEP=". '$CONDA_SH' && conda activate '$CONDA_ENV' && jarvis ppl run yaml '${RUN_PIPELINE:-$PIPELINE}'"
if getent group "$USER" >/dev/null 2>&1 && [ "$(id -gn)" != "$USER" ]; then
  echo "switching primary group $(id -gn) -> $USER for apptainer --fakeroot"
  sg "$USER" -c "$RUN_SWEEP"
else
  bash -c "$RUN_SWEEP"
fi

# ---- report ---------------------------------------------------------------
echo "=== done — results at $RESULTS ==="
if [ -f "$RESULTS" ]; then
  # grep -c already prints "0" on no match (and exits 1); `|| echo 0` would
  # append a SECOND "0" -> n="0\n0" -> `[ "$n" -ge ... ]` errors "integer
  # expression expected" (seen in job 21067). `|| true` swallows the exit
  # WITHOUT re-printing; n=${n:-0} covers the file-error (exit 2, no stdout).
  n=$(grep -c success "$RESULTS" 2>/dev/null || true); n=${n:-0}
  echo "successful rows: $n / $EXPECT"
  head -1 "$RESULTS"
  [ "$n" -ge "$EXPECT" ] && echo "ALL $EXPECT GREEN" || echo "INCOMPLETE — inspect $RESULTS / the .err log"
else
  echo "NO results.csv produced — check distributed-${SLURM_JOB_ID:-local}.err"
fi
