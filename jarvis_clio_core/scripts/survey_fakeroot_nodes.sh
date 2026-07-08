#!/usr/bin/env bash
# =============================================================================
# survey_fakeroot_nodes.sh — find Ares compute nodes that CAN'T run our
# rootless-Apptainer containers, so you can hand the admin a concrete list (or
# build a --exclude set) instead of discovering bad nodes one Slurm allocation
# at a time.
#
# THE ONE THING WE'RE LOOKING FOR
#   Our pipeline mounts FUSE (JuiceFS + CTE libfuse) inside the container, which
#   needs CAP_SYS_ADMIN, which under a non-setuid apptainer needs `--fakeroot`,
#   which needs the setuid-root helpers `newuidmap`/`newgidmap` (the distro
#   `uidmap` package) to write the subuid/subgid range mapping. A node missing
#   them FATALs `apptainer instance start --fakeroot` with "newuidmap was not
#   found in PATH" — its chimaera never starts, the cluster SWIM-marks it dead,
#   and the whole distributed run hangs (jobs 21239=comp-15, 21253=comp-17).
#   Unlike squashfuse (a plain user binary we symlink into the shared ~/.local),
#   newuidmap MUST be setuid-root and system-installed — admins only. So the
#   presence of `uidmap` on a node is the crisp, admin-actionable signal.
#
# HOW IT REACHES NODES  (honest limitation up front)
#   On Ares you generally can't ssh to a compute node you don't hold a job on,
#   so by default this surveys via `srun` (Slurm-native). `srun --immediate`
#   can only land on nodes that are ALLOCATABLE right now — a busy node is
#   reported UNREACH, not GOOD/BAD. Re-run later to cover nodes that were busy,
#   or accumulate results across runs. (The sweep's own pre-flight remains the
#   per-run safety net for whatever nodes you actually land on.)
#
# USAGE
#   scripts/survey_fakeroot_nodes.sh                  # survey all of -p compute
#   scripts/survey_fakeroot_nodes.sh -p compute       # explicit partition
#   scripts/survey_fakeroot_nodes.sh -n ares-comp-[12-31]   # a specific set
#   scripts/survey_fakeroot_nodes.sh --ssh            # reach via ssh instead of srun
#                                                     #   (only works for nodes you
#                                                     #    can ssh to, e.g. in a job)
#   scripts/survey_fakeroot_nodes.sh --deep /path/to.sif    # REAL fakeroot
#                                                     #   instance start+stop probe
#                                                     #   (definitive, heavier)
#
# OUTPUT
#   One line per node (GOOD / BAD / UNREACH) with the evidence, then a summary:
#   the GOOD list, the BAD list, and a ready-to-paste `--exclude=<csv>` you can
#   hand to sbatch. Exit 0 if no BAD nodes were found, 1 if any were.
# =============================================================================
set -uo pipefail

PARTITION="compute"
NODELIST=""
MODE="srun"          # srun | ssh
DEEP_SIF=""          # non-empty -> real fakeroot instance-start probe

while [ $# -gt 0 ]; do
  case "$1" in
    -p|--partition) PARTITION="$2"; shift 2 ;;
    -n|--nodelist)  NODELIST="$2";  shift 2 ;;
    --ssh)          MODE="ssh";     shift ;;
    --deep)         DEEP_SIF="$2";  shift 2 ;;
    -h|--help)      sed -n '2,45p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1 (try --help)" >&2; exit 2 ;;
  esac
done

# ---- resolve the node list ------------------------------------------------
# -n takes an explicit Slurm nodelist expr (ranges allowed, e.g. comp-[12-31]);
# otherwise enumerate every node in the partition. scontrol expands ranges into
# one hostname per line.
if [ -n "$NODELIST" ]; then
  NODES=$(scontrol show hostnames "$NODELIST" 2>/dev/null | sort -u)
else
  NODES=$(sinfo -h -N -p "$PARTITION" -o '%N' 2>/dev/null | sort -u)
fi
if [ -z "${NODES:-}" ]; then
  echo "ERROR: no nodes found (partition='$PARTITION', nodelist='$NODELIST'). " \
       "Is Slurm on PATH / the partition name right?" >&2
  exit 2
fi

# ---- the per-node check (runs ON the node) --------------------------------
# Default: is `uidmap` present AND is newuidmap setuid? (`[ -u ]` tests the
# setuid bit directly — a non-setuid copy can't map subuid ranges and is as
# useless as a missing one.) --deep: actually start+stop a --fakeroot instance,
# the exact op jarvis fans out, so nothing about the fakeroot path is guessed.
if [ -n "$DEEP_SIF" ]; then
  CHECK='
    pn=survey_probe_$$
    apptainer instance stop "$pn" >/dev/null 2>&1
    if apptainer instance start --fakeroot "'"$DEEP_SIF"'" "$pn" >/dev/null 2>&1; then
      apptainer instance stop "$pn" >/dev/null 2>&1
      echo "GOOD fakeroot=OK sif='"$DEEP_SIF"'"
    else
      nu=$(command -v newuidmap 2>/dev/null || echo MISSING)
      echo "BAD fakeroot=FAIL newuidmap=$nu"
    fi'
else
  CHECK='
    nu=$(command -v newuidmap 2>/dev/null || echo MISSING)
    ng=$(command -v newgidmap 2>/dev/null || echo MISSING)
    su=no; [ "$nu" != MISSING ] && [ -u "$nu" ] && su=yes
    if [ "$nu" != MISSING ] && [ "$ng" != MISSING ] && [ "$su" = yes ]; then
      echo "GOOD newuidmap=$nu newgidmap=$ng setuid=$su"
    else
      echo "BAD newuidmap=$nu newgidmap=$ng setuid=$su"
    fi'
fi

# ---- run it per node ------------------------------------------------------
run_on_node() {
  local node="$1"
  if [ "$MODE" = ssh ]; then
    # env -u LD_LIBRARY_PATH: keep conda's libs from breaking /usr/bin/ssh's
    # OpenSSL (same reason the sweep's ssh_cmd strips it).
    env -u LD_LIBRARY_PATH ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no \
      -o ConnectTimeout=5 "$node" "$CHECK" 2>/dev/null
  else
    # --immediate: don't queue behind a busy node — bail so we can mark UNREACH.
    # -t 1: cap the step at a minute. --quiet: no srun bookkeeping on stderr.
    srun --quiet --immediate --nodes=1 --ntasks=1 --nodelist="$node" \
      -p "$PARTITION" -t 1 bash -c "$CHECK" 2>/dev/null
  fi
}

echo "=== fakeroot/uidmap survey: partition=$PARTITION mode=$MODE${DEEP_SIF:+ deep=$DEEP_SIF} ==="
GOOD=""; BAD=""; UNREACH=""
for node in $NODES; do
  out=$(run_on_node "$node")
  if [ -z "$out" ]; then
    echo "  [$node] UNREACH (busy/down or unreachable — re-run to cover it)"
    UNREACH="$UNREACH $node"
    continue
  fi
  echo "  [$node] $out"
  case "$out" in
    GOOD*) GOOD="$GOOD $node" ;;
    *)     BAD="$BAD $node" ;;
  esac
done

# ---- summary --------------------------------------------------------------
csv() { echo "${1# }" | tr ' ' ','; }
echo
echo "=== summary ==="
echo "GOOD    (${GOOD:+$(echo $GOOD | wc -w)}${GOOD:+ }nodes):$GOOD"
echo "BAD     (${BAD:+$(echo $BAD | wc -w)}${BAD:+ }nodes):$BAD"
echo "UNREACH (${UNREACH:+$(echo $UNREACH | wc -w)}${UNREACH:+ }nodes):$UNREACH"
if [ -n "$BAD" ]; then
  echo
  echo "Bad nodes lack setuid uidmap (admin fix: install the 'uidmap' package,"
  echo "ideally in the compute node image so it survives re-imaging)."
  echo "To keep running meanwhile, exclude them:"
  echo "  sbatch --exclude=$(csv "$BAD") jarvis_clio_core/scripts/run_distributed_ares.sbatch"
  exit 1
fi
echo "No BAD nodes among the reachable set."
[ -n "$UNREACH" ] && echo "(NB: $(echo $UNREACH | wc -w) node(s) were UNREACH — re-run later to survey them.)"
exit 0
