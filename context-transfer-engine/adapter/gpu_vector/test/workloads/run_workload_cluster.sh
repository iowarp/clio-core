#!/bin/bash
# Run an eternia workload against a 4-node Clio cluster in docker and check
# that it still computes the right answer.
#
#   ETERNIA_BIN_DIR=<workload build root> ./run_workload_cluster.sh [lammps|gromacs|lbann|all]
#
# WHAT THIS TESTS THAT THE SINGLE-NODE RUNS DO NOT
# ------------------------------------------------
# A vector's pages are CTE blobs and blobs hash across the cluster, so with
# three peers holding storage most of the workload's page faults are served
# from another node's memory rather than locally. Everything about the paging
# path that only works when the page is local fails here, and nowhere else.
#
# Each workload is checked against the SAME reference value its single-node
# harness uses, so a cluster result that differs is a real difference and not
# a differently-configured run.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"
WORK="$REPO_ROOT/.eternia_wl_work"
WHICH="${1:-all}"

: "${ETERNIA_BIN_DIR:?set ETERNIA_BIN_DIR to the root holding lmp-build/, gmx-build/, lbann-build/}"
export HOST_WORKSPACE="$REPO_ROOT" ETERNIA_BIN_DIR
export HOST_UID=$(id -u) HOST_GID=$(id -g)

# Reference values, from the validated single-node runs. Changing a reference
# to make a test pass defeats the point of having one.
LAMMPS_REF=-4.785579     # E_pair, 10^3 lj melt at step 20, stock lj/cut
GROMACS_REF=-8491.7693   # LJ(SR) at step 0, 12-cell argon, exact lattice sum
LBANN_REF="1.93652 1.87546 1.87988"   # objective per epoch, mlp.prototext

prepare() {
  rm -rf "$WORK"; mkdir -p "$WORK"
  # LAMMPS: the shipped example, scaled down so a cluster run is affordable.
  local E="$REPO_ROOT/external/lammps/examples/ETERNIA"
  sed -e 's/region          box block 0 40 0 40 0 40/region          box block 0 10 0 10 0 10/' \
      -e 's/^run             250/run             20/' \
      -e 's/^thermo          50/thermo          20/' "$E/in.melt.eternia" > "$WORK/in.melt.small"
  # GROMACS: grompp on the host, so the container needs only mdrun.
  local G="$REPO_ROOT/external/gromacs/src/gromacs/eternia/test"
  if [ -x "$ETERNIA_BIN_DIR/gmx-build/bin/gmx" ]; then
    ( cd "$WORK" && cp "$G/topol.top" . && python3 "$G/make_argon.py" 12 >/dev/null \
      && sed 's/^nsteps .*/nsteps          = 2/' "$G/md10.mdp" > md.mdp \
      && "$ETERNIA_BIN_DIR/gmx-build/bin/gmx" grompp -f md.mdp -c conf.gro -p topol.top \
           -o argon.tpr -maxwarn 5 >/dev/null 2>&1 )
  fi
}

# Did the peers actually serve? A cluster test that silently ran single-node
# would pass while testing nothing, which is the failure mode this whole
# harness exists to avoid. Node 4 refuses to start without three peers, so
# this is a second check on the same thing rather than the only one.
peers_joined() {
  # grep -c exits non-zero on no match, so `|| echo 0` appends a second line
  # and the numeric test then chokes on "0\n0" rather than reporting cleanly.
  local n
  n=$(grep -c "serving" "$1" 2>/dev/null)
  n=${n:-0}
  [ "$n" -ge 3 ] 2>/dev/null
}

# Run one workload on node 4 with the three storage peers up.
#
# Retried, because cluster FORMATION is intermittent: the peers occasionally
# hang in init with inter-node send timeouts before any workload has started.
# That is a runtime bring-up problem, not a workload problem -- the workload
# has not run at that point -- so retrying it is honest, whereas retrying a
# failed RESULT would not be. A retry is attempted only when the peers never
# reached "serving"; a workload that ran and produced a wrong answer is
# reported as-is.
run_cluster() {
  local cmd="$1" logfile="$2" attempt=1 rc=1
  export ETERNIA_WORKLOAD_CMD="$cmd"
  cd "$SCRIPT_DIR"
  while [ "$attempt" -le "${ETERNIA_FORM_RETRIES:-3}" ]; do
    docker compose down -v --remove-orphans >/dev/null 2>&1 || true
    rm -f "$REPO_ROOT/.eternia_cluster_done" "$REPO_ROOT"/.eternia_peer_ready.* 2>/dev/null
    # Let the previous cluster's sockets and shared segments go away. This is
    # not cosmetic: at 20s, formation failed five times running -- including
    # with a trivial echo as the workload, on an idle host with a clean GPU --
    # and at 45s it succeeded first try. Back-to-back compose runs are where
    # this breaks, so the settle is the difference between a usable harness
    # and one that looks broken.
    sleep "${ETERNIA_SETTLE:-45}"
    docker compose up --abort-on-container-exit --exit-code-from wl-node4 \
      > "$logfile" 2>&1
    rc=$?
    docker compose down -v --remove-orphans >/dev/null 2>&1 || true
    if peers_joined "$logfile"; then break; fi
    echo "    (attempt $attempt: cluster never formed, retrying)" >&2
    attempt=$((attempt+1))
  done
  return $rc
}


pass=0; fail=0
report() { # name ok detail
  if [ "$2" = "ok" ]; then echo "  PASS  $1 -- $3"; pass=$((pass+1))
  else echo "  FAIL  $1 -- $3"; fail=$((fail+1)); fi
}

prepare

if [ "$WHICH" = "lammps" ] || [ "$WHICH" = "all" ]; then
  run_cluster "cd /workspace/.eternia_wl_work && /sp/lmp-build/lmp -in in.melt.small" \
              "$WORK/lammps.log"
  # docker compose prefixes every line with "wl-nodeN  | ", so the thermo
  # table has to be de-prefixed before it looks like LAMMPS output at all.
  v=$(sed 's/^wl-node[0-9]* *| *//' "$WORK/lammps.log" \
      | awk '/Step/{f=1} f&&/^ *20 /{print $3; exit}')
  if ! peers_joined "$WORK/lammps.log"; then
    report lammps bad "peers never joined -- not a cluster run"
  else
    ok=$(python3 -c "
try: print('ok' if abs(float('${v:-nan}') - ($LAMMPS_REF)) < 1e-4 else 'bad')
except Exception: print('bad')")
    report lammps "$ok" "E_pair=${v:-none} ref=$LAMMPS_REF"
  fi
fi

if [ "$WHICH" = "gromacs" ] || [ "$WHICH" = "all" ]; then
  C6=$(python3 -c "s=0.3345;e=1.045128;print(4*e*s**6)")
  C12=$(python3 -c "s=0.3345;e=1.045128;print(4*e*s**12)")
  run_cluster "cd /workspace/.eternia_wl_work && GMX_ETERNIA_NB=1 GMX_ETERNIA_C6=$C6 GMX_ETERNIA_C12=$C12 GMX_ETERNIA_RC=1.0 GMX_ETERNIA_PAGE_KB=4 GMX_ETERNIA_BLOCKS=16 GMX_ETERNIA_SLOTS=2 /sp/gmx-build/bin/gmx mdrun -s argon.tpr -nb gpu -ntmpi 1 -ntomp 2 -deffnm wl -nsteps 2" \
              "$WORK/gromacs.log"
  v=$(grep -oE "(^|[^_])E=[-0-9.]+" "$WORK/gromacs.log" | grep -oE "\-[0-9.]+" | head -1)
  if ! peers_joined "$WORK/gromacs.log"; then
    report gromacs bad "peers never joined -- not a cluster run"
  elif grep -q "RESULT INVALID" "$WORK/gromacs.log"; then
    report gromacs bad "failed page reads reported"
  else
    ok=$(python3 -c "
try: print('ok' if abs(float('${v:-nan}') - ($GROMACS_REF)) < 1e-2 else 'bad')
except Exception: print('bad')")
    report gromacs "$ok" "LJ(SR)=${v:-none} ref=$GROMACS_REF"
  fi
fi

if [ "$WHICH" = "lbann" ] || [ "$WHICH" = "all" ]; then
  P=/workspace/external/lbann/src/eternia/test/mlp.prototext
  run_cluster "cd /workspace/.eternia_wl_work && LBANN_ETERNIA_FC=1 /sp/lbann-build/bin/lbann --prototext=$P" \
              "$WORK/lbann.log"
  v=$(grep -oE "objective function : [0-9.]+" "$WORK/lbann.log" | awk '{printf "%s ",$NF}' | sed 's/ $//')
  if ! peers_joined "$WORK/lbann.log"; then
    report lbann bad "peers never joined -- not a cluster run"
  elif [ "$v" = "$LBANN_REF" ]; then
    report lbann ok "objective=[$v]"
  else
    report lbann bad "objective=[${v:-none}] ref=[$LBANN_REF]"
  fi
fi

echo
echo "  $pass passed, $fail failed"
[ "$fail" -eq 0 ]
