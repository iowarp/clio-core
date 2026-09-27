#!/usr/bin/env bash
#
# Sudo-free variant of run_clio_scratch_xfstests.sh: drive the SCRATCH-requiring
# xfstests against the clio FUSE filesystem without ever touching the real /sbin.
#
# How the scratch device works without root:
#   * The whole run executes inside `unshare -rm` (user + mount namespace), where
#     we are namespace-root and unprivileged FUSE mounts work.
#   * mount(8) only looks for `mount.fuse.<subtyp>` helpers in /sbin (a symlink to
#     /usr/sbin here). Inside the private mount namespace we bind the real
#     /usr/sbin aside, build a shadow dir of symlinks to every real entry plus our
#     mount.fuse.<subtyp> helper, and bind the shadow over /usr/sbin. The host's
#     /sbin is never modified and the change vanishes with the namespace.
#   * Like the sudo driver: two persistent "keeper" runtimes (TEST and SCRATCH,
#     fully isolated namespaces) that the per-test xfstests client mounts attach to.
#
# Hang handling: every test gets a wall-clock budget. On expiry the FUSE
# connections of our mounts are aborted (sysfs abort if writable, and by killing
# our own uniquely-named FUSE daemons, which aborts the connection in-kernel),
# then lazily unmounted, so no process is left in D state. Only our renamed
# processes are ever killed.
#
# Usage:
#   scripts/xfstests/run_clio_scratch_nosudo.sh <testlist-file> <results-file>
#   (testlist: one generic/NNN per line; tests already in <results-file> are
#    skipped, so a run can be resumed)
# Env: CLIO_BUILD_DIR (expects $CLIO_BUILD_DIR/bin), XFSTESTS_DIR,
#      CLIO_SCRATCH_PRIV (private work dir, node-local), PERTEST_TIMEOUT (150),
#      CLIO_SCRATCH_TEST_PORT / CLIO_SCRATCH_SCRATCH_PORT, CLIO_SCRATCH_SUBTYP,
#      CLIO_SCRATCH_FAILDIR (per-test FAIL/HANG diff snippets).
#
set -u

LISTFILE="${1:?usage: run_clio_scratch_nosudo.sh <testlist> <results>}"
RESULTS="${2:?usage: run_clio_scratch_nosudo.sh <testlist> <results>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_BIN="${CLIO_BUILD_DIR:-${REPO_ROOT}/build}/bin"
XFSTESTS_DIR="${XFSTESTS_DIR:-${REPO_ROOT}/external/xfstests}"
PRIV="${CLIO_SCRATCH_PRIV:-/tmp/clio_scratch_nosudo_$(id -un)}"
TEST_PORT="${CLIO_SCRATCH_TEST_PORT:-7400}"
SCRATCH_PORT="${CLIO_SCRATCH_SCRATCH_PORT:-7410}"   # >=4 above TEST_PORT
SUBTYP="${CLIO_SCRATCH_SUBTYP:-cliofsns}"
TIMEOUT="${PERTEST_TIMEOUT:-150}"
FAILDIR="${CLIO_SCRATCH_FAILDIR:-${PRIV}/fail}"
CAP_CONF="${CLIO_SCRATCH_SERVER_CONF:-${SCRIPT_DIR}/clio_xfstests_config.yaml}"

[ -x "${XFSTESTS_DIR}/check" ]      || { echo "ERROR: no xfstests at ${XFSTESTS_DIR}" >&2; exit 1; }
[ -x "${BUILD_BIN}/clio_cte_fuse" ] || { echo "ERROR: clio_cte_fuse not in ${BUILD_BIN}" >&2; exit 1; }

# --- enter user+mount namespace (namespace root; no sudo) --------------------
if [ -z "${CLIO_SCRATCH_INNS:-}" ]; then
  RESULTS="$(realpath -m "${RESULTS}")"; LISTFILE="$(realpath -m "${LISTFILE}")"
  exec env CLIO_SCRATCH_INNS=1 unshare -rm bash "$0" "${LISTFILE}" "${RESULTS}"
fi

CLF="${PRIV}/clf_${SUBTYP}"          # renamed clio_cte_fuse (pkill-immune/unique)
TEST_DIR="${PRIV}/test"
SCRATCH_MNT="${PRIV}/scratch"
KEEP_TEST="${PRIV}/keeper_test"
KEEP_SCRATCH="${PRIV}/keeper_scratch"
PRIV_XFS="${PRIV}/xfstests"
REAL_SBIN="${PRIV}/real_sbin"
SHADOW_SBIN="${PRIV}/shadow_sbin"
mkdir -p "${PRIV}" "${TEST_DIR}" "${SCRATCH_MNT}" "${KEEP_TEST}" "${KEEP_SCRATCH}" \
         "${REAL_SBIN}" "${FAILDIR}" "$(dirname "${RESULTS}")"
touch "${RESULTS}"
cp -f "${BUILD_BIN}/clio_cte_fuse" "${CLF}"

# Env shared by every clio process. The binaries' RUNPATH points at the tree they
# were built in; LD_LIBRARY_PATH pins them to BUILD_BIN's libs instead.
CLIO_ENV=(LD_LIBRARY_PATH="${BUILD_BIN}" CLIO_REPO_PATH="${BUILD_BIN}"
          CLIO_BIND_ADDR=127.0.0.1 CLIO_IPC_MODE=SHM
          CLIO_RESTART_LOG="${PRIV}/restart_log.bin")

# --- shadow /usr/sbin so mount(8) finds our mount.fuse.<subtyp> helper -------
install_mount_helper() {
  mountpoint -q "${REAL_SBIN}" || mount --bind /usr/sbin "${REAL_SBIN}" || return 1
  rm -rf "${SHADOW_SBIN}"; mkdir -p "${SHADOW_SBIN}"
  local f
  for f in "${REAL_SBIN}"/* "${REAL_SBIN}"/.[!.]*; do
    [ -e "$f" ] || [ -L "$f" ] || continue
    ln -s "$f" "${SHADOW_SBIN}/$(basename "$f")"
  done
  cat > "${SHADOW_SBIN}/mount.fuse.${SUBTYP}" <<EOF
#!/bin/bash
# mount(8) invokes: mount.fuse.${SUBTYP} <device> <mountpoint> [-o opts]
dev="\$1"; mnt="\$2"
case "\$dev" in *scratch*) port=${SCRATCH_PORT} ;; *) port=${TEST_PORT} ;; esac
exec env ${CLIO_ENV[*]} CLIO_PORT=\$port CLIO_WITH_RUNTIME=0 \\
  "${CLF}" "\$mnt" -o "fsname=\$dev,allow_other,default_permissions"
EOF
  chmod +x "${SHADOW_SBIN}/mount.fuse.${SUBTYP}"
  mount --bind "${SHADOW_SBIN}" /usr/sbin || return 1
  [ -x "/sbin/mount.fuse.${SUBTYP}" ]
}
install_mount_helper || { echo "ERROR: could not shadow /usr/sbin" >&2; exit 1; }

# --- FUSE connection abort / teardown ----------------------------------------
our_mounts() { printf '%s\n' "${TEST_DIR}" "${SCRATCH_MNT}" "${KEEP_TEST}" "${KEEP_SCRATCH}"; }

# Abort the kernel FUSE connection of every mount of ours (releases D-state).
abort_fuse_conns() {
  local m c
  for m in $(our_mounts); do
    for c in $(awk -v m="$m" '$5==m {split($3,a,":"); print a[2]}' /proc/self/mountinfo); do
      echo 1 > "/sys/fs/fuse/connections/${c}/abort" 2>/dev/null
    done
  done
  # Killing the daemon closes /dev/fuse, which aborts the connection in-kernel
  # even where the fusectl abort file is not writable from the user namespace.
  pkill -9 -x "clf_${SUBTYP}" 2>/dev/null
}
lazy_unmount_all() {
  local m
  for m in $(our_mounts); do
    fusermount3 -uz "$m" 2>/dev/null; umount -l "$m" 2>/dev/null
  done
}
cleanup() { abort_fuse_conns; lazy_unmount_all; }
trap 'cleanup; exit 0' EXIT
trap 'exit 1' INT TERM

start_keeper() {  # $1=port $2=mnt $3=log
  env "${CLIO_ENV[@]}" CLIO_PORT="$1" CLIO_WITH_RUNTIME=1 CLIO_SERVER_CONF="${CAP_CONF}" \
    "${CLF}" "$2" -o fsname=keeper -f >"$3" 2>&1 &
  disown $! 2>/dev/null
  for _ in $(seq 1 80); do mountpoint -q "$2" && return 0; sleep 0.25; done
  return 1
}
restart_keepers() {
  abort_fuse_conns; lazy_unmount_all; sleep 1
  rm -f "${PRIV}/restart_log.bin"
  start_keeper "${TEST_PORT}"    "${KEEP_TEST}"    "${PRIV}/rt_test.log" &&
  start_keeper "${SCRATCH_PORT}" "${KEEP_SCRATCH}" "${PRIV}/rt_scratch.log"
}

# --- private xfstests copy + native fuse config ------------------------------
if [ ! -x "${PRIV_XFS}/check" ]; then
  rm -rf "${PRIV_XFS}"; cp -a "${XFSTESTS_DIR}/." "${PRIV_XFS}/"
fi
rm -rf "${PRIV_XFS}/results"; mkdir -p "${PRIV_XFS}/results"
cat > "${PRIV_XFS}/local.config" <<EOF
export FSTYP=fuse
export FUSE_SUBTYP=.${SUBTYP}
export TEST_DEV=clio_test
export TEST_DIR=${TEST_DIR}
export SCRATCH_DEV=clio_scratch
export SCRATCH_MNT=${SCRATCH_MNT}
EOF

# Save the diff/out.bad snippet of a FAIL/HANG test.  $1=test $2=status $3=check-out
save_fail() {
  local t="$1" n="${1//\//_}" f
  f="${FAILDIR}/${n}.txt"
  {
    echo "== ${t}: $2"; echo "== ./check output:"; head -c 6000 "$3"
    local bad="${PRIV_XFS}/results/${t}.out.bad"
    if [ -f "${bad}" ]; then
      echo; echo "== diff tests/${t}.out results/${t}.out.bad (head):"
      diff -u "${PRIV_XFS}/tests/${t}.out" "${bad}" | head -80
    fi
    [ -f "${PRIV_XFS}/results/${t}.full" ] && { echo "== .full (tail):"; tail -30 "${PRIV_XFS}/results/${t}.full"; }
  } > "${f}" 2>&1
}

cd "${PRIV_XFS}" || exit 1
restart_keepers || { echo "ERROR: keeper runtimes failed to start" >&2; tail -5 "${PRIV}"/rt_*.log >&2; exit 1; }

n=0; since_restart=0; total=$(grep -cE '^generic/' "${LISTFILE}")
while read -r t; do
  case "$t" in generic/*) ;; *) continue ;; esac
  n=$((n+1))
  grep -qE "^${t} : " "${RESULTS}" && continue           # resume support
  { mountpoint -q "${KEEP_TEST}" && mountpoint -q "${KEEP_SCRATCH}"; } || restart_keepers
  OUT="${PRIV}/check_out.txt"
  # own process group, so a hang can take down the whole test tree
  setsid ./check "${t}" >"${OUT}" 2>&1 &
  cpid=$!; waited=0; hung=1
  while [ "${waited}" -lt "${TIMEOUT}" ]; do
    kill -0 "${cpid}" 2>/dev/null || { hung=0; break; }
    sleep 1; waited=$((waited+1))
  done
  if [ "${hung}" -eq 1 ]; then
    abort_fuse_conns; lazy_unmount_all
    kill -9 -- "-${cpid}" 2>/dev/null; kill -9 "${cpid}" 2>/dev/null
    wait "${cpid}" 2>/dev/null
    st=HANG
  else
    wait "${cpid}" 2>/dev/null
    # notrun also prints "Passed all 0 tests", so test "Not run:" first.
    if   grep -q '^Not run:'   "${OUT}"; then st=notrun
    elif grep -q '^Passed all' "${OUT}"; then st=pass
    else st=FAIL; fi
  fi
  echo "${t} : ${st}" >> "${RESULTS}"
  echo "${t} : ${st}  (${n}/${total}, ${waited}s)"
  case "${st}" in FAIL|HANG) save_fail "${t}" "${st}" "${OUT}" ;; esac
  fusermount3 -u "${TEST_DIR}" 2>/dev/null; fusermount3 -u "${SCRATCH_MNT}" 2>/dev/null
  since_restart=$((since_restart+1))
  if [ "${st}" = HANG ] || [ "${since_restart}" -ge 40 ]; then
    restart_keepers; since_restart=0
  fi
done < "${LISTFILE}"
echo "[scratch-nosudo] done ${n} tests"
