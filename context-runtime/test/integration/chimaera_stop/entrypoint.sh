#!/bin/bash
# Single entrypoint for the distributed clio_run-stop CI test (issue #710).
#
#   NODE_ID=2 -> SSH server only; Jarvis drives this node remotely via pssh.
#   NODE_ID=1 -> primary: brings up SSH + Jarvis, then runs the stop gate and
#                exits with the gate's result.
#
# The gate: start the runtime on every node via Jarvis, then stop it via
# Jarvis's PsshExec fan-out (`clio_run runtime stop`, and `stop --force`), and
# assert every node reports a clean shutdown with zero stale artifacts via
# `clio_run runtime status`.

TEST_DIR=/workspace/context-runtime/test/integration/chimaera_stop
SSH_SHARE="${TEST_DIR}/.chimaera_ssh"
HOSTFILE="${TEST_DIR}/hostfile"
CLIO_BIN=/workspace/build/bin/clio_run
export LD_LIBRARY_PATH=/workspace/build/bin:/usr/local/lib:${LD_LIBRARY_PATH:-}

# ---------------------------------------------------------------------------
# Node 2: SSH server only. The runtime here is started/stopped by Jarvis.
# ---------------------------------------------------------------------------
if [ "${NODE_ID}" = "2" ]; then
    set -e
    echo '[node2] waiting for node1 public key'
    mkdir -p /home/iowarp/.ssh && chmod 700 /home/iowarp/.ssh
    while [ ! -f "${SSH_SHARE}/node1.pub" ]; do sleep 0.5; done
    cat "${SSH_SHARE}/node1.pub" >> /home/iowarp/.ssh/authorized_keys
    chmod 600 /home/iowarp/.ssh/authorized_keys
    echo '[node2] starting sshd (foreground); ready for Jarvis'
    sudo /usr/sbin/sshd -D
    exit 0
fi

# ---------------------------------------------------------------------------
# Node 1: primary. Setup is fail-fast (set -e); the gate section manages its
# own control flow so a failing probe is reported, not swallowed.
# ---------------------------------------------------------------------------
set -e
# shellcheck source=/dev/null
source /home/iowarp/venv/bin/activate

echo '[node1] SSH setup'
mkdir -p /home/iowarp/.ssh && chmod 700 /home/iowarp/.ssh
ssh-keygen -t rsa -b 2048 -N '' -f /home/iowarp/.ssh/id_rsa -q 2>/dev/null || true
sudo mkdir -p "${SSH_SHARE}" && sudo chmod 777 "${SSH_SHARE}"
cp /home/iowarp/.ssh/id_rsa.pub "${SSH_SHARE}/node1.pub"
cat /home/iowarp/.ssh/id_rsa.pub >> /home/iowarp/.ssh/authorized_keys
chmod 600 /home/iowarp/.ssh/authorized_keys
printf 'StrictHostKeyChecking no\nUserKnownHostsFile /dev/null\n' >> /home/iowarp/.ssh/config
sudo /usr/sbin/sshd

echo '[node1] waiting for node2 sshd'
NODE2_READY=0
for _ in $(seq 1 60); do
    if ssh -o ConnectTimeout=3 -o BatchMode=yes iowarp-jarvis-stop-node2 'echo ready' 2>/dev/null; then
        NODE2_READY=1; break
    fi
    sleep 1
done
[ "$NODE2_READY" = 1 ] || { echo 'ERROR: node2 SSH not ready after 60s'; exit 1; }

echo '[node1] initializing Jarvis'
sudo mkdir -p "${TEST_DIR}/.jarvis-shared" && sudo chmod 777 "${TEST_DIR}/.jarvis-shared"
jarvis init \
    /home/iowarp/.ppi-jarvis/config \
    /home/iowarp/.ppi-jarvis/private \
    "${TEST_DIR}/.jarvis-shared" \
    +force
jarvis repo add /workspace/jarvis_clio_core +force
jarvis hostfile set "${HOSTFILE}"

echo '[node1] building pipeline (clio_runtime only, tcp:9413)'
jarvis ppl create chimaera_stop
jarvis pkg append jarvis_clio_core.clio_runtime runtime
jarvis pkg conf runtime \
    port=9413 \
    ipc_mode=tcp \
    num_threads=4 \
    client_data_segment_size=2G \
    log_level=info \
    sleep=8

# ---------------------------------------------------------------------------
# Gate. clio_run runtime status: 0=RUNNING, 1=STOPPED(clean), 2=STOPPED(stale).
# Status prints to stderr. A live-but-wedged pid prints UNRESPONSIVE yet still
# returns 1, so the UNRESPONSIVE substring is checked independently of the exit
# code -- exit-code alone would silently pass a wedged (not stale, not clean)
# runtime, which is exactly the failure #710 is about.
# ---------------------------------------------------------------------------
set +e
STATUS_CMD="LD_LIBRARY_PATH=/workspace/build/bin:/usr/local/lib ${CLIO_BIN} runtime status"

assert_all() {
    # $1 = 'running' | 'stopped'
    local want="$1" host out rc fail=0
    while read -r host; do
        [ -z "$host" ] && continue
        out=$(ssh -o BatchMode=yes "$host" "$STATUS_CMD" 2>&1); rc=$?
        echo "[status:$want] $host -> rc=$rc"
        echo "$out" | sed 's/^/    /'
        if [ "$want" = running ]; then
            [ "$rc" = 0 ] || { echo "  FAIL: $host expected RUNNING (rc=0), got rc=$rc"; fail=1; }
        else
            if echo "$out" | grep -q 'UNRESPONSIVE'; then
                echo "  FAIL: $host runtime is UNRESPONSIVE (live but wedged)"; fail=1
            elif [ "$rc" = 2 ]; then
                echo "  FAIL: $host has STALE runtime artifacts after stop"; fail=1
            elif [ "$rc" != 1 ]; then
                echo "  FAIL: $host expected STOPPED clean (rc=1), got rc=$rc"; fail=1
            fi
        fi
    done < "$HOSTFILE"
    return $fail
}

overall=0

echo '=== Phase 1: graceful stop (jarvis ppl stop -> clio_run runtime stop) ==='
jarvis ppl start || { echo 'ERROR: jarvis ppl start failed'; overall=1; }
assert_all running || { echo 'PRECONDITION FAIL: runtime not RUNNING on all nodes after start'; overall=1; }
jarvis ppl stop || echo 'note: jarvis ppl stop returned nonzero (status probe below is authoritative)'
assert_all stopped || { echo 'GATE FAIL: stale runtime after graceful stop'; overall=1; }

echo '=== Phase 2: force stop (jarvis ppl kill -> clio_run runtime stop --force) ==='
jarvis ppl start || { echo 'ERROR: jarvis ppl start (phase 2) failed'; overall=1; }
assert_all running || { echo 'PRECONDITION FAIL: runtime not RUNNING on all nodes after restart'; overall=1; }
jarvis ppl kill || echo 'note: jarvis ppl kill returned nonzero (status probe below is authoritative)'
assert_all stopped || { echo 'GATE FAIL: stale runtime after force stop'; overall=1; }

echo '[node1] jarvis ppl clean'
jarvis ppl clean || true

if [ "$overall" = 0 ]; then
    echo 'RESULT: PASS -- clio_run stop killed the whole runtime on every node, no stale artifacts'
else
    echo 'RESULT: FAIL -- see the FAIL lines above'
fi
exit $overall
