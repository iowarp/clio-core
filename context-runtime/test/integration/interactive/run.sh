#!/bin/bash
# Interactive Dashboard Cluster
#
# Starts a 4-node Chimaera runtime cluster in Docker with the dashboard
# running on node 1. A local port-forward (socat) makes port 5000
# available on this devcontainer so VS Code auto-forwards it to the host.
#
# Usage:
#   bash run.sh          # start cluster + forward (Ctrl-C to stop)
#   bash run.sh start    # start in background
#   bash run.sh stop     # stop everything
#   bash run.sh logs     # follow container logs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

if [ -n "${HOST_WORKSPACE:-}" ]; then
    export IOWARP_CORE_ROOT="${HOST_WORKSPACE}"
elif [ -z "${IOWARP_CORE_ROOT:-}" ]; then
    export IOWARP_CORE_ROOT="${REPO_ROOT}"
fi

DASHBOARD_PORT="${DASHBOARD_PORT:-5000}"
NODE1_IP="172.28.0.10"
NETWORK_NAME="interactive_interactive-cluster"

cd "$SCRIPT_DIR"

# Connect devcontainer to the Docker cluster network so socat can reach node 1
connect_network() {
    local container_id
    container_id="$(hostname)"
    if docker network inspect "$NETWORK_NAME" &>/dev/null; then
        if ! docker inspect "$container_id" --format '{{json .NetworkSettings.Networks}}' 2>/dev/null | grep -q "$NETWORK_NAME"; then
            echo "Connecting devcontainer to cluster network..."
            docker network connect "$NETWORK_NAME" "$container_id" 2>/dev/null || true
        fi
    fi
}

disconnect_network() {
    docker network disconnect "$NETWORK_NAME" "$(hostname)" 2>/dev/null || true
}

# Forward local port to node 1's dashboard inside Docker
start_port_forward() {
    # Kill any existing forward on this port
    pkill -f "dashboard_portfwd" 2>/dev/null || true
    sleep 0.5

    python3 -c "
import socket, threading, sys, os
os.environ['PROC_NAME'] = 'dashboard_portfwd'

def forward(src, dst):
    try:
        while True:
            data = src.recv(4096)
            if not data:
                break
            dst.sendall(data)
    except Exception:
        pass
    finally:
        src.close()
        dst.close()

def accept_loop():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', ${DASHBOARD_PORT}))
    srv.listen(16)
    while True:
        client, _ = srv.accept()
        try:
            remote = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            remote.connect(('${NODE1_IP}', ${DASHBOARD_PORT}))
            threading.Thread(target=forward, args=(client, remote), daemon=True).start()
            threading.Thread(target=forward, args=(remote, client), daemon=True).start()
        except Exception:
            client.close()

accept_loop()
" &
    echo $! > /tmp/dashboard-fwd.pid
    echo "Port forward: localhost:${DASHBOARD_PORT} -> ${NODE1_IP}:${DASHBOARD_PORT} (PID $(cat /tmp/dashboard-fwd.pid))"
}

stop_port_forward() {
    if [ -f /tmp/dashboard-fwd.pid ]; then
        kill "$(cat /tmp/dashboard-fwd.pid)" 2>/dev/null || true
        rm -f /tmp/dashboard-fwd.pid
    fi
    pkill -f "dashboard_portfwd" 2>/dev/null || true
}

wait_for_dashboard() {
    echo "Waiting for dashboard to become ready..."
    for i in $(seq 1 30); do
        if curl -sf "http://${NODE1_IP}:${DASHBOARD_PORT}/api/topology" >/dev/null 2>&1; then
            echo "Dashboard is ready."
            return 0
        fi
        sleep 2
    done
    echo "Warning: dashboard did not become ready within 60s (may still be initializing)"
    return 0
}

stop_all() {
    echo ""
    echo "Stopping port forward..."
    stop_port_forward

    echo "Disconnecting from cluster network..."
    disconnect_network

    echo "Stopping Docker cluster..."
    cd "$SCRIPT_DIR"
    docker compose down 2>/dev/null || true

    echo "Stopped."
}

case "${1:-foreground}" in
    start)
        echo "Starting 4-node runtime cluster with dashboard..."
        docker compose up -d

        echo "Waiting for cluster to initialize..."
        connect_network
        wait_for_dashboard

        start_port_forward

        echo ""
        echo "======================================="
        echo "  Dashboard: http://localhost:${DASHBOARD_PORT}"
        echo "======================================="
        echo ""
        echo "  View logs:   bash $0 logs"
        echo "  Stop:        bash $0 stop"
        ;;

    stop)
        stop_all
        ;;

    logs)
        docker compose logs -f
        ;;

    foreground)
        echo "Starting 4-node runtime cluster with dashboard..."
        docker compose up -d

        echo "Waiting for cluster to initialize..."
        connect_network
        wait_for_dashboard

        start_port_forward

        echo ""
        echo "======================================="
        echo "  Dashboard: http://localhost:${DASHBOARD_PORT}"
        echo "  Press Ctrl-C to stop the cluster"
        echo "======================================="
        echo ""

        trap stop_all EXIT
        # Tail node 1 logs until interrupted
        docker compose logs -f iowarp-node1
        ;;

    *)
        cat <<EOF
Usage: $0 [COMMAND]

Commands:
    (none)     Start cluster + dashboard in foreground (Ctrl-C to stop)
    start      Start cluster + dashboard in background
    stop       Stop everything
    logs       Follow Docker container logs

Dashboard: http://localhost:${DASHBOARD_PORT}
EOF
        ;;
esac
