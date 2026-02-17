#!/bin/bash
# Run IOWarp Expand/Induction Integration Test
#
# Verifies that a new node can join an existing cluster via --induct.
# - Starts a 2-node cluster (nodes 1-2) with hostfile_initial
# - Starts node 3 with --induct and hostfile_full
# - Verifies induction via log messages

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

# Export workspace path for docker-compose
if [ -n "${HOST_WORKSPACE:-}" ]; then
    export IOWARP_CORE_ROOT="${HOST_WORKSPACE}"
elif [ -z "${IOWARP_CORE_ROOT:-}" ]; then
    export IOWARP_CORE_ROOT="${REPO_ROOT}"
fi

# Configuration
POLL_INTERVAL=2
MAX_WAIT=60

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Start Docker cluster
start_docker_cluster() {
    log_info "Starting Docker cluster (2 initial nodes + 1 inducted node)..."
    cd "$SCRIPT_DIR"

    docker compose up -d

    log_info "Waiting for containers to initialize..."
    sleep 5

    docker compose ps

    log_success "Docker cluster started"
}

# Stop Docker cluster
stop_docker_cluster() {
    log_info "Stopping Docker cluster..."
    cd "$SCRIPT_DIR"
    docker compose down
    log_success "Docker cluster stopped"
}

# Verify induction succeeded by checking logs
verify_induction() {
    log_info "Verifying node induction..."
    local elapsed=0

    # Poll node 3 logs for successful induction
    log_info "Waiting for node 3 to complete induction (timeout: ${MAX_WAIT}s)..."
    while [ $elapsed -lt $MAX_WAIT ]; do
        if docker logs iowarp-expand-node3 2>&1 | grep -q "Node inducted successfully as node_id="; then
            log_success "Node 3 induction confirmed"
            break
        fi
        sleep $POLL_INTERVAL
        elapsed=$((elapsed + POLL_INTERVAL))
        log_info "  Waiting... (${elapsed}s / ${MAX_WAIT}s)"
    done

    if [ $elapsed -ge $MAX_WAIT ]; then
        log_error "Timed out waiting for node 3 induction"
        log_error "Node 3 logs:"
        docker logs iowarp-expand-node3 2>&1 | tail -30
        return 1
    fi

    # Check that nodes 1-2 registered the new node
    log_info "Checking node 1 registered the new node..."
    if docker logs iowarp-expand-node1 2>&1 | grep -q "AddNode: Registered"; then
        log_success "Node 1 registered the inducted node"
    else
        log_error "Node 1 did not register the inducted node"
        log_error "Node 1 logs:"
        docker logs iowarp-expand-node1 2>&1 | tail -30
        return 1
    fi

    log_info "Checking node 2 registered the new node..."
    if docker logs iowarp-expand-node2 2>&1 | grep -q "AddNode: Registered"; then
        log_success "Node 2 registered the inducted node"
    else
        log_error "Node 2 did not register the inducted node"
        log_error "Node 2 logs:"
        docker logs iowarp-expand-node2 2>&1 | tail -30
        return 1
    fi

    # Verify all containers are still running
    log_info "Checking all containers are still running..."
    local running
    running=$(docker compose -f "$SCRIPT_DIR/docker-compose.yml" ps --status running -q | wc -l)
    if [ "$running" -ge 3 ]; then
        log_success "All 3 containers are running"
    else
        log_error "Expected 3 running containers, found $running"
        docker compose ps
        return 1
    fi

    log_success "Induction verification passed"
}

# Usage
usage() {
    cat << EOF
Usage: $0 [COMMAND]

Commands:
    setup       Start the Docker cluster
    run         Verify induction (cluster must be running)
    clean       Stop the Docker cluster
    all         Setup, verify, and clean up (default)

Examples:
    $0 all
    $0 setup
    $0 run
    $0 clean
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        setup|run|clean|all)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

COMMAND=${COMMAND:-all}

# Main execution
log_info "IOWarp Expand/Induction Test Runner"
log_info "  Workspace path: $IOWARP_CORE_ROOT"
log_info ""

case $COMMAND in
    setup)
        start_docker_cluster
        ;;

    run)
        verify_induction
        ;;

    clean)
        stop_docker_cluster
        ;;

    all)
        # Always clean up, even on failure
        EXIT_CODE=0
        start_docker_cluster
        verify_induction || EXIT_CODE=$?
        stop_docker_cluster
        if [ $EXIT_CODE -ne 0 ]; then
            log_error "Induction test FAILED"
            exit $EXIT_CODE
        fi
        log_success "Induction test PASSED"
        ;;

    *)
        log_error "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac
