#!/bin/bash
# Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
# BSD 3-Clause License.
#
# End-to-end check of the no-flat-files GNN pipeline, at a size that runs in
# seconds:
#
#   .npz --stream--> gnn_ingest --> CTE (RAM + NVMe tiers)
#   .npz --stream--> gnn_build_csr --pipe--> gnn_aggregate --> CTE
#   CTE --readback--> compared against a numpy reference
#
# Nothing but the source .npz and the compressed store ever touches disk. This
# is the small-scale rehearsal for ogbn-papers100M; run it after touching any
# of the four tools.
#
#   BUILD=build-ci-check ./run_pipeline_test.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
BUILD="${BUILD:-build-ci-check}"
BIN="$ROOT/$BUILD/bin"
PY="${PY:-$HOME/.venv/bin/python}"
WORK="${WORK:-/tmp/gnn_pipeline_test}"
PORT="${PORT:-10778}"

N=${N:-4000}
F=${F:-64}
E=${E:-20000}
PAGE=$((F * 4 * 64))          # 64 rows per page
BLOCK=${BLOCK:-1500}          # < N so the aggregator makes several passes

command -v "$PY" >/dev/null || { echo "no python at $PY (set PY=)"; exit 1; }
for t in gnn_ingest gnn_aggregate; do
  [ -x "$BIN/$t" ] || { echo "missing $BIN/$t (build it first)"; exit 1; }
done

rm -rf "$WORK"; mkdir -p "$WORK/tier"
echo "### 0. dataset + reference"
"$PY" - "$WORK" "$N" "$F" "$E" <<'PYEOF'
import sys
import numpy as np
work, N, F, E = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rng = np.random.default_rng(21)
X = (rng.integers(0, 16, size=(N, F)) * 0.0625 - 0.5).astype(np.float16)
u = rng.integers(0, N, size=E); v = rng.integers(0, N, size=E)
np.savez(f"{work}/data.npz", node_feat=X,
         edge_index=np.stack([u, v]).astype(np.int64))
X32 = X.astype(np.float32)
adj = [set() for _ in range(N)]
for a, b in zip(u, v):
    adj[a].add(int(b)); adj[b].add(int(a))
A = np.zeros((N, F), dtype=np.float64)
for i in range(N):
    nb = sorted(adj[i])
    A[i] = (X32[i].astype(np.float64) +
            X32[nb].astype(np.float64).sum(axis=0)) / (len(nb) + 1.0)
np.save(f"{work}/ref.npy", A.astype(np.float32))
print(f"    N={N} F={F} directedE={E}")
PYEOF

cat > "$WORK/tiered.yaml" <<EOF
networking:
  port: $PORT
runtime:
  num_threads: 8
  queue_depth: 65536
compose:
  - mod_name: clio_bdev
    pool_name: "ram::chi_default_bdev"
    pool_query: local
    pool_id: "301.0"
    bdev_type: ram
    capacity: "256MB"
  - mod_name: clio_cte_compressor
    pool_name: cte_compressor
    pool_query: local
    pool_id: "600.0"
    next_pool_id: "512.0"
  - mod_name: clio_cte_core
    pool_name: cte_core
    pool_query: local
    pool_id: "512.0"
    storage:
      - path: "ram::gnn_ram_tier"
        bdev_type: "ram"
        capacity_limit: "2MB"
        score: 1.0
      - path: "$WORK/tier/nvme"
        bdev_type: "file"
        capacity_limit: "4096MB"
        score: 0.5
    dpe:
      dpe_type: "max_bw"
EOF

echo "### 1. daemon (RAM tier deliberately tiny, so everything spills to NVMe)"
CLIO_SERVER_CONF="$WORK/tiered.yaml" "$BIN/clio_run" runtime start \
    > "$WORK/daemon.log" 2>&1 &
DAEMON=$!
cleanup() { kill "$DAEMON" 2>/dev/null || true; }
trap cleanup EXIT
sleep 20
export CLIO_SERVER_CONF="$WORK/tiered.yaml"

echo "### 2. features -> CTE (no flat copy)"
"$PY" "$HERE/gnn_stream_npz.py" --zip "$WORK/data.npz" --member node_feat \
    --dtype float32 2>/dev/null \
  | "$BIN/gnn_ingest" --tag e2e_feat --page-bytes "$PAGE" 2>&1 | grep -E "DONE|FAILED"

echo "### 3. CSR -> aggregate, both streamed"
"$PY" "$HERE/gnn_stream_npz.py" --zip "$WORK/data.npz" --member edge_index \
    --dtype int64 2>/dev/null \
  | "$PY" "$HERE/gnn_build_csr.py" --nodes "$N" --edges "$E" 2>/dev/null \
  | "$BIN/gnn_aggregate" --feat-tag e2e_feat --out-tag e2e_agg --dim "$F" \
      --page-bytes "$PAGE" --block-rows "$BLOCK" 2>&1 | grep -E "DONE|rc=|FAILED"

PAGES=$(( (N + 63) / 64 ))
echo "### 4. read back and compare"
"$BIN/gnn_ingest" --tag e2e_agg --page-bytes "$PAGE" --read "$PAGES" \
    > "$WORK/got.f32" 2>/dev/null

"$PY" - "$WORK" "$F" <<'PYEOF'
import sys
import numpy as np
work, F = sys.argv[1], int(sys.argv[2])
ref = np.load(f"{work}/ref.npy")
got = np.fromfile(f"{work}/got.f32", dtype=np.float32).reshape(-1, F)[:ref.shape[0]]
d = np.abs(ref.astype(np.float64) - got.astype(np.float64))
print(f"    max_abs={d.max():.3e}  bit_exact={np.array_equal(ref, got)}")
if d.max() >= 1e-6:
    raise SystemExit("PIPELINE FAIL")
print("    PIPELINE PASS")
PYEOF

echo "### tier usage (this is where the matrix actually lives)"
du -h "$WORK"/tier/* 2>/dev/null | sed 's/^/    /'
