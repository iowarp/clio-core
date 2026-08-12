#!/bin/bash
# Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
# BSD 3-Clause License.
#
# ogbn-papers100M end to end, with no flat copy of the feature matrix anywhere.
#
#   papers100M-bin.zip --stream--> CSR file (graph only, 12.9 GiB)
#   papers100M-bin.zip --stream--> FIFO --> trainer:
#         ingest raw features into the CTE (RAM + NVMe tiers)
#      -> aggregate CTE-to-CTE in the same process
#      -> verify sampled rows against a recomputation
#      -> train on the GPU, faulting pages out of the vector
#
# The trainer does all four because a process that faults on the GPU must host
# the runtime itself (GPU queues are only created by ServerInitGpuQueues; there
# is no client-side attach). Splitting the work across a daemon would mean
# exporting the 53 GiB aggregate and re-ingesting it.
#
# The features reach the trainer through a FIFO, not a file: the store loop
# reads them once, sequentially, so nothing needs to land on disk. The CSR is a
# real file because aggregation re-reads it once per destination block.
#
#   ZIP=/path/papers100M-bin.zip ./run_papers100M.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
BUILD="${BUILD:-build-ci-check}"
BIN="$ROOT/$BUILD/bin"
PY="${PY:-$HOME/.venv/bin/python}"
ZIP="${ZIP:-$HOME/gnn/data/papers100M-bin.zip}"
WORK="${WORK:-$HOME/gnn/papers}"
PORT="${PORT:-10850}"

# OGB's published shape for ogbn-papers100M.
N=${N:-111059956}
F=${F:-128}
E=${E:-1615685872}
C=${C:-172}
PAGE_KIB=${PAGE_KIB:-1024}
# Accumulator rows per aggregation pass. Bigger = fewer passes over the 53 GiB
# of features (measured ~4.2 min each) but more RAM: rows * F * 8 bytes.
AGG_BLOCK=${AGG_BLOCK:-16000000}
EPOCHS=${EPOCHS:-3}
HIDDEN=${HIDDEN:-64}
SKIP_AGG=${SKIP_AGG:-0}     # 1 = train on raw features, no aggregation

[ -f "$ZIP" ] || { echo "no archive at $ZIP (set ZIP=)"; exit 1; }
command -v "$PY" >/dev/null || { echo "no python at $PY (set PY=)"; exit 1; }
[ -x "$BIN/test_gpu_vector_gnn_train" ] || { echo "build test_gpu_vector_gnn_train first"; exit 1; }

mkdir -p "$WORK/tier"

# ---------------------------------------------------------------------------
# Preflight. This run costs the better part of an hour before it touches the
# GPU, so the failure mode to avoid is running out of disk two thirds of the
# way in. Everything below is arithmetic on known sizes, not guesswork.
# ---------------------------------------------------------------------------
avail_gib=$(( $(df --output=avail "$WORK" | tail -1) / 1048576 ))
"$PY" - "$N" "$F" "$E" "$ZIP" "$avail_gib" "$SKIP_AGG" <<'PYEOF'
import os, sys
N, F, E = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
zip_path, avail, skip_agg = sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
GiB = 2**30
feat = N * F * 4 / GiB
stored = feat / 1.078            # measured zstd ratio on real float features
csr = (2 + (N + 1) + E) * 8 / GiB
zsz = os.path.getsize(zip_path) / GiB
csr_needed = 0.0 if skip_agg else csr
agg_needed = 0.0 if skip_agg else stored
peak_ingest = zsz + csr_needed + stored
peak_agg = csr_needed + stored + agg_needed
peak = max(peak_ingest, peak_agg)
print(f"[preflight] features {feat:.1f} GiB logical, ~{stored:.1f} GiB stored")
print(f"[preflight] csr {csr_needed:.1f} GiB, archive {zsz:.1f} GiB")
print(f"[preflight] peak disk needed ~{peak:.1f} GiB; available {avail} GiB")
if peak > avail:
    print(f"[preflight] SHORT BY ~{peak - avail:.0f} GiB -- refusing to start.")
    print("[preflight] Options: free that much, set SKIP_AGG=1 to train on raw")
    print("[preflight] features (drops the CSR and the second copy), or reduce N.")
    raise SystemExit(2)
print("[preflight] OK")
PYEOF

echo "### shapes"
"$PY" "$HERE/gnn_stream_npz.py" --zip "$ZIP" --member node_feat --info

if [ "$SKIP_AGG" = "0" ]; then
  echo "### 1. CSR (streamed from the archive; ~13 GiB on disk, re-read per pass)"
  if [ ! -s "$WORK/graph.csr" ]; then
    "$PY" "$HERE/gnn_stream_npz.py" --zip "$ZIP" --member edge_index --dtype int64 \
      | "$PY" "$HERE/gnn_build_csr.py" --nodes "$N" --edges "$E" > "$WORK/graph.csr"
  else
    echo "    reusing $WORK/graph.csr"
  fi
  ls -la "$WORK/graph.csr" | awk '{printf "    %.2f GiB\n", $5/2^30}'
fi

echo "### 2. labels"
if [ ! -s "$WORK/labels.i64" ]; then
  "$PY" "$HERE/gnn_stream_npz.py" --zip "$ZIP" --npz node-label --member node_label \
      --dtype int64 > "$WORK/labels.i64" 2>/dev/null || {
    echo "    no node_label member; synthesising"
    "$PY" -c "
import numpy as np,sys
np.arange($N,dtype=np.int64).__mod__($C).tofile('$WORK/labels.i64')"
  }
fi

cat > "$WORK/tiered.yaml" <<EOF
networking:
  port: $PORT
runtime:
  num_threads: 8
  queue_depth: 65536
gpu:
  queue_depth: 8192
compose:
  - mod_name: clio_bdev
    pool_name: "ram::chi_default_bdev"
    pool_query: local
    pool_id: "301.0"
    bdev_type: ram
    capacity: "1024MB"
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
      - path: "ram::papers_ram"
        bdev_type: "ram"
        capacity_limit: "${RAM_TIER_MB:-8192}MB"
        score: 1.0
      - path: "$WORK/tier/nvme"
        bdev_type: "file"
        capacity_limit: "${NVME_TIER_MB:-131072}MB"
        score: 0.5
    dpe:
      dpe_type: "max_bw"
EOF

echo "### 3. features -> FIFO -> trainer (ingest, aggregate, verify, train)"
FIFO="$WORK/feat.fifo"
rm -f "$FIFO"; mkfifo "$FIFO"
"$PY" "$HERE/gnn_stream_npz.py" --zip "$ZIP" --member node_feat --dtype float32 \
    > "$FIFO" 2> "$WORK/stream.log" &
STREAM=$!
trap 'kill $STREAM 2>/dev/null || true; rm -f "$FIFO"' EXIT

CSR_ENV=()
[ "$SKIP_AGG" = "0" ] && CSR_ENV=(CLIO_GNN_CSR_FILE="$WORK/graph.csr")

env -u CLIO_SERVER_CONF \
    CLIO_SERVER_CONF="$WORK/tiered.yaml" \
    CLIO_GNN_INGEST_FILE="$FIFO" \
    CLIO_GNN_LABELS_FILE="$WORK/labels.i64" \
    "${CSR_ENV[@]}" \
    CLIO_GNN_NODES="$N" CLIO_GNN_DIM="$F" CLIO_GNN_CLASSES="$C" \
    CLIO_GNN_PAGE_KIB="$PAGE_KIB" \
    CLIO_GNN_AGG_BLOCK_ROWS="$AGG_BLOCK" \
    CLIO_GNN_EPOCHS="$EPOCHS" CLIO_GNN_HIDDEN="$HIDDEN" \
    CLIO_PORT="$PORT" \
    "$BIN/test_gpu_vector_gnn_train" 2>&1 | tee "$WORK/train.log" \
  | grep -E "TRAIN\]|agg\]|PASS|FAIL|verified|self-test"

echo "### tier usage"
du -h "$WORK"/tier/* 2>/dev/null | sed 's/^/    /'
