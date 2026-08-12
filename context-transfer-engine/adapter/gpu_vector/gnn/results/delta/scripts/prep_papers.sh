#!/bin/bash
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# papers100M prep on NODE-LOCAL NVMe. Run inside the container on the A100 node.
#   NOGPU=1 ./rj.sh bash ${BENCH_ROOT}/prep_papers.sh
set -euo pipefail

ROOT=/tmp/gnn/data                 # node-local NVMe (1.5 TB)
RAWZIP=/work/hdd/bekn/rpawar/ogbraw/papers100M-bin.zip
GNN=/u/rpawar/clio-core/context-transfer-engine/adapter/gpu_vector/gnn
WORKERS="${WORKERS:-16}"

mkdir -p "$ROOT"
df -h /tmp | tail -1

# gnn_prep.py looks for <data_root>/papers100M-bin.zip; link the cached copy so
# it never re-downloads. Extraction/unpacking all land on node-local NVMe.
if [ ! -e "$ROOT/papers100M-bin.zip" ]; then
  ln -s "$RAWZIP" "$ROOT/papers100M-bin.zip"
fi

echo "############ 1) prep (extract, features.f32, labels.i64, CSR) ############"
time python3 "$GNN/gnn_prep.py" --dataset papers100M --data-root "$ROOT" \
     --out "$ROOT/papers100M"

echo "############ 2) meta ############"
cat "$ROOT/papers100M/meta.txt"
ls -la "$ROOT/papers100M/"
df -h /tmp | tail -1

echo "############ 3) 1-hop mean aggregation -> agg_features.f32 ############"
time python3 "$GNN/gnn_agg.py" --data "$ROOT/papers100M" --workers "$WORKERS"

echo "############ DONE ############"
ls -la "$ROOT/papers100M/"
df -h /tmp | tail -1
