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
# The trainer sizes pages by ROWS, not KiB (CLIO_GNN_PAGE_ROWS, default 512 =
# 256 KiB at F=128). Setting only CLIO_GNN_PAGE_KIB silently leaves it at the
# default, which is a quarter of the 1 MiB page the throughput numbers were
# measured with -- four times as many blobs and round trips for the same bytes.
PAGE_ROWS=${PAGE_ROWS:-$(( PAGE_KIB * 1024 / (F * 4) ))}
# Accumulator rows per aggregation pass. Bigger = fewer passes over the 53 GiB
# of features (measured ~4.2 min each) but more RAM: rows * F * 8 bytes.
AGG_BLOCK=${AGG_BLOCK:-16000000}
EPOCHS=${EPOCHS:-3}
HIDDEN=${HIDDEN:-64}
SKIP_AGG=${SKIP_AGG:-0}     # 1 = train on raw features, no aggregation
# 1 = if the full matrix will not fit, run on the largest leading subset that
# does, rather than refusing. Real papers100M rows either way -- just fewer of
# them -- which still puts a matrix far beyond GPU memory through the vector.
AUTOFIT=${AUTOFIT:-0}

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
# set +e: the preflight signals "cannot run" with a non-zero exit, and under
# set -e a failing command substitution would kill the script before PF_RC
# could be read, losing the explanation it just printed.
set +e
FIT_N=$("$PY" - "$N" "$F" "$E" "$ZIP" "$avail_gib" "$SKIP_AGG" "$AUTOFIT" <<'PYEOF'
import os, sys
N, F, E = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
zip_path, avail = sys.argv[4], int(sys.argv[5])
skip_agg, autofit = int(sys.argv[6]), int(sys.argv[7])
def emit(msg):
    print(msg, file=sys.stderr)
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
emit(f"[preflight] features {feat:.1f} GiB logical, ~{stored:.1f} GiB stored")
emit(f"[preflight] csr {csr_needed:.1f} GiB, archive {zsz:.1f} GiB")
emit(f"[preflight] peak disk needed ~{peak:.1f} GiB; available {avail} GiB")
if peak > avail:
    if not autofit:
        emit(f"[preflight] SHORT BY ~{peak - avail:.0f} GiB -- refusing to start.")
        emit("[preflight] Options: free that much, set SKIP_AGG=1 to train on")
        emit("[preflight] raw features, or AUTOFIT=1 to run the largest subset")
        emit("[preflight] of real rows that does fit.")
        raise SystemExit(2)
    # Largest N whose stored copies plus the fixed costs fit, with 3 GiB spare.
    copies = 1 if skip_agg else 2
    fixed = zsz + csr_needed
    room = avail - fixed - 3.0
    if room <= 0:
        emit(f"[preflight] even zero rows will not fit ({fixed:.1f} GiB fixed "
             f"vs {avail} GiB) -- refusing.")
        raise SystemExit(2)
    per_row = (F * 4 / 1.078) * copies / GiB
    fit = int(room / per_row)
    if fit < 1_000_000:
        emit(f"[preflight] only {fit} rows would fit -- not worth running.")
        raise SystemExit(2)
    emit(f"[preflight] AUTOFIT: {fit} of {N} rows ({100.0*fit/N:.0f}%), "
         f"~{fit*F*4/GiB:.1f} GiB logical")
    print(fit)
    raise SystemExit(0)
emit("[preflight] OK")
print(N)
PYEOF
)
PF_RC=$?
set -e
[ "$PF_RC" -eq 0 ] || exit "$PF_RC"
if [ -n "$FIT_N" ] && [ "$FIT_N" -lt "$N" ]; then
  echo "### running on the first $FIT_N of $N rows (AUTOFIT)"
  N="$FIT_N"
fi

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
# papers100M stores node_label as FLOAT32 with NaN for unlabeled nodes (only
# ~1.39% carry a class). Streaming it straight to int64 turns every NaN into
# INT64_MIN, which is not a class -- the first run trained against 111M
# garbage labels and reported an accuracy that meant nothing. Convert
# explicitly: NaN becomes -1 (the "no label" sentinel the trainer expects),
# everything else casts.
if [ ! -s "$WORK/labels.i64" ]; then
  "$PY" "$HERE/gnn_stream_npz.py" --zip "$ZIP" --npz node-label \
      --member node_label > "$WORK/labels.raw" 2> "$WORK/labels.log" || true
  if [ -s "$WORK/labels.raw" ]; then
    "$PY" - "$WORK" "$N" <<'PYEOF'
import sys
import numpy as np
work, N = sys.argv[1], int(sys.argv[2])
raw = np.fromfile(f"{work}/labels.raw", dtype=np.float32)
lab = np.full(N, -1, dtype=np.int64)
n = min(N, raw.shape[0])
v = raw[:n]
ok = ~np.isnan(v)
lab[:n][ok] = v[ok].astype(np.int64)
lab.tofile(f"{work}/labels.i64")
print(f"    labeled {int(ok.sum())} of {n} nodes "
      f"({100.0 * ok.sum() / max(n, 1):.2f}%), classes "
      f"{int(lab[lab >= 0].min()) if (lab >= 0).any() else -1}.."
      f"{int(lab.max())}")
PYEOF
    rm -f "$WORK/labels.raw"
  else
    echo "    no node_label member; synthesising (accuracy will be meaningless)"
    "$PY" -c "
import numpy as np
np.arange($N,dtype=np.int64).__mod__($C).tofile('$WORK/labels.i64')"
  fi
fi

cat > "$WORK/tiered.yaml" <<EOF
networking:
  port: $PORT
runtime:
  num_threads: 8
  queue_depth: 65536
  # Bound the SHM segments. Left at the default (0 = auto) each is sized to
  # total memory and then clamped to HALF of it -- independently -- so the main
  # and metadata segments together can claim 100% of RAM. Freed SHM is not
  # returned to the OS, so RssShmem climbs toward that ceiling as reads cycle
  # through the arena: measured 9 GiB after ingest, 35.6 GiB mid-epoch and
  # 60 GiB (all of memory) by the end of one epoch over a 53 GiB store, while
  # anonymous memory stayed flat at 2.3 GiB. Capping them is what lets more
  # than one epoch run.
  main_segment_size: "${MAIN_SEG:-16g}"
  metadata_segment_size: "${META_SEG:-6g}"
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
    --max-rows "$N" > "$FIFO" 2> "$WORK/stream.log" &
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
    CLIO_GNN_PAGE_ROWS="$PAGE_ROWS" \
    CLIO_GNN_AGG_BLOCK_ROWS="$AGG_BLOCK" \
    CLIO_GNN_EPOCHS="$EPOCHS" CLIO_GNN_HIDDEN="$HIDDEN" \
    CLIO_PORT="$PORT" \
    "$BIN/test_gpu_vector_gnn_train" 2>&1 | tee "$WORK/train.log" \
  | grep -E "TRAIN\]|agg\]|PASS|FAIL|verified|self-test"

echo "### tier usage"
du -h "$WORK"/tier/* 2>/dev/null | sed 's/^/    /'
