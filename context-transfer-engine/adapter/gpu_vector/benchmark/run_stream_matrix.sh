#!/usr/bin/env bash
# Copyright 2024 IOWarp - BSD 3-Clause License
#
# Sweep the gpu_vector streaming benchmark across compression ratio and
# oversubscription, raw vs lz4, with the compressor module on the path.
#
# The "VRAM" the data is sized against is the TOP STORAGE TIER (--vram-mb), not
# the physical device memory. That is the quantity the experiment is actually
# about -- how much of the working set stays in the fastest tier -- and it is
# what makes the 4x case runnable: 4x of an 8 GiB device would be 32 GiB of
# spill tier, more than this host's /dev/shm.
#
# Each run is a separate process: the benchmark composes its own runtime, and
# CLIO_INIT is once per process.
#
# usage: run_stream_matrix.sh [bin-dir] [vram-mb]
set -u

BIN_DIR="${1:-build-gv/bin}"
VRAM_MB="${2:-1024}"
BENCH="${BIN_DIR}/clio_gpu_vector_stream_bench"

if [[ ! -x "${BENCH}" ]]; then
  echo "no benchmark at ${BENCH}" >&2
  exit 1
fi

BLOCKS="${BLOCKS:-16}"
PAGES="${PAGES:-8}"
PAGE_KB="${PAGE_KB:-256}"
THREADS="${THREADS:-256}"
SPILL_MB="${SPILL_MB:-8192}"

# 0.5x is the "fits" case (strictly smaller than the tier); 2x and 4x oversubscribe.
MULTS="${MULTS:-0.5 2 4}"
ZERO_PCTS="${ZERO_PCTS:-30 80}"

echo "# gpu_vector streaming: compressor on the path"
echo "# vram(top tier)=${VRAM_MB}MB blocks=${BLOCKS} slots/blk=${PAGES} page=${PAGE_KB}KB threads=${THREADS}"

for zero in ${ZERO_PCTS}; do
  for mult in ${MULTS}; do
    total=$(awk -v v="${VRAM_MB}" -v m="${mult}" 'BEGIN{printf "%d", v*m}')
    for mode in raw lz4; do
      flag=""
      [[ "${mode}" == "lz4" ]] && flag="--compressed"
      # Fresh shm state per run: a leftover segment from a killed run makes the
      # next tier allocation fail in a way that looks like a capacity result.
      rm -f /dev/shm/clio_* 2>/dev/null
      "${BENCH}" --blocks "${BLOCKS}" --pages "${PAGES}" --page-kb "${PAGE_KB}" \
                 --total-mb "${total}" --zero-pct "${zero}" \
                 --vram-mb "${VRAM_MB}" --spill-mb "${SPILL_MB}" \
                 --threads "${THREADS}" ${flag} 2>/dev/null | grep '^GVS ' \
        || echo "GVS mode=${mode} zero=${zero}% total=${total}MB FAILED"
    done
  done
done
