#!/usr/bin/env bash
# Sweeps concurrent GPU producer blocks (1 .. 768, the RTX 4090's resident
# maximum at 256 threads) across the FOUR system arms the paper compares --
# gpuh5 (GPU-initiated sync), raw, hdf5, hostclio (Raw+CLIO) -- holding
# bytes-per-block constant at 128 MB. Each arm writes to the SAME medium
# (WS_SINK, RAM parity by default) at its own most-efficient request size, so
# the lines compare storage PATHS, not request granularity. Emits one WSRESULT
# line per point into a raw log.
#
# REPEAT ORDER IS DELIBERATE: repeats are the OUTER loop, so a given point's 8
# samples are spread across the whole campaign rather than taken back-to-back.
# Batched repeats would let GPU thermal drift become a systematic per-point bias
# (prior campaigns measured 5-20% run-to-run movement from clock state alone);
# interleaving turns it into spread the error bars actually show.
#
# RESUMABLE: every completed point is appended to the raw log tagged with its
# repeat index, and a point already present is skipped. An interrupted campaign
# resumes by re-running this script with the same log.
#
# Runs INSIDE the container (paths are /workspace). Invoke from the host with:
#   docker exec iowarp-kvhdf5-gpu bash /workspace/context-transfer-engine/adapter/kvhdf5/test/e2e/run_weak_scaling.sh
set -u -o pipefail

BUILD_DIR="${WS_BUILD_DIR:-/workspace/build}"
BIN="${BUILD_DIR}/bin/kvhdf5_e2e_tests"
OUT_DIR="${WS_OUT_DIR:-/workspace/context-transfer-engine/adapter/kvhdf5/test/e2e/results}"
LOG="${OUT_DIR}/weak_scaling_raw.log"

REPS="${WS_REPS:-8}"
BLOCKS="${WS_BLOCKS_LIST:-1 2 4 8 16 32 64 128 256 512 768}"
ARMS="${WS_ARM_LIST:-gpuh5 raw hdf5 hostclio}"

# Fixed across every point: these define the study, not the sweep.
export WS_SINK="${WS_SINK:-ram}"            # RAM parity: medium is not the wall
export WS_MEMKIND="${WS_MEMKIND:-device}"   # gpuh5: real D2H path, not a host memcpy
export WS_CHUNK_KB="${WS_CHUNK_KB:-4096}"   # gpuh5 at its efficient (bandwidth-bound) size
export WS_MB_PER_BLOCK="${WS_MB_PER_BLOCK:-128}"
export WS_KEYS_PER_BLOCK="${WS_KEYS_PER_BLOCK:-2}"
export WS_PROBE="${WS_PROBE:-0}"            # bandwidth-only campaign; no probe overhead
export WS_TMPFS_DIR="${WS_TMPFS_DIR:-/dev/shm/ws_campaign}"   # raw/hdf5 RAM sink
export WS_DISK_DIR="${WS_DISK_DIR:-${BUILD_DIR}/ws_disk}"      # raw/hdf5 file sink
export WS_BDEV_PATH="${WS_BDEV_PATH:-${BUILD_DIR}/ws_bdev.dat}"
export CUDA_MODULE_LOADING=EAGER            # hazard 5: lazy load is a device-wide sync

PER_POINT_TIMEOUT="${WS_TIMEOUT:-1800}"

mkdir -p "${OUT_DIR}" "${WS_TMPFS_DIR}"
touch "${LOG}"

if [[ ! -x "${BIN}" ]]; then
    echo "FATAL: ${BIN} not found or not executable" >&2
    exit 1
fi

# GPU quiescence gate. A busy GPU makes every number here meaningless, so refuse
# to start rather than silently publish contaminated data.
busy_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
if [[ -n "${busy_mib}" && "${busy_mib}" -gt 1024 ]]; then
    echo "FATAL: GPU has ${busy_mib} MiB in use (>1024). Free it before running." >&2
    exit 1
fi
echo "[ws] GPU baseline: ${busy_mib:-unknown} MiB used; sink=${WS_SINK} gpuh5_chunk_kb=${WS_CHUNK_KB}"

total=$(( REPS * $(echo ${BLOCKS} | wc -w) * $(echo ${ARMS} | wc -w) ))
done_n=0
start_ts=$(date +%s)

for rep in $(seq 1 "${REPS}"); do
  for arm in ${ARMS}; do
    for b in ${BLOCKS}; do
      done_n=$(( done_n + 1 ))
      key="rep=${rep} arm=${arm} blocks=${b} "
      if grep -qF "${key}" "${LOG}" 2>/dev/null; then
          echo "[ws] skip (done): ${key}"
          continue
      fi

      # Hazard 2 / 5: a stale kFile bdev backing file silently fails with rc=13,
      # and stale tmpfs/disk targets can survive. Clear per-point. (No-op for the
      # RAM bdev arms, harmless.)
      rm -f "${WS_BDEV_PATH}"
      rm -rf "${WS_DISK_DIR}"; mkdir -p "${WS_DISK_DIR}"

      elapsed=$(( $(date +%s) - start_ts ))
      echo "[ws] (${done_n}/${total}, ${elapsed}s) rep=${rep} arm=${arm} blocks=${b}"

      line=$( cd "${BUILD_DIR}" && \
              WS_BLOCKS="${b}" WS_ARM="${arm}" \
              timeout "${PER_POINT_TIMEOUT}" "${BIN}" "weak_scaling" 2>/dev/null \
              | grep -m1 '^WSRESULT ' )
      rc=$?

      if [[ -z "${line}" ]]; then
          echo "rep=${rep} arm=${arm} blocks=${b} STATUS=NORESULT rc=${rc}" >> "${LOG}"
          echo "[ws]   !! no result (rc=${rc})" >&2
          continue
      fi
      # Tag with the repeat index, then strip the WSRESULT prefix so the log is
      # uniform key=value. (The line already carries arm= and blocks=.)
      echo "rep=${rep} ${line#WSRESULT }" >> "${LOG}"
      echo "[ws]   ${line#WSRESULT }"
    done
  done
done

echo "[ws] campaign complete in $(( $(date +%s) - start_ts ))s -> ${LOG}"
