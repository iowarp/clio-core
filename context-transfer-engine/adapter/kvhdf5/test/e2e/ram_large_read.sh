#!/usr/bin/env bash
set -u
BIN="${RLR_BUILD_DIR:-/workspace/build}/bin/kvhdf5_e2e_tests"
BINDIR="${RLR_BUILD_DIR:-/workspace/build}/bin"
LOG="${RLR_BUILD_DIR:-/workspace/build}/gpuh5_ram_LARGE_read.log"
: > "$LOG"
cd "$BINDIR"
common(){ echo GSBENCH_N=16384 GSBENCH_CHUNKS=$CH GSBENCH_SUBMIT_BLOCKS=$CH GSBENCH_SNAPS=8 \
  GSBENCH_STEPS_PER=8 GSBENCH_RAW_ODIRECT=0 GSBENCH_BDEV=ram GSBENCH_DISK_DIR=/dev/shm/gsbench_raw \
  GSBENCH_HDF5_DIR=/dev/shm/gsbench_hdf5 \
  GSBENCH_BDEV_PATH="${RLR_BUILD_DIR:-/workspace/build}/gsbench_scratch/clio_bdev.dat" \
  GSBENCH_BDEV_CAP_MB=12000 GSBENCH_READ=1 GSBENCH_READ_ASYNC=1; }
run_arm(){ local name="$1" sel="$2"; shift 2
  rm -f /dev/shm/chi_* 2>/dev/null
  rm -rf /dev/shm/gsbench_raw /dev/shm/gsbench_hdf5 2>/dev/null
  echo "=== arm=$name chunks=$CH rep=$REP ===" >> "$LOG"
  env -u HDF5_VOL_CONNECTOR $(common) "$@" "$BIN" "$sel" >> "$LOG" 2>&1; }
run_async(){ rm -f /dev/shm/chi_* 2>/dev/null
  rm -rf /dev/shm/gsbench_raw /dev/shm/gsbench_hdf5 2>/dev/null
  echo "=== arm=async_VOL chunks=$CH rep=$REP ===" >> "$LOG"
  env $(common) GSBENCH_HDF5_ASYNC_FWAIT=0 \
    LD_LIBRARY_PATH="/opt/vol-async-nomemcpy/lib:/opt/hdf5ts/lib:/opt/argobots/lib:${BINDIR}" \
    HDF5_PLUGIN_PATH=/opt/vol-async-nomemcpy/lib HDF5_VOL_CONNECTOR="async under_vol=0;under_info={}" \
    "$BIN" "[gsbench_hdf5_async]" >> "$LOG" 2>&1; }
for CH in 1 4 16 64 256; do
  for REP in 1 2; do
    echo "#### chunks=$CH rep=$REP @ $(date +%H:%M:%S) (per-read=$((1024/CH))MB)" >> "$LOG"
    run_arm raw_inline "[gsbench_raw]" LD_LIBRARY_PATH="$BINDIR" GSBENCH_RAW_INLINE=1
    run_async
    run_arm hdf5_inline "[gsbench_hdf5]" LD_LIBRARY_PATH="$BINDIR" GSBENCH_RAW_INLINE=1
    run_arm hostclio "[gsbench_hostclio]" LD_LIBRARY_PATH="$BINDIR"
    run_arm gpuh5 "[gsbench_persistent]" LD_LIBRARY_PATH="$BINDIR"
    run_arm gpuh5_sync "[gsbench_persistent_sync]" LD_LIBRARY_PATH="$BINDIR"
  done
done
echo "ALLDONE" >> "$LOG"
