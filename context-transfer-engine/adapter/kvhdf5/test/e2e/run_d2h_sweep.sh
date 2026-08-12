#!/usr/bin/env bash
# Direct measurement of the producer-blocking D2H copy.
# Emits one CSV row per (arm, steps_per, N, rep) from the GSBENCH_D2H TOTALS line.
set -u

export CUDA_MODULE_LOADING=EAGER
export LD_LIBRARY_PATH=/workspace/build/bin:${LD_LIBRARY_PATH:-}
export CHI_REPO_PATH=/workspace/build/bin
export GSBENCH_D2H_TRACE=1
export GSBENCH_CHUNKS=4
export GSBENCH_RAW_ODIRECT=0
export GSBENCH_RAW_FSYNC=1
export GSBENCH_DISK_DIR=/workspace/build/gsbench_scratch/raw_out
export GSBENCH_HDF5_DIR=/workspace/build/gsbench_scratch/hdf5_out
mkdir -p "$GSBENCH_DISK_DIR" "$GSBENCH_HDF5_DIR"

BIN=/workspace/build/bin/kvhdf5_e2e_tests
REPS=${REPS:-3}

echo "kind,arm,N,snaps,steps_per,rep,copies,bytes_per_copy,copy_ms,block_ms,mean_copy_ms,GBps"

run() {  # $1=kind $2=tag $3=label $4=N $5=snaps $6=steps_per $7=rep ; rest = extra env
  local kind="$1" tag="$2" label="$3" n="$4" snaps="$5" sp="$6" rep="$7"; shift 7
  local out
  out=$(env GSBENCH_N="$n" GSBENCH_SNAPS="$snaps" GSBENCH_STEPS_PER="$sp" "$@" \
        timeout 900 "$BIN" "$tag" 2>&1 | grep -E '^GSBENCH_D2H arm=.* copies=')
  if [[ -z "$out" ]]; then
    echo "$kind,$label,$n,$snaps,$sp,$rep,FAILED,,,,," ; return
  fi
  # GSBENCH_D2H arm=X copies=N bytes_per_copy=B total_MB=M copy_ms=C block_ms=BL mean_copy_ms=MC GBps=G
  local copies bpc cms bms mcms gbps
  copies=$(sed -n 's/.*copies=\([0-9]*\).*/\1/p'            <<<"$out")
  bpc=$(  sed -n 's/.*bytes_per_copy=\([0-9]*\).*/\1/p'     <<<"$out")
  cms=$(  sed -n 's/.* copy_ms=\([0-9.]*\).*/\1/p'          <<<"$out")
  bms=$(  sed -n 's/.*block_ms=\([0-9.]*\).*/\1/p'          <<<"$out")
  mcms=$( sed -n 's/.*mean_copy_ms=\([0-9.]*\).*/\1/p'      <<<"$out")
  gbps=$( sed -n 's/.*GBps=\([0-9.]*\).*/\1/p'              <<<"$out")
  echo "$kind,$label,$n,$snaps,$sp,$rep,$copies,$bpc,$cms,$bms,$mcms,$gbps"
}

for rep in $(seq 1 "$REPS"); do
  # (1) COMPUTE-INTENSITY sweep: volume fixed (12 x 156.25 MB), steps_per varies 16x.
  #     The invariance claim: copy_ms must NOT move.
  for sp in 12 48 192 768; do
    run steps "[gsbench_raw]"  raw_threaded  6400 12 "$sp" "$rep" GSBENCH_RAW_INLINE=0
    run steps "[gsbench_raw]"  raw_inline    6400 12 "$sp" "$rep" GSBENCH_RAW_INLINE=1
    run steps "[gsbench_hdf5]" hdf5_threaded 6400 12 "$sp" "$rep" GSBENCH_RAW_INLINE=0
  done

  # (2) OUTPUT-VOLUME sweep: compute intensity fixed, bytes/copy varies 32x.
  #     The scaling claim: copy_ms proportional to bytes; GBps ~ constant.
  for n in 1600 3200 4800 6400 9024; do
    run volume "[gsbench_raw]" raw_threaded "$n" 12 48 "$rep" GSBENCH_RAW_INLINE=0
  done

  # (3) PINNED vs PAGEABLE staging buffer (hdf5 arm only knob). The pageable path is the
  #     confound flagged in the brief; show what it costs the copy itself.
  run pinned "[gsbench_hdf5]" hdf5_pinned   6400 12 48 "$rep" GSBENCH_RAW_INLINE=0 GSBENCH_HDF5_PINNED=1
  run pinned "[gsbench_hdf5]" hdf5_pageable 6400 12 48 "$rep" GSBENCH_RAW_INLINE=0 GSBENCH_HDF5_PINNED=0
done
