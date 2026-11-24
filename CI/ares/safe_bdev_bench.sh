#!/bin/bash
# safe-bdev vs plain bdev raw I/O on ONE node, ONE NVMe.
#
# Creates 4 x 4 GiB files on the node's NVMe and benchmarks, through the
# runtime's bdev data plane (clio_run_thrpt_bench bdev_write / bdev_read,
# allocation outside the timed loop):
#   bdev      one plain file bdev (baseline)
#   safe4p0   safe-bdev over 4 data members, no parity (striping)
#   safe3p1   3 data + 1 parity  (max_failures 1)
#   safe2p2   2 data + 2 parity  (max_failures 2)
#
# Usage (from a worktree):
#   CI/ares/safe_bdev_bench.sh --build build-ares --out <dir> [--threads 8]
#       [--sizes "4k 64k 1m"] [--duration 20] [--working-set 3g]
#       [--configs "bdev safe4p0 safe3p1 safe2p2"] [--submit]
# --submit wraps the run in an exclusive 1-node sbatch job.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
BUILD=build-ares OUT= THREADS=8 SIZES="4k 64k 1m" DUR=20 WS=3g SUBMIT=0
CONFIGS="bdev safe4p0 safe3p1 safe2p2" FILE_GB=4 PART=compute
EXCLUDE=ares-comp-[10,16,19,29,31]
while [ $# -gt 0 ]; do
  case $1 in
    --build) BUILD=$2; shift;; --out) OUT=$2; shift;;
    --threads) THREADS=$2; shift;; --sizes) SIZES=$2; shift;;
    --duration) DUR=$2; shift;; --working-set) WS=$2; shift;;
    --configs) CONFIGS=$2; shift;; --file-gb) FILE_GB=$2; shift;;
    --exclude) EXCLUDE=$2; shift;; --submit) SUBMIT=1;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac; shift
done
[ -n "$OUT" ] || { echo "--out required" >&2; exit 2; }
case $BUILD in /*) ;; *) BUILD=$HERE/$BUILD;; esac
mkdir -p "$OUT"
if [ $SUBMIT = 1 ]; then
  args=(--build "$BUILD" --out "$OUT" --threads "$THREADS" --sizes "$SIZES"
        --duration "$DUR" --working-set "$WS" --configs "$CONFIGS"
        --file-gb "$FILE_GB")
  exec sbatch --exclusive -N1 -p "$PART" -t 03:00:00 -J clio-sbdev-bench \
    --exclude="$EXCLUDE" -o "$OUT/slurm-%j.out" --wait \
    "$HERE/CI/ares/safe_bdev_bench.sh" "${args[@]}"
fi

BIN=$BUILD/bin
DIR=/mnt/nvme/$USER/sbdev_bench
mkdir -p "$DIR"
export CLIO_WITH_RUNTIME=0 CLIO_IPC_MODE=SHM CLIO_MEMFD_DIR=${SBDEV_MEMFD_DIR:-$DIR/memfd}
export CTP_LOG_LEVEL=warning LD_LIBRARY_PATH=$BIN PATH=$BIN:/usr/bin:/bin
mkdir -p "$CLIO_MEMFD_DIR"
hostname > "$OUT/node"
df -h /mnt/nvme > "$OUT/df.txt"
DEV=$(df --output=source /mnt/nvme | tail -1 | xargs basename)
echo "device=$DEV" >> "$OUT/node"

bdev_entry() {  # idx major [type]
  cat <<EOF
  - mod_name: clio_bdev
    pool_name: "$DIR/bdev$1.dat"
    pool_query: local
    pool_id: "$2.0"
    bdev_type: ${3:-file}
    capacity: "${FILE_GB}GB"
${SBDEV_PERSIST:+    persistence_level: $SBDEV_PERSIST}
EOF
}

write_conf() {  # config -> $OUT/<config>/clio_server.yaml
  local c=$1 d=$OUT/$1 data parity n i
  mkdir -p "$d"
  echo "$(hostname)" > "$d/hostfile"
  {
    cat <<EOF
memory:
  main_segment_size: auto
  client_data_segment_size: 4GB
networking:
  port: 9529
  hostfile: $d/hostfile
runtime:
  num_threads: 8
  queue_depth: 1024
  conf_dir: $DIR/conf_$c
compose:
EOF
    case $c in
      bdev) bdev_entry 0 301; return;;
      ram) bdev_entry 0 301 ram; return;;  # software-path ceiling, no device
      safe4p0) data=4 parity=0;;
      safe3p1) data=3 parity=1;;
      safe2p2) data=2 parity=2;;
    esac
    for i in 0 1 2 3; do bdev_entry $i $((301 + i)); done
    cat <<EOF
  - mod_name: clio_safe_bdev
    pool_name: "safe0"
    pool_query: local
    pool_id: "350.0"
    max_failures: $parity
    members:
EOF
    for i in 0 1 2 3; do
      cat <<EOF
      - pool_name: "$DIR/bdev$i.dat"
        node_id: 0
        pool_id_major: $((301 + i))
        pool_id_minor: 0
        parity: $([ $i -ge $data ] && echo true || echo false)
EOF
    done
  } > "$d/clio_server.yaml"
}

summary=$OUT/summary.tsv
echo -e "config\tmode\tio_size\tthreads\tMBps\tIOPS\tavg_lat_us\tdev_MBps_r\tdev_MBps_w" > "$summary"
for c in $CONFIGS; do
  d=$OUT/$c
  write_conf "$c"
  # Fresh files every config: no allocator/parity state carried over.
  rm -rf "$DIR"/bdev*.dat "$DIR"/conf_* "$DIR"/memfd/*
  for i in 0 1 2 3; do fallocate -l ${FILE_GB}G "$DIR/bdev$i.dat"; done
  # SBDEV_PREWRITE=1: write every byte once so no extent is unwritten
  # (isolates XFS unwritten-extent conversion from bdev's own cost).
  if [ "${SBDEV_PREWRITE:-0}" = 1 ]; then
    for i in 0 1 2 3; do
      dd if=/dev/zero of="$DIR/bdev$i.dat" bs=1M count=$((FILE_GB * 1024)) \
         oflag=direct conv=notrunc status=none
    done
  fi
  export CLIO_SERVER_CONF=$d/clio_server.yaml
  ${SBDEV_RT_WRAP:-} "$BIN/clio_run" start --fresh --no-viz > "$d/runtime.log" 2>&1 &
  RT=$!
  sleep 8
  pool=$(case $c in bdev|ram) echo 301.0;; *) echo 350.0;; esac)
  for sz in $SIZES; do
    for mode in write read; do
      tag=$c.$mode.$sz
      iostat -dxm 1 "$DEV" > "$d/$tag.iostat" 2>&1 &
      IO=$!
      timeout $((DUR * 4 + 600)) "$BIN/clio_run_thrpt_bench" \
        --test-case bdev_$mode --pool-id $pool --threads "$THREADS" \
        --io-size "$sz" --duration "$DUR" --working-set "$WS" \
        > "$d/$tag.out" 2>&1 || echo "rc=$?" >> "$d/$tag.out"
      kill $IO 2>/dev/null || true
      bw=$(grep -oP 'Bandwidth: \K[0-9.]+' "$d/$tag.out" | tail -1 || true)
      iops=$(grep -oP 'IOPS: \K[0-9.]+' "$d/$tag.out" | tail -1 || true)
      lat=$(grep -oP 'Avg latency: \K[0-9.]+' "$d/$tag.out" | tail -1 || true)
      # Device-side MB/s over the busiest $DUR seconds (fill phase excluded).
      read dr dw < <(awk -v dev="$DEV" -v n="$DUR" '$1==dev {r[++k]=$3; w[k]=$9}
        END {best=0; for (i=2;i+n-1<=k;i++){s=0;t=0; for(j=i;j<i+n;j++){s+=r[j];t+=w[j]}
             if (s+t>best){best=s+t; br=s/n; bw=t/n}} printf "%.0f %.0f\n", br, bw}' \
        "$d/$tag.iostat")
      [ -n "${iops:-}" ] && lat=$(awk -v t=$THREADS -v i=$iops "BEGIN{printf \"%.1f\", t*1e6/i}")
      echo -e "$c\t$mode\t$sz\t$THREADS\t${bw:-NA}\t${iops:-NA}\t${lat:-NA}\t$dr\t$dw" \
        | tee -a "$summary"
    done
  done
  "$BIN/clio_run" stop >> "$d/runtime.log" 2>&1 || kill -9 $RT || true
  wait $RT 2>/dev/null || true
done
rm -rf "$DIR"
column -t "$summary"
