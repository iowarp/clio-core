#!/usr/bin/env bash
# Poor man's profiler for nodes without perf: while a benchmark phase runs,
# record per-thread CPU (top -H) and N stack snapshots (eu-stack) of
# clio_run and clio_cte_fuse on THIS node into the directory $1.
#   sample_stacks.sh <outdir> [delay_s=1] [samples=8] [gap_s=0.4]
set -u
OUT=$1; DELAY=${2:-1}; N=${3:-8}; GAP=${4:-0.4}
mkdir -p "$OUT"
sleep "$DELAY"
RT=$(pgrep -u "$USER" -f '[c]lio_run start' | head -1)
FU=$(pgrep -u "$USER" -x clio_cte_fuse | head -1)
top -H -b -n 2 -d 1 -w 200 -u "$USER" > "$OUT/top.txt" 2>&1 &
for i in $(seq 1 "$N"); do
  P1=; P2=
  [ -n "$RT" ] && { eu-stack -p "$RT" > "$OUT/rt.$i.txt" 2>/dev/null & P1=$!; }
  [ -n "$FU" ] && { eu-stack -p "$FU" > "$OUT/fuse.$i.txt" 2>/dev/null & P2=$!; }
  [ -n "$P1$P2" ] && wait $P1 $P2
  sleep "$GAP"
done
wait
