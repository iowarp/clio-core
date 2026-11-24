#!/bin/bash
# SSC v3 end-to-end batch. Part 1 re-runs the original schedule (the one v3 was diagnosed on) for
# v3 and the second-best fixed setup from the grid. Part 2 is a held-out schedule nobody tuned on,
# with every policy. Rep-major order so slow drift on the machine spreads across policies.
cd "$(dirname "$0")"
DEV=/ssc-bench/live_results/e2e
HELD=/ssc-bench/live_results/e2e_heldout
HELD_SCHED="log_append:5,log_archive:4,checkpoint:5,random_ingest:5,hot_reread:5,log_archive:4,log_append:4,checkpoint:4"
mkdir -p $DEV $HELD

run() {  # dir name schedule policy [args...]
    local dir=$1 name=$2 sched=$3; shift 3
    python3 e2e.py "$@" "$dir/$name.jsonl" --schedule "$sched" > "$dir/$name.log" 2>&1
    echo "$(date +%T) $dir/$name rc=$?"
}

DEV_SCHED="checkpoint:6,hot_reread:6,log_append:6,random_ingest:6,log_archive:6,checkpoint:6"
for r in 1 2 3; do
    run $DEV agentjev-v3_r$r "$DEV_SCHED" agentjev --ssc v3
    run $DEV fixed_dram-nocache-idx-zstd_r$r "$DEV_SCHED" fixed:dram-nocache-idx-zstd
done
for r in 1 2 3; do
    for p in fixed:disk-cache-idx-zstd fixed:dram-cache-idx-lz4 fixed:dram-nocache-idx-zstd oracle; do
        run $HELD "${p/:/_}_r$r" "$HELD_SCHED" $p
    done
    run $HELD agentjev_r$r "$HELD_SCHED" agentjev --ssc v2
    run $HELD agentjev-v3_r$r "$HELD_SCHED" agentjev --ssc v3
done
echo DONE
