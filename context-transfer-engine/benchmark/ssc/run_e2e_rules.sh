#!/bin/bash
# Safety rules (declared-requirement gate + overflow reflex) on top of SSC v3.
# Part 1: a third schedule nobody tuned on, every policy. Part 2: v3+rules on the two earlier
# schedules, to complete their tables. Rep-major order.
cd "$(dirname "$0")"
R=/ssc-bench/live_results
D="checkpoint:6,hot_reread:6,log_append:6,random_ingest:6,log_archive:6,checkpoint:6"
H="log_append:5,log_archive:4,checkpoint:5,random_ingest:5,hot_reread:5,log_archive:4,log_append:4,checkpoint:4"
T="random_ingest:4,log_archive:5,hot_reread:4,log_append:5,checkpoint:5,log_archive:3,random_ingest:4,log_append:3,hot_reread:3"
mkdir -p $R/e2e_test3

run() {  # dir name schedule policy [args...]
    local dir=$1 name=$2 sched=$3; shift 3
    python3 e2e.py "$@" "$dir/$name.jsonl" --schedule "$sched" > "$dir/$name.log" 2>&1
    echo "$(date +%T) $dir/$name rc=$?"
}

for r in 1 2 3; do
    for p in fixed:disk-cache-idx-zstd fixed:dram-cache-idx-lz4 fixed:dram-nocache-idx-zstd oracle; do
        run $R/e2e_test3 "${p/:/_}_r$r" "$T" $p
    done
    run $R/e2e_test3 agentjev-v3_r$r "$T" agentjev --ssc v3
    run $R/e2e_test3 agentjev-v3-rules_r$r "$T" agentjev --ssc v3 --rules
done
for r in 1 2 3; do
    run $R/e2e agentjev-v3-rules_r$r "$D" agentjev --ssc v3 --rules
    run $R/e2e_heldout agentjev-v3-rules_r$r "$H" agentjev --ssc v3 --rules
done
for d in e2e e2e_heldout e2e_test3; do echo "== $d"; python3 e2e_compare.py $R/$d; done
echo DONE
