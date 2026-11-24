#!/bin/bash
# Fourth schedule, untouched by any design decision: SSC v3 + safety rules (with the post-swap
# canary and the 40% watermark) against the expert-rules baseline, fixed setups and the oracle.
cd "$(dirname "$0")"
R=/ssc-bench/live_results/e2e_test4
S="hot_reread:4,checkpoint:5,log_archive:4,log_append:4,random_ingest:5,hot_reread:3,log_archive:4,checkpoint:4,log_append:3"
mkdir -p $R

run() {  # name policy [args...]
    local name=$1; shift
    python3 e2e.py "$@" "$R/$name.jsonl" --schedule "$S" > "$R/$name.log" 2>&1
    echo "$(date +%T) $name rc=$?"
}

for r in 1 2 3; do
    for p in fixed:disk-cache-idx-zstd fixed:dram-cache-idx-lz4 fixed:dram-nocache-idx-zstd oracle heuristic; do
        run "${p/:/_}_r$r" $p
    done
    run agentjev-v3-rules_r$r agentjev --ssc v3 --rules
done
python3 e2e_compare.py $R
echo DONE
