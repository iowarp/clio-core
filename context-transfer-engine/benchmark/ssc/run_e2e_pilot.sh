#!/bin/bash
# Pilot on development seeds (1-3), never used by the paper benchmark (seeds 101-110, 201-203):
# checks the harness at scale and shows where the SSC misleads the model. Changes made from it
# are SSC development; the benchmark runs only after they are frozen.
cd "$(dirname "$0")"
R=${PILOT_DIR:-/ssc-bench/live_results/pilot}
G=/ssc-bench/live_results/grid_gs.jsonl
mkdir -p $R
read BEST WORST < <(python3 choose_fixed.py $G 2>/dev/null)
echo "best fixed $BEST, worst fixed $WORST"

run() {  # name schedule policy [args...]
    local name=$1 sched=$2; shift 2
    [ -s "$R/$name.jsonl" ] && { echo "skip $name"; return; }
    python3 e2e.py "$@" "$R/$name.jsonl" --schedule "$sched" --grid $G > "$R/$name.log" 2>&1
    echo "$(date +%T) $name rc=$?"
}
for s in 1 2 3; do
    run fixed-best_s$s seed:$s fixed:$BEST
    run fixed-worst_s$s seed:$s fixed:$WORST
    run heuristic2_s$s seed:$s heuristic2
    run agentjev-async_s$s seed:$s agentjev --ssc ${SSC:-v3} --rules --async-decide
    run oracle_s$s seed:$s oracle
done
python3 paper_compare.py $R --grid $G
python3 diagnose.py -q $R/agentjev-async_s*.jsonl --grid $G
echo DONE
