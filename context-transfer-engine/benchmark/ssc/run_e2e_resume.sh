#!/bin/bash
# Paper benchmark, on the dev build with the bdev free-space coalescing fix (PR 1334); oracle from
# grid_gs.jsonl measured on that build. Everything below was frozen before any of these runs: SSC v3, safety envelope 2
# (declared gate, zstd reflex at 40% watermark, post-swap canary, restart after any overflow),
# fair expert rules (heuristic2), AgentJev and qwen3-8B on the same SSC, async decisions.
#   Part A  main: 10 random schedules (seeds 101-110), 2 reps, base node (1 GB DRAM / 2 GB disk)
#   Part B  generalization: constrained node (0.5 GB DRAM / 1.5 GB disk) + two phases held out of
#           the main schedules (gs_field_reread, gs_events); measured grid first so the oracle is real
#   Part C  decision latency: AgentJev and qwen3 blocking on 3 schedules (what a stall costs)
cd "$(dirname "$0")"
R=/ssc-bench/live_results/paper
mkdir -p $R/main $R/gen $R/latency

run() {  # dir name schedule policy [args...]
    local dir=$1 name=$2 sched=$3; shift 3
    [ -s "$dir/$name.jsonl" ] && { echo "skip $name"; return; }
    python3 e2e.py "$@" "$dir/$name.jsonl" --schedule "$sched" > "$dir/$name.log" 2>&1
    echo "$(date +%T) $dir/$name rc=$?"
}
POL() {  # dir tag schedule [grid] [phases for the fixed setups]
    local d=$1 t=$2 s=$3 g=${4:-/ssc-bench/live_results/grid_gs.jsonl}
    read best worst < <(python3 choose_fixed.py $g $5 2>/dev/null)   # $5: phases (default: base five)
    run $d fixed-best_$t "$s" fixed:$best --grid $g
    run $d fixed-worst_$t "$s" fixed:$worst --grid $g
    run $d heuristic2_$t "$s" heuristic2 --grid $g
    run $d agentjev-async_$t "$s" agentjev --ssc v3 --rules --async-decide --grid $g
    run $d qwen3-async_$t "$s" llm:qwen3:8b --ssc v3 --rules --async-decide --grid $g
    run $d oracle_$t "$s" oracle --grid $g
}

# Part A
# Part A complete; resumed after host reboot 2026-11-11

# Part B (grid on the constrained node, all seven phases; then three schedules with the new phases)
export SSC_DRAM_GB=0.5 SSC_DISK_GB=1.5
G=$R/gen/grid_small.jsonl
[ "$(wc -l < $G 2>/dev/null)" -ge 336 ] || python3 run_grid.py $G --phases gs_output,gs_analysis,gs_products,gs_log,gs_archive,gs_field_reread,gs_events --repeat 2 > $R/gen/grid_small.log 2>&1
for r in 1 2; do for s in 201 202 203; do POL $R/gen s${s}_r$r gseed:$s $G gs_output,gs_analysis,gs_products,gs_log,gs_archive,gs_field_reread,gs_events; done; done
unset SSC_DRAM_GB SSC_DISK_GB

# Part C
for s in 101 102 103; do
    run $R/latency agentjev-blocking_s$s "seed:$s" agentjev --ssc v3 --rules
    run $R/latency qwen3-blocking_s$s "seed:$s" llm:qwen3:8b --ssc v3 --rules
done
for d in main gen latency; do echo "== $d"; python3 paper_compare.py $R/$d; done
echo DONE
