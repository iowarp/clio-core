#!/bin/bash
# Re-measured grid (2026-11-23): clean start between runs (measure.py settle), durable SSD + working
# cache (compose.py), 3 repeats; two node sizes. Appends; safe to re-run (run_grid skips done setups).
cd "$(dirname "$0")"
R=/ssc-bench/live_results/grid_v2
mkdir -p $R
PH=gs_output,gs_analysis,gs_products,gs_log,gs_archive,gs_field_reread,gs_events
echo "$(date +%T) start 1g" >> $R/progress.log
SSC_DRAM_GB=1 SSC_DISK_GB=2 python3 run_grid.py $R/grid_1g.jsonl --phases $PH --repeat 3 >> $R/grid_1g.log 2>&1
echo "$(date +%T) done 1g" >> $R/progress.log
SSC_DRAM_GB=0.5 SSC_DISK_GB=1.5 python3 run_grid.py $R/grid_05g.jsonl --phases $PH --repeat 3 >> $R/grid_05g.log 2>&1
echo "$(date +%T) done 05g" >> $R/progress.log
