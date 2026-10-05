#!/usr/bin/env bash
#===============================================================================
# auto_replot.sh -- redraw every nn-v2 figure (replot_all.sh, at low CPU
# priority) each time a workload finishes in run_full_all.sh, and once more
# after the last one (UNTIL, default astro-camels-6snap).
# Log: /mnt/nvme0/v2-work/runs/auto_replot.log
#===============================================================================
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
L=/mnt/nvme0/v2-work/runs/full_all.log
LOG=/mnt/nvme0/v2-work/runs/auto_replot.log
UNTIL=${UNTIL:-astro-camels-6snap}
cd "$HERE"
seen=$(grep -c "all runs done" "$L")
while true; do
  now=$(grep -c "all runs done" "$L")
  if [ "$now" -gt "$seen" ]; then
    seen=$now
    last=$(grep "all runs done" "$L" | tail -1 | awk '{print $2}')
    echo "$(date +%T) $last finished; replotting" >> "$LOG"
    nice -n 19 ./replot_all.sh >> "$LOG" 2>&1
    echo "$(date +%T) replotted ($(grep -c 'wrote' "$LOG") figures written so far)" >> "$LOG"
  fi
  grep -q "$UNTIL all runs done" "$L" && break
  sleep 60
done
echo "$(date +%T) AUTO_REPLOT_DONE" >> "$LOG"
