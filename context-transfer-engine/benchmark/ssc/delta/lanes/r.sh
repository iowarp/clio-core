# repair the one missing 0.5 GB grid cell (campaign 1: a read of a refused put crashed; fixed in workload.py)
SSC_DRAM_GB=0.5 SSC_DISK_GB=1.5 ctr python3 run_grid.py $G/grid_05g.jsonl --only disk-nocache-idx-lz4 \
  --phases h5_particles --repeat 3 > $LOGDIR/grid_05g_repair.log 2>&1
rc=$?; echo "$(date +%T) 05g repair rc=$rc rows=$(wc -l < $G/grid_05g.jsonl)"
[ $rc -eq 0 ] && touch $G/grid_05g.done
