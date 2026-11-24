#!/bin/bash
# Offline replay on a re-measured grid: $1 = grid file, $2 = DRAM GB, $3 = disk GB, $4 = out
cd "$(dirname "$0")"
G=$1; export SSC_DRAM_GB=$2 SSC_DISK_GB=$3; O=$4
rm -f $O
python3 ablate_ssc.py $O --seeds 1-3 --grid $G --variants const,rulesh,wknn10,wknn11 > /dev/null 2>&1
python3 ablate_ssc.py $O --seeds 1-3 --grid $G --variants "none@qwen3:4b,v10@qwen3:4b,v11@qwen3:4b" > /dev/null 2>&1
python3 ablate_ssc.py --summary $O
python3 - $G <<'PY'
import json, sys, statistics as st
from collections import defaultdict
acc = defaultdict(list)
for l in open(sys.argv[1]):
    r = json.loads(l); c = r["composition"]
    acc[(r["phase"], c["placement"], c["cache"], c["indexer"], c["compression"])].append(r["wall_s"])
sp = {k: (max(v) - min(v)) / st.median(v) for k, v in acc.items() if len(v) > 1}
bad = sorted(sp.items(), key=lambda x: -x[1])
print(f"\nsetups with repeat spread > 30%: {sum(1 for _, s in bad if s > 0.3)} of {len(sp)}")
for k, s in bad[:6]: print("  ", k, f"{100*s:.0f}%", [round(x, 2) for x in acc[k]])
PY
