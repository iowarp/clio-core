#!/usr/bin/env bash
#===============================================================================
# rename_workload.sh OLD NEW -- rename a staged workload everywhere it lives:
# its staged chunks (/mnt/nvme0/v2-work/OLD), its baselines (baselines/OLD,
# meta.json dataset field updated) and its run directories and files
# (runs/OLD_*). Cached learning replays of OLD are dropped.
#===============================================================================
set -euo pipefail
W=/mnt/nvme0/v2-work; O=$1; N=$2
[ -e "$W/$N" ] && { echo "$W/$N exists" >&2; exit 1; }
mv "$W/$O" "$W/$N"
[ -d "$W/baselines/$O" ] && mv "$W/baselines/$O" "$W/baselines/$N"
for r in "$W"/runs/"${O}"_*; do [ -e "$r" ] && mv "$r" "$W/runs/$N${r#$W/runs/$O}"; done
python3 - "$W/baselines/$N" "$N" <<'PY'
import glob, json, sys
for p in glob.glob(sys.argv[1] + "/**/meta.json", recursive=True):
    m = json.load(open(p)); m["dataset"] = sys.argv[2]; json.dump(m, open(p, "w"), indent=2)
PY
rm -f "$W"/runs/replay_cache/"${O}"_*
echo "renamed $O -> $N"
