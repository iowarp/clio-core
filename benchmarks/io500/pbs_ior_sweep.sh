#!/usr/bin/env bash
#PBS -l select=4
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare:daos_user
#PBS -l daos=daos_user
#PBS -q debug-scaling
#PBS -A IOWarp
#PBS -j oe
#
# ior-easy (write + read) over several clio-fs configurations in ONE job:
# each line of CONFIGS is "<tag> <VAR=value>..." applied on top of the
# defaults below; pbs_io500_aurora.sh brings clio-fs up and down per line.
#
#   qsub -l select=4 benchmarks/io500/pbs_ior_sweep.sh
set -u
HERE=/home/llogan/clio-core/benchmarks/io500
CONFIGS=${CONFIGS:-"flare MODE=flare
base
threads32 NUM_THREADS=32
window4g CLIO_FUSE_WRITE_INFLIGHT_MB=4096
cache CACHE=1
wal PERSIST=temporary
ramonly RAM_ONLY=1 RAM_PCT=400"}
# CONFIGS_FILE: read the configurations from a file instead. A line
# "@defaults <VAR=value>..." there replaces the built-in defaults
# (CACHE=0 PERSIST=volatile FUSE_PROFILE=0) for the lines after it.
[ -n "${CONFIGS_FILE:-}" ] && CONFIGS=$(cat "$CONFIGS_FILE")
DEFAULTS="CACHE=0 PERSIST=volatile FUSE_PROFILE=0"
while read -r tag vars; do
  [ -z "$tag" ] && continue
  case $tag in \#*) continue ;; esac
  if [ "$tag" = @defaults ]; then DEFAULTS=$vars; continue; fi
  echo "##### $tag: $vars"
  # shellcheck disable=SC2086
  env MODE=cliofs BENCH=ior $DEFAULTS \
      RUN_TAG="$tag" $vars bash "$HERE/pbs_io500_aurora.sh" < /dev/null 2>&1 \
    | grep -E 'IORCELL|MDCELL|IO500CELL|WFSTAGE|WFCELL|WFCAT|WFCLEAN|RESULT|SCORE|===|FAILED|rc='
done <<< "$CONFIGS"
