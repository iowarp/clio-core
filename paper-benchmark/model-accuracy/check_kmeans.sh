#!/usr/bin/env bash
#===============================================================================
# check_kmeans.sh -- confirm that the k-means consumer really ran in every
# process of one run_kmeans_parallel.sh configuration: each process log must
# show READS "KMEANS iteration i/READS done" lines with 0 chunks failed, for
# every option that has run (fixed, learn, oracle, hcompress).
#
#   check_kmeans.sh DATASET TAG PROCS READS
#
# TAG as run_kmeans_parallel.sh names it, e.g. km4b1gp2i8w1-4-5.
# Prints one line per option and process; exits 1 if any check fails.
#===============================================================================
set -u
DS=$1; TAG=$2; PROCS=$3; READS=$4
RUNS=/mnt/nvme0/v2-work/runs
bad=0
for mode in fixed learn oracle hcompress; do
  # an option not run for this tag (e.g. hcompress) is not checked
  [ -f "$RUNS/$DS-p0_${mode}_$TAG/stdout.log" ] || continue
  for ((i = 0; i < PROCS; i++)); do
    log=$RUNS/$DS-p${i}_${mode}_$TAG/stdout.log
    [ "$PROCS" = 1 ] && [ ! -f "$log" ] && log=$RUNS/${DS}-p0_${mode}_$TAG/stdout.log
    done_n=$(grep -cE "^KMEANS iteration [0-9]+/$READS done: [0-9]+ chunk\(s\) skipped, 0 failed" "$log" 2>/dev/null)
    fail_n=$(grep -cE "^KMEANS iteration [0-9]+/[0-9]+ done: .* [1-9][0-9]* failed" "$log" 2>/dev/null)
    st="ok"
    if [ "${done_n:-0}" -ne "$READS" ] || [ "${fail_n:-0}" -ne 0 ]; then st="FAIL"; bad=1; fi
    echo "$mode p$i: $done_n/$READS k-means iterations done, $fail_n with failed chunks -> $st"
  done
done
exit $bad
