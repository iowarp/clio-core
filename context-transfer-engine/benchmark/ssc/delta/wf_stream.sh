#!/bin/bash
# Keep one wf_step job in the interactive queue until the WfCommons campaign is complete.
#   bash wf_stream.sh            (on a login node; log: ssc_work/logs/wf_stream.log)
D=/u/rpawar/clio-core-ssc/ssc_work/results/wf_done
LOG=/u/rpawar/clio-core-ssc/ssc_work/logs/wf_stream.log
HERE=$(cd "$(dirname "$0")" && pwd)
while [ ! -f $D/ALL_DONE ]; do
  if ! squeue -h -u $USER -n ssc_wfstep | grep -q .; then
    j=$(sbatch --parsable $HERE/wf_step.sbatch) && echo "$(date '+%F %T') submitted $j" >> $LOG
  fi
  sleep 60
done
echo "$(date '+%F %T') ALL_DONE" >> $LOG
