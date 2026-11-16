#!/bin/bash
# ior_wrap.sh <out_prefix> <ppn> <command...>
# Runs the command; the first rank of every node runs under strace so the
# wall time of each data system call (-T) is recorded to
# <out_prefix>.<host>.strace. Used by BENCH=ior with STRACE=1 to split a
# rank's time into "inside the system call" and "in the application".
pfx=$1; ppn=$2; shift 2
rank=${PMI_RANK:-${PALS_RANKID:-0}}
if [ $(( rank % ppn )) -eq 0 ]; then
  exec strace -T -ttt -e trace=read,write,pread64,pwrite64,lseek,fsync,openat,close \
    -o "$pfx.$(hostname).strace" "$@"
fi
exec "$@"
