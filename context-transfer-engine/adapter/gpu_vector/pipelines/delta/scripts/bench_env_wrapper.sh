#!/bin/bash
# Invoked as: apptainer exec ... SIF bench_env_wrapper.sh <bench> <args...>
export PATH=/u/rpawar/eternia-4node/build-clang/bin:/usr/local/cuda-12.6/bin:${PATH:-}
export LD_LIBRARY_PATH=/u/rpawar/eternia-4node/build-clang/bin:/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}
exec "$@"
