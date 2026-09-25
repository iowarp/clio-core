## Environment

- The repository is at `/work/clio-core`. A Release build of it, with CUDA
  enabled through nvcc, is at `/work/build` (`$CLIO_BUILD_DIR`); its
  libraries are in `/work/build/bin`. The coroutine transpiler is
  `/work/build-coroc/clio-coroc` (`$COROC`). Do not modify or rebuild the
  library sources; build only your own code against this build.
- One GPU: NVIDIA RTX 4070 Laptop, 8 GB, sm_89. CUDA 12.9 at
  `/usr/local/cuda-12.9`. 16 CPU cores. **The GPU is shared with other jobs
  running at the same time: keep every test run under 500 MB of GPU memory
  in total** (use `--data-mb` <= 512 and `--cache-mb` <= 256), and expect
  timings to be noisy.
- You are working alone and unattended. Do not ask questions; make reasonable
  decisions and keep going until the deliverable is complete and you have
  tested it.

## Deliverable contract (the grader relies on exactly this)

- Put all of your code under `/work/clio-core/agent_task/`.
- `agent_task/build.sh` must build from a clean state, non-interactively, and
  produce the executable `agent_task/bin/bench`. It may assume the
  environment above.
- `bench` must accept the flags listed in the task (unknown flags are an
  error), run the whole workload in one process, and print **exactly one**
  line on stdout that starts with `RESULT ` followed by space-separated
  `key=value` pairs, containing at least the keys the task lists. Anything
  else may go to stderr. Exit code 0 on success; nonzero if anything went
  wrong -- never print a RESULT line for a run you know to be wrong.
- `faults` and `evicts` must be the real counts of page faults and page
  evictions reported by the vector for the timed region, not estimates.
- The grader runs `bench` several times with different flags, including
  configurations where the GPU cache is much smaller than the data, and
  compares the output with an independent reference and with the
  performance of other implementations of the same workload on the same
  machine and the same flags.
