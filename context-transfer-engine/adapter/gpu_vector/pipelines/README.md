# gpu_vector Jarvis pipelines

Single-node experiment pipelines for the science-workload benchmarks in
`../benchmark/` (lammps_md, gmx, lbann, grayscott, kmeans, weights),
driven by [jarvis](../../../../external/jarvis-cd) pipeline tests.

## Running

```bash
# one pipeline (env + jarvis + post figures, all handled):
./run.sh workload_understanding/kmeans_mpi_sweep.yaml
# a whole family:
./run.sh workload_understanding
# re-render figures from stored results without re-running:
../../../../.venv/bin/jarvis ppl run yaml <file>   # is what run.sh calls
../../../../.venv/bin/jarvis ppl post yaml <file>  # post: section only
```

Every sweep writes `results.csv` / `results.yaml` + its figures to
`$HOME/gv_pipeline_results/<name>/`. Per-cell benchmark logs go to
`/tmp/clio_gv/<name>/` with one uniquely-named log per parameter
combination.

## The harness contract

- **Packages**: `jarvis_clio_core.clio_gv_workload` (one cell = one
  benchmark run; builds the argv for any (workload, variant) pair) and
  `jarvis_clio_core.clio_gv_register_eval` (static fatbin analysis).
- **Full CTE stack**: the paged benches host the clio runtime
  IN-PROCESS (an external daemon cannot service another process's
  in-kernel page faults) and compose their own tier stack; passing
  `nvme_mb > 0` gives the full three-tier stack hbm + host-RAM + file.
  The MPI/NVSHMEM/NCCL baselines are CTE-free by design — that is the
  substrate comparison.
- **Warmed memory**: every cell exports `CLIO_PREFAULT=0`, which
  pre-faults the WHOLE RAM tier at compose (mem_bdev_transport.cc), so
  timings never include first-touch page population.
- **Validation**: `gates_pass=1` in results.csv only when the binary
  printed its ALL-GATES-PASS / per-gate PASS markers; post scripts
  exclude failed cells and warn.
- **VRAM**: `vram_peak_mb`/`vram_delta_mb` are sampled from
  `nvidia-smi memory.used` at 50 ms during the run — empirical, not
  analytic. Runs shorter than ~100 ms can be missed; the sweeps keep
  cells long enough where VRAM is the question.
- **post: sections**: each yaml carries an inline `post:` script (the
  jarvis pipeline-test feature) that renders PNG figures next to
  results.csv after the sweep, and can be re-run alone with
  `jarvis ppl post yaml <file>`.

## Families

### workload_understanding/ — what moves VRAM, I/O, runtime

One MPI sweep per workload over its size knob (VRAM axis), its steps
knob (runtime axis), and — for lammps_md, the one workload with a real
checkpoint phase — the checkpoint period (I/O axis, stage-D2H + durable
file write per checkpoint). Findings are written into each benchmark
subdirectory's README with the measured numbers.

### register_eval/ — register pressure per substrate

`cuobjdump --dump-resource-usage` over every built
(workload × {mpi,nccl,nvshmem,bam,paged}) binary; reports the
occupancy-binding kernel (max regs), mean regs, and worst stack frame.
No GPU needed. Headline (RTX 4070, sm_89 fatbins): every paged
workload's coroutine kernels sit at the 169-register module ceiling
(except lammps_md paged, capped at 64 by maxrregcount), vs 22–90 for
the CTE-free baselines.

### memory_pressure/ — cache size × parallelism at a 6 GB problem

Per workload: the problem size was CALIBRATED (calibrate_6gb.py, real
runs, nvidia-smi-sampled) so the MPI baseline peaks at ~6 GB VRAM; the
paged edition then runs that SAME problem with the page cache swept
1–6 GB and blocks swept from 1 up to 80% of the maximum concurrent
blocks of an RTX 5080 (84 SMs → 67 blocks at the one-block-per-SM
scale of these persistent-style kernels; on smaller GPUs the same
block counts simply oversubscribe, which is part of the measurement).

## GPU notes

Collected on: NVIDIA GeForce RTX 4070 Laptop (8 GB, sm_89). The 6 GB
target leaves ~2 GB for context + allocator slack on this card. Block
caps reference the RTX 5080 spec as the paper target; block counts are
plain pipeline vars — edit `vars:` to resweep for another card.
