# Figure 9 -- full size, campaign `full-09180417`

End-to-end wall-clock time per workload. Panel (a) is the ablation ladder, panel
(b) the external codecs. Five workloads x 14 arms = 70 arms, one SLURM job per
workload on `gpuA100x4`.

## Files

| Path | What it is |
|---|---|
| `fig9.png`, `fig9a_ablation.png`, `fig9b_baselines.png` | the plotted figure |
| `fig9_stats.csv` | the 70 plotted rows, one campaign, `compute_min`/`io_min`/`total_min`/`ratio`/`eb` |
| `data/<wl>.csv` | each job's own `fig9.csv` verbatim (its own workload filled, the other four blank) |
| `logs/<wl>-<jobid>.out` | the harness driver log: per-arm `loop=/codec=/io=/compute=/total=`, tier bytes, WAL summary, verification verdict |
| `logs/arms/<wl>/<arm>.log` | that arm's driver stdout, including its `time:` line and `VERIFIED:`/`BOUND` lines |

Jobs: Nyx 22178890, VPIC 22178891, WarpX 22178892, AI 22178893, LAMMPS 22178894.
VPIC's `NP+Tier+Async` timed out in the campaign and was rerun alone as 22188224
(`--only`), merged into `data/vpic.csv`; its log is under `logs/vpic-rerun/`.

## What the segments mean

- **Solid** = the write loop (`stage+compress`): stats, NN, quantize, codec, tier
  put, setup, scheduling.
- **Light** = `total_min - compute_min`, i.e. the input read plus the final flush.
- **Excluded from both**: H2D staging (the figure assumes the data is already on
  the GPU, as upstream's VOL write does) and, for LAMMPS, the simulate time.
- `ratio` is input bytes / stored bytes. `eb` is the relative error bound; `0` is
  lossless.

Caveats worth carrying into any claim: the durable writes of an untiered arm
happen inside the write loop, so `io_min` near zero does not mean the arm did no
I/O; and a true compute/IO split is not derivable from this instrumentation.
Run-to-run spread on the lossless, heavy-writing arms is 18-34%.
