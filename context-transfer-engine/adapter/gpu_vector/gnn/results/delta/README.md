# NCSA Delta — A100-SXM4-40GB run of the GNN / compressed-feature-store benchmarks

Measured results for `test_gpu_vector_gnn_train` and `test_gpu_vector_gnn_capacity`
on **ogbn-arxiv** and **real, non-tiled ogbn-papers100M**, on one NVIDIA
A100-SXM4-40GB (native `sm_80`, CUDA 12.6.85).

| file | what |
|---|---|
| `RESULTS.md` | the full write-up, including the limitations — **read this first** |
| `BUILD_RECIPE.md` | exact, reproducible build (apptainer, cuSZp V3.0.0, sm_80) |
| `csv/` | raw CSVs written by the tests, plus `SUMMARY.md` and the PageRank tables |
| `figures/` | `fig_training.png`, `fig_codec.png` — regenerated from `evidence/` |
| `evidence/` | the signal lines extracted from every run log (small, readable) |
| `logs-full.tar.gz` | every raw run log in full (97 MB -> 1.9 MB) |
| `scripts/` | the Slurm/apptainer drivers used to produce all of it |
| `make_figs.py`, `make_summary.py` | regenerate the figures and the summary table |

## Headline

On real, non-tiled ogbn-papers100M — 111,017,984 nodes x 128-d aggregated features
= **54,208 MiB** against **40,016 MiB** of free HBM:

```
IN-CORE: *** OOM *** cannot place 54208MiB A resident.
ETERNIA: 54208 -> 50257 MiB (1.079x);  peak GPU window = 64 MiB
         loss 5.133193 -> 4.552200 over 30 epochs
```

The in-core path cannot run this graph; Eternia trains it inside a **64 MiB** HBM
window (847x smaller), and the peak window stays **flat at 64 MiB across a 56x range
of feature-matrix sizes** (960 MiB -> 54,208 MiB).

## Read the caveats before quoting anything

`RESULTS.md` states these at length; in short:

* **Final accuracy is not a meaningful ML result.** 6.26% / 6.23% is exactly the
  172-class majority-class floor (only 1.39% of papers100M nodes are labelled, and
  30 full-batch steps do not escape it). Loss is the discriminating metric.
* **Lossless compression is only 1.079x**, so the capacity win comes from
  bounded-window streaming, *not* from compression. Only lossy cuSZp reaches 3.126x.
* **No performance claim should be drawn from these runs.** Readout throughput is
  pinned at ~246 MiB/s across both codecs and all sizes — roughly 100x below PCIe
  gen4 x16 — which bounds epoch time. See `RESULTS.md` section 4b.
* **The DGL/PyG UVA baseline was not delivered** (no torch in the container), so
  there is no comparative measurement against a zero-copy feature store.

## Reproducing

`scripts/*.sh` derive their paths from `$BENCH_ROOT` (defaulting to this directory),
so they relocate with the tree. They still assume NCSA Delta specifics — a Slurm
allocation, `apptainer`, and node-local scratch at `/tmp/gnn`. See `BUILD_RECIPE.md`.

```bash
python3 make_figs.py      # figures from the committed evidence/
python3 make_summary.py   # summary table from the committed csv/
```

`make_figs.py` supersedes `../../make_training_fig.py`, whose loss series and peak-GPU
bars were hardcoded from an older 8 GB-GPU run and read nothing from any actual run.
