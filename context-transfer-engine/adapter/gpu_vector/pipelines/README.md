# GPU vector benchmark pipelines

Six jarvis pipelines over five workloads, each driving a GPU vector whose data
does not fit on the device. All of them write to
`/home/llogan/Documents/Projects/iowarp/core/results/<pipeline>/`, which is
gitignored — results are data, not source.

    export PATH=$PWD/build-coro/bin:$PATH
    export LD_LIBRARY_PATH=/home/llogan/gnn/cuszp/lib:$LD_LIBRARY_PATH
    jarvis repo add ./jarvis_clio_core
    jarvis ppl run yaml context-transfer-engine/adapter/gpu_vector/pipelines/<file>.yaml

`jarvis ppl post yaml <file>` re-runs only the post-processing over results
already collected.

## The workloads, and why there are five

Each exercises a different **access pattern**, which is what makes the set
worth having — the page-size result below is entirely explained by them.

| pipeline | workload | pattern | reuse |
|---|---|---|---|
| `gv_weights_page_sweep` | model weights | re-read the whole matrix each pass | full |
| `gv_kmeans_page_block_sweep` | k-means | stream the point set once per pass | none within a pass |
| `gv_gnn_page_block_sweep` | GNN training | gather scattered node rows | scattered |
| `gv_grayscott_page_block_sweep` | Gray-Scott | sliding 3-plane stencil window | partial (2 of 3) |
| `gv_tiered_flush_sweep` | block flush | write a region and flush it | none |
| `gv_weights_cache_sweep` | model weights | block count vs cache size | full |

## Headline result: page size dominates, and locality explains the ordering

All measured at **2× VRAM** (16 GB/GiB datasets against an 8188 MB card),
uncompressed, with the dataset held constant across the page axis.

| workload | pattern | page-size effect | block-count effect |
|---|---|---|---|
| weights | re-read | **33.6×** (35288 → 1050 ms) | see cache sweep |
| k-means | stream | **16.9×** (65501 → 3880 ms) | 2–3× |
| Gray-Scott | sliding window | 5.5× *(indicative only)* | ~2.8× |
| GNN | gather | 4.8× (48.90 → 10.15 s) | **~1.0×** |

Page size pays most where every fetched byte is eventually used (weights) and
least where rows are scattered so a large page over-fetches (GNN). Faults
scale almost exactly inversely with page size — weights: 4,193,065 → 15,175
for a 256× page increase.

Block count is a minor axis everywhere, and for the GNN gather it is worth
essentially nothing (48.90 s vs 52.00 s at 16 KB).

## Reading the results honestly

Each pipeline's `post:` block runs trust checks *before* reporting any timing,
because several failure modes here produce plausible numbers rather than
errors:

- **`logical_mb` must be constant** across a page sweep. A run where the
  package failed to pass `--page-kb` shrank the dataset from 16 GB to 256 MB
  across the axis and reported a spurious **92×** "page-size effect".
- **`fits_in_hbm` / `hbm_used_ok`** — a kHBM tier that received nothing is a
  misconfiguration, not a result. The DPE ranks by a predicted bandwidth model
  and has been measured leaving a correctly sized HBM tier empty.
- **checksums compare with a RELATIVE TOLERANCE**, never for equality:
  `atomicAdd` makes float summation order follow the page and block layout.
  GNN is the exception — its `final_loss` is bit-identical (3.703790, spread
  0.0000%) across every page size and block count, which is the strongest
  correctness evidence in the set.
- **jarvis's `status` and exit code are not trustworthy.** It has reported
  "36 successful, 0 failed" for a sweep in which two cells hung and were
  killed, and exits 0 on an unknown subcommand. Trust the per-cell logs and
  the `completed` flag.
- **jarvis's `runtime` column is whole-process wall clock** and is 44–55%
  setup on these workloads; every pipeline reports the benchmark's own
  measured time instead.

## Resource guards, one per thing that actually cost runs

The packages refuse a cell up front rather than letting it die mid-run,
because every one of these failures looked like a hang (a truncated header, no
error):

| guard | what it caught |
|---|---|
| device memory: `blocks × slots × page_kb` + tier | 3 k-means cells (16 GB of cache on an 8 GB card) |
| host memory: matrix + pinned tier vs `MemAvailable` | GNN OOM-killed (exit 137, no message) |
| `run_timeout` per cell | a wedged cell stalling an entire 36-cell sweep |
| pre-flight wait for free VRAM | 3 GNN cells started inside the previous cell's teardown window |
| window sized in **bytes**, not pages | 2 GNN cells (a 256-page window is 2 GiB at 4 MB pages) |

Sweeps must not run concurrently: a second runtime does not error, it stalls
on a fraction of the card.

## Known issues

- **Gray-Scott is indicative, not validated.** Its field checksum is not
  reproducible run to run (3.37e-04 spread on identical settings), which
  indicates a remaining data race or stale read. Two real defects were found
  and fixed along the way — missing cross-step flush waits, and per-block
  caches holding stale copies of neighbours' planes — taking it from 7.71e-03
  to 8.39e-04, but neither closed it.
- **Gray-Scott's page axis is confounded**: one page is one XY plane, so page
  size also changes the grid geometry. Unlike the other workloads it is not a
  pure paging comparison.
- **GNN's host tier is `ram`, not `pinned`**, because the trainer transiently
  holds two host copies of the matrix and pinned pages are unswappable. That
  penalises its host tier for its allocation type rather than its speed, so
  GNN's absolute host-tier numbers are not directly comparable with the
  others'.
- Two GNN cells at 64 blocks fail sporadically to the VRAM teardown window;
  the same cells pass on other runs.
