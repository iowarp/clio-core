# GNN training on the Eternia compressed feature store — NCSA Delta A100 (sm_80)

All numbers below were measured on **one NVIDIA A100-SXM4-40GB** (`gpua022`, Slurm
job 20900276), CUDA 12.6.85, native `sm_80` SASS (no PTX-JIT; verified with
`cuobjdump`). Build recipe: `BUILD_RECIPE.md`. Raw logs: `logs/`.

> **Correction to the task premise.** The session did *not* start on a GPU node. It
> started on `dt-login01`, a login node where `nvidia-smi` fails. Every result here
> comes from a real A100 obtained via a Slurm allocation.

---

## 1. Headline (task b) — REAL, non-tiled ogbn-papers100M

`ogbn-papers100M`, 111,017,984 nodes x 128-d float32 aggregated features =
**54,208 MiB (56.8 GB)** against **40,016 MiB** of free HBM. Not tiled, not
subsampled.

```
[TRAIN] IN-CORE: *** OOM *** cannot place 54208MiB A resident.
[TRAIN] ETERNIA: stored A 54208MiB -> 50257MiB (1.079x) in 185.98s
[TRAIN] ETERNIA: epoch0 loss=3.116351 acc=0.2958 -> epoch29 loss=1.295001
        acc=0.6113 val_acc=0.6118 (10732.06s)  peak GPU window = 64MiB
[TRAIN] in-core OOM'd -> Eternia is the ONLY method that TRAINED this matrix.
Passed: 1  Failed: 0
```

**The claim holds: in-core cannot run this graph at all; Eternia trains it inside a
64 MiB HBM window — 847x smaller than the feature matrix — to 61.1% train /
61.2% validation accuracy, with loss decreasing monotonically for 30 epochs.**

This is a **capacity** result, not a performance one. The epoch time (358 s) is
bottlenecked by a readout path that runs ~100x below PCIe bandwidth — see 4b.

### The update rule matters far more than anything else

The run above uses **mini-batch SGD** (`CLIO_GNN_MINIBATCH=1`, batch =
page_rows x window = 131,072 rows -> ~847 updates/epoch). The originally-measured
configuration was **full-batch GD**, which performs exactly **one weight update per
epoch** — so 30 epochs was 30 updates, and the model never escaped the
majority-class floor (0.0626 = the share of the largest of the 172 classes):

| papers100M, 30 epochs, lr 0.2, identical data | updates | train / val acc | final loss |
|---|---|---|---|
| full-batch GD (one update per epoch) | 30 | 6.26% / 6.23% | 4.552200 |
| **mini-batch SGD** | **~25,400** | **61.13% / 61.18%** | **1.295001** |
| mini-batch SGD, 10 epochs | ~8,470 | 57.86% / 57.89% | 1.447957 |

Same features, same lr, same model, same streaming path — **9.8x the accuracy**
purely from doing more weight updates. Both configurations OOM in-core and both run
at the same 64 MiB window, so this changes nothing about the capacity claim; it
changes only whether the accuracy number is meaningful.

**Remaining caveats on the accuracy number.** Only **1.39%** of papers100M nodes
carry a label (1,546,782 / 111,059,956). The validation split here is *every 10th
node by index* (`kValStride = 10`), **not** OGB's official time-based split, so
61.18% is **not** comparable to the ogbn-papers100M leaderboard — an index split is
easier than a temporal one. Loss was still falling at epoch 30 (1.295 and
decreasing), so this is a floor for this model, not its ceiling.

---

## 2. Training correctness (task a) — ogbn-arxiv, native

```
[TRAIN] VERIFY vs in-core: max|dloss|=0.000e+00 max|dacc|=0.000e+00
        max|dvacc|=0.000e+00 -> BIT-EXACT
```

All 30 epochs agree digit-for-digit between the in-core baseline and the streamed
Eternia path. Peak GPU window **1 MiB** vs 82 MiB resident; ratio 1.067x;
0.150 s/epoch in-core vs 0.309 s/epoch streamed.

### Bit-exactness also holds under mini-batch SGD

Mini-batch updates are order-dependent, so bit-exactness was not a given. It holds
anyway — windows are issued in order on a single stream, so both paths perform the
identical update sequence. `max|dloss| = max|dacc| = max|dvacc| = 0.000e+00` on all
four arxiv configurations below:

| arxiv, lr 0.2 | updates | train acc | val acc | final loss | peak window | bit-exact |
|---|---|---|---|---|---|---|
| full-batch, 30 ep | 30 | 0.1882 | 0.1887 | 2.986905 | 32 MiB | **yes** |
| mini-batch, 5 ep, batch 4096 | ~205 | 0.3490 | 0.3486 | 2.434244 | 2 MiB | **yes** |
| mini-batch, 30 ep, batch 16384 | ~310 | 0.4515 | 0.4519 | 2.123241 | 8 MiB | **yes** |
| **mini-batch, 30 ep, batch 4096** | **~1,230** | **0.5859** | **0.5886** | **1.492313** | **2 MiB** | **yes** |

Accuracy is monotone in *update count*, independent of epoch count and batch size:
5 mini-batch epochs (~205 updates) beat 30 full-batch epochs (30 updates), 0.3490 vs
0.1882, in one-sixth the epochs. Note also that mini-batching **shrinks** the peak
window (32 MiB -> 2 MiB), because the window *is* the batch.

Raising the learning rate does **not** substitute for more updates — on a 2M-node
papers100M tile, lr 5.0 and lr 50.0 both landed *below* the majority-class floor
(0.0709 vs floor 0.1242), and lr 5.0 was still stuck at the floor after 300 epochs.
lr 0.2 was already correct; update count was the only lever.

---

## 3. Compression tradeoff (task d) — zstd vs cuSZp on identical data

Same dataset, same seed, same 30 epochs.

| (full-batch GD, 30 epochs) | zstd (lossless) | cuSZp (lossy, balanced) |
|---|---|---|
| stored | 50,257 MiB (49.1 GiB) | **17,342 MiB (16.9 GiB)** |
| ratio | 1.079x | **3.126x** |
| store time | 158.62 s | **88.82 s** |
| epoch time | 278.37 s | 272.18 s |
| peak GPU window | 64 MiB | 64 MiB |
| epoch 0 loss | 5.133193 | 5.133192 |
| epoch 29 loss | 4.552200 | 4.552200 |
| final train / val acc | 6.26% / 6.23% | 6.26% / 6.23% |

Max per-epoch loss divergence over all 30 epochs: **1.0e-06** (full-batch).

Under **mini-batch** the same comparison gives **3.0e-04** at epoch 0, falling to
2.9e-05 by epoch 2 — roughly 300x larger than full-batch, because the lossy error
now perturbs each of ~847 updates per epoch instead of a single one. Still
negligible (final accuracy 0.4979/0.4992 cuSZp vs 0.4978/0.4992 zstd at 3 epochs),
but **quote ~3e-04 for mini-batch, not 1e-06**.

> These codec runs used **full-batch GD**, which is why their accuracy sits at the
> 6.26% majority-class floor — see section 1. That is a property of the update rule,
> not of the codecs: both arms are identical to 1e-06 in loss, which is the point of
> the comparison. The codec sweep was not re-run under mini-batch.

**This is what makes the "compression expands GPU memory" claim real.** Lossless
zstd achieves only **1.079x** on these features — it does *not* meaningfully expand
capacity, and (b)'s capacity win is due entirely to bounded-window streaming, not to
compression. cuSZp delivers **3.126x** (in the 3-4x band predicted) at no measurable
cost to training, and stores *faster* because the codec runs on the GPU.

Codec identity verified in the log:
`Compressor pinned via CLIO_CTE_COMPRESS_LIB: cuszp (wire=20, preset=2)`.

### cuSZp preset comparison — "best" compresses LESS

| preset | wire/preset id | ratio | stored | store time |
|---|---|---|---|---|
| `balanced` | 20 / 2 | **3.126x** | 16.9 GiB | 88.8 s |
| `best` | 20 / 3 | 2.340x | 22.6 GiB | 95.5 s |

**`best` means best *fidelity*, not best ratio.** It applies a tighter error bound
and therefore compresses *less* (2.340x vs 3.126x) and stores slightly slower. If
the goal is capacity, `balanced` is the arm to use; `best` is the arm to cite when
fidelity is the constraint. Only `balanced` reaches the predicted 3-4x band.

---

## 4. Forward capacity sweep (task c) — peak GPU is flat in feature size

`test_gpu_vector_gnn_capacity`, page = 65536 rows (32 KiB), window = 2 pages.

### Lossless (zstd) — bit-exact everywhere

| nodes | features | in-core | ratio | store | readout | peak window | shrink | verify |
|---|---|---|---|---|---|---|---|---|
| 1.97M | 960 MiB | OK 0.25 s | 1.078x | 2.96 s | 4.04 s | 64 MiB | 15x | **BIT-EXACT** |
| 8.00M | 3,904 MiB | OK 1.00 s | 1.078x | 11.95 s | 15.99 s | 64 MiB | 61x | **BIT-EXACT** |
| 20.05M | 9,792 MiB | OK 2.49 s | 1.078x | 21.46 s | 39.80 s | 64 MiB | 153x | **BIT-EXACT** |
| 60.00M | 29,312 MiB | OK 7.46 s | 1.078x | 71.65 s | 118.88 s | 64 MiB | 458x | **BIT-EXACT** |
| **111.02M** | **54,208 MiB** | ***OOM*** | 1.078x | 141.72 s | 219.81 s | **64 MiB** | **847x** | no baseline |

`memcmp=0 max_rel=0.000e+00` at every size where a baseline exists, and the pooled
scalar matches exactly (e.g. `pool[0]=-3.279598e+08` on both paths at 60M).

### Lossy (cuSZp) — 3.1x, error ~4e-04

| nodes | features | in-core | ratio | peak window | verify |
|---|---|---|---|---|---|
| 1.97M | 960 MiB | OK | 3.073x | 64 MiB | DIFF `max_rel=3.454e-04` |
| 8.00M | 3,904 MiB | OK | 3.074x | 64 MiB | DIFF `max_rel=3.904e-04` |
| 20.05M | 9,792 MiB | OK | 3.080x | 64 MiB | DIFF `max_rel=4.344e-04` |
| 60.00M | 29,312 MiB | OK | 3.090x | 64 MiB | DIFF `max_rel=1.335e-04` |
| **111.02M** | **54,208 MiB** | ***OOM*** | **3.085x** | **64 MiB** | no baseline exists |

**The `DIFF` verdicts are not a streaming bug.** The capacity test hard-asserts
bit-exactness against the in-core path, which is unsatisfiable by construction under
a lossy codec; `max_rel ~ 1-4e-04` is cuSZp's error bound behaving correctly. The
test has no lossy mode, so it reports `Failed: 1` — worth fixing upstream. Compare
the same 1.97M point: zstd `1.078x -> BIT-EXACT`, cuSZp `3.073x -> DIFF`. The codec
is the only variable.

**Peak GPU window is 64 MiB at every size, across a 56x range of feature-matrix
sizes.** That flatness is the property the design rests on.

---

## 4b. Streaming throughput — the weak point of these results

Derived from the capacity-sweep timings above. This is the one place where the
numbers are *not* favourable, and it is invisible unless you divide them out.

| run | store MiB/s | readout MiB/s |
|---|---|---|
| zstd 111M | 383 | **246.6** |
| cuSZp 111M | 851 | **244.7** |
| zstd 60M | 409 | **246.6** |
| cuSZp 60M | 928 | **245.7** |
| zstd 20M | 456 | **246.0** |
| cuSZp 20M | 935 | **245.4** |
| zstd 8M | 327 | **244.1** |
| cuSZp 8M | 817 | **245.1** |

**Store throughput varies 2.2x between codecs (817-935 vs 327-456 MiB/s) — the GPU
codec is clearly faster. Readout varies by under 1% (244.1-246.6 MiB/s) across both
codecs and a 14x range of sizes.**

Two conclusions follow:

1. **Readout is not decompression-bound.** If it were, the codec that stores 2.2x
   faster would also read back faster. It does not — the rate is pinned. The
   bottleneck is the page-fetch/RPC path, not the compressor.
2. **246 MiB/s is roughly 100x below PCIe gen4 x16** (~25 GB/s practical). This is
   also why an epoch costs ~278 s: one epoch is essentially one readout pass
   (219.8 s readout vs 278.4 s epoch).

This does **not** affect the capacity or correctness claims — the 64 MiB bounded
window and the bit-exactness results stand independently. But it means **no
performance claim should be made from these runs**, and it is the first thing a
UVA/DGL baseline (task f, not delivered) would expose: a zero-copy feature store
would likely stream at several GB/s. This looks like a pipelining/batching problem
in the fetch path rather than a fundamental limit of the design, but it is
unmeasured — no profiling was done.

---

## 4c. Peak GPU — measured, not just the feature window

The tests report a *feature window* (64 MiB), which is the bounded streaming
buffer. That is **not** the process's GPU footprint. `test_gpu_vector_gnn_train`
now samples `cudaMemGetInfo` every window and reports
`free_at_start - free_now`, i.e. everything this process holds on the device.

| dataset | N (tiled) | in-core A resident | **PEAK GPU** | feature window |
|---|---|---|---|---|
| ogbn-arxiv | 131,072 | 64 MiB | **822 MiB** | 128 MiB |
| ogbn-products | 2,490,368 | 950 MiB | **1,652 MiB** | 100 MiB |
| **ogbn-papers100M** (zstd) | 111,017,984 | **0 (OOMs)** | **1,734 MiB** (measured) | 64 MiB |
| **ogbn-papers100M** (cuSZp) | 111,017,984 | **0 (OOMs)** | **2,858 MiB** (measured) | 64 MiB |

Decomposing the two measured points (peak = in-core A + labels + window buffers +
CUDA context) gives, for papers100M: labels 847 MiB (N x 8 B) + window buffers
215 MiB (C=172 inflates `dz2_buf`) + CUDA context ~600 MiB + CTE HBM tier 0 (it
spills to DRAM — every capacity run logged `split HBM=0MiB`) + **no** in-core
allocation, because in-core OOMs and never allocates. Total **~1.6 GiB**, i.e. the
52.9 GiB matrix trains in ~1.6 GiB of GPU — a **~33x** reduction.

**The peak is dominated by the labels (847 MiB), not by the feature window
(64 MiB).** The window is bounded and independent of N; the labels are O(N) and are
not. That is a real limit of the current design and should be stated as such.

**The papers100M rows are now measured directly** (job 20936257). The earlier
extrapolation predicted ~1,660 MiB; the measured zstd figure is 1,734 MiB, a 4%
error — but it missed cuSZp entirely. **The GPU codec costs 1.1 GiB more than the
CPU codec** (2,858 vs 1,734 MiB) because cuSZp decompresses on-device and needs
its own working buffers, which zstd does not. Quote **1.7 GiB for lossless and
2.8 GiB for lossy** — there is no single peak-GPU number for the system.

So the 52.9 GiB matrix trains in **~1.7 GiB (zstd, ~31x)** or **~2.8 GiB (cuSZp,
~19x)**. The lossy codec buys 3.126x on *stored* size at the cost of ~1.1 GiB more
*resident* GPU — a tradeoff worth stating explicitly.

---

## 5. Feature-cache prediction (task e) — reverse-PageRank on the real graph

`N=111,059,956`, directed `E=1,615,685,872`, `avg_out=avg_in=14.55`. Forward PR
converged to `L1=6.79e-08`, reverse PR to `L1=3.43e-07` at 50 iterations.

**The denser graph raises the hit rate, as predicted:**

| metric | ogbn-arxiv (published) | **ogbn-papers100M** |
|---|---|---|
| top-10% reverse-PR share of reads | 21.7% | **34.7%** |
| hit rate @ 10% budget (full-sweep) | 21.7% | **34.8%** |
| fewer decompressions vs random | 13.0% | **27.5%** |

Over one epoch's 30,484,640 reads, PR-pinning serves 19,891,001 misses vs random's
27,436,683.

### Page granularity matters more than the ranking

| page size | PR-pin | **PR-REORDER+pin** | degree-pin | LRU | random |
|---|---|---|---|---|---|
| 1 row (512 B) | 34.8 | 34.8 | **40.6** | n/a | 10.0 |
| 8 rows (4 KiB) | 19.3 | **34.8** | 21.0 | 17.5 | 10.0 |
| 64 rows (32 KiB) | 14.8 | **34.8** | 16.2 | 14.4 | 10.0 |
| 512 rows (256 KiB) | 13.9 | **34.8** | 16.0 | 13.6 | 10.0 |

At the 32 KiB page Eternia actually uses, plain PR-pinning degrades to 14.8% —
barely above LRU (14.4%) — because hot nodes scatter across pages. **Reordering
holds 34.8% at every page size.** The win comes from the layout, not the ranking.

**Caveats:** degree-pinning *beats* reverse-PR at row granularity on the full-sweep
trace (40.6% vs 34.8%); PageRank wins on the minibatch trace (48.2% vs 34.6%) and at
coarse pages with reordering. The full-sweep trace is subsampled 1-in-53 (30.5M of
1.6B accesses). Full tables: `results/pagerank_papers100M.md`.

---

## 6. NOT DELIVERED

* **(f) UVA / zero-copy baseline (DGL or PyG).** The project's apptainer image
  (`deps-nvidia.sif`) contains **no torch, no DGL, no PyG**, and the compute nodes
  have no working offline install path for them. There is therefore **no measured
  UVA epoch-throughput or peak-GPU comparison** in this report. This is a real gap
  in the requested scope, not an omission by choice.
(Everything else in the requested scope was run, including the cuSZp `best` preset.)

---

## 7. Defects found and fixed (required to run any of this)

1. **`test/CMakeLists.txt` references sources that exist in no branch** —
   `test_gpu_vector_kmeans_real_gpu.cc`, `test_gpu_vector_kmeans_capacity_gpu.cc`
   (`git log --all --diff-filter=A` finds nothing). The branch **cannot configure**.
   Guarded with `if(EXISTS ...)`.
2. **`kMaxC = 64` in `test_gpu_vector_gnn_train_gpu.cc`** — papers100M has **172**
   classes, so the test aborted on a hard `REQUIRE`. Raised to 192.
3. **`gnn_agg.py` was unrunnable at this scale** — `X.astype(np.float64)` alone is
   114 GB, `X[indices]` ~1.6 TB. Rewritten with memmap + nnz-bounded row blocks +
   `np.add.reduceat`. **Verified bit-identical** (max|diff| = 0.000e+00) to the
   original on 6 randomized graphs including all-isolated and empty-first/last-row
   edge cases.
4. **`gnn_prep.py` had no papers100M path** (raw is `.npz`, not CSV). Added
   streaming extract/convert plus an in-place uint64 key sort so peak RAM is one
   24.1 GiB array. **Verified byte-identical** to the original `build_csr` on 8
   randomized graphs.
5. **`gnn_pagerank_cache.py` hardcoded `(ogbn-arxiv)`** into the report title, so
   every non-arxiv report was mislabeled. Now derived from `--data`.

### Cosmetic issues left alone

* The train/capacity tests print `... -> NNNNMiB zstd (R x)` regardless of the active
  codec; the string is a hardcoded literal. Under cuSZp the log still says "zstd".
* The training CSV has **no column recording the update rule**, so full-batch and
  mini-batch rows are indistinguishable in `SUMMARY.md` except by their accuracy.
  Runs are separated by file instead: `gnn_train_results.csv` (full-batch),
  `gnn_minibatch.csv` (arxiv mini-batch), `gnn_papers_minibatch.csv` (papers100M
  mini-batch); `gnn_train_all.csv` is their concatenation.
* `results/gnn_cap_results.csv` has **no `lib`/`preset` column**, so zstd and cuSZp
  rows are distinguishable only by ratio. Worse, the four cuSZp points that tripped
  the bit-exact `REQUIRE` aborted *before* writing their CSV row — only the 111M
  point (which never reaches that assertion) was recorded. **The capacity tables
  above are reconstructed from the logs, which are complete; the CSV is not.**

---

## 8. Reproduce

```bash
./build.sh                                   # cuSZp V3.0.0 + clio-core, sm_80 native
./prep_papers.sh                             # papers100M -> /tmp/gnn/data/papers100M
JOB=<slurm-job> ./rj.sh bash run_exp.sh a    # arxiv correctness      (bit-exact)
JOB=<slurm-job> ./rj.sh bash run_exp.sh b    # *** HEADLINE ***       (in-core OOM)
JOB=<slurm-job> ./rj.sh bash run_exp.sh c    # capacity sweep, zstd   (bit-exact)
JOB=<slurm-job> ./rj.sh bash run_exp.sh d_bal# cuSZp train + capacity (3.1x)
JOB=<slurm-job> ./rj.sh bash run_exp.sh e    # PageRank cache table
python3 make_figs.py                         # figures from MEASURED logs
python3 make_summary.py                      # summary table
```

`make_figs.py` replaces the upstream `gnn/make_training_fig.py`, whose loss/accuracy
series and `peak = [10.24, 0.03125]` GiB bars were **hardcoded from an old 8 GB-GPU
run** and read nothing from any actual run.
