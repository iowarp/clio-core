# ogbn-papers100M end to end through the GPU vector

Two full GraphSAGE epochs over the real ogbn-papers100M feature matrix, with no
flat copy of that matrix anywhere: streamed out of the archive, into the CTE
across a RAM tier and NVMe, and faulted back into an 8 MiB GPU window.

Reproduce with:

    SKIP_AGG=1 EPOCHS=2 RAM_TIER_MB=6144 NVME_TIER_MB=102400 \
      ZIP=/path/papers100M-bin.zip ./run_papers100M.sh

## Status, honestly

The pipeline runs end to end on the real dataset and is bit-exact at a scale
where that can be checked (run_pipeline_test.sh). At FULL scale a single epoch
completes only by consuming essentially all of RAM: shared memory grows ~57 GiB
during the epoch, so the run below finished with the machine at ~0 MiB free,
and a repeat with a 4 GiB safety margin was killed by the guard mid-epoch.
Treat full-scale epochs as not safely runnable on a 60 GiB host until the
allocator defect below is fixed.

Measured mitigations that do NOT work: a larger page cache (window 4 -> 256,
64x fewer evictions) leaves segment growth unchanged at 6 per epoch on the 10M
reproducer, so the growth is not proportional to eviction count and the
per-eviction explanation is incomplete.

## What ran

| | |
|---|---|
| Nodes | 111,058,944 of 111,059,956 (the trainer tiles to a whole page count) |
| Features | 128-d float32 = **54,228 MiB (53 GiB)** |
| Labels | 1,546,782 labelled (1.39%), classes 0..171 — matches OGB |
| Store | 6 GiB RAM tier + NVMe behind it |
| GPU | RTX 4070 Laptop, 8 GiB |

## Measured

| | |
|---|---|
| Ingest | 455.8 s (~119 MB/s) |
| Epoch 0 | 279.1 s |
| Epoch 1 | 261.6 s |
| **Peak GPU, whole process** | **1,018 MiB** |
| **Feature window alone** | **8 MiB** |
| Peak host RSS (shmem) | 0.5 GiB, flat |
| New SHM segments during epochs | **0** |

The matrix is **53x** the peak GPU footprint of the whole process and **6,700x**
the resident feature window.

Learning, on real labels:

    e00  loss 5.134213  acc 0.0014  val_acc 0.0014
    e01  loss 5.092723  acc 0.0265  val_acc 0.0259

Chance is 1/172 = 0.0058, so validation accuracy after two epochs is ~4.5x
chance and loss is falling. Two epochs is far too few to compare against a
published number; this is evidence the training path is correct, not an
accuracy result.

## Things that are true of this run and easy to misread

**Compression is 1.000x, and that is correct.** The compressor keeps the
compressed form only when it reaches 87.5% of raw, because every reader pays a
decompress per page fault and storing at 93% was measured to double a paged
eval for a 7% capacity win. Real papers100M float features compress to ~92.7%
(the 1.079x in the older results), which is the wrong side of that bar, so they
are stored raw. **The 1.079x figure will not reproduce on this branch by
design.**

**The in-core baseline is SKIPPED, not OOM.** In stream mode no host copy of
the matrix exists to upload, so there is nothing to compare against. The test
says so explicitly rather than reporting a fabricated out-of-memory result.

**RETRACTED: the two-epoch numbers above were produced with a read path that
is not correct.** The at() change described below passes bit-exactness only
2 runs in 3 (measured; operator[] passes 3 in 3), and the papers100M run could
not detect that because its in-core baseline is SKIPPED -- nothing compared the
bytes. The timings and memory figures are real; **the loss and accuracy may
have been computed on corrupted reads** and should not be quoted until the run
is repeated on the reverted code.

**The at() story, and why it is reverted.** The gather kernels read through
DeviceVector's non-const operator[], which marks pages dirty. A read-only sweep
therefore left every page it touched needing a writeback; eviction re-put and
re-compressed each one, and that compress allocated a buffer the SHM allocator
would not reuse, growing a 120 MB segment per call — 497 segments and 57 GiB
during a single epoch, which exhausted the machine twice. Reading through
`at()` instead removed it: epoch time fell 32% (409 s -> 279 s) and segment
growth went to zero.

That change is REVERTED, because it is unsafe today: at() fails bit-exactness
in 2 of 3 runs where operator[] passes 3 of 3. That much is measured.

The MECHANISM IS NOT ESTABLISHED. The plausible reading is that a dirty page is
skipped as an eviction victim (the !pgi.dirty test in ClaimSlot), so dirtying
every page incidentally pins it while lanes still hold it. Against that
reading: HoldPageYield syncs all lanes before thread 0 evicts, and each block
owns its own page table, so the obvious intra-block race should already be
excluded. Restoring at() needs the failure reproduced under instrumentation
first -- not a fix aimed at this guess.

A residual remains: on compressible data the read path still decompresses, and
those allocations grow ~2.5 segments per epoch at 10M rows. It does not appear
here because papers100M is stored raw. The underlying defect is that
AllocateBuffer will not reuse a just-freed block of the same size; see the
55-second reproducer in the memory notes and CLIO_SHM_TRACE=1.
