# gpu_vector streaming with the compressor on the fault path

Measured with `clio_gpu_vector_stream_bench`: 16 blocks x 8 page slots,
256 KB pages, 256 threads, batched faulting (`--read-batch 8`), a 512 MB
`hbm` top tier and a spill tier capped at 2000 MB/s
(`CLIO_BDEV_THROTTLE_MBPS=2000 CLIO_BDEV_THROTTLE_MATCH=gv_stream_spill`).

The cap matters: without a tier that is slower than the decompressor there is
nothing for compression to save. A raw read is a copy-engine transfer costing
no CPU, so on a host where every tier runs at memory speed no CPU codec can
win. The cap is what makes the tiering question askable at all.

Throughput in GB/s. `mult` is total data / top-tier size. Every run verified
with a position-weighted checksum (sensitive to permutation, not just to
content).

| mult | zero% | mode | stored ratio | write | read  |
|------|-------|------|--------------|-------|-------|
| 0.5x | 30%   | raw  | 1.00x        | 3.06  | 27.55 |
| 0.5x | 30%   | lz4  | 1.41x        | 2.47  | 3.04  |
| 0.5x | 80%   | raw  | 1.00x        | 3.08  | 25.94 |
| 0.5x | 80%   | lz4  | 5.21x        | 2.63  | 3.90  |
| 2x   | 30%   | raw  | 1.00x        | 1.33  | 3.69  |
| 2x   | 30%   | lz4  | 1.47x        | 1.81  | 4.73  |
| 2x   | 80%   | raw  | 1.00x        | 1.31  | 3.65  |
| 2x   | 80%   | lz4  | 5.19x        | 3.07  | 5.79  |
| 4x   | 30%   | raw  | 1.00x        | 1.03  | 2.55  |
| 4x   | 30%   | lz4  | 1.43x        | 1.41  | 6.10  |
| 4x   | 80%   | raw  | 1.00x        | 1.02  | 2.55  |
| 4x   | 80%   | lz4  | 4.65x        | 3.26  | 6.04  |

## What it says

**Compression is a bad trade when the data fits and a good one when it does
not.** At 0.5x it costs 8x on reads (27.55 -> 3.04): the pages are already in
the fast tier, so a raw read is a device-to-device copy with no CPU in it and
the codec is pure added work. At 2x it is ahead, and at 4x it is 2.4x ahead.
The crossover is between fitting and not fitting, not a property of the codec.

**The compressed arm barely notices oversubscription.** Raw reads fall
27.55 -> 3.69 -> 2.55 as the working set outgrows the tier; lz4 reads go
3.04 -> 4.73 -> 6.10, i.e. they IMPROVE. Compression is not making the slow
tier faster, it is keeping the working set out of the slow tier: at 4.65x the
2 GB dataset stores in 441 MB, which fits the 512 MB top tier outright, and a
per-tier I/O trace shows the spill tier serving no reads at all.

**Ratio matters less than whether it is enough to fit.** 30% and 80% land
within noise of each other at 4x (6.10 vs 6.04) because both spend the read
phase mostly in the top tier. At 2x, where 30% still does not fit, the
difference shows (4.73 vs 5.79).

**Writes track the ratio directly**, since compression reduces what is
written: 1.02 -> 3.26 at 80%/4x.

## Reading the counters

`GVSTAT` reports the pager's own counters, and two of them look alarming and
are not:

- `faults` comes out at ~2x the page count. The write phase faults each page
  before writing it (write-allocate) and the read phase faults it again.
- `get_errors` comes out at ~1x the page count, and is benign: the write
  phase's first touch asks the CTE for a blob that does not exist yet.
  Measured at the phase boundary, the read phase adds ZERO errors.

Check `num_ok` on a batch before concluding the read path is failing.

## Open

In the compressed, multi-tier configuration the bdev trace counts fewer read
operations than pages read (4484 vs 8192 at 30%/4x), specifically for
incompressible pages resident on the spill tier. Data is correct there under
the permutation-sensitive checksum, and the gap is not on the gpu_vector side
(scalar and batched faulting produce identical counts). Cause unresolved.
