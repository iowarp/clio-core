# WarpX workload — Clio + NeuroPress, in situ

A **laser-wakefield acceleration** run in [WarpX](https://github.com/BLAST-WarpX/warpx)
(AMReX PIC, CUDA, float32) compressed **while it runs**, by a **stock,
unpatched WarpX binary**.

This is the only in-situ workload here, and the only one where Clio touches a
simulation that knows nothing about it. LAMMPS is embedded as a library; Nyx and
VPIC are patched to dump fields that are replayed afterwards. WarpX writes
openPMD-HDF5 exactly as it always does, HDF5 dlopens Clio's VOL connector
because `HDF5_VOL_CONNECTOR` says so, and the VOL chunks and compresses on the
way past — with Clio's runtime hosted inside the WarpX process.

```console
$ ldd $(which warpx.3d...) | grep -c clio
0
```

Upstream NeuroPress reaches WarpX through a **636-line patch** adding a whole
`FlushFormatGPUCompress` diagnostic backend. Clio needs **zero lines** in WarpX.

## Running

```bash
./run_sweep.sh                      # every policy, each a full WarpX run
./run_config.sh explore             # one policy
./read.sh --run static-zstd-s4      # read back through the VOL
../collect.py results/              # re-aggregate
```

Requires a stock WarpX with openPMD:

```bash
git clone --depth 1 https://github.com/BLAST-WarpX/warpx.git ~/src/warpx
cd ~/src/warpx
cmake -S . -B build-clio -DCMAKE_BUILD_TYPE=Release -DWarpX_COMPUTE=CUDA \
      -DWarpX_DIMS=3 -DWarpX_MPI=OFF -DWarpX_OPENPMD=ON \
      -DAMReX_CUDA_ARCH=8.0 -DWarpX_PRECISION=SINGLE
cmake --build build-clio -j
```

Defaults: 64×64×512 cells, 40 steps, diagnostics every 10 → 5 openPMD dumps ×
10 field components × 8 MiB = 400 MiB in 400 chunks of 1 MiB.

## The chunk size is a correctness condition, not a tuning knob

`CLIO_VOL_CHUNK_SIZE` is **1 MiB** here, not the 4 MiB the other workloads use,
and that is not a preference.

openPMD emits each AMReX box as a separate partial write, so an 8 MiB field
arrives as 1 MiB pieces at **non-contiguous** offsets — `[0,1M)`, then
`[4M,5M)`, and so on. The VOL's append path assembles only *contiguous* runs and
drops its tail on a discontinuity, so with a 4 MiB chunk **no chunk ever
completes**:

| chunk size | field blobs staged |
|---|---|
| 4 MiB (the default) | **0** |
| 1 MiB | **400** |

At 4 MiB the run still succeeds, the native `.h5` is perfect, exit status is 0,
and nothing anywhere reports a problem — the tier is simply empty of field data.
The rule is that the chunk must be **≤ the writer's contiguous write
granularity**, and this is the first workload here where that binds.

## Results

Laser acceleration, 64×64×512, 5 dumps, 400 MiB float32, 400 chunks, A100,
lossless.

| config | ratio | stored | wall |
|---|---|---|---|
| **`static-zstd-s4`** | **17.433×** | 22.9 MiB | 12.0 s |
| `explore` (ratio cap 1e6) | 17.439× | 22.9 MiB | 22 s |
| `explore` (default cap) | 16.833× | 23.8 MiB | 22.0 s |
| `static-zstd-s8` | 16.367× | 24.4 MiB | 12.0 s |
| `best` | 16.544× | 24.2 MiB | 41.0 s |
| `static-zstd` | 13.218× | 30.3 MiB | 9.0 s |
| `dynamic-ratio` | 11.995× | 33.3 MiB | 13.0 s |
| `dynamic` | 11.788× | 33.9 MiB | 14.1 s |
| `learn` | 8.762× | 45.7 MiB | 17.0 s |

Consistent with the other three workloads: **the 4-byte stride wins on float32**
(17.43× against 16.37× for 8-byte), inference under the balanced cost model
leaves most of the ratio on the table (11.79×), and online learning is the worst
option (8.76×).

Unlike the other workloads, exploration here **does** essentially reach the best
fixed codec — 16.83× against 17.43×, and 17.44× once the ratio cap is raised.

### What the ratio cap does to exploration

`RATIO_CAP` (default 100, upstream's `nn_gpu.cu` constant, overridable with
`CLIO_NEUROPRESS_RATIO_CAP` or this harness's `RATIO_CAP=`) turns out to gate
*whether exploration runs at all*, not just how candidates rank.

The exploration gate fires on `error_pct > threshold`, where

```
error_pct = |actual_cost − predicted_cost| / actual_cost
cost      = w_io · bytes / (min(ratio, RATIO_CAP) · bandwidth)      [ratio-only]
```

When a chunk's predicted *and* actual ratio both exceed the cap, both clamp to
100, the two costs are numerically identical, `error_pct` is exactly **0**, and
the gate cannot fire at **any** threshold — including 0. Measured:

| | chunks explored | ratio |
|---|---|---|
| cap 100 (default) | 78 / 400 | 16.83× |
| cap 10⁶ | 400 / 400 | 17.44× |

and every one of the 322 unexplored chunks at the default cap was one whose
ratio exceeded 100, while not a single chunk below the cap was skipped. So the
mechanism is certain; the *cost* of it here is only ~3% of ratio, because the 78
chunks the gate did select were the ones that mattered.

## Verification

Every run writes the authoritative native `.h5` alongside the tier, so `read.sh`
reads one field dataset twice — through Clio's VOL and through the native VOL —
and compares. Bytes match.

**That is necessary but not sufficient, and this workload does not yet clear the
bar the others do.** Because the native file holds the same data uncompressed, a
VOL read that misses the tier entirely still returns correct bytes. The path
trace currently shows the read taking `MISS (native + stage)` rather than
inverting a codec, so `read.sh` reports **INCONCLUSIVE** and the `verified`
column reads `n/a`. The Nyx and VPIC benchmarks have no such escape — their tier
is the only copy — and both verify by digest. Closing this for WarpX means
finding why the restart-mode probe misses blobs the writer stored; it is not
done.

## Notes

- Each policy is its own WarpX run, so unlike the Nyx and VPIC sweeps the
  policies are not guaranteed byte-identical inputs. WarpX on one GPU is
  deterministic for fixed inputs, so in practice they match.
- The per-chunk record comes from the VOL path trace via `trace_to_csv.py`,
  since the application is stock WarpX and there is no Clio driver to record
  one. That converter must read the `neuropress FINAL` line for the decision and
  the exploration log for an overridden chunk's size — reading `codec ran`
  instead reports the *primary* candidate and understated exploration by 5×
  (12.13× against 16.86×) while line counts matched 1:1.
- Particle datasets are intercepted too, but the reported numbers are
  `--fields-only`; particle arrays are small here and would dilute the field
  measurement.
- `results/` is generated output and is not tracked.
