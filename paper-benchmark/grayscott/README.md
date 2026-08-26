# Gray-Scott workload — Clio + NeuroPress, in situ

A **reaction-diffusion PDE** stepped on the GPU and compressed **while it
runs**, through Clio's HDF5 VOL connector. Fourth workload, and the only
**synthetic** one: its `(F, k)` pair selects the pattern regime, so the data's
character is a parameter rather than a property of a physics code. Upstream
NeuroPress uses Gray-Scott for the same reason
(`benchmarks/grayscott/grayscott-benchmark-pm.cu`).

The application is `bin/neuropress_grayscott_h5`. It links nothing from Clio —
`ldd` shows zero Clio libraries — and Clio enters through
`HDF5_VOL_CONNECTOR=clio`, exactly as in the WarpX workload. There is no
dump-and-replay phase: the field never lands in a file before it is
compressed.

```bash
./run_sweep.sh                                  # every policy
./run_config.sh dynamic --regime chaos          # one policy, one regime
./read.sh --run static-zstd-s4                  # cold read, separate process
../collect.py results/                          # re-aggregate
```

## Two things a pass has to prove

Bit-exactness alone proves nothing here, and the harness would happily report
it. Both of these bit while this benchmark was being written:

**The chunk size must match between the phases.** The VOL names blobs by chunk
index, so a reader using a different `CLIO_VOL_CHUNK_SIZE` asks for names
nothing ever stored, every lookup misses, and HDF5 serves the native file
instead — bit-exact, fast, and completely uninformative.

**So must the stamp granularity.** `CLIO_VOL_STAMP_GRANULARITY_NS` takes part
in the blob name; leaving it at its default in one phase and not the other
produced the same silent miss. Both phases pin it to 0.

Hence the verification here is two conditions, not one: the datasets come back
bit-exact against a reference written with plain POSIX I/O *outside* HDF5, and
the path trace shows a codec actually being **inverted**. A run that is
bit-exact with zero inversions is reported `INCONCLUSIVE`, not `pass`.

## Results

Gray-Scott 256³ spots (`Du=0.2 Dv=0.1 F=0.02 k=0.048 dt=1`), 200 steps,
snapshot every 25 → **8 frames × 64 MiB = 512 MiB float32 in 128 chunks of
4 MiB**, A100, lossless. Every configuration: 128 of 128 chunks inverted by a
codec on the read-back, 8 of 8 datasets bit-exact.

| config | ratio | stored | codec mix | wall |
|---|---|---|---|---|
| **`static-zstd-s4`** | **98.2×** | 5.2 MiB | zstd ×128 | 5.4 s |
| `static-zstd-s8` | 86.7× | 5.9 MiB | zstd ×128 | 4.7 s |
| `static-zstd` | 81.3× | 6.3 MiB | zstd ×128 | 5.3 s |
| `explore` | 80.9× | 6.3 MiB | zstd ×106, lz4 ×22 | 7.7 s |
| `best` | 46.9× | 10.9 MiB | lz4 ×84, zstd ×44 | 11.7 s |
| `dynamic-ratio` | 46.9× | 10.9 MiB | lz4 ×128 | 4.6 s |
| `dynamic` | 17.1× | 29.9 MiB | bitcomp ×116, zstd ×10, cascaded ×2 | 5.0 s |
| `learn` | 15.2× | 33.6 MiB | bitcomp ×72, cascaded ×28, snappy ×14, zstd ×14 | 4.7 s |

The same three findings as the other three workloads, on data an order of
magnitude more compressible:

* **The 4-byte stride wins on float32** — 98.2× against 86.7× for 8-byte and
  81.3× for none. Same direction as Nyx (162.9 / 141.2 / 132.6) and VPIC.
* **The balanced cost model is badly wrong on compressible data.** `dynamic`
  reaches 17.1× where 98.2× was available, and it gets there by picking
  bitcomp for 116 of 128 chunks — the fastest codec, on data whose whole value
  is in the slow entropy coders. Zeroing the two latency weights
  (`dynamic-ratio`) nearly triples it to 46.9×.
* **Online learning is again the worst arm** (15.2×), below plain inference.
* Exploration does better here than on the other workloads (80.9×, within 1 %
  of a fixed no-shuffle zstd) but still trails the right fixed stride.

`best` and `dynamic-ratio` land on exactly the same ratio (46.918×) because
best mode *is* the ratio-only objective evaluated exhaustively; the difference
is that `best` pays 11.7 s to reach the same answer 4.6 s of inference already
found.

### The regime is the dial, and it barely moves this data

`--regime spots|stripes|chaos|sparse` selects the `(F, k)` pair (`GS_REGIME` in
the deck). At 256³, 200 steps, with a fixed `nvcomp-zstd` + 4-byte shuffle:

| regime | ratio | | `dynamic` |
|---|---|---|---|
| spots | 98.2× | | 17.1× |
| stripes | 104.3× | | — |
| chaos | 96.9× | | 16.9× |

Only a 7 % spread. 200 steps is not long enough for the regimes to diverge —
the field is still close to its initial condition, which is what dominates
compressibility at this length. A run of several thousand steps would be
needed to make the dial mean what it is supposed to mean; the harness supports
it (`--steps`), and the numbers above are honest about what was measured.

## Notes

- Chunk 4 MiB works here, unlike WarpX: the application writes each snapshot as
  one contiguous `H5Dwrite`, so chunks always complete. The 1 MiB rule that
  binds WarpX comes from openPMD's per-box partial writes, not from the VOL.
- `results/` and the HDF5 files it writes are generated and untracked.
