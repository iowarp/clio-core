# VPIC in situ — Clio + NeuroPress inside the simulation

A **Weibel instability** computed by [VPIC-Kokkos](https://github.com/lanl/vpic-kokkos)
(CUDA, float32) hands its field arrays to Clio **from inside the running
process**. No `.f32` files, no replay: the compressed tier is the only copy of
the data the moment the deck returns from `begin_diagnostics`.

The offline counterpart is `paper-benchmark/vpic`, which dumps 16 flat files
per frame and sweeps them later. Same simulation, same data, same NeuroPress
knobs — what changes is that nothing is written to a filesystem in between.

```
  deck (weibel_clio.cxx, built --insitu)
    └─ begin_diagnostics ── clio_vpic_insitu_stage(k_f_d.data() + m*nv, nv)
                                 │  D2D into a Clio-registered backend
                                 ▼
                            AsyncDynamicSchedule ── NeuroPress picks a codec
                                 ▼
                            CTE tier (this process hosts the runtime)
```

## Why this one needs no de-interleaving

VPIC keeps its fields in `Kokkos::View<float*[FIELD_VAR_COUNT]>` with no
explicit layout, so a CUDA build takes the default **LayoutLeft**: element
(voxel `v`, variable `m`) sits at `v + m*n_voxels`. Every variable is therefore
already a contiguous device array at `k_f_d.data() + m*n_voxels`, and the
hand-over is a pointer and a count.

Measured, not assumed — the deck refuses if it ever stops being true:

```
[layout] host stride0=1 stride1=5832  dev stride0=1 stride1=5832  nv=5832  contig=1
```

A host/OpenMP Kokkos build defaults to LayoutRight, where the same view is
interleaved and `base + m*nv` addresses a different quantity entirely. The deck
checks `stride(0)==1 && stride(1)==nv` before every hand-over and aborts with a
named message otherwise; the adapter independently refuses a pointer that is
not device memory, which is what such a build would present.

Contrast with the sibling adapters: Nyx hands over a FAB carrying ghost cells,
so `clio_nyx_insitu` must extract the valid box with a strided
`cudaMemcpy3DAsync` first; LAMMPS' device path gathers into atom-ID order on
the GPU. VPIC needs neither.

## Building

```bash
cmake --build <clio-build> --target clio_vpic_insitu     # libclio_vpic_insitu.so
cd paper-benchmark/vpic && ./build_deck.sh --insitu       # deck links it
```

`build_deck.sh` patches the `bin/vpic` compile script VPIC generates, the same
way it already injects `-lcuda`: `-I` for this directory, `-DCLIO_VPIC_INSITU`,
`-L`/`-l`/`-rpath` for the adapter. The deck itself includes only
`clio_vpic_insitu.h` — six `extern "C"` declarations and `<stddef.h>` — so no
Clio C++ header ever goes through `nvcc_wrapper`. Without `--insitu` the deck
compiles exactly as before and the in-situ block is `#ifdef`'d out.

## Running

```bash
./run.sh --ncell 126 --steps 200 --int 25 --verify   # in situ, 8 frames, 1 GiB
./read.sh                                            # cold read, separate process
./crosscheck.sh --ncell 30 --steps 50                # bytes == the deck's own dump
```

`run.sh` writes `$STORE/compose.yaml` — the same three pools as every sibling
(`clio_bdev` 301.0, `clio_cte_compressor` 512.0 → `clio_cte_core` 513.0) — and
runs VPIC with `CLIO_SERVER_CONF` pointing at it and `CLIO_WITH_RUNTIME=1`.

| flag | |
|---|---|
| `--ncell N --steps N --int N --nppc N` | grid, length, hand-over interval, particles per cell |
| `--chunk BYTES` | bytes per compressor call (default 4 MiB) |
| `--verify` | read every blob back through the decompressor at the end of the run |
| `--learn` / `--explore K` / `--threshold X` / `--best` | the NeuroPress knobs, as in the offline sweep |
| `--static LIB` / `--static-shuffle N` | fixed codec, NeuroPress bypassed |
| `--port N` / `--store DIR` / `--restart` | one runtime per store; ports must not collide |

Environment the adapter reads directly: `CLIO_VPIC_TAG`, `CLIO_VPIC_CHUNK`,
`CLIO_VPIC_POOL`, `CLIO_VPIC_REPORT`, `CLIO_VPIC_RAW_DIR`, `CLIO_VPIC_VERIFY`.

## The runtime must be in the process

`clio_vpic_insitu_begin` refuses (exit 6) unless `CLIO_RUNTIME_MANAGER->IsRuntime()`.
The compressor chimod's tasks declare their wire format as
`SerializeStart`/`SerializeEnd`, which no archive calls, so a `DynamicSchedule`
sent to a runtime in another process arrives with `blob_name=""` and `size=0`
and stores nothing while the caller sees success. In-process the task is handed
over by pointer and no archive runs. The Nyx README records the measurement;
this adapter inherits the refusal rather than the failure.

## Measured

A100, CUDA 12.6, Weibel 126³ cells (128³ = 2,097,152 voxels with the ghost
layer), nppc 8, 200 steps, hand-over every 25 → **8 frames × 16 variables ×
8 MiB = 1,024 MiB in 256 chunks of 4 MiB**, lossless, float32.

```
[clio-vpic-insitu] stored 256 blob(s) from 8 frame(s), 1073741824 B in -> 840133230 B  (ratio 1.278)
  compressed: 200   stored raw: 56   failed: 0
   div_b_err  67108864 ->    148992  (450.4x)      rhob  67108864 -> 29121216  (2.30x)
   div_e_err  67108864 ->    148992  (450.4x)      rhof  67108864 -> 29121248  (2.30x)
         jfz  67108864 ->  58645362  (1.14x)        ex   67108864 -> 66642944  (1.01x)
  codec  nvcomp-bitcomp: 86   nvcomp-ans: 64   nvcomp-lz4: 50   (raw: 56)
  stage+compress(wait) 1.299 s   in-situ wall 30.101 s
VERIFIED: 256 of 256 blobs round-tripped bit-exact through the decompressor
```

Every blob went to the compressor as **device memory**: 256 of 256 `device=1`
in the `[np-path]` trace, zero host fallbacks, and with `--verify` off there is
no device-to-host copy on the path at all. The `--verify` digest is taken from
a separate D2H copy of the *source* and is instrumentation, not the path.

| check | how | result |
|---|---|---|
| in-process verify, 126³ | FNV-1a-64 of the decompressed bytes vs the digest of what was staged | 256 / 256 |
| cold read (`read.sh`, `CLIO_RESTART=1`, tier is the only copy) | same digest, separate process | 256 / 256 |
| `crosscheck.sh` — in-situ blobs vs the deck's own `.f32` dump, same run | digest of the file vs digest of the blob | **32 / 32 identical** |
| **literal byte comparison** | `cmp` on files, see below | **32 / 32 identical**, 0 differ |
| device residency | `[np-path]` trace | 256 / 256 `device=1`, 0 host fallbacks |

### The byte-for-byte check

The digests above are a 64-bit hash. To compare the actual bytes, run the
adapter with `CLIO_VPIC_RAW_DIR` (it writes every staged chunk to a file), then
cold-read with `--dump-decompressed` and compare the two directories:

```bash
VPIC_DUMP_FIELDS=1 VPIC_DUMP_INT=25 VPIC_DUMP_DIR=$W/dump \
CLIO_VPIC_RAW_DIR=$W/raw ./run.sh --ncell 30 --steps 50 --int 25 --verify --store $W/store

env CLIO_SERVER_CONF=$W/store/compose.yaml CLIO_WITH_RUNTIME=1 CLIO_RESTART=1 \
    CLIO_REPLAY_COMPRESSOR_POOL=512.0 \
    <build>/bin/neuropress_field_replay --readback $W/store/blobs.csv \
        --tag vpic_insitu --dump-decompressed $W/dec

for f in $W/dec/*.bin; do cmp "$f" "$W/raw/$(basename $f)" || echo "DIFFER $f"; done
```

Run at both sizes. At 126³ each variable is 8 MiB, i.e. two 4 MiB chunks, so
the per-variable comparison rejoins `chunk_0` and `chunk_1` in order — which
also checks that the chunking and the names reassemble correctly.

| | 30³ (4 MiB, 32 chunks) | **126³ (1,024 MiB, 256 chunks)** |
|---|---|---|
| decompressed vs the bytes staged to the GPU | 32 / 32 identical | **256 / 256 identical**, 1,073,741,824 B |
| decompressed vs the deck's own `.f32` files | 32 / 32 identical | **128 / 128 identical**, 1,073,741,824 B |
| md5 over both whole sets | `dc4287f264b551420fe07b7ac6ee98f4` | `39c056da640fcc64277f467b64784366` |
| codecs exercised | ans 18, raw 6, bitcomp 4, zstd 4 | bitcomp 86, ans 64, lz4 50, raw 56 |

The second row is the stronger one: it compares against the simulation's own
output, written host-side down a different code path in the same process. The
cold read of the whole 1 GiB took 8.9 s in a separate process where the
compressed tier is the only copy of the data.

## The cost model changes what gets attempted

The same in-situ workload with the latency terms zeroed
(`CLIO_NEUROPRESS_COST_W_CT=0 CLIO_NEUROPRESS_COST_W_DT=0 CLIO_NEUROPRESS_COST_W_IO=1`,
the offline sweep's `dynamic-ratio`):

| | balanced (default) | ratio-only |
|---|---|---|
| ratio | 1.278x | **1.403x** |
| stored | 801.2 MiB | **730.0 MiB** |
| compressed / raw | 200 / **56** | **256 / 0** |
| codecs | bitcomp 86, ans 64, lz4 50 | zstd 173, lz4 83 |
| stage+compress | 1.30 s | 3.25 s |
| verify, in process and cold | 256 / 256 | 256 / 256 |

Dropping the time terms buys 71 MiB at 2.5x the compression time, and the model
then attempts every chunk instead of giving up on a fifth of them. It is not
free everywhere: `div_b_err`/`div_e_err` fall from 450x to 226x and
`rhob`/`rhof` from 2.30x to 1.94x, because zstd wins those chunks from bitcomp
once the latency term stops counting. The twelve E/B/J/TCA fields move the
other way, 1.00-1.14x to 1.02-1.21x. Both numbers match the offline sweep's
`dynamic` and `dynamic-ratio` rows, which is the point of running the same
workload two ways.

The two diagnostic residuals (`div_b_err`, `div_e_err`) reach 450×; `rhob`/`rhof`
about 2.3×; the twelve E, B, J and TCA fields sit at 1.00–1.14×, which is why
the aggregate is 1.278× and why 56 near-noise chunks are stored raw. That is
the same structure the offline sweep reports, one frame interval apart.

## Notes

- **One run, not two, for the cross-check.** VPIC's current deposition
  accumulates through atomics, so two runs of the same deck are not
  bit-reproducible; comparing an in-situ run against a *separate* dump run
  fails for reasons that have nothing to do with Clio. `crosscheck.sh`
  therefore turns both paths on in one process.
- Step 0 is skipped on both paths: the field array is still identically zero
  there, and a frame of zeros would dominate any ratio it is averaged into.
- MPI: the adapter takes `rank()`/`nproc()` from the deck and gives each rank
  its own store, port and lock file (`clio_vpic_insitu_begin_mpi`), the
  topology the Nyx example established. Single rank is what is measured above.
- `store/` and the built `*.Linux` deck are generated and untracked.
