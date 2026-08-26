# LAMMPS as a library, Clio in the same process

One executable. LAMMPS is linked in as `liblammps` and driven through its C
library interface (`src/library.h`); Clio's runtime is composed from a YAML
file and hosted in that same process; the driver stands between `run`
segments, reads `Atom::x / v / f` straight out of the LAMMPS instance, and hands
them to the compressor. No HDF5 anywhere in the path, no `dump` in the deck, no
patch to LAMMPS.

Three hand-over shapes, chosen with `--order`. Two read LAMMPS' host mirror
through the C library interface. The third, `--order device`, reads the Kokkos
**device** views and hands the compressor a CUDA-IPC-registered device pointer,
so the payload is never host bytes -- still with no LAMMPS patch, because the
code that needs Kokkos lives in this driver (`lammps_device_view.cc`) rather
than in a LAMMPS `fix`. The corruption this path used to hit is fixed (see
"The device write path: a stream race, found and fixed"); **read "What is NOT
verified" before relying on it outside this configuration.**

```
 ┌──────────────────────────── one process: neuropress_lammps_lib ────────────────────────────┐
 │                                                                                            │
 │   liblammps                        driver                      Clio runtime (in-process)   │
 │   ─────────                        ──────                      ─────────────────────────   │
 │   lammps_file(in.melt_lib)  ◄──── setup                                                    │
 │   run 0                     ◄──── forces at step 0                                         │
 │     Atom::x,v,f  ──────────────► gather (ID order) ──► AsyncDynamicSchedule ──► NeuroPress │
 │   run GAP pre no post no    ◄──── advance            (host shm, float64 ctx)    picks codec │
 │     Atom::x,v,f  ──────────────► gather ───────────► AsyncDynamicSchedule ──► CTE tier     │
 │   ...                                                                                      │
 └────────────────────────────────────────────────────────────────────────────────────────────┘
```

## How LAMMPS reaches Clio: the three shapes

| | `neuropress_lammps_h5` | `neuropress_lammps_gpu_direct` | **this example** |
|---|---|---|---|
| LAMMPS binary | stock `lmp` | patched (`fix cliogpu`) | stock `liblammps`, linked in |
| Clio in the app | none -- VOL is dlopened by HDF5 | `libclio_gpu_blob.so` | compressor client, directly |
| hand-over point | `output->write` → `dump h5md` → `H5Dwrite` | `end_of_step` (fix) | between `run` segments (== after `output->write`) |
| bytes cross | GPU→host (Kokkos sync), packed, sorted, **written to .h5**, staged to Clio | device→device | `--order id/local`: gather → host shm → compressor. `--order device`: **nothing** — the chunk is gathered on the GPU straight into the buffer the compressor reads |
| native copy of the data | yes (the .h5 is authoritative) | no | no |
| where the runtime lives | in-process (`CLIO_WITH_RUNTIME=1`) | in-process | in-process |
| needs | HDF5 + VOL | a LAMMPS tree carrying the fix + adapter | a CMake build of LAMMPS with KOKKOS |

All three host Clio's runtime inside the LAMMPS process; what differs is how
the simulation's arrays get to it. The VOL route pays for a file it does not
need; the gpu-direct route keeps the bytes on the device but needs a patched
tree -- and, as of this writing, the patch's sources (`fix_cliogpu_kokkos.cpp`,
`adapter/gpu_blob/`) survive only as compiled objects, never having been
committed. This example is the one a coupling code would write: the library
interface is LAMMPS' supported embedding API, and the runtime call is the same
`CLIO_CTE_CLIENT_INIT()` the VOL makes lazily.

## Where the simulation data evolves, and when it is handed over

LAMMPS keeps the state of the system in three float64 arrays owned by `Atom`
(`src/atom.h:75`, `double **x, **v, **f`). Each is an array of row pointers
over ONE contiguous `nmax*3` block (`Memory::create`), so `x[0]` is the whole
field. `Verlet::run` (`src/verlet.cpp:229`) mutates them every timestep:

| order in a step | who | what changes |
|---|---|---|
| `initial_integrate` | `fix nve` (`fix_nve.cpp:69`) | `v += dt/2·f/m`, then `x += dt·v` |
| comm / neighbor | `Comm`, `Neighbor` | ghosts; every `neigh_modify every` steps atoms may be reordered (`atom->sort()` every 1000 by default) |
| `force_clear` + `pair->compute` | `PairLJCut` | `f` recomputed from `x` |
| `final_integrate` | `fix nve` (`fix_nve.cpp:113`) | `v += dt/2·f/m` |
| `end_of_step` | fixes that observe the finished step | **hand-over point A** (`fix cliogpu`, `fix ave/time`...) |
| `output->write` | when `ntimestep == output->next` | **hand-over point B**: dumps (`dump h5md`: `pack` → `sort` → `write_data` → `h5md_append` → `H5Dwrite`), thermo, restarts |

A frame is coherent -- x, v and f describing the same instant -- only after
`final_integrate`. Both hand-over points LAMMPS offers sit after it, and so
does this driver: it reads between `run` segments, i.e. after the last step's
`output->write`. That is the same instant a dump sees, and the cross-check
below shows it is the same bytes.

Two details that matter for whoever writes the next coupling:

* **Kokkos.** On a GPU run `Atom::x` is the host mirror of a dual view. It is
  current at exactly the hand-over points: `VerletKokkos::run` syncs to host
  before `output->write` on output steps and unconditionally at the end of
  `run` (`src/KOKKOS/verlet_kokkos.cpp:516,524`). Reading at any other time
  reads a stale image. (Upstream NeuroPress's fix calls `sync(Device, ...)`
  from `end_of_step` for the opposite direction; `fix cliogpu`'s README
  explains why that needs `need_sync_device()` first.)
* **Order.** `lammps_gather_atoms` returns the field ordered by atom ID, which
  is what `dump h5md` writes after its own `sort()`. `lammps_extract_atom`
  returns LAMMPS' internal order, which changes when atoms are sorted -- cheaper
  (no gather copy), but frames are not atom-for-atom comparable across a sort
  and are not what a dump would have written. `--order id|local` picks.

The segment boundary is exact. `run N pre no post no` continues the trajectory
without re-running `init()`/`setup()` (`src/run.cpp:168`): forces from the last
step are still valid, neighbor-list age carries over, and the timestep counter
continues. On the CPU the bytes are identical to one continuous `run STEPS` --
that is what the h5md cross-check demonstrates.

## Building

Needs a CMake build of LAMMPS with the library present (`liblammps.a` or
`.so`). The sibling examples' build is fine; nothing Clio-specific:

```bash
git clone --depth 1 --branch stable https://github.com/lammps/lammps.git ~/src/lammps
cd ~/src/lammps
cmake -S cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_MPI=OFF \
  -DPKG_KOKKOS=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON \
  -DCMAKE_CXX_STANDARD=17
cmake --build build -j
```

Then in the Clio tree:

```bash
cmake -S . -B build -DLAMMPS_SRC_DIR=$HOME/src/lammps -DLAMMPS_BUILD_DIR=$HOME/src/lammps/build
cmake --build build --target neuropress_lammps_lib
```

Those are the defaults, so with LAMMPS at `~/src/lammps/build` the target just
appears. The driver compiles against `library.h` alone (no Kokkos headers, no
nvcc). A static `liblammps.a` carries no dependency information, so the
CMakeLists reads the link line LAMMPS used for `lmp`
(`build/CMakeFiles/lmp.dir/link.txt`) and reuses it -- Kokkos, CUDA, HDF5 for
the H5MD package, OpenMP, jpeg, in that order. If LAMMPS is not found the
target is skipped with a message and nothing else is affected.

## Running

```bash
./run.sh --box 10 --steps 100 --gap 50 --verify            # 4000 atoms, CPU, 3 frames
./run.sh --box 80 --steps 300 --gap 50 --kokkos --verify   # 2M atoms, GPU, ~1 GB
./run.sh --box 10 --steps 100 --gap 50 --kokkos --order device
                                                           # GPU-resident hand-over;
                                                           # always reads back (see below)
./run.sh --box 10 --steps 100 --gap 50 --kokkos --order device --crosscheck-host
                                                           # ...and prove the GPU gather
                                                           # matches lammps_gather_atoms
./run.sh --box 10 --static nvcomp-zstd                     # pin the codec, then:
./read.sh                                                  #   cold read-back, separate process
./crosscheck_h5md.sh --box 10 --steps 100 --gap 50         # bytes == dump h5md?
```

`run.sh` writes `$STORE/compose.yaml` -- the same three pools as the siblings
(`clio_bdev` 301.0, `clio_cte_compressor` 512.0 → `clio_cte_core` 513.0) with
the same NeuroPress knobs (`--learn`, `--explore K`, `--threshold`, `--best`,
`--static LIB`, `--static-shuffle`) -- and runs the driver with
`CLIO_SERVER_CONF` pointing at it and `CLIO_WITH_RUNTIME=1`. The driver's own
options:

| flag | |
|---|---|
| `--box N --steps N --gap N` | system size (4N³ atoms), length, frame interval (frame 0 included) |
| `--chunk BYTES` | bytes per compressor call; 0 = whole field. Default 4 MiB, the siblings' size |
| `--order id\|local\|device` | `id`: atom-ID order via `lammps_gather_atoms` (default, == h5md). `local`: LAMMPS' internal order via `lammps_extract_atom`. `device`: atom-ID order gathered ON THE GPU out of the Kokkos device views -- same ordering as `id`, never host bytes. Needs `--kokkos`. |
| `--device-staging` | restore the old whole-field gather + per-chunk D2D, for measuring against the default. Off by default: the copy it restores does no work |
| `--crosscheck-host` | on `--order device`, also gather each frame with `lammps_gather_atoms` and require every chunk to match byte for byte -- same frame, same process |
| `--kokkos` | `-k on g 1 -sf kk`: LAMMPS on the GPU; the driver still reads the (synced) host mirror |
| `--verify` | read every blob back through the decompressor in the same process |
| `--readback CSV` | no simulation: read the blobs a previous run listed in its CSV (what `read.sh` does) |
| `--raw DIR` | also write each staged blob's bytes to DIR (what the cross-check compares) |

Blobs are named like h5md paths, `position/step_50/chunk_3`, under one tag.
Every chunk is submitted with `Context::data_type_ = 2` (float64) and
`error_bound_ = 0` (lossless), the same context the VOL builds for an 8-byte
`H5T_FLOAT`.

## What the runs showed

All on an A100, CUDA 12, LJ melt, float64, 4 MiB chunks unless noted.

**box 10, 100 steps, CPU** -- 4000 atoms, 9 blobs of 96,000 B:
1 compressed (force at step 0, on the lattice, `nvcomp-bitcomp` 1.28x),
8 stored raw. Round trip 9/9. Melted LJ coordinates are close to
incompressible; this example is about the path, not the ratio, same as its
siblings.

**box 80, 300 steps, GPU** -- 2,048,000 atoms, 7 frames, 984 MiB, 252 blobs:

```
stored 252 blob(s), 1032192000 B in -> 961157069 B on the tier  (ratio 1.074)
  compressed: 168   stored raw: 84
  codec  nvcomp-bitcomp : 84   nvcomp-ans : 72   nvcomp-snappy : 9   nvcomp-lz4 : 3
  time: simulate 4.1 s   stage+compress(wait) 8.1 s   total 13.6 s
VERIFIED: 252 of 252 blobs round-tripped bit-exact through the decompressor
```

1.074x is the number the HDF5 example reports for the same payload (~1.07x),
which is what it should be: same bytes, same selector. Per field, from the
CSV: positions compress on every frame (1.18x on the lattice, 1.16x melted,
`nvcomp-bitcomp`, 12/12 chunks), velocities barely (raw at step 0, 1.03x
afterwards), forces only on the lattice frame (1.39x at step 0, raw after).

**`--static nvcomp-zstd`** forces every blob through a codec (9/9 compressed,
1.25x), which is what makes the cold read-back meaningful:

```
$ ./read.sh
cold read-back of 9 blob(s) listed in .../store/blobs.csv
VERIFIED: 9 of 9 blobs round-tripped bit-exact through the decompressor
-- blobs the compressor inverted a codec for: 9
```

A different process, `CLIO_RESTART=1` replaying the metadata log, no LAMMPS,
and -- unlike the HDF5 example -- no native file to fall through to. The
compressed tier is the only copy; bytes that come back correct came from it,
and `stored_compressed=1 -> inverting codec` in the trace says a codec ran to
produce them.

**Cross-check against `dump h5md`.** `crosscheck_h5md.sh` runs the stock `lmp`
on the sibling's deck with plain HDF5 (no VOL, no Clio), digests each frame of
`/particles/all/{position,velocity,force}/value`, and compares with the digests
the driver recorded:

```
  position/step_0/chunk_0      h5md=62385809d0d70bd5 driver=62385809d0d70bd5 same
  ...
  force/step_100/chunk_0       h5md=1a06e889a6c96f27 driver=1a06e889a6c96f27 same
9 frame(s) byte-identical to dump h5md, 0 differ
```

This is the line that justifies the "same instant, same bytes" claim above,
and it says two things at once: the ID-ordered gather reproduces h5md's sorted
output, and segmented `run GAP pre no` reproduces the continuous trajectory.
CPU only -- the GPU run is not bit-reproducible, as the sibling's `--cpu` note
explains.

## `--order device`: the GPU-resident hand-over

`library.h` has no device accessor -- `lammps_extract_atom` returns the host
mirror and `lammps_gather_atoms` copies out of it -- so `--order id` and
`--order local` hand the compressor host bytes even when LAMMPS is running on
the GPU. Measured, with `--kokkos`, on the path trace: `device=0` on every
blob.

`--order device` goes through `lammps_device_view.cc` instead, the one
translation unit here compiled with LAMMPS' own `nvcc_wrapper` and LAMMPS' own
flags. It reads `LAMMPS::atomKK` (`lammps.h:44`, public) to reach
`AtomKokkos::k_x / k_v / k_f` (`atom_kokkos.h:34-36`, public DualViews),
gathers into atom-ID order **on the device** with a Kokkos kernel, stages into
a registered `kDeviceMem` backend, and submits that. Same trace: `device=1` on
every blob.

The earlier gpu-direct attempt needed a patched LAMMPS only because it put this
code in a `fix`. In the driver it needs no LAMMPS change at all.

### The ID gather is exactly `lammps_gather_atoms`

`lammps_gather_atoms` zero-fills a `natoms*3` buffer and writes
`copy[3*(tag[i]-1)+j] = array[i][j]` for each owned atom
(`library.cpp:3663-3672`). `clio_lmp_device_gather_id` does the same
permutation, with the same zero fill, in two Kokkos kernels over `k_tag` and
the field's device view.

Verified directly -- same process, same GPU trajectory, both orders taken at
the same instant, so any difference would be the gather rather than the
physics:

```
step 0    x  host_gather=62385809d0d70bd5  device_gather=62385809d0d70bd5  IDENTICAL
step 0    v  host_gather=bc91d38b3114b7de  device_gather=bc91d38b3114b7de  IDENTICAL
step 0    f  host_gather=d29a529281c550ef  device_gather=d29a529281c550ef  IDENTICAL
step 50   ... IDENTICAL x3
step 100  ... IDENTICAL x3
```

`62385809d0d70bd5` is the digest `crosscheck_h5md.sh` records for
`position/step_0` from a stock `dump h5md`, so the GPU gather reproduces h5md's
ordering.

**`crosscheck_h5md.sh` cannot be run against `--order device` end to end**, and
not because of the gather: the crosscheck's reference is a **CPU** `lmp` run,
`--order device` requires the GPU, and the GPU trajectory is not
bit-reproducible against the CPU one -- which is why that script says it is
"only meaningful for a CPU run". The control settles it: the existing,
unmodified `--order id --kokkos` scores exactly the same 1 of 9 against the CPU
reference (only `position/step_0`, the initial lattice, is CPU/GPU-identical).
`--order device` is precisely as h5md-comparable as `--order id` is on the GPU,
and the digest comparison above is the rigorous form of the check. On the CPU,
`--order id` still scores 9 of 9.

### What it refuses to do

Each of these could have been continued past by quietly doing something else,
and the something else would have produced blobs that pass every check while
not being what was asked for:

| condition | exit | why not carry on |
|---|---|---|
| `atomKK == nullptr` (no `-k on g 1 -sf kk`, or a liblammps without KOKKOS) | 3 | the atom arrays are host-resident; there is no device view. Asked of LAMMPS, not inferred from `--kokkos`, because the flag is only what this driver passes in |
| `need_sync<Device>()` non-zero for x, v, f or tag | 3 | the device image is behind the host one; reading it stores an earlier step under this step's name. A stale **tag** is worse: right coordinates, wrong atoms |
| `AllocateAndRegisterGpuBackend` returns null | 4 | staging through host memory instead would turn a GPU-resident run into a host one with nothing in the log to show it |
| the registered buffer is not device memory | 5 | the compressor would get host bytes while the run reported GPU residency |
| `nlocal != natoms` (multi-rank) | 1 | the device views hold one rank's atoms; storing them unqualified publishes a fraction of the system as the whole of it |

Measured: without `-k on g 1 -sf kk` the run exits **3** and says so. The other
four are code paths that could not be provoked here without injecting faults --
stated as unverified rather than claimed.

### The device write path: a stream race, found and fixed

Device-staged chunks whose codec ran **used to** reach the tier corrupted --
the read side rejected them with `header is not valid (magic/version
mismatch)`, and the damage was on disk: a cold read from a separate process
failed the same blobs. It needed three things at once, and removing any one
made it clean: device staging, a codec that actually ran, and LAMMPS/Kokkos
sharing the process. That last condition is why the standalone GPU examples
never showed it.

**Root cause: a CUDA stream-ordering violation, in Clio, not in this driver.**
The 24-byte `CompressionHeader` was written with `GpuApi::Memcpy` -- a
`cudaMemcpy` on the **legacy default stream**, from pageable stack memory,
which returns as soon as the source is staged while the DMA to the device is
still outstanding. The bdev worker then read those bytes back on a
`cudaStreamNonBlocking` stream, which is not ordered against the legacy one,
and copied whatever the previous occupant of that recycled allocation had left
in the first 24 bytes. The payload never raced: the codec synchronizes its own
stream before returning. Kokkos mattered only because it parks work on the
legacy stream, widening the window -- without it the DMA retired long before
the reader arrived and the race was essentially never lost.

Fixed by writing the header with `ctp::DeviceAwareMemcpy`, which synchronizes
its own private stream. `GpuApi::Synchronize()` would also have worked but was
rejected: on the legacy stream it blocks a compressor worker until the
simulation's entire queued `run` segment retires.

| configuration | before | after |
|---|---|---|
| `--order device --kokkos --static nvcomp-zstd` | 5 of 8 runs failed | **0 of 8** |
| `--order device --kokkos` (dynamic) | 3 of 10 failed | **0 of 6** |
| cold `./read.sh`, separate process, `CLIO_RESTART=1` | failed | **9 of 9 bit-exact** |
| 50 steps, 500 atoms, 12 KB chunks | -- | **9 of 9 bit-exact** |

Note the last row: 12 KB chunks against the 4 MiB everything else was measured
at, so the fix holds across a ~350x size difference.

`--order device` still **reads every blob back**, whether or not `--verify` was
passed. That was added while the corruption was unexplained and is now
redundant insurance; see "Next actions" for the case for removing it.

### The staging copy, removed

`--order device` used to gather a whole field into a scratch buffer and then
copy each chunk out of it into the registered backend the compressor reads —
**one payload-sized D2D per chunk that moved bytes without changing any**. The
gather needed a whole-field destination because it is a *scatter*: thread `i`
writes to `(tag(i)-1)*3`, which can land anywhere in the field.

Naming a destination **window** removes the need. `clio_lmp_device_gather_id_window`
takes `[dst_elem_off, +dst_elem_count)` in field-element space; each thread
still computes its global destination index and writes only when it falls
inside the window, shifted to a local one. The chunk is produced directly where
the compressor will read it, and the scratch field stops existing.

Measured with `nsys --trace=cuda`, LJ melt, 20 steps, 3 frames, `--order device`:

| | box 10, 32 KB chunks (27 chunks) | box 30, 4 MiB chunks (9 chunks) |
|---|---|---|
| D2D copies, `--device-staging` | 27 | 9 |
| D2D bytes, `--device-staging` | 864,000 | 23,328,000 |
| D2D copies, default | **0** | **0** |
| D2D bytes, default | **0** | **0** |
| H2D bytes, staged → default | 14,580,472 → 14,580,384 | 448,472,307 → 448,474,095 |
| D2H bytes, staged → default | 21,117,180 → 21,117,092 | 867,279,257 → 867,281,045 |

**Every payload-sized D2D is gone**; H2D and D2H move by ~88 B and ~1.8 KB of
control traffic, i.e. not at all. The whole-field scratch allocation is gone
with it — 2.59 MB at box 30, and it scales with the system.

It is **not** a speed fix, and the numbers say so plainly: 0.103 ms of GPU
memcpy time out of a 1.8 s run, and total wall time 1.828 s staged against
1.812 s direct, which is noise. What it removes is bytes and a buffer.

The cost is that the source is re-read once per chunk, since which atoms fall
in a window is not known without reading `tag`. At one chunk per field — the
usual case, a field is 96 KB at box 10 against a 4 MiB default chunk — it is a
strict win: one fewer full-field read and one fewer full-field write. At two
chunks it is even. Beyond that it trades device-local reads for a device-local
copy plus a whole extra field of device memory. Nothing here crosses PCIe
either way.

`--device-staging` restores the old path so the two can be measured against
each other rather than taken on faith. That is how the table above was made:
`PROFILE="nsys profile --trace=cuda -o /tmp/x" ./run.sh --order device ...`.

### Why the round-trip check could not validate this, and what did

`--verify` reads each blob back and compares it against a digest **taken from
the buffer that was submitted**. A gather that wrote the wrong bytes into that
buffer produces a blob that round-trips perfectly and is simply wrong — so the
existing check could not have caught a window bug. Neither could comparing
digests between two runs: **LJ force summation on the GPU is not
bit-reproducible**, and two runs of identical code already disagree on `force`
at step 0. (Positions and velocities at step 0 do match, which is what makes
the cause identifiable rather than mysterious.)

`--crosscheck-host` is the check that works. For each frame it also calls
`lammps_gather_atoms` — LAMMPS' own reference permutation, the one `dump h5md`
writes — and requires every device-gathered chunk to match it byte for byte,
**same frame, same process**, which sidesteps run-to-run nondeterminism
entirely.

```console
$ ./run.sh --box 10 --steps 20 --gap 10 --kokkos --order device \
           --crosscheck-host --chunk 32768
CROSSCHECK: 27 of 27 device-gathered chunk(s) match lammps_gather_atoms byte
for byte, same frame same process
```

Passing at 1, 3 and 7 chunks per field, and with `--device-staging` too. And it
has teeth — with a deliberate `+1` injected into the window offset:

```
CROSSCHECK MISMATCH position step 0 chunk 0: first differing element 2 of 4096 (field element 2)
CROSSCHECK MISMATCH position step 0 chunk 1: first differing element 0 of 4096 (field element 4096)
```

A first difference at exactly a chunk boundary names the bug on sight. The run
exits non-zero (6, or 3 when the bad window also trips the gather's own bounds
refusal), because a correctly-round-tripping blob of the wrong bytes is the one
outcome that must never exit 0.

## What is NOT verified

Be careful about reading the results above as broader than they are.

- **Refusals 2-5 have never fired.** Only refusal 1 (no `atomKK`, i.e. missing
  `-k on g 1 -sf kk`) was provoked and confirmed to exit 3. Stale device views,
  a failed backend allocation, a registered pointer that is not device memory,
  and the multi-rank `nlocal != natoms` case are all code paths nobody has
  triggered. They are written and reviewed, not tested.
- **The cross-process path is untested.** Every run here uses
  `CLIO_WITH_RUNTIME=1`, so client and runtime share a PID and
  `IpcManager::ToFullPtr` takes its same-process shortcut: it dereferences the
  raw device pointer directly and the CUDA IPC handle machinery
  (`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`) is never exercised. A
  separate-runtime deployment is unproven.
- **`--verify` is not independent of the codec.** It reads back through the
  same Clio decompressor that wrote the data, so a codec that corrupted
  symmetrically would pass -- and, more to the point, it digests the buffer
  that was *submitted*, so it cannot see a gather that filled that buffer
  wrongly. Two checks cover what it cannot:
  `crosscheck_h5md.sh` compares against a stock `dump h5md` and is independent
  of Clio entirely, but is meaningful only for a **CPU** run with `--order id`,
  because a GPU run is not bit-reproducible. `--crosscheck-host` is the GPU
  path's equivalent: it compares each device-gathered chunk against
  `lammps_gather_atoms` in the **same frame and the same process**, which is
  what makes it immune to that nondeterminism. It is independent of the codec
  because it runs before compression.
- **Single-node, single-rank only.** MPI is refused, not supported.

## Next actions

In the order worth doing them:

1. **Test the cross-process path.** Run the Clio runtime as a separate process
   rather than `CLIO_WITH_RUNTIME=1`, and confirm the blobs still arrive
   device-resident and bit-exact. This is the difference between "works in this
   configuration" and "works", and it is the only one of these that could
   invalidate the design rather than just leave it under-tested.
2. **Provoke refusals 2-5.** Fault injection: force
   `AllocateAndRegisterGpuBackend` to return null, hand `store_frame` a host
   pointer, mark a view stale before the read. A refusal that has never fired is
   a claim, not a guarantee.
3. **Drop the unconditional read-back**, once 1 and 2 hold. Restore `--verify`
   to opt-in so `--order device` stops paying for a bug that no longer exists.
4. **Fix `GpuApi::Memcpy`'s contract.** It reads as a completed copy but is
   asynchronous for pageable-H2D *and* for D2D. This example hand-compensates
   with an explicit `Synchronize()` after its D2D staging copy -- now only on
   `--device-staging`, since the default path has no D2D to compensate for and
   ends in a `Kokkos::fence` instead; two other call sites rely on the wrong
   assumption, one saved only by an incidental `cudaFree` sync. Either rename it (`MemcpyEnqueue`) or give it
   `DeviceAwareMemcpy`'s private-stream-and-sync shape. The header bug was one
   instance of this trap; the trap is still set.
5. **Fix `warn` in the paper-benchmark scripts.** The log-level parser accepts
   `warning`, not `warn` -- `std::stoi("warn")` throws and the level silently
   stays at debug. Fixed here and in `read.sh`, still wrong in seven
   `paper-benchmark/*/{run_config,read}.sh` files, where it means those runs
   have been logging at debug all along.

## Notes

- **Storing raw is the normal outcome here, not a failure.** Melted LJ float64
  coordinates are close to incompressible, so on the default settings most
  chunks come back with `compress_lib_ == 0`: a codec was selected and run, it
  did not shrink the chunk, and the ORIGINAL bytes went to the tier. The run
  still returns 0, because storing raw is the correct answer -- writing more
  bytes than the caller gave us is not. What changed is that the compressor now
  says so at WARNING with the blob's name and the sizes it achieved
  (`compressor_runtime.cc`, "Compression not beneficial for blob '...'"),
  instead of at DEBUG. The summary's `stored raw:` count is the aggregate of
  the same thing. Use `--static <lib>` when you need every chunk to genuinely
  go through a codec.

- **Log level: "warning", not "warn".** `Logger::Logger`
  (`clio_ctp/util/logging.h:143-163`) matches the full level names and then
  falls through to `std::stoi`, which throws on `"warn"` and leaves the
  COMPILE-TIME default -- kDebug in this build. `run.sh` and `read.sh` used to
  pass `warn`, so every run of this example was silently at debug: measured,
  `warn` produced 386 DEBUG lines out of 540 where `warning` produces 0 out of 50. Both now pass
  `warning`. The same typo is still present in `paper-benchmark/*/run_config.sh`
  and `paper-benchmark/*/read.sh` (7 files), which are out of this example's
  scope. (`../neuropress_lammps_h5/lmp_common.sh` passes `debug` deliberately --
  its `lmp_write.sh` greps the log for `kept=` -- and is unaffected.)

- `CLIO_WITH_RUNTIME=1` is load-bearing, for the reason the sibling README
  gives: a compressor task sent to a runtime in another process arrives with a
  host pointer that means nothing there. The driver calls
  `CLIO_CTE_CLIENT_INIT()` first thing; with that variable set, the runtime
  comes up inside `main` before LAMMPS is even opened.
- LAMMPS with Kokkos/CUDA and Clio's CUDA code coexist in the process for
  every HOST-staged order -- `--order id --kokkos` was verified like the
  others. An earlier version of this note said they coexist "without incident"
  full stop, which was only ever measured for the host orders. With
  `--order device` they did not, until the stream race documented above was
  fixed -- Kokkos parking work on the legacy default stream is precisely what
  made that race lose. They coexist; the ordering has to be explicit.
- Serial only (`BUILD_MPI=OFF`, as the siblings). Under MPI,
  `lammps_gather_atoms` still returns the global ID-ordered array on every
  rank -- rank 0 would stage it -- or each rank stages its own
  `lammps_extract_atom` slice under a rank-qualified name; `--order local`
  refuses to run when `nlocal != natoms` rather than silently storing a
  fraction of the system.
- The staging copy (`gather` → `memcpy` into shm) is the cost of reading the
  host mirror on `--order id` / `--order local`. The C library interface
  exposes no device view, which is why `--order device` reaches past it into
  LAMMPS' C++ headers -- see that section above.

- **Cross-process runtime is untested for `--order device`.** Everything here
  runs with the runtime hosted in this process (`CLIO_WITH_RUNTIME=1`). The
  gpu-direct sibling's README claims device payloads now cross a process
  boundary by CUDA IPC handle (`Context::blob_is_device_`); that claim has not
  been re-checked here and nothing in this example depends on it. One thing
  that WOULD break out of process is addressing several chunks as offsets into
  one registered backend: `IpcManager::ToFullPtr` case 4 resolves a same-PID
  allocation through `ShmPtr::off_`, but case 5 -- the cross-process path --
  resolves through the backend's own `device_ptr` and ignores `off_`, so every
  chunk would silently come back as chunk 0. This driver gives each chunk its
  own backend for that reason.
- Every in-process runtime binds a TCP port, 9413 by default here and in all
  the siblings. Two Clio-hosting processes on one machine -- this example and
  another harness, say -- collide with `Failed to start main server on
  0.0.0.0:9413 ... Address already in use`, and the driver exits before
  writing anything. `run.sh` now says so and takes `--port N` (or `PORT=`);
  `read.sh` reads the port back from the store's compose.yaml. That is what
  the one "flaky" failure during development turned out to be.
