# LAMMPS as a library, Clio in the same process

One executable. LAMMPS is linked in as `liblammps` and driven through its C
library interface (`src/library.h`); Clio's runtime is composed from a YAML
file and hosted in that same process; the driver stands between `run`
segments, reads `Atom::x / v / f` straight out of the LAMMPS instance, and hands
them to the compressor. No HDF5 anywhere in the path, no `dump` in the deck, no
patch to LAMMPS.

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
| bytes cross | GPU→host (Kokkos sync), packed, sorted, **written to .h5**, staged to Clio | device→device | gather (ID order) → host shm → compressor |
| native copy of the data | yes (the .h5 is authoritative) | no | no |
| where the runtime lives | in-process (`CLIO_WITH_RUNTIME=1`) | in-process | in-process |
| needs | HDF5 + VOL | a LAMMPS tree carrying the fix + adapter | a CMake build of LAMMPS |

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
| `--order id\|local` | gather in atom-ID order (default, == h5md) or LAMMPS' internal order |
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

## Notes

- `CLIO_WITH_RUNTIME=1` is load-bearing, for the reason the sibling README
  gives: a compressor task sent to a runtime in another process arrives with a
  host pointer that means nothing there. The driver calls
  `CLIO_CTE_CLIENT_INIT()` first thing; with that variable set, the runtime
  comes up inside `main` before LAMMPS is even opened.
- LAMMPS with Kokkos/CUDA and Clio's CUDA code coexist in the process without
  incident; the Kokkos run above was verified like the others.
- Serial only (`BUILD_MPI=OFF`, as the siblings). Under MPI,
  `lammps_gather_atoms` still returns the global ID-ordered array on every
  rank -- rank 0 would stage it -- or each rank stages its own
  `lammps_extract_atom` slice under a rank-qualified name; `--order local`
  refuses to run when `nlocal != natoms` rather than silently storing a
  fraction of the system.
- The staging copy (`gather` → `memcpy` into shm) is the cost of reading the
  host mirror. Keeping the bytes on the GPU is the gpu-direct example's
  territory and needs device views, which the C library interface does not
  expose.
- Every in-process runtime binds a TCP port, 9413 by default here and in all
  the siblings. Two Clio-hosting processes on one machine -- this example and
  another harness, say -- collide with `Failed to start main server on
  0.0.0.0:9413 ... Address already in use`, and the driver exits before
  writing anything. `run.sh` now says so and takes `--port N` (or `PORT=`);
  `read.sh` reads the port back from the store's compose.yaml. That is what
  the one "flaky" failure during development turned out to be.
