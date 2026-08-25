# Nyx in situ: AMReX device memory handed to NeuroPress inside the simulation

One process. Nyx runs the Sedov blast on the GPU, Clio's runtime is composed
from a YAML file and hosted in that same Nyx process, and Nyx's **own in-situ
callback** (`Nyx::updateInSitu`) hands `fab.dataPtr(comp)` -- AMReX device
memory, uncopied -- to the compressor. NeuroPress picks a codec per chunk and
Clio stores the result. With `--hook insitu` Nyx writes **no plotfile and no
checkpoint at all**: the compressed tier is the only copy of the data in
existence, and a separate process reads it back bit-exact.

```
 ┌────────────────────── one process: nyx_HydroTests ──────────────────────────┐
 │                                                                             │
 │  Nyx / AMReX (CUDA)          patch (+339 lines, Nyx_output.cpp)   Clio      │
 │  ──────────────────          ───────────────────────────────────  ────      │
 │  Amr::coarseTimeStep                                                        │
 │    Nyx::advance      (GPU)                                                  │
 │    Nyx::post_timestep                                                       │
 │      updateInSitu()  ──────► Gpu::Device::synchronize()                     │
 │                              MFIter over State_Type                         │
 │        fab.dataPtr(comp) ──► clio_nyx_insitu_stage(dev_ptr, geometry)       │
 │            (device mem)          │  dlopen'd libclio_nyx_insitu.so          │
 │                                  │  cudaMemcpy3DAsync: valid box out of     │
 │                                  │  the ghost-inclusive FAB, own stream     │
 │                                  ├─ chunk -> registered kDeviceMem backend  │
 │                                  └─ AsyncDynamicSchedule ──► NeuroPress ──► │
 │  ... 200 steps ...                                            CTE tier      │
 └─────────────────────────────────────────────────────────────────────────────┘
```

This is the counterpart of `../../../../paper-benchmark/nyx`, which is the
offline route: patch Nyx to dump raw fields, sweep the files later. That
route's README says an in-situ shape is "not available" because Nyx has no
library interface to embed. It is available -- just in the other direction.
LAMMPS could be linked in and read from the outside; Nyx has to **call out**,
and it already has the callback to call out from.

## Q1 first: is `fab.dataPtr(comp)` device memory?

Yes -- plain `cudaMalloc` device memory, not managed/UVM and not pinned host.
Measured, not inferred. `NYX_PTR_PROBE=1` prints it from inside Nyx:

```
[ptr-probe] sizeof(amrex::Real)=4 The_Arena: isDeviceAccessible=1 isHostAccessible=0
                                  isManaged=0 isDevice=1 isPinned=0
[ptr-probe] MultiFab nComp=6 nGrow=(1,1,1) local_size=1 size=1
[ptr-probe] fab index=0 validbox=((0,0,0) (31,31,31) (0,0,0)) numPts(valid)=32768
                       fabbox=((-1,-1,-1) (32,32,32) (0,0,0)) numPts(fab)=39304 nComp=6
[ptr-probe]   comp=0 (density) ptr=0x79281a880000 rc=0 type=2 (DEVICE) device=0
                                                  devicePointer=0x79281a880000 hostPointer=0
[ptr-probe]   comp=1 (xmom)    ... type=2 (DEVICE) ...   (and comps 2-5 likewise)
[ptr-probe] comp stride (elements) = 39304   fab.box().numPts()=39304   validbox().numPts()=32768
```

`cudaPointerGetAttributes` returns `type=2` (`cudaMemoryTypeDevice`) with a
null `hostPointer`, and `The_Arena()` reports `isDevice=1 isManaged=0`. So the
target shape is reachable: there is a real device pointer to hand over and no
host mirror anywhere in the path.

**But it is a runtime property, not a build property.** `amrex.the_arena_is_managed=1`
on the command line makes the same build allocate managed memory instead, and
then the same pointer is `type=3`. That is why the adapter re-checks **per
blob** rather than trusting the build -- and it is one of the three refusals
that has actually been fired (below).

That last probe line is also the layout fact everything else follows from: a
FArrayBox component is contiguous over the **grown** box (32³ valid, 34³ with
`nGrow=1`), and components are consecutive copies of the grown box. The valid
region is therefore a strided 3D sub-volume, not a prefix.

## Q2: which hook, and why

`Nyx::updateInSitu()` (`Source/IO/Nyx_output.cpp:1261`), Nyx's own in-situ
callback. `Nyx::post_timestep` fires it at level 0 whenever
`(nstep+1) % insitu.int == 0` (`Nyx.cpp:1691`).

| | |
|---|---|
| it is already the in-situ hook | upstream calls SENSEI, Ascent/Conduit and Reeber from it; its body is otherwise empty in a build with none of them |
| it lives in the I/O file | `Source/IO/Nyx_output.cpp`, beside the raw-dump patch. Nothing in the physics or the timestepping is touched |
| it has a cadence knob already | `insitu.int` / `insitu.start`, parsed by Nyx (`Nyx.cpp:429`) |
| it needs no plotfile | so `amr.plot_files_output=0` can be set and Clio genuinely becomes the only I/O |

`Nyx::writePlotFile` is offered as a second hook (`NYX_CLIO_INSITU_HOOK=plotfile`)
for one reason: it fires on the same MultiFab, in the same instant, as the
native plotfile Nyx then writes -- which is what makes the cross-check below
possible. It is not the recommended production shape, because it forces a
native plotfile to exist.

Two consequences of choosing `updateInSitu` that a reader should know:

* **The cadence is offset by one step from `plot_int`.** `insitu.int=10` fires
  after steps 9, 19, 29 ...; `plot_int=10` fires after 10, 20, 30. That is
  Nyx's arithmetic (`(nstep+1) % insitu_int`), not this patch's. Blobs are
  named with `parent->levelSteps(0)` as seen in the callback, so they read
  `step_00009`, `step_00019`, ...
* **`rho_E` and `rho_e` are pre-reset.** `post_timestep` runs
  `reset_internal_energy_nostore` and `compute_new_temp` *after* the in-situ
  callback (`Nyx.cpp:1713-1722`). The state captured here is exactly what
  SENSEI and Ascent see through the same callback, but it is not identical to
  a plotfile written later in the same step.

## Q3: what a chunk is, and ghost cells

**A blob is one (component, box, chunk).** Every FArrayBox of the level-0
`State_Type` MultiFab, every component (`density`, `xmom`, `ymom`, `zmom`,
`rho_E`, `rho_e`), cut into `--chunk` byte pieces. The default is 4 MiB -- the
same as `../neuropress_field_replay`, the LAMMPS examples and
`paper-benchmark/nyx` -- so the ratios are comparable across all of them. At
128³ float32 a component is 8 MiB, so 2 chunks; 6 components × 20 frames =
**240 blobs, 960 MiB**, against the offline route's 252 blobs / 1,008 MiB.

**Ghost cells are excluded.** A FAB component is contiguous over the grown
box, so the valid region is extracted with one `cudaMemcpy3DAsync` on the
adapter's own stream -- a strided device-to-device copy, no kernel and no host
bounce. The reasons for excluding them:

* a ghost cell is a copy of a neighbour's interior cell, so storing them
  inflates the ratio with redundancy nothing asked for;
* at an output point they are not guaranteed to have been filled;
* with them included a blob corresponds to no region of the domain, so it
  cannot be compared with anything the application itself writes -- and the
  cross-check below is the strongest evidence this example has.

`--ghosts` stores the whole grown FAB instead, for anyone who wants to measure
the difference. (It is bigger: 34³ vs 32³ is +20% of payload at that size.)

## Q4: streams

AMReX creates its own streams with `cudaStreamCreate`
(`AMReX_GpuDevice.cpp:544`) and does not park kernels on the legacy default
stream, so it does not reproduce the condition that made Kokkos widen the
stream race fixed in 6b6e28a7. Both sides are explicit anyway:

* the Nyx side calls `amrex::Gpu::Device::synchronize()` before the first
  stage of a frame. Not `Gpu::streamSynchronize()`: that syncs only the
  *current* stream and `MFIter` rotates through a pool. `post_timestep` does
  not synchronise before the in-situ callback, so without this the FAB can
  still be being written when Clio reads it;
* the adapter issues every copy on its own `cudaStreamNonBlocking` stream and
  synchronises before publishing the pointer. It deliberately does **not** use
  `ctp::GpuApi::Memcpy`, which is a legacy-default-stream `cudaMemcpy` and is
  not host-synchronous for device-to-device.

Measured rather than assumed -- the failure mode is nondeterministic, so
repeats:

| configuration | runs | result |
|---|---|---|
| `--static nvcomp-zstd --verify` (every blob through a codec), 96 blobs each | **8** | 0 failures, 768/768 bit-exact |
| cold `./read.sh` on each of those 8 stores | **8** | 0 failures, 768/768 bit-exact, 768 codec inversions |
| dynamic `--verify`, 96 blobs each | **6** | 0 failures, 576/576 bit-exact, 576/576 `device=1` |
| 128³ dynamic, 240 blobs | 4 | 0 failures |
| 64³ decomposed into 8 boxes, `--verify` | 1 | 0 failures, 96/96 bit-exact, 96/96 `device=1` |

## Q5: MPI

**Single rank only, and refused otherwise.** `NProcs() > 1` aborts with a
named message: each rank owns a subset of the boxes, and blob names carry only
the box index, so under MPI several ranks would publish different data under
the same name and a reader would get a fraction of the domain presented as the
whole of it. The fix is a naming change (rank- or box-qualified), not a code
change to the path -- see "Next actions". The Nyx build used here is
`Nyx_MPI=NO`, so this refusal cannot currently be provoked.

## How the two halves are joined

Nyx is an AMReX **executable**. There is no `liblammps` equivalent to link in,
so the patch is unavoidable -- but it can be made to cost Nyx nothing:

* `libclio_nyx_insitu.so` is **dlopen'd** at first use. Nyx's build is
  untouched: no CMake change, no Clio in its link line, nothing to configure.
  Exactly the relationship HDF5 has with a VOL connector.
* The ABI (`clio_nyx_insitu.h`) is plain C over a `void*` and integers, so the
  library needs **no AMReX** and cannot go out of step with a particular AMReX
  build. Everything that touches an AMReX type lives on the Nyx side.
* Cadence, chunk size, tag, report path and verification are environment
  variables, so the patch has no options of its own to keep in sync.

The patch touches `Source/IO/Nyx_output.cpp` and nothing else: +339 lines, -0,
all in one block, plus one call in `Nyx::updateInSitu()` and one in
`Nyx::writePlotFile()`. No physics, no timestepping, no numerics.

## Building

Nyx, with **both** patches -- this one is a diff against the raw-dump patch's
result, because the plotfile hook exists to be compared against that dump:

```bash
git clone --depth 1 --recursive https://github.com/AMReX-Astro/Nyx.git ~/src/Nyx
cd ~/src/Nyx
git apply <clio>/paper-benchmark/nyx/patches/nyx-raw-field-dump.patch
git apply <clio>/context-transfer-engine/compressor/example/neuropress_nyx_insitu/patches/nyx-clio-insitu.patch
cmake -S . -B build-clio -DCMAKE_BUILD_TYPE=Release \
      -DNyx_MPI=NO -DNyx_OMP=NO -DNyx_HYDRO=YES -DNyx_HEATCOOL=NO \
      -DNyx_GPU_BACKEND=CUDA -DAMReX_CUDA_ARCH=Ampere \
      -DAMReX_PRECISION=SINGLE -DAMReX_PARTICLES_PRECISION=SINGLE
cmake --build build-clio --target nyx_HydroTests -j
```

Clio:

```bash
cmake --build <clio>/build --target clio_nyx_insitu
cmake --build <clio>/build --target neuropress_field_replay   # the cold reader
```

`git apply --check` was run on a clean copy: the patch applies to a
raw-dump-patched tree and reproduces exactly the tree these numbers were
measured on. It does **not** apply to a vanilla Nyx (its anchors are the
raw-dump patch's own lines) -- see "Next actions".

## Running

```bash
./run.sh                                             # 128^3, 200 steps, every 10
./run.sh --ncell 64 --steps 20 --int 5 --verify      # small, read back in-process
./run.sh --static nvcomp-zstd --static-shuffle 4     # pin the codec, then:
./read.sh                                            #   cold read, separate process
./crosscheck_plotfile.sh                             # bytes == AMReX's own plotfile?
```

`run.sh` writes `$STORE/compose.yaml` -- the same three pools as every sibling
(`clio_bdev` 301.0, `clio_cte_compressor` 512.0 → `clio_cte_core` 513.0), the
same NeuroPress knobs (`--learn`, `--explore K`, `--threshold`, `--best`,
`--static LIB`, `--static-shuffle`) -- and runs Nyx with `CLIO_SERVER_CONF`
pointing at it and `CLIO_WITH_RUNTIME=1`.

| flag | |
|---|---|
| `--ncell N --steps N --int N` | grid, length, in-situ interval |
| `--chunk BYTES` | bytes per compressor call; 0 = whole component. Default 4 MiB |
| `--hook insitu\|plotfile\|both` | which callback, see Q2 |
| `--ghosts` | store the grown FAB rather than the valid box |
| `--verify` | read every blob back through the decompressor at the end of the run |
| `--bin PATH` | a different Nyx binary (e.g. the float64 build) |

Environment the adapter reads directly: `CLIO_NYX_TAG`, `CLIO_NYX_CHUNK`,
`CLIO_NYX_POOL`, `CLIO_NYX_REPORT`, `CLIO_NYX_RAW_DIR`, `CLIO_NYX_VERIFY`.
`RAW_DIR=` and `NYX_DUMP_DIR=` in front of `run.sh` also write the staged bytes
and the raw-dump `.f32` files for external comparison.

## What the runs showed

A100, CUDA 12.6, Sedov blast, 128³ float32, 200 steps, `insitu.int=10` → 20
frames, 6 components, 4 MiB chunks: **240 blobs, 960 MiB**. Lossless
throughout (`error_bound_ = 0`). Every blob `device=1` in the `[np-path]`
trace.

```
[clio-nyx-insitu] stored 240 blob(s) from 20 frame(s), 1006632960 B in -> 43136691 B  (ratio 23.336)
  compressed: 240   stored raw: 0   failed: 0
     density  167772160 -> 5913036  (28.373x)      xmom  ... (18.183x)
       rho_E  167772160 -> 4214698  (39.806x)      ymom  ... (19.527x)
       rho_e  167772160 -> 9581604  (17.510x)      zmom  ... (29.913x)
  codec nvcomp-bitcomp:160  nvcomp-zstd:30  nvcomp-ans:22  nvcomp-cascaded:20  nvcomp-snappy:8
  stage+compress(wait) 2.422 s   in-situ wall 5.070 s
VERIFIED: 240 of 240 blobs round-tripped bit-exact through the decompressor
```

| config | ratio | on the tier |
|---|---|---|
| `--static nvcomp-zstd --static-shuffle 4` | **157.7×** | 6.08 MiB |
| `--static nvcomp-zstd --static-shuffle 8` | 136.6× | 7.02 MiB |
| `--static nvcomp-zstd` (no shuffle) | 128.3× | 9.36 MiB |
| `dynamic` (NeuroPress inference, default cost) | 23.3× | 41.1 MiB |

The shuffle ordering reproduces the offline benchmark's headline on the
in-situ path: on float32 data a 4-byte stride beats an 8-byte one beats none,
and the default cost model leaves most of the available ratio on the table
(23× where 158× is reachable) because it charges compression time and this
data's value is in the slow entropy coders. Same finding, live.

**Cost.** 128³, 200 steps, native output suppressed, three runs each:

| | runs | wall |
|---|---|---|
| Nyx alone, hook loaded but `NYX_CLIO_INSITU` unset | 3 | 4.01 / 4.83 / 4.24 s |
| `./run.sh` (Nyx + Clio runtime up and down + 960 MiB compressed) | 3 | 7.78 / 7.34 / 7.26 s |
| of which the adapter's own `stage+compress(wait)` | 3 | 2.42 / 2.12 / 2.01 s |

So in-situ compression roughly doubles this run. That is a property of the
configuration, not a verdict: dumping 48 MiB every 10 steps of a 4-second
simulation is a deliberately punishing output rate, and the alternative being
compared against is not "free", it is writing 960 MiB to disk.

**Bit-reproducible.** Six dynamic runs of the same configuration produced
byte-identical blob digests (`md5sum` of the name/size/digest columns matched
across all six) and the identical 24.277× ratio. Unlike the Kokkos LAMMPS
trajectory, a single-rank fixed-grid Nyx run repeats exactly, so policies can
be compared on identical bytes without the two-phase dump the offline route
needs for that.

**float64 works too.** The same patch on `build-clio-f64`
(`-DAMReX_PRECISION` unset) submits with `data_type_ = 2`: 64³, 2 frames, 24
blobs of 2 MiB (= 6 × 64³ × 8 B per frame), all `device=1`, 24/24 bit-exact.
The width comes from `sizeof(amrex::Real)` at the call site, so nothing has to
be told which build it is.

## Verification

Four independent things, with sample sizes.

**1. Device residency, per blob.** The `[np-path]` trace prints
`device=<0|1>` for every `DynamicSchedule` from the compressor's own side,
after `IpcManager::ToFullPtr` has resolved the pointer. Across the runs
reported here: **240/240** on the 128³ run, **576/576** across the six 64³
dynamic runs, **24/24** on the float64 run. No blob was ever `device=0`.

**2. Bit-exact round trip, same process.** `--verify` reads every blob back
through the decompressor into a fresh buffer and compares FNV-1a-64 with the
digest of the bytes staged. 240/240, 768/768 (8 static runs), 576/576 (6
dynamic runs), 24/24 (float64), 12/12 (several smaller runs). No failures.

**3. Cold read, separate process.** `./read.sh` runs
`bin/neuropress_field_replay --readback`, which has no Nyx, no AMReX and no
simulation dependency of any kind; it shares only the store directory and the
CSV of names/sizes/digests. `CLIO_RESTART=1` replays the metadata log. With
`--hook insitu` there is no plotfile, no checkpoint and no `.f32` dump to fall
through to -- the compressed tier is the only copy of the data. Measured:
**240/240** on the 128³ store and **768/768** across the eight static stores,
with `stored_compressed=1 -> inverting codec` on every one of them, so a codec
really ran to produce those bytes.

**4. Independent cross-check against AMReX's own output.** This is the one
that does not go through Clio on the reference side.
`./crosscheck_plotfile.sh` runs with `--hook plotfile`, so the in-situ
hand-over and the native plotfile come from the same MultiFab at the same
instant; then it parses the plotfile's `Level_0/Cell_D_00000` directly (AMReX
VisMF: an ASCII FAB header, then component-major raw values over the box) and
digests each component's valid box.

```
-- plt00000: real=4B npts=32768 ncomp=13
   density/step_00000/fab0000/chunk_0   plotfile=b05fcf6c86b62325 blob=b05fcf6c86b62325 same
   ... 5 more ...
-- plt00010:
   density/step_00010/fab0000/chunk_0   plotfile=cf8de130671d2ad5 blob=cf8de130671d2ad5 same
   ... 5 more ...

12 blob(s) byte-identical to the native plotfile, 0 differ
```

12 of 12. Repeated with `--ncell 64 --max-grid 32`, so the grid decomposes
into **8 FArrayBoxes** and the loop over boxes actually iterates:

```
-- plt00000: 8 FAB(s), real=4B npts=32768 ncomp=13
-- plt00010: 8 FAB(s), real=4B npts=32768 ncomp=13

96 blob(s) byte-identical to the native plotfile, 0 differ
```

96 of 96. That settles four things at once: the valid-box extraction picks
exactly the region a plotfile stores, the per-box geometry is right for boxes
that do not start at the domain origin, the device-to-device path delivers the
right bytes, and the box/component naming lines up with AMReX's own MFIter
order. `--verify` and `read.sh` both read back through the same decompressor
that wrote the data, so a symmetric corruption would pass them; this check has
no Clio in it at all on the reference side.

## What the raw dump actually dumps

The cross-check run also compares against the `.f32` files
`paper-benchmark/nyx`'s raw-dump patch writes in the same run. **9 of 12
differ.** This is a finding about that patch, not about this one.

`nyx_dump_raw_fields` writes `mfi.validbox().numPts()` elements starting at
`fab.dataPtr(comp)`. But `dataPtr(comp)` is the start of that component's
**grown** box, so at 32³ it writes the first 32,768 of 39,304 ghost-inclusive
cells -- not the valid box, and not any contiguous region of the domain.
Demonstrated directly: with `--ghosts` (which stores the whole grown
component) the first 131,072 bytes of the blob are byte-identical to the raw
dump's file, for `density`, `xmom` and `rho_E`.

The 3 that do match are the step-0 momenta, which are uniformly zero, so the
mis-slice is invisible.

It matters for the offline benchmark's numbers. Replaying both byte sets
through the **same** `neuropress_field_replay` invocation, same policy, same
128³ simulation, same 288 MiB payload:

| bytes replayed | dynamic ratio |
|---|---|
| the raw dump's slice | 14.8× |
| the actual valid box | **32.4×** |

So the gap between this example's 23.3× and the offline benchmark's reported
11.9× for `dynamic` is dominated by *what the bytes are*, not by the in-situ
path. (The in-situ run and a replay of the in-situ bytes agree exactly --
32.351× both ways -- which is a nice consistency check on its own.)

Fixing it is a one-line change on the offline side (walk the valid box, or use
a ghost-free copy), but it changes every number in
`paper-benchmark/nyx/README.md`, so it is left for the human to decide. It is
listed under "Next actions".

## What it refuses to do

Each of these could have been continued past by quietly doing something else,
and the something else would produce blobs that pass every check above while
not being what was asked for.

| condition | exit | why not carry on | fired? |
|---|---|---|---|
| `NYX_CLIO_INSITU=1` and `dlopen` or `dlsym` fails | 1 (Abort) | the run asked for its output to go to Clio; a run that quietly did not is a run whose output does not exist | **yes** |
| `NYX_CLIO_INSITU=1` on a non-GPU build | Abort | there is no device pointer to hand over; staging host bytes and calling it GPU-direct is the exact claim this example exists to make honestly | no (needs a CPU build) |
| `NProcs() > 1` | Abort | blob names carry only the box index; ranks would collide and publish a fraction of the domain as the whole | no (`Nyx_MPI=NO`) |
| `finestLevel() > 0` | Abort | the hook stores level 0; finer levels would be silently dropped and a coarse image published as the state | **yes** (`amr.max_level=1`) |
| component stride != `box().numPts()` | Abort | the geometry handed to the extraction would be wrong in a way that compresses, stores and decompresses perfectly | no |
| `cudaPointerGetAttributes` says the pointer is not `cudaMemoryTypeDevice` | 5 | the compressor would report GPU residency for memory that is not GPU-resident | **yes** (`amrex.the_arena_is_managed=1`) |
| `cudaPointerGetAttributes` fails outright | 5 | an unconfirmed pointer is not a device pointer | no |
| `AllocateAndRegisterGpuBackend` returns null | 4 | staging through host memory instead would silently turn a GPU-resident run into a host one | no |
| the Clio-registered buffer is not device memory | 5 | same, one layer further in | no |
| a staging slot is smaller than the chunk (layout changed between frames) | 4 | the slot pool cannot express it; a regrid would do this | no |
| element width changes mid-run | 3 | half the blobs would carry the wrong `data_type_` | no |
| the region does not fit inside the FAB | 3 | an out-of-bounds strided read | no |

Three of the eleven have actually been fired. The managed-memory one is worth
singling out: `amrex.the_arena_is_managed=1` is a supported AMReX runtime
option, so this is not a hypothetical, and it produced exactly the intended
outcome:

```
[clio-nyx-insitu] REFUSING: density/step_00009/fab0000: memory is managed/UVM
  (cudaPointerGetAttributes type=3), not device memory. The compressor would
  report GPU residency for something that is not GPU-resident.
$ echo $?
5
```

Refusals inside the adapter print and `_exit()` with a distinct code rather
than returning, because the process is hosting a Clio runtime with live worker
threads and running static destructors underneath them turns a clear refusal
into a hang. Refusals on the Nyx side use `amrex::Abort`, which is what the
rest of Nyx does.

## What is NOT verified

- **Eight of the eleven refusals have never fired.** dlopen failure, AMR
  levels and managed memory were provoked and confirmed. Multi-rank, a CPU
  build, a null backend allocation, a non-device registered buffer, a changed
  component stride, a changed chunk layout, a changed element width and an
  out-of-range region are written and reviewed, not tested. Several need
  fault injection or a second build to reach.
- **The cross-process runtime is untested.** Every run here uses
  `CLIO_WITH_RUNTIME=1`, so client and runtime share a PID and
  `IpcManager::ToFullPtr` takes its same-process shortcut: it dereferences the
  raw device pointer directly and the CUDA IPC machinery
  (`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`) is never exercised. A
  separate-runtime deployment is unproven. This is the same gap the LAMMPS
  device example reports, and it is the one thing here that could invalidate
  the design rather than just leave it under-tested.
- **Multi-box is verified but only at one decomposition.** `64³` with
  `max_grid_size=32` gives 8 FArrayBoxes; that was run (96 blobs, 96/96
  bit-exact, 96/96 `device=1`, 96/96 identical to the plotfile). A
  non-cubic or uneven decomposition, and more boxes than the slot pool sees in
  its first frame, have not been tried.
- **AMR is refused, not supported.** Everything measured is `max_level=0`.
- **Particles are not touched.** Nyx is a cosmology code; this stores the
  hydro `State_Type` MultiFab only. Dark-matter particles, `DiagEOS_Type`,
  gravity and any derived field are outside the hook.
- **Only the Sedov blast.** A real cosmological run (`Nyx_HEATCOOL=ON`,
  particles, gravity, AMR) has not been run at all, and its ratios would be
  much lower -- the offline README's note about Sedov being chosen for spread
  rather than realism applies here unchanged.
- **The integrity check is not independent of the codec** for `--verify` and
  `read.sh`. `crosscheck_plotfile.sh` is the independent one, and it has been
  run at 32³ (12 blobs, 1 box) and 64³/`max_grid_size=32` (96 blobs, 8
  boxes) -- not at full size, and not against a `--hook insitu` frame (the
  plotfile hook is what makes a same-instant reference exist at all).
- **The timing numbers are three runs each on an otherwise idle A100,** with
  one other Clio runtime alive on the machine for part of that window. They
  are indicative, not a benchmark.

## Next actions

In the order worth doing them:

1. **Decide what to do about the raw dump's slice.** The offline benchmark's
   files are not the fields they claim to be, and the correction moves
   `dynamic` from 14.8× to 32.4× on the sample measured here. Either fix
   `nyx_dump_raw_fields` to walk the valid box and re-run
   `paper-benchmark/nyx`, or document what the files actually contain. Doing
   neither leaves two Nyx results in the tree that disagree for an unstated
   reason.
2. **Test the cross-process runtime.** Host the Clio runtime in a separate
   process and confirm the blobs still arrive device-resident and bit-exact.
   Note the trap the LAMMPS example documents: `ToFullPtr` case 5 resolves a
   registered backend through its own `device_ptr` and ignores `ShmPtr::off_`,
   so addressing several chunks as offsets into one backend would silently
   return chunk 0 every time -- which is why this adapter gives every chunk its
   own backend.
3. **Qualify blob names by rank and lift the MPI refusal.** `rank%04d` (or the
   global box id, which is already unique across ranks) in the name is most of
   the work; the rest is deciding whether a reader wants per-box blobs or a
   reassembled field.
4. **Make the patch apply to a vanilla Nyx.** It currently anchors on the
   raw-dump patch's lines. Splitting the plotfile hook into its own optional
   hunk would let the in-situ patch stand alone, at the cost of the
   cross-check needing both.
5. **Provoke the remaining refusals.** Fault injection: force
   `AllocateAndRegisterGpuBackend` to return null, hand `stage` a host
   pointer, change `max_grid_size` between frames. A refusal that has never
   fired is a claim, not a guarantee.
6. **Run a realistic cosmology configuration** before any of these ratios are
   quoted as what Nyx compresses to.

## Notes

- **The patch is inert when off.** With `NYX_CLIO_INSITU` unset, every
  function it adds returns before doing anything and nothing is dlopen'd.
  Verified: the same binary run with only `NYX_DUMP_FIELDS=1` produces the
  raw-dump route's 12 `.f32` files and exits 0, exactly as before.
- **`CLIO_WITH_RUNTIME=1` is load-bearing**, for the reason the sibling
  READMEs give: a compressor task sent to a runtime in another process arrives
  with a device pointer that means nothing there. The adapter calls
  `CLIO_CTE_CLIENT_INIT()` on first use, which brings the runtime up inside
  the Nyx process.
- **Suppressing Nyx's own output takes two flags, not one.** `amr.plot_int=-1`
  is not enough: `nyx_main.cpp:165-171` writes a final checkpoint *and* a
  final plotfile after the loop regardless -- 360 MB at 128³, which was there
  in the first version of these runs. `run.sh --hook insitu` now passes
  `amr.plot_files_output=0 amr.checkpoint_files_output=0`, and the work
  directory afterwards is empty. That is what makes "the tier is the only
  copy" a fact rather than a slogan.
- **Every blob is submitted with `error_bound_ = 0`** (lossless) and
  `data_type_` 1 or 2 from `sizeof(amrex::Real)` -- the same context the HDF5
  VOL builds for a 4- or 8-byte `H5T_FLOAT`.
- **The digest costs a device-to-host copy per chunk.** It is instrumentation,
  not part of the path: nothing on the compressor side reads those bytes. It
  is taken from the extraction scratch (already fenced) rather than from the
  registered staging buffer, so it cannot be read out of a copy that has not
  landed. It is currently unconditional because the CSV it feeds is what the
  cold read replays.
- **Log level: "warning", not "warn".** `Logger::Logger`
  (`clio_ctp/util/logging.h:143-163`) matches the full level names and then
  falls through to `std::stoi`, which throws on `"warn"` and leaves the
  compile-time default (kDebug) in place. `run.sh` and `read.sh` pass
  `warning`. The same typo is still present in seven
  `paper-benchmark/*/{run_config,read}.sh` files.
- **Every Clio runtime binds a TCP port** and one process per port is the
  limit. `run.sh` picks a free one unless `--port` or `PORT=` is given; a
  collision kills the client before it writes anything and produces an empty
  log rather than an error.
- `store/` is generated output.
