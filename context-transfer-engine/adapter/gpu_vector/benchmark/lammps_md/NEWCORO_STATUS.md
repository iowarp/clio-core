# lammps_md on clio-coroc — where it stands

Builds, links and runs. Faults inside the first kernel.

```
compute-sanitizer memcheck, --lattice 8 --steps 2

Invalid __global__ read of size 4 bytes
  at BuildListKernel(...)+0x7f0c0
  by thread (0,0,0) in block (0,0,0)
  Access to 0x8 is out of bounds
  Host Frame: clio::gv_bench::md::LaunchBuildList(...)
```

## What the address tells us

`0x8` is a null pointer plus a field offset, not a stray index — an
out-of-range array subscript would land near a real allocation, and the
sanitizer says the nearest one is 8.6 GB away. So some pointer is null
when it is read.

It is a **`__global__`** read, which rules out the one suspect that looked
obvious: `CLIO_SHARED_PERSIST` binds `s_tbl` into `YieldTls().persist_`,
and that is SHARED memory. The re-derived reference is not the culprit.

It fires on **block 0, thread 0, on the first launch**, so it is not a
replay bug either — no park has happened yet, nothing has been restored
from a frame. Whatever is null was null on the straight-line path.

## Located exactly

Rebuilt with -lineinfo, the sanitizer names it:

```
BuildListCoro(...)+0x7e980  in clio_lammps_md_paged_newcoro.cc:783
BuildListKernel(...)+0x68d0 in clio_lammps_md_paged_newcoro.cc:2816
```

Line 783 is

```cpp
CO_AWAIT(nl.CoBeginFlush(0, row * rowlist, rowlist));
```

so the null is reached inside `CoBeginFlush`, which touches `Tasks()` and
then `FlushBusy()`. An access at 0x8 is a small field offset, consistent
with `Tasks()` returning null for `nl` -- but `BuildListKernel` calls
`nl.Init(yv.Block())` exactly as the original launcher does, and the
argument list matches the transpiled signature parameter for parameter.

## Ruled out, second round

- **The awaiter shape.** These verbs used to name their awaiter
  (`FlushWait w = FlushWait{this}; CO_AWAIT(w.Take())`), which makes it an
  ordinary hoist: value-initialised above the switch, so `v` is null when
  the park guard calls Ready() on the way in. That looked like the answer.
  It is now a temporary -- `CO_AWAIT(FlushWait{this}.Take())`, the shape
  the tool is designed around, which gets its own awaiter slot assigned
  immediately before the case label -- and the fault is byte-identical at
  the same line. Kept anyway, because it is the correct shape and the
  other five still pass with it.

## Ruled out, third round

- **A stale frame mistaken for a replay.** `Frame::Replaying()` returned
  `!fresh_`, and `fresh_` is false whenever the depth already had a frame
  -- which is not the same as "this call parked here". A chain that
  resumed, ran on, and then made a DIFFERENT call at the same depth would
  re-attach to the old frame and Pop a save list belonging to its
  predecessor, overwriting its own freshly-assigned awaiter. That would
  read as a null receiver at the park guard, which is exactly the
  symptom. It is now `!fresh_ && resume_point_ != 0`, since Done() zeroes
  the resume point on every normal return. The fault is unchanged.

  The guard is kept: it is strictly more correct, and the other five
  still pass with it, including gmx's out-of-core path, which parks
  heavily (495 evictions), and grayscott's bit-identical checksum.

## Two readings of `0x8`, and how to tell them apart

Both fit a 4-byte read at a small offset through a null pointer:

1. `FlushWait::Ready()` calling `v->FlushBusy()`, which reads
   `Tasks()->flush_busy` -- i.e. the awaiter's DeviceVector is null.
2. `Frame::Frame()` doing `lane_->cur_depth_++`. `cur_depth_` is at
   offset 8 of `YieldLaneHeader` (after `sp_` and `live_depth_`), so a
   null lane gives precisely a 4-byte access to 0x8.

Reading (2) would mean `YieldLane()` is null, which should have faulted
at BuildListCoro's own Frame long before line 783 -- unless the awaits at
598-670 are all in branches not taken at this lattice size, which would
make 783 the first Frame constructed in the kernel. That is checkable and
is the next thing to check: print or trap on `YieldLane() == nullptr` at
the top of BuildListCoro, or run with --lattice small enough to force the
earlier branches and see whether the fault moves.

## The remaining suspect



A hoisted declaration is emitted value-initialised above the switch and
its initializer becomes an assignment at the original site. A pointer
whose declaration-with-initializer did not survive that split would read
as null exactly like this. `BuildListCoro` is the largest function in the
benchmark and hoists on the order of seventy values, so there is a lot of
surface for one of them to have lost its initializer.

Next step is mechanical rather than clever: dump the generated
`BuildListCoro` prologue, list every hoisted pointer, and check each has a
corresponding assignment on the path to its first use. The generated file
carries `#line` directives back to the source, so each one maps to the
declaration it came from.

## Ruled out

- **Lane overflow.** md sizes lanes at 2560 bytes in `md_common.h` and
  coroc's save lists here are large, but the fault is byte-identical at
  16384. Not space.
- **The re-derived reference**, as above: wrong memory space.

## What already works

Everything up to the kernels: nvcc builds a 3.9 MB binary, it links, the
runtime starts, all six paged vectors allocate (110 MB of device storage
for 19.2 MB of data), and the neighbour-list geometry is computed. The
transpiler reports 23 suspending functions across the benchmark and its
`DeviceVector` verbs with no diagnostics.
