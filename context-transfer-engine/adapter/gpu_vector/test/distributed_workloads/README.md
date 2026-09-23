# Distributed paged-workload harness

Runs a paged workload benchmark across a multi-container Clio cluster and
checks that it still computes the right answer.

```
BUILD_DIR=build-gv ./run_workloads_distributed.sh {kmeans|weights|gmx|grayscott|lbann|lammps_md|gnn|all}

GVW_NODES=4                     # 4 containers instead of 2
GVW_VARIANT=sycl                # the SYCL editions (needs their own build tree)
GVW_VARIANT=newcoro             # the clio-coroc editions (see below)
GVW_VARIANT=newcoro_sycl        # both at once
```

From ctest:

```
ctest -L gv_distributed          # paged CUDA, 2 nodes
ctest -L gv_dist4                # 4 nodes, and the baselines at 4 ranks
ctest -L gv_dist_sycl            # paged SYCL, 2 nodes
ctest -L gv_dist_newcoro         # the clio-coroc lowering, CUDA, 2 nodes
ctest -L gv_dist_ooc_newcoro     # the same under eviction
ctest -L gv_dist4_newcoro        # the same at 4 nodes
ctest -L gv_dist_newcoro_sycl    # the clio-coroc lowering on DPC++
ctest -L gv_dist_ooc_newcoro_sycl
ctest -L distributed             # all of it
```

The docker gates are opt-in:
`-DCLIO_CORE_ENABLE_DOCKER_TESTS=ON`; the SYCL ones additionally need
`-DCLIO_GV_SYCL_BUILD_DIR=build-syclreg` because DPC++ and nvcc cannot share a
build tree, and the newcoro ones `-DCLIO_GV_COROC=ON` (or a path) with
`tools/coroc/build.sh` having been run.

## The four editions, and why the newcoro ones are gated separately

| GVW_VARIANT | binary | lowered by | compiled by |
|---|---|---|---|
| *(unset)* | `clio_<wl>_paged_bench` | the compiler's `co_await` | nvcc / clang |
| `sycl` | `clio_<wl>_paged_bench_sycl` | the compiler's `co_await` | DPC++ |
| `newcoro` | `clio_<wl>_paged_newcoro` | `tools/coroc` | nvcc |
| `newcoro_sycl` | `clio_<wl>_paged_newcoro_sycl` | `tools/coroc` | DPC++ |

Same science, same decks, same gates -- only the lowering and the compiler
move, so a divergence between two editions is a real difference rather than a
differently-configured run. Every gate key the harness greps for
(`centroid_checksum=`, `checksum_total=`, `v_checksum=`, `E0=`, `evicts=`,
`ALL GATES PASS`) and the `GS_NO_HALO` negative control exist unchanged in the
ported sources, so the comparison logic above needed no changes at all.

The newcoro editions earn their own gates because a suspend point is exactly
where a page fault, an eviction and a cross-node fetch happen. A lowering that
drops a live value across a resume, or replays a frame that already ran,
produces a plausible wrong number under precisely the conditions this harness
creates -- and nothing else in the tree runs them across nodes.

**And they are registered at 4 nodes too, for all six.** The tempting
argument against is that node count varies the DECOMPOSITION (extra halo
seams, smaller bands, the page-ownership constraint) while the newcoro axis
varies the LOWERING, so crossing them buys nothing. The 4-node CUDA set is
the evidence that this is wrong: every defect it ever found -- grayscott's
second interior seam, lbann's sub-page band (which HUNG rather than failed),
lammps_md's z-plane geometry -- appears only at 4. Each of those is a suspend
point 2 nodes never reaches, and a suspend point is exactly where the two
lowerings can differ.

Resident only at 4 nodes: the 4-node decks already quarter the per-node
working set, and stacking `GVW_OOC` on that is a different pressure question
that the 2-node OOC set, whose decks are calibrated for it, answers better.

## The newcoro binaries are built by the harness, not by cmake

They are not CMake targets and cannot easily be: the transpile is a
source-to-source pass that runs *before* the compiler, against flags taken
from an already-configured build. So `run_one` calls
`benchmark/build_newcoro.sh` (or `run_newcoro_sycl.sh`) with `NEWCORO_RUN=0`
and lets it write the binary into the same `bin/` the containers mount.

It rebuilds on **every** invocation, not only when the binary is missing. The
pipeline has two inputs a timestamp cannot see past -- the benchmark source
and the transpiler itself -- and a gate that silently ran last week's lowering
is worse than no gate. One TU, well under a minute.

```
GVW_NEWCORO_BUILD=0    use whatever is already in bin/
GVW_CUDA_BUILD_DIR     the CUDA tree supplying the transpile's -I/-D set.
                       Only needed for newcoro_sycl, where BUILD_DIR is the
                       DPC++ tree and cannot answer that question.
```

`reput_stale` is skipped under a newcoro variant rather than failed: it is a
self-checking CTE probe, not a paged benchmark, and has no ported edition.

### The register ceiling nvcc has to be told about

Every paged kernel carries `__launch_bounds__(256, 4)`, which on a
65536-register SM asks ptxas for 64 registers. In the CMake build that budget
is met by `cmake/ClioCoroRegCap.cmake`, an LLVM pass -- a clang plugin, with no
way to run it from nvcc. Without it `-rdc=true` alone breaks the build:
separate compilation leaves `DeviceVector::CoFetch` a real call, allocated on
its own at 190 registers, and ptxas refuses the mismatch outright:

```
ptxas error : Entry function '...StepKernel...' with max regcount of 64
              calls function '...CoFetch...' with regcount of 190
```

Measured: grayscott fails this way on sm_89 while the other five link, the
difference being which calls survive inlining rather than anything about the
kernels. `build_newcoro.sh` therefore passes `-maxrregcount`, derived from the
same three inputs as the launch bounds and the cmake module so the three
cannot drift apart.

## What this covers that the md harnesses do not

The other harnesses here are md-specific: four of them run
`clio_lammps_md_paged_bench`, and `workloads/` runs the external
LAMMPS/GROMACS/LBANN forks. Every other workload had no distributed harness at
all, so their `--nodes` support could be written and never exercised.

Own port (9425) and subnet (172.26.0.0/24), so it can be up alongside the md
harnesses.

## Why a distributed run of these was impossible before

Every paged bench used to write its own config and `Setenv`
`CLIO_SERVER_CONF` with `overwrite=1`, so a harness-supplied cluster config was
clobbered a line later and each node stood up its own single-host runtime on
the same port. Two of those collide with `Address already in use`, and the
survivor waits forever for a peer that never arrives. All six now leave an
already-set `CLIO_SERVER_CONF` alone.

## The gate is against a single-node reference, not against exit 0

Each workload runs single-node first and the distributed result is compared
against it. These benches report checksums whose entire purpose is to be
configuration-independent, so a run that merely exits 0 proves nothing: the
failure mode they actually have is a plausible, wrong number.

| workload | 2 nodes vs 1 | notes |
|-----------|--------------|-------|
| kmeans    | rel ~2e-09 | `--data-mb` is the GLOBAL problem, split; tolerance is for atomicAdd ordering |
| weights   | exact | integer accumulation commutes, so bit-equal at any node count |
| gmx       | bit-exact | fixed point: CONSERVATION, MESH and GATHER all have no tolerance |
| grayscott | rel 0 | 3D stencil, the only one exchanging a halo every step |
| lbann     | max abs 0 | elementwise vs the dense reference, not a digest |
| lammps_md | rel 0 | `E0 = -592121.595111`, the value `distributed_md_bench/` documents |
| gnn       | bit-exact | `logit_digest` is an integer sum over logit bit patterns; see below |

All six also pass at 4 nodes, and all six pass in their SYCL edition. gnn
passes at 2 and 4 nodes and out of core, in both CUDA editions. It has no SYCL
edition yet (no `sycl/` launcher), so it is absent from every SYCL gate list.

### gnn: a 2-layer GraphSAGE forward, sharded by page ownership

`benchmark/gnn/clio_gnn_paged_bench` is the multi-node edition of the GNN
tooling. The burst bench and `gnn_aggregate` read a prepared dataset from
local disk and own the whole matrix, so they are single-node only. This bench
synthesises features and the graph from a hash, so every node and the
single-node reference agree on the input without a dataset in the containers.

Each node owns a contiguous run of pages in two regions, X (features) and H
(layer-1 embeddings). Neighbours are drawn from the whole graph, so about
(N-1)/N of the edges read a peer's page. Layer 1 reads peer X pages at
generation 1 (published by the seed), and layer 2 reads peer H pages at
generation 2 (published by layer 1). Own pages are fetched at generation 0,
for the reason grayscott documents: a page this node wrote and never
re-fetched stays at generation 0, so demanding one of it stalls.

Nothing uses atomics, and each output element is summed in a fixed order. So
the logits are BIT-IDENTICAL at any node count, block count and lowering, and
the gate is an equality test on an integer digest. Measured on the
`--vertices 8192` deck (RTX 4070 Laptop):

| run | digest | witness `remote_edges` | control `GNN_NO_REMOTE=1` |
|---|---|---|---|
| 1 node (reference) | 10160982883994938968 | 0 | -- |
| 2 nodes, co_await and newcoro | 10160982883994938968 | 22463 | 13839825605177781443 |
| 4 nodes, co_await and newcoro | 10160982883994938968 | 33766 | 16573396213974516880 |
| 2 nodes OOC (`--slots 24`), both | 10160982883994938968 | 22463 | skipped (OOC) |

OOC evictions were 184 to 202 single-node and 106 to 109 distributed,
depending on the edition. The negative control drops peer-owned neighbours
but keeps them in the mean's divisor, so it computes a different function by
design and has to move. Independently of node count, every node also checks
its own logits against a host float reference (`ref_err`, about 2e-7, gated
at 1e-4). That catches a run where every node count is equally wrong, which
the 1-vs-N comparison alone cannot.

### Which newcoro gates have actually been RUN

Registering a gate is not evidence it passes. These were executed, on an RTX
4070 Laptop against a `cuda-release` tree (nvcc 12.9, sm_89):

| label | gates | status |
|---|---|---|
| `gv_dist4_newcoro`     | 7 | **all run, all pass** (gnn added later, run on the same machine) |
| `gv_dist_newcoro`      | 7 | **all run, all pass** |
| `gv_dist_ooc_newcoro`  | 7 | **all run, all pass** (lammps_md included -- see the frame-log section) |
| `gv_dist_newcoro_sycl` | 6 | REGISTERED BUT NEVER RUN -- needs a DPC++ tree |
| `gv_dist_ooc_newcoro_sycl` | 6 | REGISTERED BUT NEVER RUN -- same |

The SYCL rows are honest gaps, not implied passes. They are gated behind
`CLIO_GV_SYCL_BUILD_DIR`, so they simply do not register without one; nobody
gets a false green from them, but nobody should quote them as coverage
either.

### Measured: the newcoro editions at 4 nodes

The clio-coroc lowering, all six workloads, `GVW_NODES=4`, against each
workload's own single-node reference:

| workload | reference | 4 nodes | gate |
|---|---|---|---|
| kmeans    | 30719.999993 | 30719.999674 | rel 1.04e-08 <= 1e-4 |
| weights   | `checksum=OK` | `checksum=OK` | exact |
| gmx       | ALL GATES PASS | ALL GATES PASS | exact (fixed point) |
| grayscott | 36410.579344 | 36410.579344 | rel 0 <= 1e-9 |
| lbann     | ALL GATES PASS | ALL GATES PASS | exact |
| lammps_md | -592121.595111 | -592121.595111 | rel 0 <= 1e-6 |

The whole table was re-measured a second time, from deleted binaries, with the
harness doing its own transpile and build (no `GVW_NEWCORO_BUILD=0`), and all
six reproduced. kmeans lands on a slightly different checksum each run --
30719.999674 then 30719.999666 -- which is the atomicAdd ordering the bench
documents and is why that row has a tolerance rather than an equality test.
The weights witness reproduced EXACTLY (4258726497366525 ->
17035619285236281), as integer accumulation should.

lammps_md is in this set although the co_await 4-node set (`gv_dist4`) skips
it, and it reproduces `E0 = -592121.595111` -- the value
`distributed_md_bench/` documents -- on the `--lattice 28 --steps 20` deck at
four nodes.

The two anti-vacuity checks both fired and both moved, which is what makes the
row above mean anything:

- **weights witness**, `checksum_total`: 4258726497366525 at one node,
  17035619285236281 at four -- a ratio of 4.0002. Not 1.0 (no reduction) and
  not exactly 4.0 (four nodes on the same shard), so the four really did sum
  four different adjacent shards and combine them.
- **grayscott negative control**, `GS_NO_HALO=1`: 81072.776895 against the
  real run's 36410.579344. Disabling the exchange changes the answer, so the
  exchange is load-bearing rather than incidentally correct. The control's own
  value is NOT reproducible and is not meant to be -- three runs of the same
  configuration gave 81072.776895, 34388.929012 and (at 2 nodes) 31331.363277,
  because generation 0 reads whatever replica is lying around. The real run
  was 36410.579344 every single time. Only "control != real" is the gate.

### Measured: the newcoro editions at 2 nodes

Resident (`gv_dist_newcoro`) and under eviction (`gv_dist_ooc_newcoro`), same
decks as the co_await gates:

| workload | resident ref -> 2 nodes | OOC ref -> 2 nodes | OOC evictions 1-node / 2-node |
|---|---|---|---|
| kmeans    | 30719.999775 -> 30719.999578 | 30719.998967 -> 30720.000317 | 12288 / 6144 |
| weights   | OK -> OK | OK -> OK | 448 / 448 |
| gmx       | ALL GATES PASS | ALL GATES PASS | 247 / 117 |
| grayscott | 36410.579344 -> 36410.579344 | 381349.522809 -> 381349.522809 | 2177 / 865 |
| lbann     | ALL GATES PASS | ALL GATES PASS | 792 / 427 |
| lammps_md | -592121.595111 -> -592121.595111 | -592121.595111 -> -592121.595111 | 2234 / 555 |

The resident weights witness reads 4258726497366525 -> 8517538851720295, a
ratio of 2.00002 -- the same number the co_await edition measures, which is
the strongest single piece of evidence that the two lowerings are doing the
same distributed work and not merely arriving at the same checksum.

**The grayscott negative control's value is edition-dependent, and that is
fine.** `GS_NO_HALO=1` pins the halo fetches to generation 0, so what the
kernel reads is whatever replica happens to be lying around -- the newcoro run
measures 31331.363277 at 2 nodes where this README records 36104.119147 for
co_await. The gate asserts only that the control DIFFERS from the real run,
which is the property that makes the real run's pass meaningful; the control's
own number is not a reference value and should not be treated as one.

## OOC lammps_md: the defect that was in the coroc backend, and the fix

`cte_gv_ooc_newcoro_lammps_md` was the one newcoro gate that failed, and it
is registered and passing now. What was wrong, what fixed it, and the
measurements on both sides -- because for a while it looked like two
unrelated bugs, and it was one.

### Symptoms

With default slots, `--blocks <= 8` died at `CUDA Error 719`; the vector's
fatal channel named it:

```
[yield] DEVICE FATAL 101 (101=depth 102=frame 103=coro-frame): block=7 lane=0 need=8
```

At the OOC deck's `--slots 28` it did not trap at all -- every block parked
forever and the driver gave up after 2,000,000 pumped rounds with
`STATICS GATE: FAIL ... dev=0.0000000` (nothing computed). Both reproduced
single-node. Both were clio-coroc regressions: the co_await edition, built
from the same tree with clang-18 and run on the same GPU, passed every one
of those decks bit-exactly, including the OOC deck at 3013 evictions.

### Cause

The coroc backend's `clio::co::Frame` pre-claimed a fixed-size frame at
function ENTRY and recorded where it landed in `frame_off_[kYieldMaxDepth]`
(8), so that the next kernel entry could find the same frame again by depth.
That made call-chain depth a resource with a compile-time cap, and lammps_md
is the one workload whose fetch/evict chain is deep enough to exceed it. The
requirement moved with pressure -- `--blocks 16+` fit in 8, `--blocks 8`
needed 9-16, `--blocks 4` needed more than 16 -- so raising the constant only
moved the cliff (measured: at 16, `--blocks 8` passed and `--blocks 4` still
trapped). The livelock was the same bookkeeping failing differently.

### Fix: the lane region is a LIFO log, and nothing else

The cap was never necessary, because park and resume are already a stack:
the unwind on a park pushes innermost-first (C, then B, then A), and
re-entry resumes outermost-first (A, then B, then C). So `Frame` now claims
space at PARK time, not entry time -- `Push` appends `[vars..., trailer{start,
state}]` and bumps `sp_`, `Pop` reads the top entry back and unbumps, and a
function entering the chain asks one question: is the log non-empty? If so
the top entry is necessarily its own. `Done()` has nothing to do. The only
limit left is `bytes_per_lane`, which is the actual memory. The transpiler's
emitted API (`Resume/Replaying/Push/Pop/Done`) is unchanged, so no generated
code changed -- the benches just rebuild against the new header.

### After

| deck | before | after |
|---|---|---|
| `--lattice 28 --steps 10 --blocks 8` | `CUDA Error 719`, depth trap | PASS, `E0=-592121.595111` |
| `--lattice 28 --steps 10 --blocks 4` | trap even at `kYieldMaxDepth=16` | PASS, `E0=-592121.595111` |
| the OOC deck (`--slots 28 --page-kb 32 --rowchunk 1`) | livelock, `dev=0` | PASS, `E0=-592121.595111` |
| `cte_gv_ooc_newcoro_lammps_md`, 2 nodes | not registered | PASS, rel 0, **2234 / 555 evictions** |

The other eleven newcoro gates were re-run on the new backend and
reproduced to the digit (weights witnesses identical, eviction counts within
one).

### Occupancy and spills, since the register cap is the other half of this

From each binary's real launch records (`nsys`; `ncu` is blocked in the
devcontainer), sm_89, 36 SMs. Time-weighted over science kernels:

| | per-SM ceiling | limiter | actual fill at the deck |
|---|---|---|---|
| lammps_md newcoro   | 66% | REG=64, 256 thr -> 4 blk/SM | 28% (grid 64) |
| lammps_md co_await  | 67% | same                        | 15% (grid 1-64) |
| lammps_md MPI       | 70% | 50-72 regs at 64 thr        | 15% (grid 128) |
| lammps_md NVSHMEM   | 41% | 86-90 regs at 64 thr        | 15% |
| kmeans newcoro      | 67% | REG=64                      | **2%** (`--blocks 8` on 36 SMs) |
| kmeans MPI          | 100% | 40 regs                    | 31% |

Both paged editions sit on the same 66.7% ceiling by construction
(`__launch_bounds__(256, 4)` + the reg cap). At these decks nobody is
occupancy-limited; everybody is GRID-limited on a 36-SM part, and for the
paged benches "occupancy" is the `--blocks` knob until blocks exceed ~4x SMs.

Where the two lowerings really differ is what the 64-register cap COSTS.
Local-memory instructions in the SASS (lower bounds; the paged kernels run
past 4096 instructions):

| kernel | newcoro STL/LDL | co_await | MPI |
|---|---|---|---|
| ListForceKernel | >= 875 / 585 | 62 / 40 | 60 / 14 |
| BuildListKernel | >= 819 / 362 | 59 / 36 | 60 / 14 |
| MDIntegrateKernel | 257 / 223 | 60 / 36 | 0 / 0 |

TERMINOLOGY, because it bit once already: `cuobjdump`'s `STACK:` field is
ptxas's per-thread LOCAL-MEMORY frame -- register spills plus locals that
cannot be enregistered (dynamically indexed arrays, printf argument blocks).
It is a compiler artifact. It is NOT the yield stack / park log, which is the
`cudaMalloc`'d lane arena the host allocates and Push/Pop write to; that one
costs zero registers and is the same memory the co_await edition used for its
coroutine frames. Every "stack" number in this file is the local-memory frame.

### Where the registers actually go -- measured by subtraction

Natural allocation (no launch bounds, no cap), so the numbers are demand
rather than a ceiling. MPI's kernels are the floor: same physics, raw
pointers, 12-76 registers. The coroutine editions are 138-255. Taking
`ThermoKernel` apart one piece at a time, inlined:

| variant | registers | local mem |
|---|---|---|
| full: Frame + switch + Push/Pop + `CoFetch`/`CoHoldPage` | 182 | 272 B |
| Frame + switch + Push/Pop, page-cache calls removed       |  60 |   8 B |
| plain loop, no frame, page-cache calls removed            |  44 |   8 B |
| MPI `ThermoKernel`                                        |  28 |   0   |

So: the coroutine machinery itself (Frame, switch, saving 25 locals at 4
suspend points) is **16 registers**. The kernel scaffolding
(`CLIO_GPU_INIT`, `CLIO_COROC_RUN` persist save/restore, `UnpinRange`) is
another **16**. The remaining **122 registers and essentially all the local
memory are `DeviceVector::CoFetch` and `CoHoldPage`** -- the set-associative
page lookup, claim/admit, generation checks, the `OnceWait` retry loops,
block-wide votes and a dozen device `printf` sites in the fatal paths, all
`__forceinline__`d into every kernel. co_await inlines the same runtime and
lands at 192 for the same reason.

Two things that were suspected and measured NOT to matter for this kernel:
the transpiler's push-everything save list (worth ~16 here, more on kernels
with 92 hoisted locals like ListForce), and the `case`-labels-inside-loops
control flow (a nested-dispatch rewrite of ThermoCoro moved it by 3). The
earlier wording in this section that blamed the hoist/byte-copy lowering for
the register count was wrong by an order of magnitude and is superseded by
the table above. What the coroc lowering does add over co_await is spill
traffic under the cap, because everything it hoists is kept live everywhere.

## checksum=OK alone is a vacuous gate -- weights needed a witness

Each node compares its own partial `got` against its own partial `want`, so a
reduction that silently did nothing still reports OK. The bench therefore also
prints `checksum_total`, the reduced whole-model number, and the harness fails
if it is unchanged from the single-node run.

Measured 2-node/1-node ratio: **2.00002**. That number is the proof --
`1.0` would mean no reduction happened, exactly `2.0` would mean both nodes
summed the same shard, and `2.00002` means they summed different adjacent
shards and combined them.

## grayscott has a negative control, because its gate has a tolerance

`GS_NO_HALO=1` forces the halo fetches back to generation 0. The harness runs
it after a pass and FAILS if the answer is unchanged: an exchange whose absence
changes nothing is not load-bearing, and a tolerant checksum would hide that.
With the demand: `36410.579344`. Without: `36104.119147`.

Contrast gmx, whose fixed-point gates are exact and fail loudly on their own --
that is why gmx caught a stale-page bug that grayscott's gate would have
absorbed.

## Traps this harness has already stepped in

- **In a devcontainer, `/workspace` is not the path the daemon binds.** The
  repo's own devcontainer talks to the HOST's docker daemon, which resolves
  bind sources in the host's filesystem -- so `-v /workspace:/workspace` hands
  it a path that does not exist, docker creates an empty directory, and every
  node dies with `sh: 1: .../clio_<wl>_paged_newcoro: not found` and exit 127.
  That reads as a missing binary while the binary is sitting right there. The
  script now asks the daemon for this container's own mounts and rewrites the
  root through the one that contains it; `HOST_WORKSPACE` still overrides.
  Note this applies to every bind in the compose file, not just the workspace.
- **`CUDA_HOME` must name a toolkit the DAEMON can see, with the soname the
  build actually links.** Two separate traps, and the second hides behind the
  first. A host can have `/usr/local/cuda-12.9/` present but EMPTY while the
  devcontainer's copy of that same path is complete, so the mount succeeds and
  contributes nothing; and a host that only ships CUDA 13 gives
  `libcudart.so.13` to a tree built by nvcc 12, which wants `.12`. Both land as
  `error while loading shared libraries: libcudart.so.12` and exit 127. Check
  what the DAEMON sees (`docker run --rm -v /usr/local:/hl:ro alpine ls
  /hl/cuda-*/lib64/libcudart.so.*`), not what `ls` shows here. The fallback,
  when no daemon-visible toolkit matches, is to copy the one library into the
  build's `bin/` -- it is already mounted, already on `LD_LIBRARY_PATH` and
  already the `$ORIGIN` of every binary there.
- **The `CUDA_HOME` the compose binds is not the one the BUILD wants, and
  three requirements were sharing that one variable.** The mount needs a
  toolkit the docker daemon can see; nvcc needs one matching the libraries the
  benches link; the transpiler's clang needs one it can actually parse (clang
  18 cannot read CUDA 13 headers at all). On a host whose only daemon-visible
  toolkit is CUDA 13, satisfying the mount broke the other two and EVERY
  transpile died on

  ```
  crt/math_functions.hpp: error: expected function body after function declarator
  fatal error: 'texture_fetch_functions.h' file not found
  ```

  which reads as a broken transpiler rather than a mis-pointed toolkit. The
  harness now unsets `CUDA_HOME` for the build sub-process (so
  `build_newcoro.sh` takes nvcc from PATH) and `GVW_BUILD_CUDA_HOME` overrides
  when the two must differ. Worth noting HOW this was missed for a while: the
  binaries had always been built BY HAND, with `CUDA_HOME` unset, and the
  gates were then run with `GVW_NEWCORO_BUILD=0`. The binaries passed; the
  gate path itself had never executed. Run at least one gate without
  `GVW_NEWCORO_BUILD=0` before believing a newcoro label.
- **libnvcomp is not in the deps image.** Containers died at exec with 127
  before any clio code ran. It is mounted, as the md harness does. Same for the
  DPC++ prefix (`DPCPP_HOME`) for the SYCL editions.
- **The cluster compose schema is not the one the benches write inline.** Pool
  ids are `"major.minor"` STRINGS -- a bare `pool_id: 512` is rejected at load
  with `Invalid UniqueId format` -- and a core pool declares `storage:`, not
  `tiers:`.
- **Neither node may leave while its peer is still paging.** The runtime dies
  with the process and a peer whose pages live on the departing node waits
  forever, so each node touches `.done_<id>` on the shared mount and waits.
- **An unset variable is not an absent one.** docker-compose passes
  `GS_NO_HALO=` through as an empty string, and `getenv` returns non-null for
  that, so a presence test enabled the negative control in EVERY run -- a
  correct build reported the control's wrong answer and looked like a
  regression. Flags test for non-empty.
- **Generation 0 means "any version will do".** It was the cause of three
  separate defects: grayscott's stale halo, lbann's stale peer W2, and lbann's
  own verification read, which reported a whole weight update as error while
  the vector held the right bytes. A fetch that does not name a generation
  accepts whatever is lying around.
- **A node's band must own whole pages.** Splitting off a page boundary makes
  two nodes write the same page and clobber each other -- silently for lbann's
  biases, and as a HANG at 4 nodes. lbann rejects such a geometry now.

## CI

No `add_test` for the underlying scripts themselves, matching the other
cluster harnesses -- the ctest entries above wrap them and carry the
environment. These need docker, the nvidia container toolkit and a GPU, which
is why they are opt-in rather than on by default.
