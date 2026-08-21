# eternia-MD, NVSHMEM edition: the scale-out baseline

`clio_md_nvshmem_bench` runs the same melt deck as `clio_gpu_vector_md_bench`
with one substitution: simulation state lives in the NVSHMEM **symmetric
heap**, cut into z-plane slabs across PEs, instead of on paged `gv::Vector`s.

It exists to make the capacity claim falsifiable. The eternia bench says "the
ceiling is the tier stack, not VRAM". The obvious retort is "or you could just
use more GPUs". This is that retort, written out, gated the same way, and
priced.

|                       | `clio_gpu_vector_md_bench`      | `clio_md_nvshmem_bench`        |
|-----------------------|---------------------------------|--------------------------------|
| substrate             | paged `gv::Vector`              | NVSHMEM symmetric heap         |
| capacity ceiling      | HBM → RAM → NVMe tier stack     | `npes` × VRAM                  |
| a miss costs          | a page fault, a coroutine park  | a `getmem` from the owning PE  |
| out of core           | yes, by design                  | **no** — nothing spills        |
| per-step host traffic | zero                            | zero                           |
| per-step fabric traffic | zero                          | the halo (reported in a ledger)|

## What is identical, on purpose

Physics, layout and decomposition are carried over unchanged, because a
baseline that differs in two things measures neither:

- atoms bin-major in padded bins, sentinel `type = -1` in `x.w`;
- work unit = a chunk of x-rows within one z-plane, so the chunk's stencil is
  three z-planes × (`rowchunk` + 2) y-rows, each a contiguous element run;
- Verlet list transposed and padded, entries packed stencil-relative as
  `(q << 16) | slot-within-row`;
- the gates: step-0 statics against a host double reference *and* the
  published stock LAMMPS PE/atom, resort continuity, NVE drift, and the
  ballistic gate (bitwise against a host float replica).

The one structural simplification: nothing here can suspend, so block-uniform
tables published into shared memory stay valid. Distribution costs a copy;
paging costs a suspension. That is the whole trade in one sentence.

## The substitution, precisely

Where the eternia kernel writes

```cpp
Held<float> h = co_await x.HoldPage(base, len);   // may fault to a tier
```

this kernel writes `StageStencil(...)`, which resolves each stencil span to
either a bare local pointer or a block-collective `nvshmemx_getmem_block` from
the PE that owns it, landing in per-block scratch — the structural analogue of
a page-cache slot, and reported as such in the footprint line.

Migration is the part the paged design does not have to write at all: an atom
that leaves its slab claims a slot with `nvshmem_uint32_atomic_fetch_add` on
the **owner's** bin counter, and its four floats go over as element puts.

## Measured (2026-08-20, RTX 4070 Laptop, sm_89, NVSHMEM 3.7.2)

Gate deck: `--md --lattice 20 --steps 50 --rebin 10 --temp 3.0 --cap 48
--blocks 128 --threads 64 --drift-tol 5e-3` (32k atoms, 11³ bins). The
256k-atom timing deck is in the next section.

| PEs | statics | resort | NVE | ms/step | staged/step | migrants |
|-----|---------|--------|-----|---------|-------------|----------|
| 1   | PASS    | PASS (rel 0.0) | PASS (3.97e-3) | 1.28 | 0 | 0 |
| 1 `--force-remote` | PASS | PASS (rel 0.0) | PASS | 2.30 | 4.97 MB | 0 |
| 2   | PASS    | PASS (rel 0.0) | PASS (3.97e-3) | 9.08 | 0.60 MB (100% peer) | 891 |

Every configuration reproduces the same physics as the paged bench: PE/atom
−6.7733683 against stock's published −6.7733681, and a pair count of exactly
864000 matching the independent host double reference. Resort continuity is
**bitwise** at 2 PEs while 891 atoms cross the slab boundary, which is the
gate that actually exercises the migration path.

## The MPI sibling, and what it exposed

`clio_md_mpi_bench` is the same kernels with one variable changed: two-sided,
host-staged `MPI_Sendrecv` of the boundary planes before each launch instead
of one-sided `nvshmemx_getmem_block` inside it. It was built to check that
the NVSHMEM bench was a fair baseline. It found that it was not, for a reason
that has nothing to do with communication.

Interleaved, 256k atoms, best config for each (`--rowchunk 1 --threads 128`),
3 reps, spread under 0.5%:

| configuration | ms/step | note |
|---|---|---|
| stock `lj/cut/kk` | 3.63 | |
| MPI, 1 rank | **1.47** | no communication at all |
| NVSHMEM, 1 PE | 1.77 | no communication at all |
| MPI, 2 ranks | 2.33 | one GPU, time-sliced |
| NVSHMEM, 2 PEs | 8.23 | one GPU, time-sliced, **limited MPG** |

**Finding 1 — the rdc tax, and it is real everywhere.** At one rank *neither*
bench communicates, yet they differ by 20%. That is not transport. NVSHMEM's
device library is a relocatable-device-code archive, so linking it forces
`-rdc=true` on the whole binary and blocks whole-program device optimisation.
Compiling the MPI bench — same source, same kernels — with `-rdc=true` alone
reproduces it exactly:

| MPI bench, 1 rank | ms/step |
|---|---|
| default | 1.472 |
| `-rdc=true` | 1.740 |
| (NVSHMEM bench, 1 PE) | 1.768 |

So **+18% on the compute kernels, paid before a single byte moves**, purely
for linking NVSHMEM. This is machine-independent and worth knowing before
choosing NVSHMEM for a kernel-bound code.

**Finding 2 — the 2-rank row is NOT a transport result, and must not be
quoted as one.** With two PEs on one GPU NVSHMEM runs in "limited MPG" mode
with no direct peer mapping, and its device-side remote reads collapse. The
arithmetic, from the ledgers: at 2 PEs the force pass goes 130 → 644 ms while
only 11000 of 189750 spans are remote, i.e. ~46.7 µs per remote span. The
same staging copy issued to *itself* within one process (`--force-remote` at
1 PE: 1.770 → 2.402 ms/step over 189750 spans) costs ~0.33 µs. A remote span
is **140× a local one** here. On real multi-GPU with NVLink a `getmem` is a
direct peer load, not a proxied round trip, so that 140× is an artifact of
this box and nothing else.

**What survives the machine, and is worth keeping:**

- *Bytes moved.* One-sided per-block pull moves **3.0×** the halo of a bulk
  pre-staged exchange (5.11 vs 1.70 MB/step), because each block re-pulls the
  boundary rows it needs — 11000 span fetches against 110 plane exchanges.
  Lazy and fine-grained costs redundancy; that is algorithmic, not hardware.
- *Migration.* Two-sided messaging has no remote atomic, so an atom leaving
  a slab must be packed, exchanged and unpacked. The MPI resort phase is
  221.8 ms against NVSHMEM's 126.7 — one-sided genuinely wins here, and that
  direction should hold on real hardware.
- *Both agree, exactly.* All gates pass in all five configurations, and the
  two transports migrate an identical **5137** atoms with resort continuity
  bitwise. That is the check that they are the same simulation.

## Against stock LAMMPS, and against paging

The deck all three run: 256k atoms (`lattice 40`, rho 0.8442), lj/cut 2.5,
skin 0.3, rebuild every 10 steps, T = 3.0, 100 steps. Stock is `lj/cut/kk`
(KOKKOS CUDA, `-k on g 1 -sf kk -pk kokkos newton on neigh half`) from
`lbann-stack/lmp-kk-build`. Each configuration tuned for itself; runs
INTERLEAVED (stock, NVSHMEM, paged, repeat ×3) because this laptop GPU's
clock state moves the absolute numbers by ~40% across a session while the
ratios hold — a non-interleaved comparison here is worthless.

| code | substrate | ms/step | rebuild-matched | Matom-step/s | vs stock |
|------|-----------|---------|-----------------|--------------|----------|
| stock `lj/cut/kk` | LAMMPS device arrays | 3.64 | 3.64 | 70.3 | 1.00× |
| `clio_md_mpi_bench` | plain CUDA + MPI halo | 1.47 | **1.52** | 174.4 | **2.40× faster** |
| `clio_md_nvshmem_bench` | symmetric heap | 1.76 | **1.81** | 145.1 | 2.01× faster |
| `clio_gpu_vector_md_bench` | paged `gv::Vector` | 7.05 | **7.31** | 36.3 | 2.01× slower |

The MPI row is the honest single-rank compute number for this physics, since
it carries neither the rdc tax nor any communication; the NVSHMEM row is the
same kernels plus 18% of relocatable-device-code overhead.

Spread across 3 interleaved reps was under 5% for all three. Best configs:
NVSHMEM `--rowchunk 1 --threads 128`; paged `--rowchunk 2 --page-kb 69
--threads 64` (the paged bench loses badly at 128 threads — 11.0 ms/step —
which is the coroutine register ceiling, exactly as its own notes predict).

"Rebuild-matched" corrects the one scale mismatch in the decks: stock does
**10** neighbour builds in 100 steps, both benches do **9** (their loop skips
step 0). Per-rebuild cost measured by differencing against `--rebin 0` in the
same clock regime — 4.9 ms NVSHMEM, 26.1 ms paged — and one is added back.
It is a 3% effect on the NVSHMEM number and 4% on the paged one; it does not
move any conclusion, but it is the kind of thing that should not be left for
a reader to discover.

### Is this the same problem? An audit

Scale and problem size are identical and verified from both logs. The
numerical and algorithmic *conventions* are not, and they do not all push the
same way:

| aspect | stock | both benches | same |
|---|---|---|---|
| atoms | 256000 | 256000 | ✅ |
| box edge | 67.183848 | 67.1838 | ✅ |
| lattice / density | fcc 0.8442 | fcc 0.8442 | ✅ |
| cutoff / skin / rlist | 2.5 / 0.3 / 2.8 | 2.5 / 0.3 / 2.8 | ✅ |
| timestep | 0.005 (verified explicitly) | 0.005 | ✅ |
| initial temperature | 3.0 | 3.0 | ✅ |
| steps | 100 | 100 | ✅ |
| rebuild cadence | every 10, `check no` | every 10 | ✅ (10 vs 9 builds, corrected above) |
| neighbour bins | 2.8, 24³ | 2.921, 23³ | ≈ |
| **precision** | **double** | **float** | ❌ favours the benches |
| **ghosts** | **70348 ghost atoms + per-step comm** | **none (minimum image)** | ❌ favours the benches |
| **spatial sort** | every 1000 steps (default) | every rebuild | ❌ favours the benches |
| **list convention** | **half, newton on — 37.9 neigh/atom, 9.70M pairs** | **full, newton off — ~75.7/atom, 19.4M pair evals** | ❌ favours stock |
| list storage | compact dynamic, 9.70M entries | padded fixed, 56M slots (214 MB) | ❌ favours stock |
| initial velocities | uniform RNG, seed 87287 | deterministic analytic, momentum-zeroed, scaled to T=3 | ❌ same macrostate, different microstate |

So the honest reading of "2× faster than LAMMPS" is: the benches carry a real
2× advantage in precision and pay a real 2× penalty in pair evaluations, plus
a ghost-comm and sorting advantage that stock's own timing cannot cleanly
separate (on one rank KOKKOS attributes ~78% of the loop to `Comm`, which is
mostly the async sync point rather than genuine ghost traffic). The
defensible claim is **"at or about stock speed, in single precision"** — not
a win over LAMMPS. What the number does establish is that the
reimplementation is not leaving an order of magnitude on the floor, which is
the precondition for the substrate comparison below meaning anything.

**The substrate cost, which is the point.** Every caveat in the audit above
cancels between the two benches: they are the same source lineage, same
precision, same full-list/newton-off convention, same ghost-free minimum
image, same sort cadence, same rebuild count. They differ *only* in how a
stencil row is reached. That difference is **4.0×** (1.81 vs 7.31 ms/step
rebuild-matched), and the paged run held its resident contract — zero faults,
zero evictions — so it is the hold machinery itself being measured (coroutine
frames, pins, guards), not tier traffic. Per phase, in one clock regime:

| phase (100 steps) | NVSHMEM | paged |
|---|---|---|
| force | 130.0 ms | 393.7 ms |
| list build + resort | 50.6 ms | 307.2 ms |
| integrate (kick) | 10.1 ms | 75.8 ms |

Paging buys a capacity ceiling that is not VRAM. On this deck it costs about
4× against the scale-out way of buying the same thing, and about 2× against
stock — and that is the number the comparison is actually for, because it is
the only one where nothing else differs.

### Two numbers in the PE table are not performance numbers

- **The 2-PE ms/step is meaningless as scaling.** This machine has one GPU, so
  the two PEs time-slice it in NVSHMEM's "limited MPG" mode. The run is a
  *correctness* result for the decomposition, not a throughput one. Real
  scaling numbers need one GPU per PE.
- **`--force-remote` is a floor, not the cost of distribution.** It routes
  self-owned rows through a self-PE `getmem`, so the 1.28 → 2.30 ms/step
  delta prices the staging *copy* with no fabric involved. A peer copy over
  PCIe or NVLink is strictly worse than that.

## Build and run

NVSHMEM is not part of the default build:

```bash
cmake -B build -DCLIO_CORE_ENABLE_NVSHMEM=ON ...      # NVSHMEM_HOME if needed
cmake --build build --target clio_md_nvshmem_bench
```

The target is built by **nvcc** through a custom rule, not
`add_cuda_executable`: the NVSHMEM device library is an rdc archive needing
`-rdc=true` and nvlink, while this tree points `CMAKE_CUDA_COMPILER` at clang
for the device-coroutine kernels and turns separable compilation off there.
The two cannot share a target. It links nothing from clio — a baseline that
shares a substrate with the thing it is measured against is not a baseline.

```bash
# one PE (unique-id bootstrap, no launcher needed)
build/bin/clio_md_nvshmem_bench --md --lattice 20 --steps 50 --rebin 10 \
    --temp 3.0 --cap 48 --blocks 128 --threads 64 --drift-tol 5e-3

# N PEs, one GPU each (MPI bootstrap, compiled in when MPI is found)
mpirun -n 4 build/bin/clio_md_nvshmem_bench --md --lattice 40 ...
```

Useful flags: `--force-remote` (stage every row, exercising the remote path on
one GPU), `--no-list` (cell-direct stencil scan instead of the Verlet list),
`--rowchunk` (rows per staging batch), `MD_NOCOMPUTE=1` (keep every stage,
delete the pair loop — the difference is the staging cost).

The run refuses up front, with numbers, when per-PE state will not fit VRAM.
That refusal is the point of the file: this baseline has nowhere to spill.
