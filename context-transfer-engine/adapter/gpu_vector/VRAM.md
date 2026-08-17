# What determines VRAM in the eternia workloads

Measured on an RTX 4070 Laptop (8188 MiB, compute 8.9) with the `vram_probe`
sampler. Every number below is peak GPU memory for the whole device, including
a 15 MiB idle baseline.

## Measure it at 50 ms or you will get the wrong number

Peaks are transient. The same LBANN configuration reads **5487 MiB** when
`nvidia-smi` is sampled every 200 ms and **7569 MiB** at 50 ms, reproducibly.
Every figure here is from 50 ms sampling; anything coarser understates the peak
by enough to invalidate a sizing decision.

## The two paths behave completely differently

**Stock kernels: VRAM scales with the problem.** The data is resident, so
memory is a linear function of problem size.

**The paged path: VRAM is almost independent of the problem.** The data lives
in the CTE, so what costs VRAM is the cache and tier configuration. LAMMPS
paged, at fixed cache settings:

| atoms | peak VRAM |
|---|---|
| 62,500 | 3037 MiB |
| 665,500 | 3109 MiB |
| 1,372,000 | 3113 MiB |

A 22x increase in atoms costs 76 MiB. This is the property the whole design
exists for, and it is why the paged path can be sized by configuration alone.

## Stock-path models

Linear fits over the measured range. `atoms` in thousands.

| workload | model | measured points |
|---|---|---|
| LAMMPS `lj/cut/kk` | `226 MiB + 0.49 x atoms` | 535 @ 665k, 1661 @ 2.92M, 3627 @ 6.91M, 4067 @ 7.81M |
| GROMACS nbnxm | `159 MiB + 0.140 x atoms` | 201 @ 216k, 401 @ 1.73M, 975 @ 5.83M, 4045 @ 28.1M |
| LBANN `El::Gemm` | not linear -- see below | 3289 @ W=1024 MiB, 4467 @ 1089, 4599 @ 1156, 7569 @ 1600 |

**LBANN does not fit a line.** VRAM holds W, the gradient dW, optimizer state
and activations, and the allocator moves in steps: width 16384 gives 3289 MiB
and 16896 gives 4467 MiB, a 1178 MiB jump for 65 MiB more weights. Pick an
LBANN size by measuring, not by interpolating.

Configurations that land on 4 GB:

| workload | configuration | peak VRAM |
|---|---|---|
| LAMMPS | 7,812,500 atoms (`lattice_cells 125`) | 4067 MiB |
| GROMACS | 28,094,464 atoms (`cells 304`) | 4045 MiB |
| LBANN | `width 16896` | 4467 MiB |

## Paged-path variables

Four things move VRAM, and the dataset is not one of them.

### 1. The `hbm` tier's `capacity_limit`

A GPU allocation taken at startup, whatever the workload does. The LAMMPS
example config reserves **1 GB** and the test configs 64 MB. This dominates
small configurations and is the first thing to check when a paged run looks
unexpectedly heavy.

### 2. `blocks`, 3. `slots`, 4. `page_kb`

The page cache is `blocks x slots x page_kb` **per vector**, and LAMMPS creates
FOUR paged vectors with different per-block slot counts:

| vector | slots | set by |
|---|---|---|
| `x` | 16 | the `slots` keyword |
| `f` | 8 | fixed |
| `type` | 4 | fixed |
| `neigh` | 4 | fixed |

So the `slots` keyword controls only one of the four, and the effective total is
`slots_x + 16` pages per block. That is why raising `slots` from 16 to 64
behaves differently from raising `blocks` or `page_kb` by the same factor.

Measured at 62,500 atoms with a 1 GB hbm tier:

| blocks | slots | page_kb | peak VRAM |
|---|---|---|---|
| 16 | 16 | 256 | 1429 MiB |
| 64 | 16 | 64 | 1467 MiB |
| 64 | 16 | 256 | 2125 MiB |
| 64 | 64 | 256 | 2899 MiB |
| 128 | 16 | 256 | 2907 MiB |
| 64 | 16 | 1024 | 4621 MiB |

**Do not size from `blocks x slots x page_kb` alone.** It predicts the `slots`
axis well -- 16 to 64 slots added 768 MB of cache and 774 MiB of VRAM, almost
exactly 1:1 -- but underestimates the other two badly. Raising `page_kb` from
256 to 1024 added 768 MB of cache by that formula and **2496 MiB** of actual
VRAM; raising `blocks` from 64 to 128 added 256 MB and cost 782 MiB. The excess
also scales with `blocks x page_kb`, which is the shape of per-block staging and
writeback buffers rather than of the cache itself.

The practical rule: `page_kb` is the most expensive knob, `blocks` next,
`slots` the cheapest per byte of cache. Measure the corner you intend to run.

## Accuracy is a VRAM-adjacent variable

Worth recording next to the memory numbers because it changes which
configuration is correct, not merely which is fast. At 28,094,464 atoms:

| kernel | per-atom LJ energy | relative error |
|---|---|---|
| exact lattice sum | -4.914218 | -- |
| paged (double accumulation) | -4.914219 | **1.0e-07** |
| GROMACS stock (mixed precision) | -4.9007 | **2.7e-03** |

GROMACS's own kernel is 0.27% low at this scale, against 9.8e-05 at 216k atoms
and 1e-07 for the paged kernel. Its single-precision accumulation degrades with
pair count, so a large stock run is not a trustworthy reference for a small
paged one.
