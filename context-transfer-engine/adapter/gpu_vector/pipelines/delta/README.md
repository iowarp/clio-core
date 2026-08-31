# gpu_vector on NCSA Delta (A100)

The CUDA-clang gpu_vector stack, built and swept on one Delta A100. The
science is unchanged from `../memory_pressure/` — what lives here is the
Delta-specific toolchain and the SLURM wrapper the login node forces on us.

```
env.sh                              build+run environment (source it)
configure.sh                        configure + build (cuda-clang, sm_80)
gv_memory_pressure_a100_smoke.yaml  6 cells, 10 min — the plumbing check
gv_memory_pressure_a100_1n.yaml     96 cells — the experiment
gv_variant_matrix_a100.yaml         24 cells — does every substrate run?
```

## Build

```bash
./configure.sh              # configure + build everything
./configure.sh --configure  # configure only
BUILD_DIR=/somewhere JOBS=32 ./configure.sh
```

Then, for just the pieces the sweep needs:

```bash
source env.sh
cmake --build "$CLIO_GV_BUILD" -j 32 --target \
    clio_kmeans_paged_bench clio_weights_paged_bench \
    clio_grayscott_paged_bench clio_gmx_paged_bench \
    clio_lbann_paged_bench clio_lammps_md_paged_bench
```

### Why the toolchain is spelled out by hand

gpu_vector's device API is C++20 coroutines end to end, so the tree refuses
to build it under nvcc (`CLIO_GPU_CLANG`, see the top-level `CMakeLists.txt`).
On Delta that forces three choices that no module set makes for you:

| Piece | What we use | Why not the obvious thing |
|---|---|---|
| CUDA | `/sw/external/cuda/cuda-12.6.3` (outside lmod) | The default `cudatoolkit/26.5_13.2` is CUDA 13.2, which clang cannot parse (`crt/math_functions.hpp: expected function body after function declarator`). The nvhpc `cuda/12.9` module has no curand headers in its `cuda/` tree, and clang's runtime wrapper includes `curand_mtgp32_kernel.h` unconditionally. 12.6 is also the newest CUDA clang 19 claims to know. |
| clang | `llvm/19.1.7`, by absolute path | `module load llvm` swaps out `gcc-native` (Lmod compiler family) and deactivates `cray-mpich`. We want gcc as the HOST compiler and clang only as `CMAKE_CUDA_COMPILER`. |
| C++ deps | the spack instance at `$SPACK_ROOT` | The `iowarp` conda env this tree used to build against no longer exists. |

`env.sh` also clears `CPATH`: the cudatoolkit module puts the 13.2 include
dir there, and `CPATH` **outranks `--cuda-path`**, so unloading the module is
not by itself enough.

### The spack instance must not live under the source tree

It used to be `clio-core/spack`. CMake then refuses to generate — every
external dependency directory ends up in an exported
`INTERFACE_INCLUDE_DIRECTORIES`, and CMake rejects an exported path
"prefixed in the source directory" (`lightbeam` on zmq/sodium/bsd, `aio` on
libaio). The real instance now lives at `/u/llogan/spack`, with
`clio-core/spack` left as a compat symlink so the absolute prefixes baked
into the installed RPATHs still resolve; the `.pc`/`.cmake` files in the
install tree were rewritten to the new prefix. `env.sh` canonicalizes
`SPACK_ROOT` through `readlink -f` for the same reason — a search prefix
reached through the symlink makes `find_path` record the source-prefixed
path all over again.

## Run

```bash
source env.sh                                            # not needed to submit,
jarvis ppl submit gv_memory_pressure_a100_smoke.yaml     # but handy for `which`
jarvis ppl submit gv_memory_pressure_a100_1n.yaml
jarvis ppl submit gv_memory_pressure_a100_1n.yaml +no_submit   # script only
jarvis ppl post   yaml gv_memory_pressure_a100_1n.yaml         # re-render figures
```

Both use jarvis-cd's native `scheduler:` block (Mode A: the whole sweep in
one allocation) — there is no separate `.sbatch` wrapper and no
`jarvis hostfile set`. The generated script lands in
`~/.ppi-jarvis/shared/<pipeline>/submit.slurm`.

Results go to `$HOME/gv_pipeline_results/<name>/`; per-cell benchmark logs
go to `/tmp/clio_gv/<name>/` on the compute node and do **not** survive the
job.

Two things that will bite you:

- **The pipeline YAML must be on a shared filesystem.** The job script runs
  `jarvis ppl run yaml <path>` on the compute node; a path under a
  node-local or session-local `/tmp` fails there with
  `Pipeline file not found`.
- **Clear the output directory before a fresh sweep.** jarvis resumes by
  ROW COUNT, not by parameters — see the RESUME note at the top of either
  YAML.

## NVSHMEM multi-node (the thing docker CI could not do)

`753798ef` recorded NVSHMEM multi-node as NOT SOLVED in containers, with four
causes eliminated. It works here, and the container dead-ends were not the
reason. Full recipe, all of it load-bearing, in `env.sh`:

```bash
NVSHMEM_BOOTSTRAP=PMI  NVSHMEM_BOOTSTRAP_PMI=PMI2
NVSHMEM_REMOTE_TRANSPORT=libfabric  NVSHMEM_LIBFABRIC_PROVIDER=cxi
FI_CXI_OPTIMIZED_MRS=0  FI_CXI_DISABLE_HMEM_DEV_REGISTER=1
srun --mpi=pmi2 -n N --ntasks-per-node=1 <bench>
```

Four separate failures had to be cleared, each with its own signature:

1. **MPI cannot bootstrap it here.** NVSHMEM's MPI bootstrap is a dlopen'd
   plugin built against OpenMPI (needs `libmpi.so.40`), but Delta's MPI is
   cray-mpich; and the HPC-X OpenMPI inside nvhpc cannot connect across
   Slingshot at all -- `ucp_ep_create failed: Destination is unreachable`
   from its UCX PML. PMI sidesteps both.
2. **`NVSHMEM_BOOTSTRAP` takes the FAMILY, not the variant.** Passing `PMI2`
   to it fails with `bootstrap_preinit failed`; the variant belongs in
   `NVSHMEM_BOOTSTRAP_PMI`. PMIX is unusable -- its plugin needs HPC-X's
   libpmix, which has an undefined `opal_libevent2022_evthread_use_pthreads`.
3. **The transport must be named.** Left alone NVSHMEM defaults to `ibrc`,
   enumerates InfiniBand devices, finds none on a Slingshot machine, and
   fails with `building transport map failed`.
4. **CXI needs the tuning NVSHMEM asks for in its own strings** -- the
   libfabric transport literally carries "FI_CXI_OPTIMIZED_MRS is set. This
   may cause a hang at runtime if the value is not 0". Without it the run
   reaches the data path and hangs in `nvshmemt_libfabric_quiet` with
   `Connection timed out`.

The benches must be built with `-DCLIO_GV_NVSHMEM_MPI_BOOTSTRAP=OFF`, which
makes them call `nvshmem_init()` and honour the above. Their previous
non-MPI path was hardcoded to ONE PE
(`nvshmemx_set_attr_uniqueid_args(0, 1, ...)`), so the only build that could
be distributed was the MPI one -- which is the build that cannot run here.

### Measured

NVIDIA's own `shmem_put_bw`, 2 nodes over Slingshot:

| size | GB/s | size | GB/s |
|---|---|---|---|
| 4 B | 0.00048 | 256 KiB | 10.25 |
| 16 KiB | 0.636 | 1 MiB | 10.57 |
| 64 KiB | 2.639 | 4 MiB | **11.33** |

Our benches, `--steps/--iters 20`:

| workload | 4 PEs / 4 nodes | result |
|---|---|---|
| kmeans | 294.7 ms (2 nodes: 455.6 ms) | ALL GATES PASS |
| grayscott | 27.7 ms | ALL GATES PASS |
| weights | 114.5 ms | ALL GATES PASS |
| lammps_md | 21.7 ms (1.086 ms/step) | gates pass; ledger shows 3.30 MB/step, **100% from a PEER** |
| gmx | ran | **CONSERVATION GATE FAIL** |
| lbann | ran | **LOSS + WEIGHT GATE FAIL** |

### Two real bugs this immediately exposed

All six initialise and run multi-node; two are numerically wrong once there
is more than one PE, which nothing could have caught while NVSHMEM was
single-node only.

- **gmx**: `CONSERVATION GATE: PASS (exact)` at **1 PE**, `FAIL` at **2 and
  4 PEs**. Not the transport and not the parameters -- a multi-PE bug in
  `clio_gmx_nvshmem_bench`.
- **lbann**: fails at 4 PEs on both the loss gate (0.41861 vs 0.42488) and
  the weight digest. The default `--lr 0.01` also diverges to NaN at this
  size, as the memory-pressure notes already warned; `--lr 0.0001` gives the
  finite-but-wrong numbers above, so the divergence and the multi-PE bug are
  two separate things.

## Substrates

All 25 (workload × substrate) binaries build. Measured by
`gv_variant_matrix_a100.yaml` on 2 A100s (job 21606871):

| workload | paged | mpi | nvshmem | nccl |
|---|---|---|---|---|
| kmeans | ✅ | ✅ | ✅ | — |
| weights | ✅ | ✅ | ✅ | — |
| grayscott | ✅ | ✅ | ✅ | — |
| gmx | ✅ | ✅ | ✅ | — |
| lbann | ✅ | ✅ | ✅ | — |
| lammps_md | ✅ | ✅ | ✅ | ✅ |

**BaM is not in the study** — dropped by decision. For the record it built
for all six and passed five (kmeans 6.6 s, gmx 1.4 s, lbann 1.9 s,
lammps_md 2.4 s, weights 143 s); `grayscott × bam` hung, emitting no output
and dying on the timeout at both 120 s and 240 s. Undiagnosed, and now moot.

### The NCCL holes are missing code, not missing capability

`—` above means there is no `clio_<wl>_nccl_bench.cc` at all: lammps_md is
the only workload with an NCCL edition in the tree. That is not because the
others cannot use NCCL — every one of their MPI editions is built from
collectives with a direct NCCL equivalent:

| workload | MPI primitives used | NCCL equivalent |
|---|---|---|
| kmeans / weights / gmx | Allreduce, Bcast | `ncclAllReduce`, `ncclBroadcast` |
| grayscott | + Sendrecv (halo) | + `ncclSend`/`ncclRecv` |
| lbann | + Allgather | + `ncclAllGather` |
| lammps_md | Allreduce, Sendrecv | already implemented |

So five NCCL editions are writable against the existing MPI siblings.

NVSHMEM runs 2 real PEs on 2 GPUs (`PEs=2` in the bench output), not the
single-PE degenerate case — see the bootstrap note below for why that
distinction needed work.

**`grayscott × bam` HANGS.** It emits no output at all and is killed by
`timeout_sec`; reproduced at 120 s and at 240 s, while the other five BaM
cells finish in 1.4–143 s. Not diagnosed. Every other substrate for
grayscott passes, so this is specific to the BaM edition.

### Getting the comm substrates to run

Three separate things had to be true, and none is the default:

1. **nvcc needs gcc ≤ 13.** The mpi/nccl/nvshmem benches are NOT clang
   targets — each workload's CMakeLists builds them with a custom nvcc
   command, since a baseline that linked clio would not be a baseline. nvcc
   12.6 rejects the gcc-toolset-14 host compiler outright. `env.sh` sets
   `NVCC_PREPEND_FLAGS=-ccbin <gcc-13>`, which reaches every nvcc invocation
   without touching six CMakeLists.

2. **The MPI must be OpenMPI, not cray-mpich.** NVSHMEM's MPI bootstrap is a
   dlopen'd plugin NVIDIA built against OpenMPI and hard-requires
   `libmpi.so.40`; under cray-mpich it fails with *"Bootstrap unable to load
   'nvshmem_bootstrap_mpi.so.3' — libmpi.so.40: cannot open shared object"*.
   The benches' non-MPI path is single-PE by construction
   (`nvshmemx_set_attr_uniqueid_args(0, 1, ...)`), so it cannot answer
   whether NVSHMEM communicates at all. The baselines therefore build and
   run against the HPC-X OpenMPI inside nvhpc — the MPI that NVSHMEM and
   NCCL ship alongside. Use the concrete `hpcx-2.50/ompi4` path, not
   `comm_libs/hpcx`, whose dispatcher resolves by CUDA driver version and
   fails on a login node that has no driver.

3. **OpenMPI must not bind processes.** Inside a Slurm cgroup cpuset,
   `mpirun`'s own affinity fails with *"hwloc_set_cpubind returned Error for
   bitmap 0"* and the app never starts. The launcher needs `--bind-to none`
   so Slurm owns affinity. This is why `clio_gv_workload` gained an
   `mpi_launcher` option: its hardcoded `mpirun -n N --oversubscribe` is
   both OpenMPI-specific and unbindable here.

(For the record, `srun -n {n}` with cray-mpich also works for **mpi** and
**nccl** — measured 6/6 and 1/1. It is only NVSHMEM that forces OpenMPI, and
running the whole matrix on one MPI is worth more than mixing two.)

## Substrate comparison: kmeans at 32 GB

Driven by `clio_gv_kmeans` (one package per benchmark) through
`gv_kmeans_substrate_a100.yaml`, at MATCHED geometry -- same problem, same
blocks, same threads, `--repeat 1`, `iters=20` -- on one A100.

ms per iteration:

| blocks | paged | mpi | nvshmem | paged/mpi |
|---|---|---|---|---|
| 32 | 7740.5 | 6988.0 | 6741.1 | 1.11 |
| **64** | **6160.3** | **6142.1** | 5941.1 | **1.003** |
| 128 | 11708.7 | 6528.3 | 6441.0 | 1.79 |
| 256 | 8940.9 | 5801.2 | 5676.4 | 1.54 |
| 512 | 7190.5 | 5996.7 | 5723.7 | 1.20 |

**At 64 blocks the paged vector is within 0.3% of MPI**; best-to-best it is
6% (6160 vs 5801). Earlier figures of 3.9x / 18.9x / 40x came from an
UNCONTROLLED comparison in which the paged side was pinned to 32 blocks by
a sweep ladder while each baseline used its own default (the MPI editions
default to `blocks=64, threads=256`). Those numbers were launch geometry,
not the vector. That is the whole reason `clio_gv_workload` was replaced
here.

### Why paged degrades above 108 blocks: wave quantization

The paged coroutine kernels measure REG=192, so on a 65536-register SM only
`65536/(192*256) = 1` block per SM is resident -- 108 blocks on this GPU,
12.5% occupancy. Asking for more does not raise occupancy, it adds WAVES,
and a partial final wave idles most of the device. Total work is fixed, so
time should scale as `waves/blocks`:

| blocks | waves | waves/blocks (norm 512) | measured (norm 512) |
|---|---|---|---|
| 128 | 2 | 1.600 | 1.628 |
| 256 | 3 | 1.200 | 1.243 |
| 512 | 5 | 1.000 | 1.000 |

Within 3.6% -- that is the entire >108 curve, including why 128 blocks is
the WORST point (its second wave carries 20 blocks on 108 SMs) and why it
recovers monotonically to 512. Faults are not the driver: they FALL
(4439 -> 3144 -> 2256) while time falls, at ~0.7% of page touches.

MPI shows no such effect because REG=32 gives it 8 blocks/SM = 864
resident, so 128-512 blocks all fit one wave. Same workload, same geometry;
the difference is register pressure.

Below 108 blocks something else dominates -- 64 blocks (59% of SMs) BEATS
108 (100% of SMs) with zero faults at both, which the wave model does not
explain. For k-means the likely cause is contention on the shared 16
centroids x 32 dims reduction, but that is unconfirmed.

### Register pressure, and what `__launch_bounds__` does about it

`cmake/ClioCoroRegCap.cmake` explains the mechanism: NVPTX has no tail
calls, so CoroSplit merges every resume segment into one function and the
register allocator takes the LIVENESS UNION across all suspend points. The
count is therefore near-independent of the kernel body.

`clio_lammps_md_paged_bench` is the only paged bench at 64 registers, and
the only one whose kernels carry `__launch_bounds__` (md_common.h:
`MD_LB_THREADS=256`, `MD_LB_BLOCKS=4`). An A/B on that single TU, with the
regcap pass stripped so the annotation is isolated:

```
MD_LB_BLOCKS=4  (__launch_bounds__(256,4))  ->  REG=64    STACK:16  LOCAL:0
MD_LB_BLOCKS=0  (compiled out)              ->  REG=192   STACK:16  LOCAL:0
```

So 192 is what the coroutine lowering produces for BOTH workloads, and the
annotation is what cuts it -- to exactly `65536/(256*4) = 64`, with no
local-memory spill. It is not a guarantee: `__launch_bounds__` lowers to
`.maxntid`/`.minnctapersm` and ptxas may exceed the request, and the cap is
architecture-relative (it fixes an OCCUPANCY target, which is 64 registers
only on a 65536-register SM).

`clio_coro_regcap()` is the measured belt-and-braces version, also wired
only to lammps_md. It is a silent no-op unless `llvm-config` is findable,
which it was not here until env.sh started exporting
`CLIO_DELTA_LLVM_CONFIG` -- clang is referenced by absolute path precisely
so LLVM's bin stays off PATH.

Untried: `__launch_bounds__(256, 4)` on the other five paged benches.
Registers are verifiable statically in seconds; whether it makes anything
FASTER is a separate question, since kmeans at 32 GB looks
bandwidth-bound (all three substrates converge to 5700-6200 ms/iter at
their best). The confident prediction is only that it removes the
128-block cliff, by taking residency from 108 to 432.

### Checksums are block-count sensitive on EVERY substrate

| blocks | mpi | nvshmem | paged |
|---|---|---|---|
| 32 | 5601.878 | 5744.866 | 5841.250 |
| 64 | 5744.866 | 5744.866 | 5756.924 |
| 128 | 5799.815 | 5744.866 | 5809.408 |
| 256 | 6267.482 | 6085.158 | 5807.762 |
| 512 | 5718.794 | 6020.632 | 5858.314 |

MPI varies as much as anything else (5602-6267, +-5.6%), so this is NOT an
NVSHMEM defect as an earlier partial dataset suggested. Reduction order and
seeding evidently depend on block count. Until that is understood,
`centroid_checksum` cannot gate correctness ACROSS block counts -- only
within one.

## Status

Built and run against clang 19.1.7 / CUDA 12.6.3 / sm_80, on an
A100-SXM4-40GB (driver 595.71.05), Delta partition `gpuA100x4`.

All six paged benchmarks and all eleven `test_gpu_vector_*` targets compile.
The smoke sweep passes its gates on all six workloads at the 6 GB anchors:

| workload | bench_ms | VRAM peak | faults |
|---|---|---|---|
| kmeans | 3293 | 6941 MB | 0 |
| weights | 792 | 7041 MB | 5984 |
| grayscott | 2907 | 6943 MB | 3000 |
| gmx | 1123 | 6843 MB | — |
| lbann | (21.0 s/step) | 6579 MB | — |
| lammps_md | 85840 (4.29 s/step) | 7077 MB | 1648 |

(cache_level 4, blocks_level 3 = 32 blocks; job 21582692 on gpua006.)

Source changes needed to get here: `benchmark/lammps_md/md_kernels.h`
called `__ldcg`/`__ldcv`, which clang's CUDA headers do not provide —
`__clang_cuda_runtime_wrapper.h` deliberately skips `sm_32_intrinsics.h`, so
the calls resolved only against the `__half`/`__nv_bfloat16` overloads
`cuda_fp16.hpp` brings in and failed. The two loads are now emitted as
`ld.global.cg.f32` / `ld.global.cv.f32` directly (`ProbeLoadCG`/`ProbeLoadCV`
in that header).

`clio_lammps_md_mpi_bench.cc` and `clio_lammps_md_nccl_bench.cc` both used
`std::string` without including `<string>` — latent until nvcc compiled them
against gcc-13's libstdc++, where it is no longer pulled in transitively.
