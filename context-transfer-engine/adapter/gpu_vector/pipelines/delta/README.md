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

## Substrates

All 25 (workload × substrate) binaries build, and every one that exists runs
on Delta. Measured by `gv_variant_matrix_a100.yaml` on 2 A100s (job
21606871):

| workload | paged | mpi | nvshmem | bam | nccl |
|---|---|---|---|---|---|
| kmeans | ✅ | ✅ | ✅ | ✅ | — |
| weights | ✅ | ✅ | ✅ | ✅ | — |
| grayscott | ✅ | ✅ | ✅ | **hangs** | — |
| gmx | ✅ | ✅ | ✅ | ✅ | — |
| lbann | ✅ | ✅ | ✅ | ✅ | — |
| lammps_md | ✅ | ✅ | ✅ | ✅ | ✅ |

`—` is by design: NCCL has an edition only for lammps_md.

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
