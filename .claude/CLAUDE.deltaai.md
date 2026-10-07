# CLAUDE.md — DeltaAI Agent Instructions

> For Claude Code, Codex, Gemini CLI, or any AI coding agent working with DeltaAI.
> Project: CIS250329 (IOWarp) | Slurm account: `bekn-dtai-gh`

---

## Architecture: This is NOT a normal Linux box

DeltaAI runs **aarch64 ARM** (NVIDIA Grace CPUs) on **SLES 15.6** with **64KB memory pages**.

**What this means for you:**
- x86 binaries fail silently or crash with "Exec format error"
- jemalloc-based tools (rattler-build, some Rust CLI tools) crash: `Unsupported system page size`
- `$OSTYPE` is `linux`, not `linux-gnu` — many detection scripts break
- System GCC is 7.5 (ancient). **Always use `/usr/bin/gcc-13` and `/usr/bin/g++-13`**
- Claude Code native binary does not work here (x86 only)

## Access Pattern

You **cannot SSH programmatically** to DeltaAI. Auth requires interactive NCSA password + Duo MFA.

**If operating from a local workstation (recommended):**
1. Human SSHes to DeltaAI and starts `tmux`
2. Agent sends commands via `tmux send-keys -t SESSION_NAME "command" Enter`
3. Agent reads output via `tmux capture-pane -t SESSION_NAME -p`

**Monitoring pattern (don't sleep randomly):**
```bash
while true; do
  out=$(tmux capture-pane -t SESSION -p | tail -3)
  echo "$out" | grep -qE 'username@.*>$' && break
  sleep 3
done
```

## Slurm Commands

```bash
# Interactive GPU session
srun --account=bekn-dtai-gh --partition=ghx4-interactive \
  --nodes=1 --gpus-per-node=1 --cpus-per-task=16 \
  --mem=64G --time=00:30:00 --pty bash

# Batch job
sbatch --account=bekn-dtai-gh --partition=ghx4 job.slurm

# NEVER use mpirun. Only srun.
# Interactive costs 2x. Prefer batch for real work.
```

## Compiler Selection

```bash
# WRONG — GCC 7.5, no C++17 support
cmake -B build

# RIGHT — GCC 13, full modern C++ support
cmake -DCMAKE_C_COMPILER=/usr/bin/gcc-13 \
      -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
      -B build
```

The Cray compiler wrappers (`cc`/`CC`) exist but CMake doesn't recognize `CC` as a C++ compiler. Use gcc-13/g++-13 directly.

## Conda is the Package Manager

No apt/yum for user packages. Use conda:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ENV_NAME
conda install -c conda-forge PACKAGE
```

When building C++ projects with conda deps, you MUST pass include/lib paths explicitly:
```bash
cmake \
  -DCMAKE_C_FLAGS="-I$CONDA_PREFIX/include" \
  -DCMAKE_CXX_FLAGS="-I$CONDA_PREFIX/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$CONDA_PREFIX/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L$CONDA_PREFIX/lib" \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
  ...
```

## Storage Rules

| Path | Use |
|------|-----|
| `/u/$USER` | Small configs only (100GB quota) |
| `/work/hdd/bekn/$USER/` | Primary workspace for everything |
| `/tmp` | Burst buffer, purged after job ends |

Never build in HOME. Always use `/work/hdd/bekn/$USER/`.

## Known Gotchas

1. **msgpack cmake naming** — conda `msgpack-cxx` provides `msgpack-cxx-config.cmake` but many projects `find_package(msgpack)` expecting `msgpackConfig.cmake`. Create symlinks:
   ```bash
   mkdir -p $CONDA_PREFIX/lib/cmake/msgpack
   ln -sf $CONDA_PREFIX/lib/cmake/msgpack-cxx/msgpack-cxx-config.cmake \
     $CONDA_PREFIX/lib/cmake/msgpack/msgpackConfig.cmake
   ln -sf $CONDA_PREFIX/lib/cmake/msgpack-cxx/msgpack-cxx-config-version.cmake \
     $CONDA_PREFIX/lib/cmake/msgpack/msgpackConfigVersion.cmake
   ln -sf $CONDA_PREFIX/lib/cmake/msgpack-cxx/msgpack-cxx-targets.cmake \
     $CONDA_PREFIX/lib/cmake/msgpack/msgpack-cxx-targets.cmake
   ```

2. **Module system differs between login and compute nodes** — `module spider` results may vary. Always verify on the node type where you'll build.

3. **No io_uring** — SLES 15.6 kernel may not support it. Disable with `-DCLIO_CORE_ENABLE_IO_URING=OFF`.

4. **Ninja from conda** — system cmake is 3.20 (old). Install cmake + ninja from conda for better compatibility.

5. **Reservation flag** — During system updates, jobs may need `--reservation=update`. Check MOTD on login.

## IOWarp Clio Core

Already built and installed at `$CONDA_PREFIX/iowarp_core/` in the `iowarp` conda environment.

Source: `~/projects/iowarp/clio-core/`
Build: `~/projects/iowarp/clio-core/build/`

To rebuild after changes:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate iowarp
cd ~/projects/iowarp/clio-core
cmake --build build -j16
cmake --install build
```

## GPU Info

- NVIDIA GH200 120GB per superchip
- 4 superchips per node (4 GPUs)
- CUDA 12.8, Driver 570.172
- SM architecture: 9.0 (Hopper)
- Use `nvidia-smi` on compute nodes (no GPUs on login nodes)
