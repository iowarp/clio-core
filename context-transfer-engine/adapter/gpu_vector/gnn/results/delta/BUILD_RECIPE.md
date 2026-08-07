# Exact build + run recipe (NCSA Delta, A100-SXM4-40GB, sm_80 native)

Everything below was actually executed. Paths are literal.

## 0. Environment facts you must know first

* `dt-login01.delta.ncsa.illinois.edu` is a **login node with no GPU** (`nvidia-smi`
  fails). All GPU work requires a Slurm allocation.
* Delta A100 nodes (`gpua*`): 4x **A100-SXM4-40GB**, 64 cores, 257 GB RAM,
  **1.5 TB node-local NVMe at `/tmp`** (and `/local`). Outbound network works.
* Storage constraints that forced the layout used here:
  * `/u/rpawar` (home): 100 GB quota, ~64 GB free -> too small for papers100M.
  * `/projects/bekn`: **100% full, 0 bytes available**.
  * `/work/hdd/bekn`, `/work/nvme/bekn`: shared allocation quotas, ~250 GB headroom.
  * => papers100M derived data (~140 GB) lives on **node-local NVMe** `/tmp/gnn`;
    the 60 GB raw zip is cached on `/work/hdd/bekn/rpawar/ogbraw/` so a lost job
    does not force a re-download.

## 1. Repo

```bash
cd /u/rpawar/clio-core
git fetch origin gnn-lossless-zstd
git checkout -B gnn-lossless-zstd FETCH_HEAD     # a0e991f1e
git submodule update --init --recursive          # only 'docs'; not needed to build
```

### REQUIRED FIX: the branch does not configure as-is
`context-transfer-engine/adapter/gpu_vector/test/CMakeLists.txt` declares targets
`test_gpu_vector_kmeans_real` and `test_gpu_vector_kmeans_capacity` whose sources
(`test_gpu_vector_kmeans_real_gpu.cc`, `test_gpu_vector_kmeans_capacity_gpu.cc`)
**were never committed to any branch** (`git log --all --diff-filter=A` finds
nothing). CMake fails at generate time. Both target blocks are now wrapped in
`if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/<src>")`.

## 2. Toolchain: the project's apptainer image

`/u/rpawar/containers/deps-nvidia.sif` (10.5 GB) carries the documented CUDA build
deps: **nvcc 12.6.85, g++ 13.3.0, cmake 3.28.3, python 3.12 (numpy 2.4.6, pandas
3.0.3), libzstd**. It has **no torch / ogb / dgl** — hence the raw `.npz` is parsed
directly with numpy, and the DGL/PyG UVA baseline (task f) is not runnable here.

Two container gotchas:
* Delta's Cray PE exports `CC=cc` / `CXX=CC`; those wrappers do not exist inside the
  container and CMake's compiler detection fails. The build script does
  `unset CC CXX FC F77 F90 CFLAGS CXXFLAGS LDFLAGS` and pins `/usr/bin/{gcc,g++}`.
* `nvidia-smi` inside the container prints
  `Failed to initialize NVML: Driver/library version mismatch` (container NVML 595.71
  vs host driver 570.148.08). This is **cosmetic**: the CUDA runtime works. Verified
  with a real kernel — `cudaGetDeviceCount -> no error, n=1`,
  `device0: NVIDIA A100-SXM4-40GB sm_80 42.4 GB`, kernel result correct.

## 3. cuSZp V3.0.0 (lossy arm)

```bash
cd /u/rpawar/cuSZp
git fetch --tags origin
git checkout cuSZp-V3.0.0        # da90afb
```

```bash
cmake -S /u/rpawar/cuSZp -B /u/rpawar/cuSZp/build-a100 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=80 \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
      -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
      -DCMAKE_INSTALL_PREFIX=/u/rpawar/cuSZp/install
cmake --build /u/rpawar/cuSZp/build-a100 -j 16
cmake --install /u/rpawar/cuSZp/build-a100
```

**Two upstream quirks, reported for exact reproduction:**
1. `cmake/Installing.cmake:6` **FORCE-overrides** `CMAKE_INSTALL_PREFIX` to
   `<src>/install`. Passing `-DCMAKE_INSTALL_PREFIX=...install-a100` is silently
   ignored, so the real prefix is `/u/rpawar/cuSZp/install`.
2. `CMakeLists.txt:22` does `set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 75 80 86)`
   (plain `set`, directory scope), which **overrides** `-DCMAKE_CUDA_ARCHITECTURES=80`.
   The flag is effectively a no-op upstream. sm_80 **is** in that list, so the
   library still contains native sm_80 SASS — the intent ("native, no PTX-JIT") holds.

## 4. clio-core CUDA build

```bash
cmake -S /u/rpawar/clio-core -B /u/rpawar/clio-core/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCLIO_CORE_ENABLE_CUDA=ON \
      -DCLIO_CTE_ENABLE_COMPRESS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=80 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
      -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
      -DCLIO_CUSZP_ROOT=/u/rpawar/cuSZp/install
cmake --build /u/rpawar/clio-core/build --target test_gpu_vector_gnn_train    -j 16
cmake --build /u/rpawar/clio-core/build --target test_gpu_vector_gnn_capacity -j 16
```

**Native sm_80 verified** (not PTX-JIT), via `cuobjdump`:

```
test_gpu_vector_gnn_train:    arch = sm_80
test_gpu_vector_gnn_capacity: arch = sm_80
```

Driver script: `/u/rpawar/gnnbench/build.sh` (run it inside the container).

## 5. Source changes required to run papers100M at all

`test/test_gpu_vector_gnn_train_gpu.cc`:
* **`kMaxC` 64 -> 192.** ogbn-papers100M has **172 classes**; the stock
  `REQUIRE(C <= kMaxC)` aborts immediately. This is a hard blocker, not a tuning knob.
* Added a deterministic **train/val split** (every 10th node held out, index-based):
  validation nodes are scored but contribute neither loss nor gradient. Needed for
  the "final train+val acc" column and to quantify lossy accuracy loss. It preserves
  bit-exactness (verified: `max|dvacc| = 0` on arxiv).
* CSV now records dataset, codec lib/preset, val accuracies, store time and
  per-epoch times.

`gnn/gnn_prep.py`: new streaming **papers100M** path (npz -> bare .npy unpack so
arrays larger than RAM can be memory-mapped; chunked f32 conversion; an in-place
sort CSR builder for 1.6e9 directed edges that peaks at one 24.1 GiB key array).
Verified **byte-identical** to the original `build_csr` on 8 randomized graphs.

`gnn/gnn_agg.py`: rewritten chunked/mmap/multiprocess. The original is unrunnable at
this scale (`X.astype(np.float64)` = 114 GB; `X[indices]` ~ 1.6 TB; `np.add.at` over
3.2e9 edges). New version uses `np.add.reduceat` over nnz-bounded row blocks with
float64 accumulation. Verified **bit-identical** to the original on 6 randomized
graphs including empty-row / empty-graph edge cases.

`gnn/gnn_pagerank_cache.py`: accepts an `edge_index.npy` (`--edges`), and bounds the
trace (`--max-batches`, `--sweep-sample`, `--lru-limit`) — a full papers100M epoch is
~108k minibatches and the LRU is a Python loop over 3.2e9 accesses. Defaults are
unchanged, and the arxiv run **reproduces the previously published numbers exactly**
(reverse-PR L1 = 2.09e-06, top-10% capture 21.7%, 13.0% fewer decompressions).

## 6. Data

```bash
# arxiv / products (small; kept on $HOME)
python3 gnn/gnn_prep.py --dataset arxiv    --data-root /u/rpawar/gnnbench/data
python3 gnn/gnn_prep.py --dataset products --data-root /u/rpawar/gnnbench/data
python3 gnn/gnn_agg.py  --data /u/rpawar/gnnbench/data/arxiv    --workers 4
python3 gnn/gnn_agg.py  --data /u/rpawar/gnnbench/data/products --workers 6

# papers100M: 60.3 GB raw. snap.stanford.edu throttles a single connection to
# ~3 MB/s; 16 parallel range requests reach ~57 MB/s (19x faster).
aria2c -x16 -s16 -k10M -c -d /work/hdd/bekn/rpawar/ogbraw -o papers100M-bin.zip \
  http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip
# then, ON the compute node (node-local NVMe):
bash /u/rpawar/gnnbench/prep_papers.sh      # prep + gnn_agg, ~35 min total
```

Observed dataset facts (all match the published OGB statistics):

| dataset | N | F | undirected E | C | features.f32 | agg_features.f32 |
|---|---:|---:|---:|---:|---:|---:|
| ogbn-arxiv | 169,343 | 128 | 2,315,598 | 40 | 82.7 MiB | 82.7 MiB |
| ogbn-products | 2,449,029 | 100 | 123,718,152 | 47 | 934.2 MiB | 934.2 MiB |
| ogbn-papers100M | 111,059,956 | 128 | **3,228,124,712** | **172** | **56.86 GB** | **56.86 GB** |

papers100M: directed E = 1,615,685,872 -> symmetrised 3,231,371,744 -> deduped
3,228,124,712 (0.10% reciprocal/self pairs removed), avg degree 29.07.
Only **1,546,782 / 111,059,956 nodes (1.39%) are labelled**.

## 7. Running

```bash
# hold a node, then srun --overlap into it (see rj.sh)
sbatch /u/rpawar/gnnbench/holder.sbatch          # 1 GPU, 32 c, 224 GB, gpuA100x4
JOB=<jobid> ./rj.sh bash /u/rpawar/gnnbench/run_exp.sh a       # arxiv correctness
JOB=<jobid> EPOCHS_P=30 ./rj.sh bash run_exp.sh b              # papers100M headline
```

Per-binary runtime env (from the build dir's `bin/`):
`LD_LIBRARY_PATH=. CLIO_REPO_PATH=. CLIO_BIND_ADDR=127.0.0.1 CLIO_PORT=<unique>`.

**`CLIO_GNN_DRAM_MIB` was set to 140000, not 200000.** The training test holds a
**57 GB host copy of A** *plus* the compressed DRAM tier (~50 GB); with a 224 GB
cgroup limit, 200000 MiB risks OOM-killing the job. 140 GB comfortably exceeds the
~50 GB actually stored.
