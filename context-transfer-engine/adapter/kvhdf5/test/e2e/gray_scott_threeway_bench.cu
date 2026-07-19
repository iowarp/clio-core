#ifndef _GNU_SOURCE
#define _GNU_SOURCE  // expose O_DIRECT from <fcntl.h> (raw-arm cache-bypass parity)
#endif
/*
 * THREE-WAY Gray-Scott I/O benchmark (the paper figure the advisor asked for):
 * ONE shared computation, THREE storage paths, wall-clock compared:
 *
 *   - raw   : no CLIO. GPU compute -> D2H to pinned host -> pwrite + fsync to disk.
 *   - sync  : CLIO, fused submit-AND-WAIT snapshot (GPU blocks on each PutBlob).
 *   - async : CLIO, fire-all snapshot + drain-at-end (server I/O overlaps GPU compute).
 *
 * SAME COMPUTATION across all three by construction: every arm runs the identical
 * `GsStepKernel` (one thread per cell) through the identical timed `RunSim` loop;
 * only the per-snapshot "sink" differs. So the comparison isolates the I/O + storage
 * backend, not the math.
 *
 * ROUTING AROUND THE iowarp ~16-large-backend ceiling (ADVISOR-REPORT §6a; the limit
 * is by COUNT not bytes, and dataset REUSE hangs — see refire probe): each arm reaches
 * the ~2 GB target with FEWER, BIGGER snapshots (<=~12 distinct datasets, each large),
 * keeping the proven fresh-dataset-per-snapshot recipe. The sync and async arms run in
 * SEPARATE processes (one arm per TEST_CASE invocation), so neither exceeds the ceiling.
 *
 * FAIR STORAGE: the CLIO arms can target a kFile bdev (O_DIRECT to a real file, cache-
 * bypassing) to match the raw arm's disk, or a kRam bdev for a RAM baseline — selected
 * by GSBENCH_BDEV. The raw arm always writes real files + fsync.
 *
 * All knobs are ENV VARS so scaling to 2 GB / switching to disk needs NO recompile:
 *   GSBENCH_N            grid dim (NxN float32)              default 512
 *   GSBENCH_CHUNKS       chunks per snapshot dataset         default 4
 *   GSBENCH_SNAPS        number of snapshots (<= ~12!)       default 4
 *   GSBENCH_STEPS_PER    sim steps between snapshots         default 8
 *   GSBENCH_BDEV         ram | file  (CLIO arms)             default ram
 *   GSBENCH_BDEV_CAP_MB  bdev capacity (MB)                  default 512
 *   GSBENCH_BDEV_PATH    kFile path (CLIO arms)              default ./gsbench_bdev.dat
 *   GSBENCH_DISK_DIR     raw-arm output dir                  default ./gsbench_raw_out
 *
 * Each arm prints ONE machine-parseable RESULT line (ms, MB, MB/s, checksum). A wrapper
 * runs all three processes and builds the relative table; the shared checksum proves all
 * three computed identical bytes. Cases HIDDEN ([.]); CLIO arms RUN AT num_threads=1.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/singletons.h>
#include <clio_ctp/util/gpu_api.h>

#include <clio_cte/kvhdf5/cpu_dataset.h>      // Layout
#include <clio_cte/kvhdf5/gpu_cte_dataset.h>
#include <clio_cte/kvhdf5/tag_path.h>         // CanonicalTag

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if !CTP_IS_DEVICE_PASS
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/types.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <catch2/catch_test_macros.hpp>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#ifndef O_DIRECT
#define O_DIRECT 040000  // Linux x86/arm value; fallback if _GNU_SOURCE didn't expose it
#endif
#endif

using kvhdf5::byte_t;

namespace {
struct GsParams { float Du, Dv, F, k, dt; };
}  // namespace

// ---- shared computation: identical for every arm ---------------------------

// One Gray-Scott step, one thread per cell, periodic BCs. Pure CUDA.
__global__ void GsStepKernel(const float* u, const float* v, float* un, float* vn,
                             GsParams p, unsigned N) {
    unsigned cells = N * N;
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= cells) return;
    unsigned x = gid % N, y = gid / N;
    unsigned xm = (x == 0) ? (N - 1) : (x - 1);
    unsigned xp = (x == N - 1) ? 0u : (x + 1);
    unsigned ym = (y == 0) ? (N - 1) : (y - 1);
    unsigned yp = (y == N - 1) ? 0u : (y + 1);
    float uc = u[gid], vc = v[gid];
    float lap_u = u[y*N+xm] + u[y*N+xp] + u[ym*N+x] + u[yp*N+x] - 4.f*uc;
    float lap_v = v[y*N+xm] + v[y*N+xp] + v[ym*N+x] + v[yp*N+x] - 4.f*vc;
    float uvv = uc * vc * vc;
    un[gid] = uc + p.dt * (p.Du * lap_u - uvv + p.F * (1.f - uc));
    vn[gid] = vc + p.dt * (p.Dv * lap_v + uvv - (p.F + p.k) * vc);
}

// ---- CLIO snapshot kernels (device-facing handle) --------------------------

// Bulk stage a snapshot into its registered per-chunk device backends with a FULL grid
// (gridDim.y = chunk, gridDim.x = blocks/chunk), saturating HBM. This is split OUT of the
// submit kernels: the old design fused the copy into the single <<<1,256>>> CLIO-producer
// block, so 156 MB moved at ~1 GB/s (~160 ms/snapshot) and dominated BOTH arms — dwarfing
// the actual compute and masking any async advantage. `src` is the flat masked grid; chunk
// c's bytes are src[c*size .. c*size+size). Word-wise (float-grid sizes are 4-byte
// multiples). Being its OWN completed kernel gives kernel-boundary ordering, so the staged
// writes are visible to the subsequent submit kernel and the server's readback — no
// __threadfence_system needed (that fence was only for the old same-kernel copy+enqueue).
__global__ void TwCopyKernel(kvhdf5::GpuDatasetHandle h, const byte_t* src) {
    uint32_t c = blockIdx.y;
    uint64_t n = h.Size(c);
    const uint32_t* s = reinterpret_cast<const uint32_t*>(src + uint64_t(c) * n);
    uint32_t* d = reinterpret_cast<uint32_t*>(h.Data(c));
    uint64_t words = n >> 2;
    uint64_t gid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    for (uint64_t i = gid; i < words; i += stride) d[i] = s[i];
}

// SYNC submit: per chunk, fused Write(c) = submit-AND-WAIT (GPU blocks). Data already
// staged by TwCopyKernel. Single block (CLIO producer guard: thread-0 enqueues).
__global__ void TwSnapSyncKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = 0; c < h.Count(); ++c) {
        h.Write(c);
        __syncthreads();
    }
}

// ASYNC FIRE only: fire every chunk's PutBlob, no wait. Data already staged by
// TwCopyKernel. Drained later so the puts run on the server WHILE the subsequent sim
// steps run on the GPU.
__global__ void TwSnapFireKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = 0; c < h.Count(); ++c) {
        h.WriteAsync(c);
        __syncthreads();
    }
}

// Drain a fired snapshot's outstanding puts.
__global__ void TwDrainKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = 0; c < h.Count(); ++c) { h.WriteWait(c); __syncthreads(); }
}

// XOR the snapshot with a fixed random mask (word-wise) into a scratch buffer, so the
// PERSISTED bytes are high-entropy / incompressible — otherwise the Gray-Scott field is
// mostly zeros and a compressing filesystem (e.g. btrfs zstd) makes the "disk I/O" nearly
// free, voiding any disk comparison. Deterministic (fixed mask) => byte-identical across
// arms => checksums still match.
__global__ void MaskKernel(uint32_t* dst, const uint32_t* src, const uint32_t* mask,
                           unsigned words) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < words) dst[i] = src[i] ^ mask[i];
}

#if !CTP_IS_DEVICE_PASS

namespace {

constexpr GsParams kGs{0.16f, 0.08f, 0.055f, 0.062f, 1.0f};

// ---- env-var config --------------------------------------------------------

unsigned EnvU(const char* k, unsigned dflt) {
    const char* v = std::getenv(k);
    if (!v || !*v) return dflt;
    long x = std::strtol(v, nullptr, 10);
    return x > 0 ? static_cast<unsigned>(x) : dflt;
}
std::string EnvS(const char* k, const char* dflt) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::string(v) : std::string(dflt);
}

struct Cfg {
    unsigned N       = EnvU("GSBENCH_N", 512);
    unsigned chunks  = EnvU("GSBENCH_CHUNKS", 4);
    unsigned snaps   = EnvU("GSBENCH_SNAPS", 4);
    unsigned steps_per = EnvU("GSBENCH_STEPS_PER", 8);
    std::string bdev = EnvS("GSBENCH_BDEV", "ram");
    unsigned cap_mb  = EnvU("GSBENCH_BDEV_CAP_MB", 512);
    std::string bdev_path = EnvS("GSBENCH_BDEV_PATH", "./gsbench_bdev.dat");
    std::string disk_dir  = EnvS("GSBENCH_DISK_DIR", "./gsbench_raw_out");
    // Raw arm O_DIRECT (cache-bypass) for the disk comparison; set 0 for buffered writes
    // when the raw target is a RAM tier (tmpfs, which rejects O_DIRECT) — the fair
    // software-path comparison vs CLIO's kRam bdev.
    unsigned raw_odirect  = EnvU("GSBENCH_RAW_ODIRECT", 1);
    // Raw arm fsync per snapshot: DURABILITY PARITY with CLIO (whose bdev waits for each
    // write to commit). Without it, a buffered raw write just lands in RAM/page-cache and
    // isn't really persisted — doing less work than CLIO. Default on.
    unsigned raw_fsync    = EnvU("GSBENCH_RAW_FSYNC", 1);
    // XOR each snapshot with a random mask so persisted bytes are incompressible (else the
    // mostly-zero Gray-Scott field compresses away on btrfs zstd, voiding disk comparisons).
    unsigned incompressible = EnvU("GSBENCH_INCOMPRESSIBLE", 1);
    // Raw writer structure: 0 = background thread (I/O overlaps compute); 1 = inline
    // synchronous (GPU idle during the write, matching host-CLIO / sync-CLIO). Use inline
    // for a storage-path comparison free of this box's GPU-concurrent-I/O throttle.
    unsigned raw_inline   = EnvU("GSBENCH_RAW_INLINE", 0);
    // Place the CLIO snapshot DATA backend in pinned host memory (kPinnedHost)
    // instead of on-GPU (kDeviceMem). The in-process bdev server's device->host
    // readback does not overlap the producer's compute (its first per-run D2H
    // stalls for the whole compute window), so with kDeviceMem the async drain
    // serializes after compute and barely beats sync. Pinned data removes the
    // server D2H, letting disk writes pipeline under compute: at steps_per=192,
    // N=6400 async goes ~836 -> ~955 MB/s. Trade-off: the producer kernel writes
    // mapped host over PCIe (slower submit) and it regresses the disk-bound /
    // low-compute regime, so it is off by default.
    unsigned data_pinned  = EnvU("GSBENCH_DATA_PINNED", 0);

    unsigned steps() const { return snaps * steps_per; }
    uint64_t cells() const { return uint64_t(N) * N; }
    uint64_t grid_bytes() const { return cells() * sizeof(float); }
    // total bytes persisted by the whole run (one v-grid per snapshot).
    uint64_t total_bytes() const { return grid_bytes() * snaps; }
};

// ---- shared sim scaffolding ------------------------------------------------

struct Grids { float *u_curr, *u_next, *v_curr, *v_next; };
Grids MakeGrids(unsigned N) {
    uint64_t cells = uint64_t(N) * N, bytes = cells * sizeof(float);
    Grids g{};
    REQUIRE(cudaMalloc(&g.u_curr, bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&g.u_next, bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&g.v_curr, bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&g.v_next, bytes) == cudaSuccess);
    std::vector<float> u0(cells, 1.0f), v0(cells, 0.0f);
    unsigned lo = N/2 - 3, hi = N/2 + 3;
    for (unsigned y = lo; y < hi; ++y)
        for (unsigned x = lo; x < hi; ++x) v0[uint64_t(y)*N + x] = 1.0f;
    ctp::GpuApi::Memcpy(g.u_curr, u0.data(), bytes);
    ctp::GpuApi::Memcpy(g.v_curr, v0.data(), bytes);
    return g;
}
void FreeGrids(Grids& g) {
    cudaFree(g.u_curr); cudaFree(g.u_next); cudaFree(g.v_curr); cudaFree(g.v_next);
}

// Makes each snapshot INCOMPRESSIBLE: Apply(v) XORs the grid with a fixed random mask into
// a scratch device buffer and returns it, so the persisted bytes are high-entropy (the
// mostly-zero Gray-Scott field would otherwise compress away on btrfs zstd). Mask is fixed
// (seed) => byte-identical across arms => checksums still match. If disabled, Apply is a
// pass-through. One shared scratch is safe: the MaskKernel and each arm's subsequent
// copy/D2H are serialized on the default stream, so scratch is consumed before the next
// Apply overwrites it.
struct Masker {
    bool on_;
    unsigned cells_;
    uint32_t* d_mask_ = nullptr;
    uint32_t* d_scratch_ = nullptr;
    Masker(unsigned N, bool on) : on_(on), cells_(N * N) {
        if (!on_) return;
        uint64_t bytes = uint64_t(cells_) * sizeof(uint32_t);
        REQUIRE(cudaMalloc(&d_mask_, bytes) == cudaSuccess);
        REQUIRE(cudaMalloc(&d_scratch_, bytes) == cudaSuccess);
        std::vector<uint32_t> mask(cells_);
        std::mt19937 rng(0xC0FFEEu);            // fixed => identical across arms
        for (auto& x : mask) x = rng();
        ctp::GpuApi::Memcpy(d_mask_, mask.data(), bytes);
    }
    ~Masker() { if (on_) { cudaFree(d_mask_); cudaFree(d_scratch_); } }
    // Returns an incompressible view of v (scratch), or v itself if disabled.
    float* Apply(float* v) {
        if (!on_) return v;
        unsigned t = 256, b = (cells_ + t - 1) / t;
        MaskKernel<<<b, t>>>(d_scratch_, reinterpret_cast<uint32_t*>(v), d_mask_, cells_);
        return reinterpret_cast<float*>(d_scratch_);
    }
};

// FNV-1a over a host byte buffer (cross-arm "identical computation" proof).
uint64_t Fnv1a(const void* data, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

// GPU-timeline phase tracer (GSBENCH_TRACE=1), used by the sync & async CLIO arms to
// answer the advisor's question: is async's I/O actually OVERLAPPING compute, or is
// compute so cheap there's nothing to overlap? It splits each snapshot interval on the
// GPU's own timeline into:
//   compute_ms  — the steps_per GsStepKernels leading up to the snapshot,
//   submit_ms   — the snapshot's mask + PutBlob-task emplace kernel (async: fire only;
//                 sync: fire-AND-device-wait, so sync's submit bucket carries the wait),
//   drain_ms    — (async only) the single TAIL TwDrainKernel spin-wait at the very end.
// So the hypothesis reads directly off the buckets: async should have near-zero submit_ms
// and a drain_ms that is SMALL vs total compute if I/O kept up (good overlap), or LARGE if
// the server fell behind (backlog => async collapses toward sync). sync has no drain; its
// per-snapshot wait is inside submit_ms.
//
// Mechanics: everything runs on the default stream, so cudaEvents recorded at the phase
// boundaries measure GPU-serialized work in order. We ONLY cudaEventRecord in the hot loop
// (async, ~1us, no stall) and read every cudaEventElapsedTime AFTER the closing Synchronize
// — a mid-loop cudaEventSynchronize would serialize the stream and destroy the overlap we
// are measuring. NOTE: submit_ms includes the per-snapshot MaskKernel (identical work in
// both arms, so it cancels in the sync-vs-async comparison).
struct PhaseTrace {
    bool on_;
    unsigned snaps_;
    bool has_drain_;
    std::vector<cudaEvent_t> cs_, ce_, se_;   // compute-start, compute-end(=submit-start), submit-end
    cudaEvent_t ds_ = nullptr, de_ = nullptr; // drain-start, drain-end (async only)
    PhaseTrace(bool on, unsigned snaps, bool async)
        : on_(on), snaps_(snaps), has_drain_(async) {
        if (!on_) return;
        cs_.resize(snaps); ce_.resize(snaps); se_.resize(snaps);
        for (unsigned i = 0; i < snaps; ++i) {
            cudaEventCreate(&cs_[i]); cudaEventCreate(&ce_[i]); cudaEventCreate(&se_[i]);
        }
        if (has_drain_) { cudaEventCreate(&ds_); cudaEventCreate(&de_); }
    }
    ~PhaseTrace() {
        if (!on_) return;
        for (auto e : cs_) cudaEventDestroy(e);
        for (auto e : ce_) cudaEventDestroy(e);
        for (auto e : se_) cudaEventDestroy(e);
        if (has_drain_) { cudaEventDestroy(ds_); cudaEventDestroy(de_); }
    }
    void CompStart(unsigned si)  { if (on_) cudaEventRecord(cs_[si]); }
    void CompEnd(unsigned si)    { if (on_) cudaEventRecord(ce_[si]); }
    void SubmitEnd(unsigned si)  { if (on_) cudaEventRecord(se_[si]); }
    void DrainStart()            { if (on_ && has_drain_) cudaEventRecord(ds_); }
    void DrainEnd()              { if (on_ && has_drain_) cudaEventRecord(de_); }
    // Read the deltas + print the per-snapshot series and totals. MUST be called after the
    // caller's RunSim returns (i.e. after the closing Synchronize) so every event is done.
    void Report(const char* arm) {
        if (!on_) return;
        double tot_c = 0, tot_s = 0;
        std::fprintf(stderr, "GSBENCH_TRACE arm=%s per-snapshot (GPU-timeline ms):\n", arm);
        for (unsigned i = 0; i < snaps_; ++i) {
            float c = 0, s = 0;
            cudaEventElapsedTime(&c, cs_[i], ce_[i]);
            cudaEventElapsedTime(&s, ce_[i], se_[i]);
            tot_c += c; tot_s += s;
            std::fprintf(stderr, "GSBENCH_TRACE   snap=%2u compute_ms=%8.3f submit_ms=%8.3f\n",
                         i, c, s);
        }
        float drain = 0;
        if (has_drain_) cudaEventElapsedTime(&drain, ds_, de_);
        std::fprintf(stderr,
            "GSBENCH_TRACE arm=%s TOTALS compute_ms=%.3f submit_ms=%.3f drain_ms=%.3f "
            "(sum=%.3f)\n",
            arm, tot_c, tot_s, double(drain), tot_c + tot_s + double(drain));
    }
};

// The ONE timed sim loop shared by every arm. `snap(si, v_curr)` persists snapshot si;
// `finalize()` runs inside the timed region (async drain). No per-step Synchronize so
// the stream can overlap; one Synchronize closes the region. Returns wall-clock ms and
// accumulates a checksum of every snapshot's v-grid (host-read) for cross-arm equality.
// `trace` (optional) records GPU-timeline phase events; nullptr = off.
template <class SnapFn, class FinalizeFn>
double RunSim(const Cfg& cfg, Grids& g, SnapFn snap, FinalizeFn finalize,
              uint64_t* checksum_out, PhaseTrace* trace = nullptr) {
    unsigned N = cfg.N;
    uint64_t cells = cfg.cells();
    unsigned threads = 256, blocks = unsigned((cells + threads - 1) / threads);
    ctp::GpuApi::Synchronize();  // settle the seed before timing
    auto t0 = std::chrono::steady_clock::now();
    unsigned si = 0;
    for (unsigned step = 1; step <= cfg.steps(); ++step) {
        // First step of a snapshot interval => mark where this snapshot's compute begins.
        if (trace && (step - 1) % cfg.steps_per == 0) trace->CompStart(si);
        GsStepKernel<<<blocks, threads>>>(g.u_curr, g.v_curr, g.u_next, g.v_next,
                                          kGs, N);
        std::swap(g.u_curr, g.u_next);
        std::swap(g.v_curr, g.v_next);
        if (step % cfg.steps_per != 0) continue;
        if (trace) trace->CompEnd(si);   // compute done; snap() is the submit phase
        snap(si, g.v_curr);
        if (trace) trace->SubmitEnd(si);
        ++si;
    }
    if (trace) trace->DrainStart();
    finalize();                          // async: tail-drain all outstanding puts
    if (trace) trace->DrainEnd();
    ctp::GpuApi::Synchronize();
    auto t1 = std::chrono::steady_clock::now();
    (void)checksum_out;  // checksum computed by arms post-run from persisted bytes
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void PrintResult(const char* arm, const Cfg& cfg, double ms, uint64_t checksum) {
    double mb = double(cfg.total_bytes()) / (1024.0 * 1024.0);
    std::fprintf(stderr,
        "GSBENCH_RESULT arm=%s N=%u chunks=%u snaps=%u steps=%u bdev=%s "
        "MB=%.1f ms=%.2f MBps=%.1f checksum=%llu\n",
        arm, cfg.N, cfg.chunks, cfg.snaps, cfg.steps(), cfg.bdev.c_str(),
        mb, ms, mb / (ms / 1000.0), (unsigned long long)checksum);
}

// ---- CLIO env bring-up (configurable bdev) ---------------------------------

struct BenchEnv {
    clio::cte::core::TagId probe_tag;
    BenchEnv(const Cfg& cfg) {
        using namespace std::chrono_literals;
        namespace bdev = clio::run::bdev;
        std::fprintf(stderr, "[bench] bringing up server (bdev=%s cap=%uMB)\n",
                     cfg.bdev.c_str(), cfg.cap_mb);
        if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer))
            throw std::runtime_error("CLIO_INIT(kServer) failed");
        if (!clio::cte::core::CLIO_CTE_CLIENT_INIT())
            throw std::runtime_error("CLIO_CTE_CLIENT_INIT failed");
        auto* cte = CLIO_CTE_CLIENT;
        cte->Init(clio::cte::core::kCtePoolId);
        clio::cte::core::CreateParams params;
        auto ct = cte->AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                   clio::cte::core::kCtePoolName,
                                   clio::cte::core::kCtePoolId, params);
        ct.Wait();
        if (ct->GetReturnCode() != 0) throw std::runtime_error("CTE create failed");
        std::this_thread::sleep_for(50ms);

        const clio::run::u64 cap = clio::run::u64(cfg.cap_mb) << 20;
        const bool is_file = (cfg.bdev == "file");
        const bdev::BdevType type = is_file ? bdev::BdevType::kFile
                                            : bdev::BdevType::kRam;
        // For kFile the bdev name IS the on-disk file path (O_DIRECT). For kRam it is
        // just an identifier.
        const std::string name = is_file ? cfg.bdev_path : std::string("gsbench_ram");
        clio::run::PoolId bdev_pool_id(960, 0);
        bdev::Client bclient(bdev_pool_id);
        auto bc = bclient.AsyncCreate(clio::run::PoolQuery::Dynamic(), name, bdev_pool_id,
                                      type, cap);
        bc.Wait();
        if (bc->GetReturnCode() != 0) throw std::runtime_error("bdev create failed");
        std::this_thread::sleep_for(50ms);
        auto rt = cte->AsyncRegisterTarget(name, type, cap, clio::run::PoolQuery::Local(),
                                           bdev_pool_id);
        rt.Wait();
        if (rt->GetReturnCode() != 0) throw std::runtime_error("RegisterTarget failed");
        std::this_thread::sleep_for(50ms);
        std::fprintf(stderr, "[bench] server ready\n");
    }
};

clio::cte::core::TagId MakeTag(const char* name) {
    auto t = CLIO_CTE_CLIENT->AsyncGetOrCreateTag(name);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    return t->tag_id_;
}

std::vector<byte_t> HostReadBlob(clio::cte::core::TagId tag, const std::string& name,
                                 uint64_t size) {
    ctp::ipc::FullPtr<char> buf = CLIO_CPU_IPC->AllocateBuffer(size);
    REQUIRE(!buf.IsNull());
    std::memset(buf.ptr_, 0, size);
    ctp::ipc::ShmPtr<> shm = buf.shm_.template Cast<void>();
    auto t = CLIO_CTE_CLIENT->AsyncGetBlob(tag, name, clio::run::u64(0), size,
                                           clio::run::u32(0), shm);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    std::vector<byte_t> out(size);
    std::memcpy(out.data(), buf.ptr_, size);
    return out;
}

// Pre-create `snaps` snapshot datasets (distinct path => distinct tag). Kept well under
// the ~16-large-backend ceiling by the caller (snaps <= ~12).
void MakeSnapDatasets(clio::run::IpcManager* ipc, clio::run::IpcManagerGpuInfo gpu_info,
                      const char* prefix, const Cfg& cfg,
                      std::vector<kvhdf5::GpuCteDataset>& out,
                      std::vector<clio::cte::core::TagId>& tags) {
    kvhdf5::Layout layout{/*dims=*/{cfg.cells()},
                          /*chunk_dims=*/{cfg.cells() / cfg.chunks},
                          /*elem_size=*/sizeof(float)};
    REQUIRE(layout.ChunkCount() == cfg.chunks);
    const auto data_kind = cfg.data_pinned
        ? kvhdf5::GpuCteDataset::MemKind::kPinnedHost
        : kvhdf5::GpuCteDataset::MemKind::kDeviceMem;
    out.clear(); tags.clear(); out.reserve(cfg.snaps);
    for (unsigned s = 0; s < cfg.snaps; ++s) {
        char path[160];
        std::snprintf(path, sizeof(path), "%s/v/step_%04u", prefix, s);
        out.emplace_back(kvhdf5::GpuCteDataset::FromPath(
            ipc, gpu_info, /*gpu_id=*/0, CLIO_CTE_CLIENT, path, layout,
            /*pool_size=*/0, data_kind));
        tags.push_back(MakeTag(kvhdf5::tagpath::CanonicalTag(path).c_str()));
    }
}

// Read every snapshot back and fold into one checksum (proves what was persisted).
uint64_t ChecksumSnapshots(const std::vector<clio::cte::core::TagId>& tags,
                           const Cfg& cfg) {
    const uint64_t chunk_bytes = cfg.grid_bytes() / cfg.chunks;
    uint64_t h = 1469598103934665603ull;
    for (unsigned s = 0; s < cfg.snaps; ++s)
        for (unsigned c = 0; c < cfg.chunks; ++c) {
            auto got = HostReadBlob(tags[s], std::to_string(c), chunk_bytes);
            h = Fnv1a(got.data(), got.size(), h);
        }
    return h;
}

// Run a CLIO arm (async=false -> sync). Returns ms; fills checksum from persisted bytes.
double RunClioArm(const Cfg& cfg, bool async, const char* prefix, uint64_t* checksum) {
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

    std::vector<kvhdf5::GpuCteDataset> ds;
    std::vector<clio::cte::core::TagId> tags;
    MakeSnapDatasets(ipc, gpu_info, prefix, cfg, ds, tags);

    Masker masker(cfg.N, cfg.incompressible != 0);
    Grids g = MakeGrids(cfg.N);
    PhaseTrace trace(EnvU("GSBENCH_TRACE", 0) != 0, cfg.snaps, async);
    // Grid sizing for the decoupled bulk copy: one gridDim.y per chunk, enough blocks/chunk
    // (each thread copies one 4-byte word, grid-strided) to saturate HBM.
    const uint64_t copy_chunk_bytes = cfg.grid_bytes() / cfg.chunks;
    const unsigned copy_words = unsigned(copy_chunk_bytes / sizeof(uint32_t));
    unsigned copy_bpc = (copy_words + 255) / 256;
    if (copy_bpc < 1) copy_bpc = 1;
    if (copy_bpc > 2048) copy_bpc = 2048;
    auto snap = [&](unsigned si, float* v_curr) {
        float* src = masker.Apply(v_curr);   // incompressible view (or v_curr if disabled)
        const byte_t* bsrc = reinterpret_cast<const byte_t*>(src);
        TwCopyKernel<<<dim3(copy_bpc, cfg.chunks), 256>>>(ds[si].Handle(), bsrc);  // multi-block stage
        if (async) {
            TwSnapFireKernel<<<1, 256>>>(ds[si].Handle());
        } else {
            TwSnapSyncKernel<<<1, 256>>>(ds[si].Handle());
            // Bound the CUDA pending-launch queue. The sync arm's in-kernel SubmitWait
            // spins waiting for the IN-PROCESS server to flip the completion flag. If the
            // host races ahead and fills the ~1024-deep launch queue (once the run's total
            // kernels, 12*(steps_per+2), exceed it) it blocks inside cudaLaunchKernel while
            // the GPU is stalled on that spin — and the server can't make forward progress
            // from the same process → DEADLOCK (repros at steps_per>=96). A per-snapshot
            // sync caps in-flight work to one interval. Free for the sync arm (it already
            // waits on every put); the async arm must NOT do this — racing ahead IS its
            // overlap, and it never wedges (its waits are deferred to the end drain).
            ctp::GpuApi::Synchronize();
        }
    };
    auto finalize = [&]() {
        if (async) for (auto& d : ds) TwDrainKernel<<<1, 32>>>(d.Handle());
    };
    double ms = RunSim(cfg, g, snap, finalize, nullptr, &trace);
    FreeGrids(g);
    trace.Report(async ? "async" : "sync");   // GSBENCH_TRACE=1: emit phase breakdown
    *checksum = ChecksumSnapshots(tags, cfg);
    return ms;
}

// Host-driven CLIO arm: the SAME host flow as the raw arm (synchronous D2H into a host
// buffer, then a host-side write call), but the sink is a CLIO PutBlob instead of a
// file write+fsync. It does NOT use the GPU-producer model — no device-side task
// submission, no GpuCteDataset / registered device backends. So comparing it against:
//   - the RAW arm      isolates the STORAGE-PATH cost (CLIO server+bdev vs a plain file);
//   - the SYNC arm     isolates the SUBMISSION MODEL (host-orchestrated vs GPU-producer).
// Durable like CLIO-sync: each PutBlob is waited (bdev write completion).
double RunHostClioArm(const Cfg& cfg, const char* prefix, uint64_t* checksum) {
    const uint64_t gbytes = cfg.grid_bytes();
    const uint64_t chunk_bytes = gbytes / cfg.chunks;

    // One tag per snapshot (distinct path -> tag), same key scheme as the CLIO arms so
    // ChecksumSnapshots reads them back identically. No GPU backends => not bound by the
    // ~16-large-backend ceiling, but we keep cfg.snaps equal for a matched comparison.
    std::vector<clio::cte::core::TagId> tags;
    tags.reserve(cfg.snaps);
    for (unsigned s = 0; s < cfg.snaps; ++s) {
        char path[160];
        std::snprintf(path, sizeof(path), "%s/v/step_%04u", prefix, s);
        tags.push_back(MakeTag(kvhdf5::tagpath::CanonicalTag(path).c_str()));
    }

    // Host staging buffer (shm) that PutBlob DMAs from — the host-side counterpart of raw's
    // pinned D2H buffer.
    ctp::ipc::FullPtr<char> buf = CLIO_CPU_IPC->AllocateBuffer(gbytes);
    REQUIRE(!buf.IsNull());

    Masker masker(cfg.N, cfg.incompressible != 0);
    Grids g = MakeGrids(cfg.N);
    auto snap = [&](unsigned si, float* v_curr) {
        // Same host D2H as raw: pull the (incompressible) current grid into the host buffer.
        cudaMemcpy(buf.ptr_, masker.Apply(v_curr), gbytes, cudaMemcpyDeviceToHost);
        // Then persist each chunk to CLIO from the host (synchronous wait == durable).
        for (unsigned c = 0; c < cfg.chunks; ++c) {
            ctp::ipc::ShmPtr<> shm = buf.shm_.template Cast<void>();
            shm.off_ += uint64_t(c) * chunk_bytes;   // point at chunk c within the buffer
            auto t = CLIO_CTE_CLIENT->AsyncPutBlob(tags[si], std::to_string(c),
                                                   clio::run::u64(0), chunk_bytes, shm);
            t.Wait();
            REQUIRE(t->GetReturnCode() == 0);
        }
    };
    auto finalize = [&]() {};
    double ms = RunSim(cfg, g, snap, finalize, nullptr);
    FreeGrids(g);
    *checksum = ChecksumSnapshots(tags, cfg);
    return ms;
}

// ---- raw (no-CLIO) disk arm -----------------------------------------------

void EnsureDir(const std::string& d) {
    if (mkdir(d.c_str(), 0755) != 0 && errno != EEXIST)
        throw std::runtime_error("mkdir " + d);
}
// Write in <=1 MiB O_DIRECT chunks (matches the CLIO bdev's block loop; a single huge
// O_DIRECT pwrite from CUDA-pinned memory hit a ~5x-slow kernel path on this box).
void WriteAllAt(int fd, off_t off, const void* data, size_t bytes) {
    const auto* p = static_cast<const uint8_t*>(data);
    constexpr size_t kBlk = 1u << 20;
    while (bytes) {
        size_t want = bytes < kBlk ? bytes : kBlk;
        ssize_t n = pwrite(fd, p, want, off);
        if (n < 0) { if (errno == EINTR) continue; throw std::runtime_error("pwrite"); }
        p += n; off += n; bytes -= size_t(n);
    }
}

// A competent (deliberately NOT maximally-tuned) decoupled writer: ONE background thread
// + a small pinned-buffer pool (standard double-buffered checkpoint I/O). It lets disk
// I/O OVERLAP the subsequent sim steps instead of stalling inline — the same structural
// benefit CLIO gets by offloading I/O to its server. No libaio / queue-depth / thread
// fan-out (that would be an "expert" baseline; we keep it fair). O_DIRECT for cache-bypass
// parity with CLIO's kFile bdev.
class BgWriter {
public:
    BgWriter(std::string path, uint64_t gbytes, uint64_t wbytes, unsigned nbuf,
             unsigned snaps, bool odirect, bool do_fsync)
        : gbytes_(gbytes), wbytes_(wbytes), fsync_(do_fsync) {
        // ONE pre-allocated checkpoint file, snapshots written at distinct offsets — same
        // as CLIO's bdev (single truncated file, offset per blob). Avoids the per-snapshot
        // file create/O_TRUNC + async-discard churn that throttled the 12-fresh-files
        // pattern ~5x when the writes were spaced out by GPU work.
        int flags = O_WRONLY | O_CREAT | O_TRUNC | (odirect ? O_DIRECT : 0);
        fd_ = open(path.c_str(), flags, 0644);
        REQUIRE(fd_ >= 0);
        REQUIRE(ftruncate(fd_, off_t(snaps) * off_t(wbytes_)) == 0);  // preallocate size
        bufs_.resize(nbuf);
        for (unsigned i = 0; i < nbuf; ++i) {
            REQUIRE(cudaMallocHost(reinterpret_cast<void**>(&bufs_[i]), wbytes_)
                    == cudaSuccess);                 // pinned: fast D2H
            std::memset(bufs_[i] + gbytes_, 0, wbytes_ - gbytes_);  // O_DIRECT pad tail
            free_.push_back(int(i));
        }
        th_ = std::thread([this] { Run(); });
    }
    // Producer (CUDA thread): grab a free buffer to D2H into (blocks if all in flight).
    uint8_t* Acquire(int* idx) {
        std::unique_lock<std::mutex> lk(m_);
        cv_free_.wait(lk, [this] { return !free_.empty(); });
        int i = free_.back(); free_.pop_back();
        *idx = i; return bufs_[i];
    }
    // Producer: buffer filled (D2H complete) -> hand to the writer.
    void Submit(int idx, unsigned s) {
        { std::lock_guard<std::mutex> lk(m_); work_.push_back({idx, s}); }
        cv_work_.notify_one();
    }
    // Drain + join (call INSIDE the timed region). Yields writer busy-ms. Checksum is
    // computed by the caller AFTER timing (readback), matching the CLIO arms.
    void Finish(double* writer_ms) {
        { std::lock_guard<std::mutex> lk(m_); done_ = true; }
        cv_work_.notify_one();
        th_.join();
        if (fd_ >= 0) close(fd_);
        for (auto* b : bufs_) cudaFreeHost(b);
        *writer_ms = writer_ms_;
        REQUIRE(!err_);
    }
private:
    void Run() {
        for (;;) {
            std::pair<int, unsigned> job;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_work_.wait(lk, [this] { return !work_.empty() || done_; });
                if (work_.empty() && done_) return;
                job = work_.front(); work_.pop_front();
            }
            auto t0 = std::chrono::steady_clock::now();
            uint8_t* buf = bufs_[job.first];
            // NB: checksum is computed AFTER timing (post-run readback, like the CLIO arms)
            // — folding a scalar FNV over 1.9 GB here would add ~6 s to the timed region and
            // unfairly penalize raw vs CLIO (whose ChecksumSnapshots runs post-timing).
            WriteAllAt(fd_, off_t(job.second) * off_t(wbytes_), buf, wbytes_);  // offset slot
            if (fsync_ && fdatasync(fd_) != 0) err_ = true;  // durability parity with CLIO
            writer_ms_ += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            { std::lock_guard<std::mutex> lk(m_); free_.push_back(job.first); }
            cv_free_.notify_one();
        }
    }
    int fd_ = -1; uint64_t gbytes_, wbytes_; bool fsync_ = true;
    std::vector<uint8_t*> bufs_;
    std::mutex m_; std::condition_variable cv_free_, cv_work_;
    std::vector<int> free_; std::deque<std::pair<int, unsigned>> work_;
    bool done_ = false, err_ = false; std::thread th_;
    double writer_ms_ = 0;
};

// Read the persisted checkpoint back and fold FNV in snapshot order — AFTER timing (a
// scalar FNV over 1.9 GB is ~6 s and must NOT sit in the timed region; the CLIO arms
// likewise checksum post-timing).
uint64_t RawReadbackChecksum(const std::string& path, const Cfg& cfg,
                             uint64_t gbytes, uint64_t wbytes) {
    uint64_t h = 1469598103934665603ull;
    int fd = open(path.c_str(), O_RDONLY);
    REQUIRE(fd >= 0);
    std::vector<uint8_t> rb(gbytes);
    for (unsigned s = 0; s < cfg.snaps; ++s) {
        off_t base = off_t(s) * off_t(wbytes);
        size_t got = 0;
        while (got < gbytes) {
            ssize_t n = pread(fd, rb.data() + got, gbytes - got, base + off_t(got));
            REQUIRE(n > 0);
            got += size_t(n);
        }
        h = Fnv1a(rb.data(), gbytes, h);
    }
    close(fd);
    return h;
}

// Raw arm (no CLIO): identical sim + timed loop; each snapshot D2Hs the (incompressible)
// v-grid to host and persists it into one pre-allocated file. Two structures selected by
// GSBENCH_RAW_INLINE:
//   0 = background writer thread — I/O overlaps the next sim steps (the natural design, but
//       this box throttles GPU-concurrent disk I/O ~5x, penalizing the overlap);
//   1 = inline synchronous — GPU idle during the write, matching host-CLIO / sync-CLIO, for
//       a storage-path comparison free of that throttle.
double RunRawArm(const Cfg& cfg, uint64_t* checksum) {
    EnsureDir(cfg.disk_dir);
    const uint64_t gbytes = cfg.grid_bytes();
    constexpr uint64_t kAlign = 4096;
    const uint64_t wbytes = (gbytes + kAlign - 1) & ~(kAlign - 1);  // O_DIRECT length
    const std::string path = cfg.disk_dir + "/checkpoint.bin";
    Masker masker(cfg.N, cfg.incompressible != 0);

    if (cfg.raw_inline) {
        int flags = O_WRONLY | O_CREAT | O_TRUNC | (cfg.raw_odirect ? O_DIRECT : 0);
        int fd = open(path.c_str(), flags, 0644);
        REQUIRE(fd >= 0);
        REQUIRE(ftruncate(fd, off_t(cfg.snaps) * off_t(wbytes)) == 0);
        uint8_t* buf = nullptr;
        REQUIRE(cudaMallocHost(reinterpret_cast<void**>(&buf), wbytes) == cudaSuccess);
        std::memset(buf + gbytes, 0, wbytes - gbytes);
        Grids g = MakeGrids(cfg.N);
        double write_ms = 0;
        auto snap = [&](unsigned si, float* v_curr) {
            cudaMemcpy(buf, masker.Apply(v_curr), gbytes, cudaMemcpyDeviceToHost);
            auto a = std::chrono::steady_clock::now();          // GPU idle during this write
            WriteAllAt(fd, off_t(si) * off_t(wbytes), buf, wbytes);
            if (cfg.raw_fsync) fdatasync(fd);
            write_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - a).count();
        };
        auto finalize = [&]() {};
        double ms = RunSim(cfg, g, snap, finalize, nullptr);
        FreeGrids(g);
        close(fd);
        cudaFreeHost(buf);
        std::fprintf(stderr, "[raw] total=%.1f ms  write=%.1f ms  (inline, GPU idle)\n",
                     ms, write_ms);
        *checksum = RawReadbackChecksum(path, cfg, gbytes, wbytes);
        return ms;
    }

    BgWriter writer(path, gbytes, wbytes, /*nbuf=*/3, cfg.snaps,
                    /*odirect=*/cfg.raw_odirect != 0, /*do_fsync=*/cfg.raw_fsync != 0);
    Grids g = MakeGrids(cfg.N);
    auto snap = [&](unsigned si, float* v_curr) {
        int idx;
        uint8_t* buf = writer.Acquire(&idx);
        // SYNCHRONOUS D2H on the DEFAULT stream (ordered after the step / mask kernel), then
        // hand the buffer to the writer, which writes it while the NEXT sim steps run.
        cudaMemcpy(buf, masker.Apply(v_curr), gbytes, cudaMemcpyDeviceToHost);
        writer.Submit(idx, si);
    };
    double writer_ms = 0;
    auto finalize = [&]() { writer.Finish(&writer_ms); };  // drain inside timed region
    double ms = RunSim(cfg, g, snap, finalize, nullptr);
    FreeGrids(g);
    std::fprintf(stderr,
        "[raw] total=%.1f ms  writer-busy=%.1f ms (overlapped with compute)  nbuf=3\n",
        ms, writer_ms);
    *checksum = RawReadbackChecksum(path, cfg, gbytes, wbytes);
    return ms;
}

}  // namespace

// ---- the three arm TEST_CASEs (run each in its OWN process) ----------------

// raw arm needs no CLIO server.
TEST_CASE("GSBENCH raw disk (no CLIO)", "[.][integration][gpu][gsbench][gsbench_raw]") {
    Cfg cfg;
    uint64_t checksum = 0;
    double ms = RunRawArm(cfg, &checksum);
    PrintResult("raw", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

TEST_CASE("GSBENCH clio sync", "[.][integration][gpu][gsbench][gsbench_sync]") {
    Cfg cfg;
    static BenchEnv env(cfg);
    uint64_t checksum = 0;
    double ms = RunClioArm(cfg, /*async=*/false, "results/gsbench/sync", &checksum);
    PrintResult("sync", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

// Host-driven CLIO (raw's flow, CLIO sink): isolates the storage path vs raw, and the
// submission model vs the GPU-producer sync/async arms.
TEST_CASE("GSBENCH clio host-driven", "[.][integration][gpu][gsbench][gsbench_hostclio]") {
    Cfg cfg;
    static BenchEnv env(cfg);
    uint64_t checksum = 0;
    double ms = RunHostClioArm(cfg, "results/gsbench/hostclio", &checksum);
    PrintResult("hostclio", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

TEST_CASE("GSBENCH clio async", "[.][integration][gpu][gsbench][gsbench_async]") {
    Cfg cfg;
    static BenchEnv env(cfg);
    uint64_t checksum = 0;
    double ms = RunClioArm(cfg, /*async=*/true, "results/gsbench/async", &checksum);
    PrintResult("async", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

#endif  // !CTP_IS_DEVICE_PASS

#else
#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
