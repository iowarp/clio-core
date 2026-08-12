/*
 * WEAK SCALING BENCHMARK — synthetic, compute-free submit/transport probe.
 *
 * This is the direct test of the reframed C2 claim ("metadata updates without
 * contention"): give the runtime an increasing number of independent concurrent
 * GPU submitters and show that adding submitters adds bandwidth, not contention.
 *
 * ARMS ARE THE LINES (WS_ARM). The figure compares the four systems the paper
 * pits against each other, not GPUH5's internal bdev tiers:
 *   gpuh5    — GPU-initiated, one PutBlob per block in-kernel (the code below).
 *   raw      — host: D2H to pinned host, then a buffered file write (RunHostArm).
 *   hdf5     — host: D2H to pinned host, then H5Dwrite               (RunHostArm).
 *   hostclio — host: D2H to shm, then host-side PutBlob-and-wait     (RunHostArm).
 * The three host arms drain through a SINGLE host pipeline, so they are flat
 * across block count by construction; only gpuh5's per-block submitter turns GPU
 * concurrency into bandwidth. Async/overlap arms are excluded on purpose: this
 * figure measures I/O bandwidth, and async bandwidth is a compute/IO-overlap
 * artifact this compute-free study removes. WS_SINK fixes the medium (ram parity
 * by default) so the lines are comparable.
 *
 * The GPUH5 STRUCTURE below. One GpuCteDataset of C = WS_BLOCKS * WS_KEYS_PER_BLOCK chunks of
 * WS_CHUNK_MB MB. One kernel launch of WS_BLOCKS blocks x 256 threads. Block b
 * owns chunks [b*WS_KEYS_PER_BLOCK, (b+1)*WS_KEYS_PER_BLOCK) and loops
 * WS_ITERS times, iteration i writing chunk b*WS_KEYS_PER_BLOCK +
 * (i % WS_KEYS_PER_BLOCK) — DESIGN.md's ping-pong scheme (>=2 keys avoids the
 * degenerate same-key-back-to-back case). h.Write(c) is Send().Wait(), i.e.
 * synchronous submit-and-wait; async is disabled by construction, so this
 * isolates submission + transport from overlap. No compute is performed, so
 * bytes/elapsed genuinely IS bandwidth here (unlike the write-runtime figures,
 * where MBps folds compute into the denominator).
 *
 * MEMKIND (gpuh5 arm). The dataset's DATA-buffer placement (MemKind, where the
 * client-side staging buffer lives) defaults to kDeviceMem — the real
 * device-initiated D2H path. kPinnedHost would make the server's "I/O" a
 * host->host memcpy (the GPU moves no data), inflating bandwidth; it is opt-in
 * via WS_MEMKIND only. The kPinnedHost forcing seen in gsbench applies to
 * PERSISTENT grids (an in-kernel wait can deadlock against the server's D2H);
 * this kernel is not persistent (blocks exit after WS_ITERS).
 *
 * TIMING. Chunks are filled by a SEPARATE prologue kernel (WsFillKernel) before
 * the timed region, with one __threadfence_system() per block after filling,
 * so fill cost never lands in the measurement. A throwaway warm-up launch of
 * the exact submit-kernel template instantiation (WS_ITERS=0, so no Write() is
 * actually issued) runs before the timed launch: CUDA's lazy module loading
 * makes a kernel's FIRST launch a device-wide synchronizing operation, and
 * without this warm-up that stall would land inside the timed region on the
 * very first point measured. The timed region itself wraps ONLY the submit
 * kernel, using CUDA events (never a host clock — a host-observed block on
 * this path has been measured to read ~2.4x the true cost).
 *
 * VERIFICATION. GpuCteDataset::ThrowIfIoFailed() is called after
 * ctp::GpuApi::Synchronize() following the timed region — not a GetBlob spot-
 * check. A PutBlob whose bytes don't fit the registered bdev target silently
 * does nothing (see backend_ceiling_test.cu); at high WS_BLOCKS that would
 * fabricate bandwidth from writes that never landed. ThrowIfIoFailed() surfaces
 * that loudly instead.
 *
 * PROBE (WS_PROBE=1). Per-block clock64() deltas are taken around each
 * h.Write(c) inside the timed kernel, accumulated into a device array sized
 * WS_BLOCKS * WS_ITERS words, copied back once after the kernel and reduced to
 * a median/p99 (in microseconds, converted from cycles via the device's clock
 * rate — comparable across points only while the campaign's GPU clocks stay
 * locked, which is the runner's job, not this binary's). This is the kernel's
 * OWN instrumentation, gated by a compile-time template parameter (kProbing)
 * selected at launch by the WS_PROBE env var — never a runtime branch inside
 * the timed loop, and never surfaced on GpuDatasetHandle's public signature
 * (following the existing convention: no probing in the public API). It also
 * calls h.Write<false>(c) to shed the handle's OWN internal SendIn submit-probe
 * registers, which this bench does not use.
 *
 * ENV VARS:
 *   WS_ARM             gpuh5 | raw | hdf5 | hostclio             default gpuh5
 *   WS_SINK            ram | file (storage medium, all arms)      default ram
 *   WS_BLOCKS          producer grid (blocks)                    default 1
 *   WS_MB_PER_BLOCK    bytes/block held constant (weak scaling)   default 128
 *   WS_CHUNK_KB        gpuh5 request size, KB (load-bearing)      default 64
 *   WS_ITERS           gpuh5 iterations/block (derived from above)
 *   WS_KEYS_PER_BLOCK  gpuh5 ping-pong keys per block             default 2
 *   WS_MEMKIND         device | pinned (gpuh5 staging placement)  default device
 *   WS_BDEV            clio bdev tier override                    default per WS_SINK
 *   WS_BDEV_CAP_MB     clio bdev capacity, MB; 0 = auto-size      default 0
 *   WS_BDEV_PATH       kFile backing path                         default ./ws_bdev.dat
 *   WS_TMPFS_DIR       raw/hdf5 RAM-sink dir                      default /dev/shm/ws
 *   WS_DISK_DIR        raw/hdf5 file-sink dir                     default ./ws_raw_out
 *   WS_PROBE           0|1, gpuh5 per-submission clock64() probe  default 0
 *
 * OUTPUT. Exactly one machine-readable line to STDOUT, prefixed "WSRESULT ":
 *   WSRESULT arm=<name> blocks=<n> bdev=<name> sink=<name> chunk_kb=<n>
 *            iters=<n> keys=<n> mb_per_block=<n> bytes=<total_bytes>
 *            elapsed_ms=<f> bw_gbps=<f> submit_med_us=<f> submit_p99_us=<f>
 *            verified=<ok|FAIL|na>
 * submit_med_us/submit_p99_us are -1 unless WS_ARM=gpuh5 and WS_PROBE=1.
 *
 * Registered as a HIDDEN ([.][ws]) Catch2 case named "weak_scaling" — one
 * point per process, matching how the gsbench arms and backend_ceiling_test's
 * env-driven probes run (a driver script sweeps WS_ARM/WS_BLOCKS across
 * separate process invocations).
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/singletons.h>
#include <clio_ctp/util/gpu_api.h>

#include <clio_cte/kvhdf5/dataset_meta.h>      // Layout
#include <clio_cte/kvhdf5/gpu_cte_dataset.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if !CTP_IS_DEVICE_PASS
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/types.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <catch2/catch_test_macros.hpp>

#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#if GSBENCH_HAVE_HDF5
#include <hdf5.h>
#endif
#endif

using kvhdf5::byte_t;

namespace {

// Fill pattern for chunk c, byte i. Distinct per chunk so a stuck/aliased
// chunk index would show up under a content readback; this bench itself
// verifies via ThrowIfIoFailed()'s runtime status (see file header), not a
// byte comparison, but the pattern costs nothing and keeps the buffers
// non-degenerate.
__host__ __device__ inline byte_t WsPattern(uint32_t c, uint64_t i) {
    return static_cast<byte_t>((c * 131u + static_cast<unsigned>(i) * 7u) & 0xFFu);
}

/**
 * Prologue: fill every chunk with its pattern. One block per chunk (launched
 * <<<C, 256>>>), grid-stride so an undersized launch would just leave chunks
 * unfilled rather than misbehave. Separate from the timed submit kernel so
 * fill cost never lands in the measured region. One __threadfence_system()
 * per block after its own fills so the CPU-side Put in the later timed kernel
 * observes the bytes.
 */
__global__ void WsFillKernel(kvhdf5::GpuDatasetHandle h) {
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        byte_t* dst = h.Data(c);
        const uint64_t n = h.Size(c);
        for (uint64_t i = threadIdx.x; i < n; i += blockDim.x)
            dst[i] = WsPattern(c, i);
    }
    __threadfence_system();
}

// Flat-buffer fill for the host-mediated arms (raw/hdf5/hostclio). Those arms
// stage from ONE device source buffer that the timed loop D2H's B times; it is
// filled once here, off the timed path. Grid-stride so the launch config is
// irrelevant. Non-degenerate bytes (WsPattern) so no sink can dedup/compress it
// away, though the RAM/tmpfs sinks used here do neither.
__global__ void WsFillFlatKernel(byte_t* p, uint64_t n) {
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
        p[i] = WsPattern(static_cast<uint32_t>(i >> 16), i);
}

/**
 * Timed submit kernel. Block b loops `iters` times over its
 * `keys_per_block` chunks, iteration i writing chunk
 * b*keys_per_block + (i % keys_per_block) (the ping-pong scheme). `iters==0`
 * is a legal no-op launch (used as the pre-timing warm-up, so the module load
 * happens before the region CUDA events wrap).
 *
 * kProbing gates this bench's OWN clock64() instrumentation (compile-time
 * template flag, per-block deltas by thread 0 — h.Write's producer guard is
 * already thread-0-only, so this matches exactly what that thread waited on).
 * probe_deltas is a device array of blocks*iters entries; null and untouched
 * when kProbing is false. Write<false> sheds the handle's own internal
 * SendIn submit-probe registers, which this bench does not use.
 */
template<bool kProbing>
__global__ void WsSubmitKernel(kvhdf5::GpuDatasetHandle h, uint32_t iters,
                               uint32_t keys_per_block,
                               unsigned long long* probe_deltas) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    const uint32_t b = blockIdx.x;
    for (uint32_t i = 0; i < iters; ++i) {
        const uint32_t c = b * keys_per_block + (i % keys_per_block);
        unsigned long long t0 = 0;
        if (kProbing && threadIdx.x == 0) t0 = clock64();
        __syncthreads();
        h.Write<false>(c);
        __syncthreads();
        if (kProbing && threadIdx.x == 0) {
            const unsigned long long t1 = clock64();
            probe_deltas[static_cast<uint64_t>(b) * iters + i] = t1 - t0;
        }
    }
}

// EXPLICIT INSTANTIATION, and it is load-bearing. WsSubmitKernel is a template
// whose ONLY launch site lives in the host-only `#if !CTP_IS_DEVICE_PASS` block
// below. The device pass therefore never sees an instantiation, emits no device
// code for either variant, and every launch fails at runtime with "invalid
// device function" — which, without the cudaGetLastError check at the timed
// region, surfaces as a 0 ms kernel and a fabricated multi-TB/s bandwidth.
// Instantiating both variants here (outside that guard) is what puts them in the
// device image.
template __global__ void WsSubmitKernel<true>(kvhdf5::GpuDatasetHandle, uint32_t,
                                              uint32_t, unsigned long long*);
template __global__ void WsSubmitKernel<false>(kvhdf5::GpuDatasetHandle, uint32_t,
                                               uint32_t, unsigned long long*);

}  // namespace

#if !CTP_IS_DEVICE_PASS

namespace {

// ---- env-var config, matching the convention in gray_scott_threeway_bench.cu
// and backend_ceiling_test.cu ------------------------------------------------

unsigned EnvU(const char* k, unsigned dflt) {
    const char* v = std::getenv(k);
    if (!v || !*v) return dflt;
    long x = std::strtol(v, nullptr, 10);
    return x > 0 ? static_cast<unsigned>(x) : dflt;
}
// Like EnvU but 0 is a meaningful value (WS_PROBE=0, WS_BDEV_CAP_MB=0 for
// auto-sizing both need this; EnvU would coerce 0 back to the default).
unsigned EnvU0(const char* k, unsigned dflt) {
    const char* v = std::getenv(k);
    if (!v || !*v) return dflt;
    long x = std::strtol(v, nullptr, 10);
    return x >= 0 ? static_cast<unsigned>(x) : dflt;
}
std::string EnvS(const char* k, const char* dflt) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::string(v) : std::string(dflt);
}

clio::run::bdev::BdevType BdevTypeFromName(const std::string& name) {
    if (name == "noop")   return clio::run::bdev::BdevType::kNoop;
    if (name == "hbm")    return clio::run::bdev::BdevType::kHbm;
    if (name == "pinned") return clio::run::bdev::BdevType::kPinned;
    if (name == "ram")    return clio::run::bdev::BdevType::kRam;
    if (name == "file")   return clio::run::bdev::BdevType::kFile;
    throw std::runtime_error("weak_scaling_bench: unknown WS_BDEV '" + name +
                             "' (want noop|hbm|pinned|ram|file)");
}

// ---- CLIO env bring-up (configurable bdev) — the exact pattern of
// gray_scott_threeway_bench.cu's BenchEnv, parameterized by this bench's own
// env vars instead of Cfg. kFile uses the on-disk path AS the bdev name;
// every other type uses a plain identifier. ---------------------------------
struct WsEnv {
    WsEnv(const std::string& bdev_name, clio::run::bdev::BdevType type,
          unsigned cap_mb, const std::string& bdev_path) {
        using namespace std::chrono_literals;
        std::fprintf(stderr, "[ws] bringing up server (bdev=%s cap=%uMB)\n",
                     bdev_name.c_str(), cap_mb);
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

        const clio::run::u64 cap = clio::run::u64(cap_mb) << 20;
        const bool is_file = (type == clio::run::bdev::BdevType::kFile);
        const std::string name = is_file ? bdev_path : (std::string("ws_bdev_") + bdev_name);
        clio::run::PoolId bdev_pool_id(960, 0);
        clio::run::bdev::Client bclient(bdev_pool_id);
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
        std::fprintf(stderr, "[ws] server ready\n");
    }
};

// Median + p99 (in microseconds) of a device-cycle-count probe array, reduced
// host-side via the device's clock rate. Comparable across points only while
// GPU clocks are locked for the whole campaign (the runner's job).
void ReduceProbe(const std::vector<unsigned long long>& cycles,
                 double* submit_med_us, double* submit_p99_us) {
    if (cycles.empty()) return;
    int device = 0;
    cudaGetDevice(&device);
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device);
    const double cycles_per_us = clock_khz > 0 ? (double(clock_khz) / 1000.0) : 1.0;

    std::vector<double> us(cycles.size());
    for (size_t i = 0; i < cycles.size(); ++i)
        us[i] = double(cycles[i]) / cycles_per_us;
    std::sort(us.begin(), us.end());

    *submit_med_us = us[us.size() / 2];
    const size_t p99_idx = static_cast<size_t>(0.99 * double(us.size() - 1));
    *submit_p99_us = us[p99_idx];
}

// ---- host-mediated arms: raw / hdf5 / hostclio -----------------------------
//
// The three cross-system comparison arms, stripped of Gray-Scott compute from
// gray_scott_threeway_bench.cu (RunRawArm / Hdf5Sink / RunHostClioArm). They
// share ONE shape: a single device source buffer (per_block_bytes, filled once
// off the clock), D2H-staged into a host buffer, then written to the arm's
// sink. The timed loop runs B iterations reusing the buffers, so it moves
// B * per_block_bytes with a bounded ~3*per_block_bytes footprint -- the host
// analog of the gpuh5 arm's ping-pong, and what keeps B=768 off a 96 GB file.
//
// TIMING is a HOST steady_clock around { D2H + sink-write }: there is no kernel
// to wrap, the sequence is host-serial (D2H is synchronous on the default
// stream), and the host clock is the correct instrument for a host-driven write
// pipeline. (The "never time this with a host clock" rule from the gpuh5 path
// was about timing a lone cudaMemcpy whose cost is dominated by an unrelated
// compute tail; here the D2H+write IS the thing being measured.) One warm
// iteration runs BEFORE the clock to fault in sink pages, prime the D2H path,
// and force lazy module load. No compute is in the loop, so bytes/elapsed is
// genuinely bandwidth.
//
// X-AXIS NOTE: block count is the GPU producer's concurrency. These arms drain
// through a SINGLE host pipeline regardless of B, so their curve is flat by
// construction -- that is the intended finding (host-mediated I/O does not turn
// GPU concurrency into bandwidth), not a defect.

struct HostArmResult {
    double elapsed_ms = 0.0;
    uint64_t total_bytes = 0;
    bool ok = true;
    std::string err;
};

void WsMkdir(const std::string& d) {
    if (mkdir(d.c_str(), 0755) != 0 && errno != EEXIST)
        throw std::runtime_error("mkdir " + d + ": " + std::strerror(errno));
}

// pwrite the whole buffer at offset 0 in <=1 MiB blocks (a single huge pwrite
// from pinned memory hit a ~5x-slow kernel path on this box; see gsbench L2411).
void WsWriteAll(int fd, const void* data, uint64_t bytes) {
    const auto* p = static_cast<const uint8_t*>(data);
    off_t off = 0;
    constexpr uint64_t kBlk = 1u << 20;
    while (bytes) {
        uint64_t want = bytes < kBlk ? bytes : kBlk;
        ssize_t n = pwrite(fd, p, static_cast<size_t>(want), off);
        if (n < 0) { if (errno == EINTR) continue; throw std::runtime_error("pwrite"); }
        p += n; off += n; bytes -= uint64_t(n);
    }
}

HostArmResult RunHostArm(const std::string& arm, const std::string& sink,
                         unsigned blocks, uint64_t per_block_bytes,
                         clio::cte::core::TagId tag,
                         const std::string& tmpfs_dir,
                         const std::string& disk_dir) {
    HostArmResult r;
    r.total_bytes = uint64_t(blocks) * per_block_bytes;
    const bool ram = (sink != "file");

    // ---- device source, filled once off the timed path ----
    byte_t* dsrc = ctp::GpuApi::Malloc<byte_t>(per_block_bytes);
    {
        const unsigned threads = 256;
        const uint64_t g = (per_block_bytes + threads - 1) / threads;
        const unsigned grid = static_cast<unsigned>(std::min<uint64_t>(65535, g));
        WsFillFlatKernel<<<grid, threads>>>(dsrc, per_block_bytes);
        const cudaError_t fl = cudaGetLastError();
        const cudaError_t fe = cudaDeviceSynchronize();
        if (fl != cudaSuccess || fe != cudaSuccess) {
            r.ok = false;
            r.err = std::string("flat fill failed: ") +
                    cudaGetErrorString(fl != cudaSuccess ? fl : fe);
            ctp::GpuApi::Free(dsrc);
            return r;
        }
    }

    try {
        if (arm == "raw") {
            byte_t* hbuf = nullptr;
            if (cudaMallocHost(reinterpret_cast<void**>(&hbuf), per_block_bytes) != cudaSuccess)
                throw std::runtime_error("cudaMallocHost");
            const std::string dir = ram ? tmpfs_dir : disk_dir;
            WsMkdir(dir);
            const std::string path = dir + "/ws_raw.bin";
            // tmpfs REJECTS O_DIRECT (gsbench L616); only the disk sink cache-bypasses.
            const int flags = O_WRONLY | O_CREAT | O_TRUNC | (ram ? 0 : O_DIRECT);
            int fd = open(path.c_str(), flags, 0644);
            if (fd < 0) throw std::runtime_error("open " + path);
            cudaMemcpy(hbuf, dsrc, per_block_bytes, cudaMemcpyDeviceToHost);  // warm
            WsWriteAll(fd, hbuf, per_block_bytes);
            const auto t0 = std::chrono::steady_clock::now();
            for (unsigned b = 0; b < blocks; ++b) {
                cudaMemcpy(hbuf, dsrc, per_block_bytes, cudaMemcpyDeviceToHost);
                WsWriteAll(fd, hbuf, per_block_bytes);
            }
            r.elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            close(fd);
            cudaFreeHost(hbuf);

        } else if (arm == "hdf5") {
#if GSBENCH_HAVE_HDF5
            byte_t* hbuf = nullptr;
            if (cudaMallocHost(reinterpret_cast<void**>(&hbuf), per_block_bytes) != cudaSuccess)
                throw std::runtime_error("cudaMallocHost");
            const std::string dir = ram ? tmpfs_dir : disk_dir;
            WsMkdir(dir);
            const std::string path = dir + "/ws_hdf5.h5";
            hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            if (file < 0) throw std::runtime_error("H5Fcreate " + path);
            const hsize_t dims[1] = {static_cast<hsize_t>(per_block_bytes)};
            hid_t space = H5Screate_simple(1, dims, nullptr);
            hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
            // Allocate file space up front + never fill: we overwrite every byte,
            // so a fill pass would zero per_block_bytes right before we write it,
            // and sparse pages would otherwise fault in on the FIRST timed write
            // (gsbench L2686-2689).
            H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_EARLY);
            H5Pset_fill_time(dcpl, H5D_FILL_TIME_NEVER);
            hid_t dset = H5Dcreate2(file, "v", H5T_NATIVE_UCHAR, space, H5P_DEFAULT,
                                    dcpl, H5P_DEFAULT);
            if (dset < 0) throw std::runtime_error("H5Dcreate2");
            cudaMemcpy(hbuf, dsrc, per_block_bytes, cudaMemcpyDeviceToHost);  // warm
            if (H5Dwrite(dset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, hbuf) < 0)
                throw std::runtime_error("H5Dwrite (warm)");
            const auto t0 = std::chrono::steady_clock::now();
            for (unsigned b = 0; b < blocks; ++b) {
                cudaMemcpy(hbuf, dsrc, per_block_bytes, cudaMemcpyDeviceToHost);
                if (H5Dwrite(dset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, hbuf) < 0)
                    throw std::runtime_error("H5Dwrite");
            }
            r.elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(file);
            cudaFreeHost(hbuf);
#else
            throw std::runtime_error("hdf5 arm needs GSBENCH_HAVE_HDF5=1");
#endif

        } else if (arm == "hostclio") {
            // Host staging buffer (shm) that PutBlob DMAs from; register it so the
            // D2H rides the pinned fast path (gsbench L2344), else this arm is
            // handicapped vs gpuh5's device-initiated path.
            ctp::ipc::FullPtr<char> buf = CLIO_CPU_IPC->AllocateBuffer(per_block_bytes);
            if (buf.IsNull()) throw std::runtime_error("AllocateBuffer");
            const bool reg =
                (cudaHostRegister(buf.ptr_, per_block_bytes, cudaHostRegisterDefault)
                 == cudaSuccess);
            std::fprintf(stderr, "[ws:hostclio] shm staging registered=%d\n", int(reg));
            constexpr uint64_t kChunk = 1u << 20;   // 1 MiB PutBlobs (fair natural size)
            auto put_all = [&]() {
                uint64_t off = 0;
                unsigned c = 0;
                while (off < per_block_bytes) {
                    const uint64_t want = std::min<uint64_t>(kChunk, per_block_bytes - off);
                    ctp::ipc::ShmPtr<> shm = buf.shm_.template Cast<void>();
                    shm.off_ += off;
                    auto t = CLIO_CTE_CLIENT->AsyncPutBlob(tag, std::to_string(c),
                                                           clio::run::u64(0), want, shm);
                    t.Wait();
                    if (t->GetReturnCode() != 0)
                        throw std::runtime_error("PutBlob rc != 0");
                    off += want; ++c;
                }
            };
            cudaMemcpy(buf.ptr_, dsrc, per_block_bytes, cudaMemcpyDeviceToHost);  // warm
            put_all();
            const auto t0 = std::chrono::steady_clock::now();
            for (unsigned b = 0; b < blocks; ++b) {
                cudaMemcpy(buf.ptr_, dsrc, per_block_bytes, cudaMemcpyDeviceToHost);
                put_all();
            }
            r.elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            if (reg) cudaHostUnregister(buf.ptr_);

        } else {
            throw std::runtime_error("unknown WS_ARM '" + arm + "'");
        }
    } catch (const std::exception& e) {
        r.ok = false;
        r.err = e.what();
    }

    ctp::GpuApi::Free(dsrc);
    return r;
}

}  // namespace

TEST_CASE("weak_scaling", "[.][ws]") {
    const unsigned blocks = EnvU("WS_BLOCKS", 1);
    // WS_ARM selects the WRITE PATH (the figure's lines): gpuh5 (GPU-initiated,
    // one PutBlob per block in-kernel) vs the three host-mediated arms raw /
    // hdf5 / hostclio (GPU produces the data, a host pipeline drains it). See
    // agents/weak-scaling/DESIGN.md section 9.
    const std::string arm = EnvS("WS_ARM", "gpuh5");
    const bool is_gpuh5 = (arm == "gpuh5");
    // WS_SINK fixes the storage medium so the arms are comparable (RAM parity is
    // the default: this is a no-submission-contention figure, so the medium must
    // not be the wall). ram: gpuh5/hostclio -> kRam bdev, raw/hdf5 -> tmpfs.
    // file: gpuh5/hostclio -> kFile, raw/hdf5 -> disk dir.
    const std::string sink = EnvS("WS_SINK", "ram");
    const bool sink_file = (sink == "file");
    const std::string bdev_name = EnvS("WS_BDEV", sink_file ? "file" : "ram");
    const std::string bdev_path = EnvS("WS_BDEV_PATH", "./ws_bdev.dat");
    const std::string tmpfs_dir = EnvS("WS_TMPFS_DIR", "/dev/shm/ws");
    const std::string disk_dir  = EnvS("WS_DISK_DIR", "./ws_raw_out");

    // gpuh5-only knobs. REQUEST SIZE IS THE LOAD-BEARING KNOB for gpuh5: at 4 MB a
    // SINGLE outstanding request already saturates the D2H link (measured: 4 MB /
    // 272 us = 15 GB/s at blocks=1), so concurrency has no headroom left to add
    // bandwidth and gpuh5's own curve is flat by construction. Requests must be
    // small enough that one submitter is LATENCY-bound; then bandwidth rises with
    // concurrency until the link fills. Hence KB, not MB. (The host arms use their
    // own natural request sizes inside RunHostArm; this knob does not touch them.)
    const unsigned chunk_kb = EnvU("WS_CHUNK_KB", 64);
    // Bytes per block is held CONSTANT (that is what makes this weak scaling), for
    // ALL arms; gpuh5's iteration count is derived so shrinking the request does
    // not shrink the work. WS_ITERS still overrides for diagnostics.
    const unsigned mb_per_block = EnvU("WS_MB_PER_BLOCK", 128);
    const unsigned derived_iters =
        static_cast<unsigned>((uint64_t(mb_per_block) * 1024ull) / uint64_t(chunk_kb));
    const unsigned iters = EnvU("WS_ITERS", derived_iters);
    const unsigned keys = EnvU("WS_KEYS_PER_BLOCK", 2);
    const bool probe_on = EnvU0("WS_PROBE", 0) != 0;

    const clio::run::bdev::BdevType type = BdevTypeFromName(bdev_name);

    const uint64_t chunk_bytes = uint64_t(chunk_kb) << 10;
    const uint64_t per_block_bytes = uint64_t(mb_per_block) << 20;
    REQUIRE(iters > 0);
    const uint64_t total_chunks = uint64_t(blocks) * uint64_t(keys);
    REQUIRE(total_chunks > 0);
    REQUIRE(total_chunks <= 0xFFFFFFFFull);

    // bdev capacity: gpuh5 lands total_chunks*chunk_kb; hostclio overwrites a
    // single per_block_bytes footprint; raw/hdf5 don't use the bdev at all.
    const unsigned cap_mb_env = EnvU0("WS_BDEV_CAP_MB", 0);
    unsigned cap_mb;
    if (cap_mb_env > 0) {
        cap_mb = cap_mb_env;
    } else {
        const uint64_t needed_mb = is_gpuh5
            ? (total_chunks * uint64_t(chunk_kb) + 1023ull) / 1024ull
            : uint64_t(mb_per_block);   // hostclio overwrites one per-block footprint
        const uint64_t auto_cap = needed_mb + needed_mb / 4;  // +25% headroom
        cap_mb = static_cast<unsigned>(std::max<uint64_t>(64, auto_cap));
    }

    static WsEnv env(bdev_name, type, cap_mb, bdev_path);

    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);

    auto tag_t = CLIO_CTE_CLIENT->AsyncGetOrCreateTag("weak_scaling");
    tag_t.Wait();
    REQUIRE(tag_t->GetReturnCode() == 0);
    const clio::cte::core::TagId tag = tag_t->tag_id_;

    // Outputs, filled by whichever arm runs below.
    double out_elapsed_ms = 0.0;
    uint64_t out_total_bytes = 0;
    std::string verified = "FAIL";
    double submit_med_us = -1.0, submit_p99_us = -1.0;

    if (is_gpuh5) {

    kvhdf5::Layout layout{/*dims=*/{total_chunks * chunk_bytes},
                          /*chunk_dims=*/{chunk_bytes}, /*elem_size=*/1};
    REQUIRE(layout.Valid());
    REQUIRE(layout.ChunkCount() == total_chunks);

    // MemKind selects where the STAGING buffer lives, which is orthogonal to
    // WS_BDEV (where bytes land). It matters enormously to what this bench
    // measures: with kPinnedHost the payload already sits in host memory, so the
    // server's "I/O" is a host->host memcpy and the GPU moves no data at all --
    // that is not GPU I/O bandwidth. kDeviceMem makes the server D2H the payload,
    // which is the real device-initiated path. The kPinnedHost forcing seen in
    // gsbench applies to PERSISTENT grids, where an in-kernel wait can deadlock
    // against the server's D2H; this kernel is not persistent (blocks exit after
    // WS_ITERS), so kDeviceMem is the default here and pinned is opt-in.
    const std::string memkind_name = EnvS("WS_MEMKIND", "device");
    const kvhdf5::GpuCteDataset::MemKind memkind =
        (memkind_name == "pinned") ? kvhdf5::GpuCteDataset::MemKind::kPinnedHost
                                   : kvhdf5::GpuCteDataset::MemKind::kDeviceMem;
    // grid_size == blocks matches the actual submit-kernel launch config
    // grid_size == blocks matches the actual submit-kernel launch config
    // (pool_size 0 => M == N == total_chunks, so pooling never kicks in and
    // every chunk keeps its own dedicated data buffer).
    kvhdf5::GpuCteDataset ds(ipc, gpu_info, /*gpu_id=*/0, tag, layout,
                             /*pool_size=*/0, memkind,
                             /*grid_size=*/blocks);
    REQUIRE(ds.ChunkCount() == total_chunks);

    kvhdf5::GpuDatasetHandle h = ds.Handle();

    // ---- Prologue: fill every chunk, off the timed path. ----
    WsFillKernel<<<static_cast<unsigned>(total_chunks), 256>>>(h);
    {
        const cudaError_t fl = cudaGetLastError();
        if (fl != cudaSuccess)
            FAIL("fill kernel LAUNCH failed: " << cudaGetErrorString(fl));
        const cudaError_t fe = cudaDeviceSynchronize();
        if (fe != cudaSuccess)
            FAIL("fill kernel EXEC failed: " << cudaGetErrorString(fe));
    }

    // ---- Probe buffer (device), only allocated when WS_PROBE=1. ----
    const size_t probe_n = size_t(blocks) * size_t(iters);
    unsigned long long* probe_dev = nullptr;
    if (probe_on && probe_n > 0)
        probe_dev = ctp::GpuApi::Malloc<unsigned long long>(probe_n * sizeof(unsigned long long));

    // ---- Warm-up: force this exact kernel instantiation's module resident
    // before the timed region (CUDA lazy module loading is a device-wide sync
    // on first launch). iters=0 so no Write() actually fires. ----
    if (probe_on) WsSubmitKernel<true><<<blocks, 256>>>(h, /*iters=*/0, keys, probe_dev);
    else          WsSubmitKernel<false><<<blocks, 256>>>(h, /*iters=*/0, keys, probe_dev);
    ctp::GpuApi::Synchronize();

    // ---- Timed region: CUDA events around the submit kernel only. ----
    cudaEvent_t ev_start = nullptr, ev_stop = nullptr;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);
    if (probe_on) WsSubmitKernel<true><<<blocks, 256>>>(h, iters, keys, probe_dev);
    else          WsSubmitKernel<false><<<blocks, 256>>>(h, iters, keys, probe_dev);
    const cudaError_t launch_err = cudaGetLastError();
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    const cudaError_t exec_err = cudaDeviceSynchronize();
    // A failed launch or a kernel that faulted would otherwise be reported as an
    // absurdly fast run (0 elapsed => fabricated bandwidth), which is exactly the
    // fraud mode this bench exists to avoid. Fail loudly instead.
    if (launch_err != cudaSuccess)
        FAIL("submit kernel LAUNCH failed: " << cudaGetErrorString(launch_err));
    if (exec_err != cudaSuccess)
        FAIL("submit kernel EXEC failed: " << cudaGetErrorString(exec_err));
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    ctp::GpuApi::Synchronize();

    // ---- Verification: ThrowIfIoFailed(), not a GetBlob spot-check. ----
    // kNoop DISCARDS by definition, so "did the bytes land" is not a question that
    // has an answer for it -- ThrowIfIoFailed always fires. Exempt it and report
    // verified=na rather than either lying (ok) or failing the point (FAIL). This
    // is safe precisely because the noop line is not a storage-bandwidth claim: it
    // is the submit-path ceiling, where no bytes were ever supposed to land. Every
    // backend that DOES claim to store is still guarded.
    const bool is_noop = (type == clio::run::bdev::BdevType::kNoop);
    bool io_ok = true;
    std::string io_err;
    if (!is_noop) {
        try {
            ds.ThrowIfIoFailed("weak_scaling");
        } catch (const std::exception& e) {
            io_ok = false;
            io_err = e.what();
        }
    }

    if (probe_on && probe_dev != nullptr) {
        std::vector<unsigned long long> probe_host(probe_n);
        ctp::GpuApi::Memcpy(probe_host.data(), probe_dev,
                            probe_n * sizeof(unsigned long long));
        ReduceProbe(probe_host, &submit_med_us, &submit_p99_us);
    }
    if (probe_dev != nullptr) ctp::GpuApi::Free(probe_dev);

    out_elapsed_ms = double(elapsed_ms);
    out_total_bytes = uint64_t(blocks) * uint64_t(iters) * chunk_bytes;
    verified = is_noop ? "na" : (io_ok ? "ok" : "FAIL");
    if (!io_ok)
        std::fprintf(stderr, "[weak_scaling] verification FAILED: %s\n", io_err.c_str());
    CHECK(io_ok);

    } else {
        // ---- host-mediated arm (raw / hdf5 / hostclio) ----
        HostArmResult hr = RunHostArm(arm, sink, blocks, per_block_bytes, tag,
                                      tmpfs_dir, disk_dir);
        out_elapsed_ms = hr.elapsed_ms;
        out_total_bytes = hr.total_bytes;
        verified = hr.ok ? "ok" : "FAIL";
        if (!hr.ok)
            std::fprintf(stderr, "[weak_scaling] arm=%s FAILED: %s\n",
                         arm.c_str(), hr.err.c_str());
        CHECK(hr.ok);
    }

    const double elapsed_s = out_elapsed_ms / 1000.0;
    const double bw_gbps =
        elapsed_s > 0.0 ? (double(out_total_bytes) / elapsed_s) / 1e9 : 0.0;

    std::fprintf(stdout,
        "WSRESULT arm=%s blocks=%u bdev=%s sink=%s chunk_kb=%u iters=%u keys=%u "
        "mb_per_block=%u bytes=%llu elapsed_ms=%.3f bw_gbps=%.3f "
        "submit_med_us=%.3f submit_p99_us=%.3f verified=%s\n",
        arm.c_str(), blocks, bdev_name.c_str(), sink.c_str(), chunk_kb, iters, keys,
        mb_per_block, (unsigned long long)out_total_bytes, out_elapsed_ms, bw_gbps,
        submit_med_us, submit_p99_us, verified.c_str());
    std::fflush(stdout);
}

#endif  // !CTP_IS_DEVICE_PASS

#else

// Non-GPU build: nothing to test here.

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
