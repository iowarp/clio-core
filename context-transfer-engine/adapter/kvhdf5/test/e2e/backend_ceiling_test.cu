/*
 * BACKEND CEILING PROBE — is there really a "~16 live large GPU backend" limit?
 *
 * ============================ WHY THIS FILE EXISTS ==========================
 *
 * Folklore (ADVISOR-REPORT §6a, quoted in gray_scott_threeway_bench.cu:17,
 * gray_scott_async_e2e_test.cu:20, dataset_refire_probe_test.cu:6, and
 * e2e/CMakeLists.txt:29) holds that a process may hold only ~16 live "large"
 * device-memory backends registered via
 * IpcManager::AllocateAndRegisterGpuBackend, and that past that point data
 * SILENTLY CORRUPTS — no throw, no error code. The Gray-Scott benchmark routes
 * around it by capping snapshots at 12 and running one arm per process.
 *
 * That claim has never been tested. A static search of the whole registration
 * chain finds NO fixed-size table, array, bitmap, or slot limit:
 *
 *   - IpcManager::AllocateAndRegisterGpuBackend  ipc_manager.cc:2824
 *   - storage is gpu::IpcManager::PerGpuDeviceState::client_backends, an
 *     UNBOUNDED std::unordered_map<u64, ClientBackend>
 *                                              gpu_ipc_manager.h:186
 *   - insertion (plain operator[])              gpu2cpu_init_hip.cc:139-145
 *   - key = (alloc_id.major_ << 32) | alloc_id.minor_, where major_ = PID and
 *     minor_ = shm_count_.fetch_add(1)          ipc_manager.cc:2851
 *
 * MemoryBackendId::ToIndex() (memory_backend.h:118) computes `major_*2 + minor_`,
 * which WOULD collide wildly — but it has zero callers. There is no device-side
 * backend table either; GPU kernels never resolve AllocatorIds, only the host
 * does (ipc_manager.h:954).
 *
 * So the ceiling is unexplained by the code, and this probe exists to decide
 * empirically whether it exists at all, and if so on which axis.
 *
 * ============================== WHAT IT MEASURES ============================
 *
 * The core method is WRITE-ALL-THEN-VERIFY-ALL. Every backend gets a pattern
 * keyed to its own index; ALL N are allocated and written before ANY is checked.
 * Verifying each backend right after its own write would pass even under total
 * corruption, because the damage (a map overwrite, a VA collision, an allocator
 * handing the same memory out twice) is done by a LATER registration to an
 * EARLIER backend. Corruption is therefore detected by CONTENT, never by the
 * absence of a thrown exception — that is the entire point.
 *
 * Each probe additionally records, for every backend, the AllocatorId
 * (major/minor), the derived 64-bit map key, and the base pointer, and asserts
 * key uniqueness — this directly tests the "silent map overwrite" hypothesis.
 *
 * The probes are env-driven so a driver script can sweep them in SEPARATE
 * processes (the alleged ceiling is per-process, so each N needs a fresh one):
 *
 *   BC_N      backend count                        (default 24)
 *   BC_BYTES  bytes per backend                    (default 156 MiB, the
 *                                                   Gray-Scott snapshot size)
 *   BC_KIND   0 = kPinnedHost, 1 = kManagedUvm, 2 = kDeviceMem (default 2)
 *   BC_LIVE   if > 0, keep at most this many backends live at once and free the
 *             oldest as new ones are made — separates LIVE count from CUMULATIVE
 *             churn. Only the live set is verified.
 *
 * ============================== FINDINGS ===================================
 *
 * THE ~16-BACKEND CEILING DOES NOT EXIST. It is an artifact. Measured:
 *
 *   - 100 live 156 MiB kDeviceMem backends (15.6 GiB) register and verify
 *     CLEAN — 6x the alleged limit. Same for kPinnedHost (40) and kManagedUvm
 *     (24). Churn (100 allocs, 4 live at a time) is CLEAN. Every AllocatorId is
 *     unique; zero map-key collisions. There is no count limit at any MemKind.
 *
 * WHAT IS ACTUALLY BROKEN: a PutBlob whose bytes do not fit in the registered
 * bdev target SILENTLY DOES NOTHING. The blob never lands; a later GetBlob
 * returns nothing and leaves the caller's buffer untouched; no error surfaces.
 *
 * The runtime detects the failure correctly and sets a nonzero return code
 * (core_runtime.cc:1155 `task->return_code_ = 10 + alloc_result`, from
 * ExtendBlob's error_code=3 on a short allocation). The GPU producer path then
 * THROWS THAT CODE AWAY: GpuDatasetHandle::SubmitWait (gpu_dataset_handle.h:112)
 * does `fut.Wait();` and never reads task->return_code_, and Write()/Read()/
 * Submit() all return void — so a producer kernel cannot learn its Put failed.
 * That discarded return code is the entire bug. Nothing else is wrong.
 *
 * WHY IT LOOKED LIKE "~16": the break is on BYTES, not count. Corruption starts
 * the instant cumulative blob bytes exceed bdev capacity, and the apparent
 * "ceiling N" is just floor(capacity / blob_size). Measured against the shared
 * 64 MiB kRam bdev, corrupted == N - floor(64/blob_MiB) EXACTLY, every time:
 *
 *     blob 8 MiB -> breaks after N=8      blob 2 MiB -> breaks after N=32
 *     blob 4 MiB -> breaks after N=17     blob 1 MiB -> breaks after N=64
 *
 * At 4 MiB blobs the cliff lands on 17 — which is where "a ~16 backend ceiling"
 * was almost certainly born. Gray-Scott runs 156.25 MiB snapshots against a
 * 3072 MiB bdev (run_threeway_bench.sh:54,59), giving a true limit of ~19
 * snapshots; capping snaps at 12 stayed under it by luck, not by design.
 *
 * SAFE OPERATING RULE: sum(blob bytes) < bdev capacity. Backend COUNT is not a
 * constraint. See agents/paper-writing/traces/05-backend-ceiling.md.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/singletons.h>
#include <clio_runtime/ipc_manager.h>
#include <clio_ctp/util/gpu_api.h>

#include <clio_cte/kvhdf5/chunking.h>
#include <clio_cte/kvhdf5/gpu_cte_dataset.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <set>
#include <string>
#include <vector>

#if !CTP_IS_DEVICE_PASS
#include <catch2/catch_test_macros.hpp>
#endif
#include "cte_env.h"

using MemKind = clio::run::gpu::IpcManager::MemKind;
using kvhdf5::byte_t;

namespace {

/**
 * Byte pattern for backend `idx` at offset `i`. Distinct per backend, cheap to
 * recompute host-side, and non-constant along the buffer so a partial or
 * offset-shifted copy is still caught.
 */
__host__ __device__ inline unsigned char BcByte(unsigned idx, unsigned long long i) {
    return static_cast<unsigned char>((idx * 131u + static_cast<unsigned>(i) * 7u) & 0xFFu);
}

/** Grid-stride fill of a device-accessible buffer with backend `idx`'s pattern. */
__global__ void BcFillKernel(unsigned char *buf, unsigned long long n, unsigned idx) {
    unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    for (unsigned long long i = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
         i < n; i += stride) {
        buf[i] = BcByte(idx, i);
    }
}

/** Fill a dataset's device buffer with pattern `idx` (separate kernel). */
__global__ void BcDsFillKernel(byte_t *buf, unsigned long long n, unsigned idx) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
         i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
        buf[i] = static_cast<byte_t>(BcByte(idx, i));
    }
}

/** Submit the pre-built PutBlob task from the kernel via the handle. */
__global__ void BcDsWriteKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    h.Write();
}

/** Submit the pre-built GetBlob task from the kernel via the handle. */
__global__ void BcDsReadKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    h.Read();
}

}  // namespace

#if !CTP_IS_DEVICE_PASS

namespace {

struct Backend {
    ctp::ipc::AllocatorId id;
    char *base = nullptr;
    unsigned long long bytes = 0;
    unsigned idx = 0;
};

unsigned long long EnvU64(const char *name, unsigned long long dflt) {
    const char *v = std::getenv(name);
    return (v && *v) ? std::strtoull(v, nullptr, 10) : dflt;
}

const char *KindName(MemKind k) {
    switch (k) {
        case MemKind::kPinnedHost: return "kPinnedHost";
        case MemKind::kManagedUvm: return "kManagedUvm";
        case MemKind::kDeviceMem:  return "kDeviceMem";
    }
    return "?";
}

/** Abort-free CUDA error probe: report and clear, so we can prove silence. */
bool BcCudaCheck(const char *what, unsigned idx) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[cuda] %s (backend %u): %s\n", what, idx,
                     cudaGetErrorString(e));
        return false;
    }
    return true;
}

/**
 * Verify backend `b` holds its own pattern. Checks three 64 KiB windows (head,
 * middle, tail) rather than every byte: a corrupting event (map overwrite, VA
 * aliasing, double-hand-out) replaces the whole region, so any window catches
 * it, and sampling keeps a 40 x 156 MiB sweep fast.
 *
 * Returns the number of mismatched bytes found, and reports what the bytes
 * ACTUALLY were — if they match another backend's pattern, that identifies the
 * aliasing partner and is the single most diagnostic fact available.
 */
unsigned long long VerifyBackend(const Backend &b, unsigned n_total) {
    const unsigned long long kWin = 64ull << 10;
    unsigned long long win = (b.bytes < kWin) ? b.bytes : kWin;
    unsigned long long offs[3] = {0, (b.bytes - win) / 2, b.bytes - win};

    std::vector<unsigned char> host(win);
    unsigned long long bad = 0;

    for (int w = 0; w < 3; ++w) {
        cudaError_t e = cudaMemcpy(host.data(), b.base + offs[w], win,
                                   cudaMemcpyDefault);
        if (e != cudaSuccess) {
            std::fprintf(stderr,
                         "[FAIL] backend %u window %d: cudaMemcpy failed: %s\n",
                         b.idx, w, cudaGetErrorString(e));
            return win;
        }
        for (unsigned long long i = 0; i < win; ++i) {
            unsigned char want = BcByte(b.idx, offs[w] + i);
            if (host[i] == want) continue;
            if (bad == 0) {
                // Whose pattern IS this, if anyone's?
                int culprit = -1;
                for (unsigned c = 0; c < n_total; ++c) {
                    if (host[i] == BcByte(c, offs[w] + i)) { culprit = static_cast<int>(c); break; }
                }
                std::fprintf(stderr,
                             "[CORRUPT] backend %u @off %llu: got 0x%02X want 0x%02X"
                             " (byte matches backend %d's pattern)\n",
                             b.idx, offs[w] + i, host[i], want, culprit);
            }
            ++bad;
        }
    }
    return bad;
}

/**
 * The probe proper. Allocates+registers `n` backends of `bytes` each, fills each
 * from a GPU kernel, and only then verifies all live ones.
 */
void RunCeilingProbe(unsigned n, unsigned long long bytes, MemKind kind,
                     unsigned live_cap) {
    auto *ipc = CLIO_IPC;
    REQUIRE(ipc != nullptr);

    std::fprintf(stderr,
                 "\n=== BC PROBE: n=%u bytes=%llu (%.1f MiB) kind=%s live_cap=%u"
                 " total=%.2f GiB ===\n",
                 n, bytes, bytes / 1048576.0, KindName(kind), live_cap,
                 (double)n * bytes / (1024.0 * 1024 * 1024));

    std::deque<Backend> live;
    std::set<unsigned long long> keys_seen;
    unsigned key_collisions = 0;
    unsigned alloc_failures = 0;

    for (unsigned i = 0; i < n; ++i) {
        char *base = nullptr;
        auto id = ipc->AllocateAndRegisterGpuBackend(0, kind, bytes, &base);

        if (id.IsNull() || base == nullptr) {
            std::fprintf(stderr,
                         "[alloc-fail] backend %u: AllocateAndRegisterGpuBackend"
                         " returned null (id.IsNull=%d base=%p)\n",
                         i, (int)id.IsNull(), (void *)base);
            ++alloc_failures;
            break;  // a hard failure is NOT silent corruption — stop and report
        }

        unsigned long long key = (static_cast<unsigned long long>(id.major_) << 32) |
                                 static_cast<unsigned long long>(id.minor_);
        bool fresh = keys_seen.insert(key).second;
        if (!fresh) {
            ++key_collisions;
            std::fprintf(stderr,
                         "[KEY-COLLISION] backend %u reuses map key %llu"
                         " (alloc_id=%u.%u) — an earlier backend was just"
                         " SILENTLY OVERWRITTEN in client_backends\n",
                         i, key, id.major_, id.minor_);
        }

        std::fprintf(stderr,
                     "[alloc] backend %2u alloc_id=(%u.%u) key=%llu base=%p\n",
                     i, id.major_, id.minor_, key, (void *)base);

        // Fill from the GPU, exactly as a real producer kernel would.
        unsigned long long threads = 256;
        unsigned long long blocks = (bytes + threads - 1) / threads;
        if (blocks > 65535) blocks = 65535;
        BcFillKernel<<<(unsigned)blocks, (unsigned)threads>>>(
            reinterpret_cast<unsigned char *>(base), bytes, i);
        cudaDeviceSynchronize();
        BcCudaCheck("fill kernel", i);

        Backend b;
        b.id = id; b.base = base; b.bytes = bytes; b.idx = i;
        live.push_back(b);

        // Churn mode: retire the oldest so only `live_cap` are ever live.
        if (live_cap > 0 && live.size() > live_cap) {
            Backend old = live.front();
            live.pop_front();
            ipc->FreeGpuBackend(0, old.id);
            switch (kind) {
                case MemKind::kPinnedHost: ctp::GpuApi::FreeHost(old.base); break;
                default:                   ctp::GpuApi::Free(old.base); break;
            }
        }
    }

    // ---- VERIFY ALL LIVE BACKENDS, only now that every write has landed ----
    unsigned corrupted = 0;
    unsigned long long total_bad = 0;
    for (const auto &b : live) {
        unsigned long long bad = VerifyBackend(b, n);
        if (bad) { ++corrupted; total_bad += bad; }
    }

    std::fprintf(stderr,
                 "=== BC RESULT: n=%u kind=%s live=%zu | alloc_failures=%u"
                 " key_collisions=%u corrupted_backends=%u bad_bytes=%llu ===\n",
                 n, KindName(kind), live.size(), alloc_failures, key_collisions,
                 corrupted, total_bad);
    std::fprintf(stderr, "=== BC VERDICT: %s ===\n",
                 (corrupted == 0 && key_collisions == 0 && alloc_failures == 0)
                     ? "CLEAN" : "BROKEN");

    // Free whatever is still live.
    for (const auto &b : live) {
        ipc->FreeGpuBackend(0, b.id);
        switch (kind) {
            case MemKind::kPinnedHost: ctp::GpuApi::FreeHost(b.base); break;
            default:                   ctp::GpuApi::Free(b.base); break;
        }
    }

    CHECK(key_collisions == 0);
    CHECK(corrupted == 0);
    CHECK(alloc_failures == 0);
}

/**
 * DATASET-LEVEL probe — the shape the Gray-Scott benchmark actually uses, and
 * the one the corruption was originally reported against.
 *
 * Each GpuCteDataset allocates THREE backends (gpu_cte_dataset.h Init): a
 * kPinnedHost task backend, a kDeviceMem data backend (the "large" one), and a
 * tiny kDeviceMem descriptor backend. So N datasets => 3N registrations, and the
 * benchmark's "one fresh dataset per snapshot" is exactly this loop.
 *
 * Again WRITE-ALL-THEN-VERIFY-ALL: every dataset is created and PutBlob'd first,
 * and only then is each one GetBlob'd back and checked. The device buffer is
 * zeroed before the Get so a stale buffer cannot masquerade as a good read —
 * without that, a GetBlob that silently no-ops would still "pass".
 */
void RunDatasetProbe(unsigned n, unsigned long long bytes,
                     const kvhdf5::itest::ClioCteEnv &env) {
    auto *ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);

    std::fprintf(stderr,
                 "\n=== BC DATASET PROBE: n=%u datasets x %llu B (%.2f MiB each,"
                 " %.1f MiB total, %u backend registrations) ===\n",
                 n, bytes, bytes / 1048576.0, (double)n * bytes / 1048576.0, 3 * n);

    std::deque<kvhdf5::GpuCteDataset> sets;
    std::vector<std::string> names(n);
    unsigned puts_reported_failed = 0;

    // ---- Phase 1: create every dataset and PutBlob a distinct pattern. ----
    for (unsigned i = 0; i < n; ++i) {
        names[i] = "bc_" + std::to_string(i);
        sets.emplace_back(ipc, gpu_info, /*gpu_id=*/0, env.tag_id,
                          names[i].c_str(), static_cast<clio::run::u32>(bytes));
        byte_t *data = sets.back().DeviceData();

        unsigned threads = 256;
        unsigned blocks = (unsigned)((bytes + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;
        BcDsFillKernel<<<blocks, threads>>>(data, bytes, i);
        ctp::GpuApi::Synchronize();

        BcDsWriteKernel<<<1, 32>>>(sets.back().Handle());
        ctp::GpuApi::Synchronize();
        BcCudaCheck("dataset put", i);

        // Read back the return code the runtime already recorded in the (pinned,
        // CPU-readable) task slot. Without this the Put's failure is invisible:
        // the kernel's Write() is void and SubmitWait() discards the code.
        int put_rc = sets.back().PutStatus(0);
        if (put_rc != 0) {
            ++puts_reported_failed;
            std::fprintf(stderr,
                         "[put] dataset %2u ('%s') FAILED, rc=%d — data did NOT"
                         " land (13 == bdev out of capacity)\n",
                         i, names[i].c_str(), put_rc);
        } else {
            std::fprintf(stderr, "[put] dataset %2u ('%s') ok\n", i, names[i].c_str());
        }
    }

    // ---- Phase 2: only NOW read each one back and verify. ----
    unsigned corrupted = 0;
    std::vector<byte_t> host(static_cast<size_t>(bytes));
    std::vector<byte_t> zeros(static_cast<size_t>(bytes), byte_t{0});

    for (unsigned i = 0; i < n; ++i) {
        byte_t *data = sets[i].DeviceData();
        ctp::GpuApi::Memcpy(data, zeros.data(), bytes);  // poison before Get

        BcDsReadKernel<<<1, 32>>>(sets[i].Handle());
        ctp::GpuApi::Synchronize();
        BcCudaCheck("dataset get", i);

        ctp::GpuApi::Memcpy(host.data(), data, bytes);

        unsigned long long bad = 0;
        for (unsigned long long j = 0; j < bytes; ++j) {
            byte_t want = static_cast<byte_t>(BcByte(i, j));
            if (host[j] == want) continue;
            if (bad == 0) {
                int culprit = -1;
                for (unsigned c = 0; c < n; ++c) {
                    if (host[j] == static_cast<byte_t>(BcByte(c, j))) {
                        culprit = static_cast<int>(c);
                        break;
                    }
                }
                std::fprintf(stderr,
                             "[CORRUPT] dataset %u @%llu: got 0x%02X want 0x%02X"
                             " (matches dataset %d's pattern; 0x00 => Get returned"
                             " nothing and the poison survived)\n",
                             i, j, (unsigned)host[j], (unsigned)want, culprit);
            }
            ++bad;
        }
        if (bad) ++corrupted;
        std::fprintf(stderr, "[get] dataset %2u: %s (%llu bad bytes)\n", i,
                     bad ? "CORRUPT" : "ok", bad);
    }

    // The headline number for the fix: how many of the silently-lost Puts did
    // PutStatus() actually report? If puts_reported_failed == corrupted, the
    // failure is no longer silent — it is detectable at the moment it happens.
    std::fprintf(stderr,
                 "=== BC DATASET RESULT: n=%u | corrupted_datasets=%u"
                 " puts_reported_failed=%u ===\n"
                 "=== BC DATASET VERDICT: %s (data loss %s) ===\n",
                 n, corrupted, puts_reported_failed,
                 corrupted == 0 ? "CLEAN" : "BROKEN",
                 corrupted == 0 ? "none"
                     : (puts_reported_failed == corrupted ? "REPORTED (not silent)"
                                                          : "SILENT"));

    // Every lost blob must have been reported by PutStatus(). This is the
    // assertion the fix exists to satisfy: data loss may happen (the bdev really
    // is full), but it must never be SILENT.
    CHECK(puts_reported_failed == corrupted);
    CHECK(corrupted == 0);
}

/**
 * CHURN probe — the ADVISOR-REPORT §6a configuration, which the all-live probe
 * above does NOT cover.
 *
 * §6a reported corruption at the 17th dataset with 256 KiB and 512 KiB chunks
 * (invariant to chunk size => count-gated, not byte-gated), and clean runs at
 * 4 KiB to 200. Crucially it used "one FRESH dataset per snapshot": each dataset
 * is created, written, and DESTROYED before the next. RunDatasetProbe keeps every
 * dataset alive, so it never exercises the create/free path at all — a different
 * experiment entirely.
 *
 * Here: create dataset i, fill with pattern i, PutBlob, destroy it. Then, only
 * after all N are gone, re-open each blob through a fresh dataset and verify.
 * Blobs live in the bdev, not in the (destroyed) backends, so the readback is
 * legitimate. Still write-all-then-verify-all.
 *
 * Sizes here stay far below the 64 MiB bdev cap (100 x 256 KiB = 25 MiB), so the
 * capacity bug is NOT a confound and any failure is a genuine count/churn limit.
 */
void RunChurnProbe(unsigned n, unsigned long long bytes,
                   const kvhdf5::itest::ClioCteEnv &env) {
    auto *ipc = CLIO_CPU_IPC;
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);

    std::fprintf(stderr,
                 "\n=== BC CHURN PROBE (ADVISOR §6a shape): n=%u fresh datasets"
                 " x %llu B (%.0f KiB each, %.1f MiB total vs 64 MiB bdev) ===\n",
                 n, bytes, bytes / 1024.0, (double)n * bytes / 1048576.0);

    std::vector<std::string> names(n);
    unsigned puts_reported_failed = 0;

    // ---- Phase 1: create -> fill -> Put -> DESTROY, N times. ----
    for (unsigned i = 0; i < n; ++i) {
        names[i] = "churn_" + std::to_string(i);
        kvhdf5::GpuCteDataset ds(ipc, gpu_info, /*gpu_id=*/0, env.tag_id,
                                 names[i].c_str(), static_cast<clio::run::u32>(bytes));
        unsigned threads = 256;
        unsigned blocks = (unsigned)((bytes + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;
        BcDsFillKernel<<<blocks, threads>>>(ds.DeviceData(), bytes, i);
        ctp::GpuApi::Synchronize();

        BcDsWriteKernel<<<1, 32>>>(ds.Handle());
        ctp::GpuApi::Synchronize();
        BcCudaCheck("churn put", i);

        int rc = ds.PutStatus(0);
        if (rc != 0) {
            ++puts_reported_failed;
            std::fprintf(stderr, "[put] churn %2u FAILED rc=%d\n", i, rc);
        }
        // ds destructor frees + unregisters all three backends here.
    }
    std::fprintf(stderr, "[churn] %u datasets created+destroyed\n", n);

    // ---- Phase 2: re-open each blob and verify its pattern survived. ----
    unsigned corrupted = 0;
    int first_bad = -1;
    std::vector<byte_t> host(static_cast<size_t>(bytes));
    std::vector<byte_t> zeros(static_cast<size_t>(bytes), byte_t{0});

    for (unsigned i = 0; i < n; ++i) {
        kvhdf5::GpuCteDataset ds(ipc, gpu_info, /*gpu_id=*/0, env.tag_id,
                                 names[i].c_str(), static_cast<clio::run::u32>(bytes));
        ctp::GpuApi::Memcpy(ds.DeviceData(), zeros.data(), bytes);  // poison

        BcDsReadKernel<<<1, 32>>>(ds.Handle());
        ctp::GpuApi::Synchronize();
        BcCudaCheck("churn get", i);
        ctp::GpuApi::Memcpy(host.data(), ds.DeviceData(), bytes);

        unsigned long long bad = 0;
        for (unsigned long long j = 0; j < bytes; ++j)
            if (host[j] != static_cast<byte_t>(BcByte(i, j))) ++bad;

        if (bad) {
            if (first_bad < 0) {
                first_bad = static_cast<int>(i);
                std::fprintf(stderr,
                             "[CORRUPT] churn dataset %u (the %u-th): %llu bad bytes\n",
                             i, i + 1, bad);
            }
            ++corrupted;
        }
    }

    std::fprintf(stderr,
                 "=== BC CHURN RESULT: n=%u chunk=%lluB | corrupted=%u"
                 " first_bad_index=%d (i.e. the %d-th) puts_reported_failed=%u ===\n"
                 "=== BC CHURN VERDICT: %s ===\n",
                 n, bytes, corrupted, first_bad,
                 first_bad < 0 ? 0 : first_bad + 1, puts_reported_failed,
                 corrupted == 0 ? "CLEAN" : "BROKEN");

    CHECK(corrupted == 0);
}

MemKind KindFromEnv() {
    switch (EnvU64("BC_KIND", 2)) {
        case 0:  return MemKind::kPinnedHost;
        case 1:  return MemKind::kManagedUvm;
        default: return MemKind::kDeviceMem;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// HIDDEN ([.]) env-driven probe. The sweep driver runs this once per N, each in
// its own process. This is the instrument, not a pass/fail assertion about the
// codebase — see the [!mayfail] case below for that.
// ---------------------------------------------------------------------------
TEST_CASE("BCPROBE registered GPU backend sweep",
          "[.][integration][gpu][bcprobe]") {
    (void)kvhdf5::itest::SharedCteEnv();
    RunCeilingProbe(static_cast<unsigned>(EnvU64("BC_N", 24)),
                    EnvU64("BC_BYTES", 156ull << 20),
                    KindFromEnv(),
                    static_cast<unsigned>(EnvU64("BC_LIVE", 0)));
}

// ---------------------------------------------------------------------------
// HIDDEN ([.]) dataset-level probe — N datasets (3N backend registrations),
// PutBlob all, then GetBlob all and verify. Env: BC_DS_N, BC_DS_BYTES.
//
// Blobs default to 1 MiB because the shared test env registers a 64 MiB kRam
// bdev (cte_env.h); larger blobs at high N would exhaust the bdev and we would
// be measuring THAT, not a backend ceiling.
// ---------------------------------------------------------------------------
TEST_CASE("BCPROBE dataset sweep", "[.][integration][gpu][bcds]") {
    auto &env = kvhdf5::itest::SharedCteEnv();
    RunDatasetProbe(static_cast<unsigned>(EnvU64("BC_DS_N", 24)),
                    EnvU64("BC_DS_BYTES", 1ull << 20), env);
}

// ---------------------------------------------------------------------------
// HIDDEN ([.]) churn probe — reproduces the ADVISOR-REPORT §6a configuration
// (fresh dataset per snapshot, created and DESTROYED each iteration).
// Env: BC_CH_N (default 100), BC_CH_BYTES (default 256 KiB).
// ---------------------------------------------------------------------------
TEST_CASE("BCPROBE churn sweep (fresh dataset per snapshot)",
          "[.][integration][gpu][bcchurn]") {
    auto &env = kvhdf5::itest::SharedCteEnv();
    RunChurnProbe(static_cast<unsigned>(EnvU64("BC_CH_N", 100)),
                  EnvU64("BC_CH_BYTES", 256ull << 10), env);
}

// ---------------------------------------------------------------------------
// NEGATIVE RESULT, GUARDED: there is no backend-count ceiling.
//
// 24 live 156 MiB kDeviceMem backends — 1.5x the alleged ~16 limit, at the exact
// per-snapshot size Gray-Scott uses (3.7 GiB, comfortably inside the 4090's
// 24 GB, so this is not VRAM exhaustion either). This PASSES, and it is a plain
// (non-mayfail) case on purpose: it is the regression guard that keeps the
// debunked ceiling debunked. If someone ever reintroduces a real slot limit,
// this goes red. Hidden ([.]) only because it is slow and allocates GBs.
// ---------------------------------------------------------------------------
TEST_CASE("BACKEND CEILING: 24 live 156 MiB kDeviceMem backends stay intact",
          "[.][integration][gpu][bcceiling]") {
    (void)kvhdf5::itest::SharedCteEnv();
    RunCeilingProbe(/*n=*/24, /*bytes=*/156ull << 20, MemKind::kDeviceMem,
                    /*live_cap=*/0);
}

// ---------------------------------------------------------------------------
// THE REAL DEFECT — known-failing, allowed to fail.
//
// [!mayfail] — Catch2 lets this fail WITHOUT failing the run, so the bug stays
// visible in CI and we notice loudly the day it gets fixed. Also [.] (hidden),
// following the convention of every other case in this directory.
//
// WHAT BREAKS: 80 x 1 MiB blobs are Put against the shared env's 64 MiB kRam
// bdev. The first 64 land. Blobs 64..79 DO NOT FIT — and PutBlob reports success
// anyway. Each of those 16 blobs then reads back as the 0x00 poison this probe
// wrote before the Get: the data is simply gone, silently.
//
// AT WHAT N: exactly N > floor(bdev_capacity / blob_bytes). Nothing to do with
// backend count (100 live backends are fine — see the case above).
//
// CAUSE (confirmed, not hypothesised): the runtime DOES detect the failure and
// records it as task->return_code_ = 10 + alloc_result = 13 (core_runtime.cc:1155,
// from ExtendBlob's error_code=3). The GPU producer path discards it —
// GpuDatasetHandle::SubmitWait (gpu_dataset_handle.h:112) calls fut.Wait() and
// never reads the code, and Write()/Read() return void. The storage layer is
// correct; only the reporting is broken.
//
// WHY IT STILL FAILS (and should): RunDatasetProbe now calls the new
// GpuCteDataset::PutStatus() (gpu_cte_dataset.h) after each Put, which reads that
// code out of the pinned, CPU-readable task slot — so the loss is now REPORTED
// ("dataset 64 FAILED, rc=13") rather than silent, and the
// `puts_reported_failed == corrupted` assertion PASSES. What remains failing is
// `corrupted == 0`: the bdev genuinely IS full, so those blobs genuinely ARE
// lost. That residual failure is honest and expected — overflowing storage must
// lose data. The bug was never the loss; it was the SILENCE.
//
// So: if this case ever reports "data loss SILENT" again, the reporting path has
// regressed and that is a real bug. "data loss REPORTED (not silent)" is the
// fixed state.
// ---------------------------------------------------------------------------
TEST_CASE("SILENT DATA LOSS: PutBlob past bdev capacity reports success",
          "[.][!mayfail][integration][gpu][bcoverflow]") {
    auto &env = kvhdf5::itest::SharedCteEnv();
    // 80 MiB of blobs into a 64 MiB bdev: blobs 64..79 are silently dropped.
    RunDatasetProbe(/*n=*/80, /*bytes=*/1ull << 20, env);
}

#endif  // !CTP_IS_DEVICE_PASS

#else

// Non-GPU build: nothing to test here.

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
