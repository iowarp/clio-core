/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * TRANSPARENT LOSSLESS-COMPRESSED PAGE FAULTING.
 *
 * Sequential read of a dataset many times larger than the HBM cache, where
 * every page is stored COMPRESSED and is decompressed on the fault path without
 * the kernel knowing. The kernel does nothing but read_range; compression is
 * entirely a property of how the tag is stored.
 *
 * WHY LOSSLESS, AND WHY NOT cuSZp. The existing compressed gpu_vector tests pin
 * CLIO_CTE_COMPRESS_LIB=cuszp, which CMake describes as a GPU "error-bounded
 * LOSSY compressor for floating-point data". Lossy is fine for simulation
 * fields; it is NOT acceptable for model weights, which must round-trip
 * bit-exactly. This test therefore pins a LOSSLESS codec (zstd by default --
 * CompressionFactory registers it via CreateLossless<ZstdWithModes>) and
 * asserts EXACT equality on every byte, not a PSNR bound.
 *
 * WHY A CPU CODEC IS THE RIGHT CHOICE ON THE FAULT PATH. The documented
 * deadlock behind the older host-orchestrated workarounds is that a GPU
 * compressor's GetBlob services a fault by launching a decompression kernel --
 * so a spin-waiting on-device fault and the decompressor contend for the same
 * GPU. zstd/lz4 decompress on the HOST, so the fault is serviced entirely off
 * the GPU, exactly like the uncompressed kRam path that already works. The
 * compressed fault path and the deadlock hazard are therefore separable, and
 * this test covers the half that should work.
 *
 * Topology: the clio_cte_compressor chimod is composed IN FRONT of clio_cte_core
 * (next_pool_id 512.0), so PutBlob/GetBlob transparently compress/decompress.
 * The vector never sees it.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include "simple_test.h"

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <clio_ctp/util/gpu_api.h>
#include <clio_ctp/introspect/system_info.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>

using namespace std::chrono_literals;

namespace gv  = clio::cte::gpu_vector;
namespace dev = cte::gpu::dev;

namespace {

bool g_initialized = false;

constexpr clio::run::u64 kPageSizeBytes = 64 * 1024;
/** The compressor chimod's pool (compose: pool_id 600.0,
 *  next_pool_id 512.0). Page traffic routed here is compressed. */
inline clio::run::PoolId CompressorPool() {
  return clio::run::PoolId(600, 0);
}

constexpr clio::run::u32 kCachePages    = 8;

void EnsureInit() {
#if !CTP_IS_DEVICE_PASS
  if (g_initialized) return;
  const char *port_env = std::getenv("CLIO_PORT");
  int port = port_env ? std::atoi(port_env) : 10644;
  std::string cfg_path =
      "/tmp/gpu_vec_compress_fault_" + std::to_string(port) + ".yaml";
  {
    std::ofstream cfg(cfg_path);
    cfg << "networking:\n  port: " << port << "\n\n"
        << "runtime:\n  num_threads: 8\n  queue_depth: 65536\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n"
        << "    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"4096MB\"\n\n"
        // Compressor sits IN FRONT of cte_core: every blob is transparently
        // compressed on Put and decompressed on Get (i.e. on the fault path).
        << "  - mod_name: clio_cte_compressor\n"
        << "    pool_name: cte_compressor\n    pool_query: local\n"
        << "    pool_id: \"600.0\"\n    next_pool_id: \"512.0\"\n\n"
        << "  - mod_name: clio_cte_core\n"
        << "    pool_name: cte_core\n    pool_query: local\n    pool_id: \"512.0\"\n"
        << "    storage:\n      - path: \"ram::cte_ram_tier1\"\n"
        << "        bdev_type: \"ram\"\n        capacity_limit: \"4096MB\"\n"
        << "        score: 1.0\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
  }
  ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", cfg_path.c_str(), 1);

  // Pin a LOSSLESS codec unless the harness already chose one. Weights must
  // round-trip bit-exactly, so a lossy pin would be a silent correctness bug.
  if (std::getenv("CLIO_CTE_COMPRESS_LIB") == nullptr) {
    ctp::SystemInfo::Setenv("CLIO_CTE_COMPRESS_LIB", "zstd", 1);
  }
  std::fprintf(stderr, "[INIT] compressed-fault: compose=%s lib=%s\n",
               cfg_path.c_str(), std::getenv("CLIO_CTE_COMPRESS_LIB"));

  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
  std::this_thread::sleep_for(1s);  // let compose pools initialize
  g_initialized = true;
#endif
}

}  // namespace

/** Read the whole compressed-backed extent through read_range. */
__global__ void CompressFaultReadKernel(clio::run::IpcManagerGpuInfo info,
                                        gv::DeviceView<uint8_t> view,
                                        uint8_t *out, clio::run::u64 total) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  dev::vector<uint8_t> v(view, g_ipc_manager_ptr);
  if (blockIdx.x != 0) return;
  v.read_range(0, total, [out](clio::run::u64 i, uint8_t b) { out[i] = b; });
  (void)g_ipc_manager;
}

#if !CTP_IS_DEVICE_PASS

namespace {

/**
 * Compressible-but-realistic payload. Pure random bytes would not compress at
 * all (making the test vacuous: it would prove nothing about the decompression
 * path), and a constant would compress absurdly. This mixes a smooth ramp with
 * a small repeating perturbation, roughly like quantized weight data: real
 * entropy, real redundancy.
 */
uint8_t PayloadByte(clio::run::u64 i) {
  return static_cast<uint8_t>(((i / 64) & 0x3F) + ((i * 7u) & 0x0F));
}

/**
 * Populate `tag`, and PROVE the compressor actually ran.
 *
 * Without this check the test is vacuous: if the compressor chimod silently
 * passed data through (misconfigured compose, unknown CLIO_CTE_COMPRESS_LIB
 * falling back to none, backend compiled out), every byte would still
 * round-trip and every assertion downstream would still pass. PutBlobTask's
 * context_ is INOUT and carries actual_original_size_/actual_compressed_size_,
 * so the shrink is observable -- assert on it.
 */
clio::cte::core::TagId PutCompressedPages(const std::string &tag,
                                          clio::run::u64 nbytes,
                                          clio::run::u32 npages) {
  gv::Vector<uint8_t> vec(tag, /*nblocks=*/1, /*gpu_id=*/0, kCachePages,
                          /*host_pages_per_block=*/0, kPageSizeBytes,
                          /*cache_period_us=*/20000, gv::CacheMode::kLegacy,
                          /*manager_threads_per_block=*/32,
                          /*allow_cold_miss_fault=*/true,
                          /*storage_pool_id=*/CompressorPool());
  auto tag_id = vec.TagId();
  // PutBlob through the COMPRESSOR pool (600), not the core: the compressor
  // compresses and forwards to its next_pool_id (the core, 512), which is
  // also where the tag lives. Going straight to the core -- as this test
  // originally did -- stores the pages verbatim and silently tests nothing.
  clio::cte::core::Client cte_c(CompressorPool());
  auto *cte = &cte_c;
  clio::run::u64 orig_total = 0, comp_total = 0;

  for (clio::run::u32 p = 0; p < npages; ++p) {
    auto buf = CLIO_IPC->AllocateBuffer(kPageSizeBytes);
    auto *bytes = reinterpret_cast<uint8_t *>(buf.ptr_);
    std::memset(bytes, 0, kPageSizeBytes);
    const clio::run::u64 base = static_cast<clio::run::u64>(p) * kPageSizeBytes;
    const clio::run::u64 n =
        (base + kPageSizeBytes <= nbytes) ? kPageSizeBytes : (nbytes - base);
    for (clio::run::u64 j = 0; j < n; ++j) bytes[j] = PayloadByte(base + j);

    // dynamic_compress_ = 1 => STATIC: use the pinned library rather than
    // letting the predictor pick (which could pick a lossy one).
    clio::cte::core::Context ctx;
    ctx.dynamic_compress_ = 1;
    ctx.compress_preset_ = 2;  // BALANCED

    std::string blob_name = tag + "_b0_pi" + std::to_string(p);
    auto task = cte->AsyncPutBlob(tag_id, blob_name, 0, kPageSizeBytes,
                                  buf.shm_.template Cast<void>(), 1.0f, ctx, 0);
    task.Wait();
    REQUIRE(task->return_code_ == 0);
    orig_total += task->context_.actual_original_size_;
    comp_total += task->context_.actual_compressed_size_;
  }
  std::fprintf(stderr,
               "[PUT] %u pages -> '%s' | original=%llu B compressed=%llu B "
               "(%.2fx)\n",
               npages, tag.c_str(), (unsigned long long) orig_total,
               (unsigned long long) comp_total,
               comp_total ? (double) orig_total / (double) comp_total : 0.0);

  // The compressor must actually have run and actually have shrunk the data.
  // If either is false the round-trip below proves nothing about compression.
  REQUIRE(orig_total > 0);
  REQUIRE(comp_total > 0);
  REQUIRE(comp_total < orig_total);
  return tag_id;
}

}  // namespace

/**
 * The core claim: a kernel that only calls read_range reads an entire
 * losslessly-compressed dataset many times larger than its cache, bit-exactly,
 * with every page arriving via a demand fault that decompressed transparently.
 */
TEST_CASE("gpu_vector: transparent lossless-compressed page faulting",
          "[gpu_vector][cte][compress][fault][lossless]") {
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->GetGpuIpcManager() != nullptr);
  auto gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);

  // 128 pages against an 8-page cache == 16x oversubscribed.
  const clio::run::u32 npages = 128;
  const clio::run::u64 nbytes =
      static_cast<clio::run::u64>(npages) * kPageSizeBytes;
  const std::string tag = "gpu_vector_compress_fault";

  const auto tag_id_for_probe = PutCompressedPages(tag, nbytes, npages);

  // DISCRIMINATOR: read page 0 back HOST-side through the compressor, before
  // any GPU involvement. This separates two very different failures:
  //   - if this succeeds, storing/looking up compressed blobs is fine and the
  //     defect is specific to the device fault (PodGetBlob) path;
  //   - if this fails too, the compressor cannot find what it just stored, and
  //     the GPU path is an innocent bystander.
  {
    clio::cte::core::Client probe(CompressorPool());
    auto pbuf = CLIO_IPC->AllocateBuffer(kPageSizeBytes);
    std::memset(pbuf.ptr_, 0, kPageSizeBytes);
    std::string p0 = tag + "_b0_pi0";
    auto g = probe.AsyncGetBlob(tag_id_for_probe, p0, 0, kPageSizeBytes,
                               /*flags=*/0, pbuf.shm_.template Cast<void>(),
                               clio::run::PoolQuery::Dynamic());
    g.Wait();
    auto *pb = reinterpret_cast<uint8_t *>(pbuf.ptr_);
    clio::run::u64 pbad = 0;
    for (clio::run::u64 i = 0; i < kPageSizeBytes; ++i) {
      if (pb[i] != PayloadByte(i)) ++pbad;
    }
    std::fprintf(stderr,
                 "[PROBE] host-side compressed GetBlob '%s': rc=%d mismatches=%llu/%llu\n",
                 p0.c_str(), (int) g->return_code_,
                 (unsigned long long) pbad,
                 (unsigned long long) kPageSizeBytes);
    CLIO_IPC->FreeBuffer(pbuf);
  }

  auto *out = ctp::GpuApi::MallocHost<uint8_t>(nbytes);
  REQUIRE(out != nullptr);
  std::memset(out, 0, nbytes);

  std::fprintf(stderr,
               "[CFAULT] %llu B, %u pages, cache=%u (%.2fx oversubscribed)\n",
               (unsigned long long) nbytes, npages, kCachePages,
               static_cast<double>(npages) / kCachePages);
  {
    gv::Vector<uint8_t> vec(tag, /*nblocks=*/1, /*gpu_id=*/0, kCachePages,
                            /*host_pages_per_block=*/0, kPageSizeBytes,
                            /*cache_period_us=*/20000, gv::CacheMode::kLegacy,
                            /*manager_threads_per_block=*/32,
                            /*allow_cold_miss_fault=*/true,
                            /*storage_pool_id=*/CompressorPool());
    CompressFaultReadKernel<<<1, 32>>>(gpu_info, vec.Device(), out, nbytes);
    ctp::GpuApi::Synchronize();

    auto s = vec.StatsSnapshot();
    std::fprintf(stderr,
                 "[CFAULT] resolve=%llu cold=%llu fault=%llu hits=%llu\n",
                 s.resolve_total, s.resolve_cold_miss, s.resolve_fault_get,
                 s.resolve_hits);
    REQUIRE(s.resolve_fault_get == npages);
    REQUIRE(s.resolve_cold_miss == npages);
    REQUIRE(s.resolve_hits == 0);
  }

  // LOSSLESS means exact. No tolerance, no PSNR.
  clio::run::u64 bad = 0;
  for (clio::run::u64 i = 0; i < nbytes; ++i) {
    if (out[i] != PayloadByte(i)) {
      if (bad < 4) {
        std::fprintf(stderr, "[CFAULT] mismatch @%llu: got %u want %u\n",
                     (unsigned long long) i, out[i], PayloadByte(i));
      }
      ++bad;
    }
  }
  std::fprintf(stderr, "[CFAULT] mismatches=%llu / %llu\n",
               (unsigned long long) bad, (unsigned long long) nbytes);
  REQUIRE(bad == 0);
  ctp::GpuApi::FreeHost(out);
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

#else

int main() { return 0; }

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
