/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * MAPPING MODEL TENSORS AS GPU VECTORS, across ggml tensor formats.
 *
 * A "tensor mapped as a vector" means: the tensor's raw bytes live in CTE as
 * page blobs under a tag, a gpu_vector is opened over that tag, and a GPU
 * kernel reads the whole tensor back through read_range -- faulting pages in on
 * demand because the cache is deliberately smaller than the tensor.
 *
 * WHY RAW BYTES (Vector<uint8_t>) FOR QUANTIZED FORMATS. Only F32/F16/BF16 are
 * scalar-per-element. Everything else is BLOCK quantized: Q4_K packs 256
 * weights into a 144-byte super-block (2 halves + 12 scale bytes + 128 nibble
 * bytes), so there is no C++ element type whose sizeof() is the "element" size.
 * The honest mapping is therefore bytes, with the format table below recording
 * each type's real (block_elems, block_bytes) so tensor sizes are computed the
 * way ggml computes them rather than guessed. Sizes are taken from
 * ggml-common.h's own static_asserts (K_SCALE_SIZE=12, QK_K=256, QK4_0=32).
 *
 * WHY SYNTHETIC TENSORS AND NOT A .gguf FILE. These are unit tests: they must
 * run everywhere, so they cannot depend on a multi-GB model being present. The
 * shape used below (4864 x 896) is qwen2.5-0.5b's blk.N.ffn_down.weight and the
 * byte layouts are ggml-exact, so the geometry exercised is the real thing.
 *
 * NOT COVERED HERE: reading tensor extents out of an actual .gguf. That needs a
 * GGUF header parser (or linking ggml into a CUDA TU) and is tracked separately;
 * end-to-end against a real model file currently goes through the CAE
 * ModelWeightsAssimilator path in clio_model_weights_roundtrip.cc.
 *
 * The last page of a tensor is almost never full, so it is zero-padded on write
 * and only the real byte range is verified -- same as the CAE assimilator does.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include "simple_test.h"

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <clio_ctp/util/gpu_api.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace gv  = clio::cte::gpu_vector;
namespace dev = cte::gpu::dev;

namespace {

bool g_initialized = false;

// 64 KiB pages: the geometry the model-weights assimilator uses.
constexpr clio::run::u64 kPageSizeBytes = 64 * 1024;
// Cache deliberately far smaller than any tensor below, so every case faults.
constexpr clio::run::u32 kCachePages = 8;

void EnsureInit() {
#if !CTP_IS_DEVICE_PASS
  if (g_initialized) return;
  std::fprintf(stderr, "[INIT] Starting Clio server (tensor map test)\n");
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);
  cte_client->Init(clio::cte::core::kCtePoolId);
  clio::cte::core::CreateParams params;
  auto create_task = cte_client->AsyncCreate(
      clio::run::PoolQuery::Dynamic(), clio::cte::core::kCtePoolName,
      clio::cte::core::kCtePoolId, params);
  create_task.Wait();
  REQUIRE(create_task->GetReturnCode() == 0);
  std::this_thread::sleep_for(50ms);

  const clio::run::u64 kRamCapacity = 2ULL << 30;
  clio::run::PoolId bdev_pool_id(972, 0);
  clio::run::bdev::Client bdev_client(bdev_pool_id);
  auto bdev_create = bdev_client.AsyncCreate(
      clio::run::PoolQuery::Dynamic(), std::string("tensor_map_ram"),
      bdev_pool_id, clio::run::bdev::BdevType::kRam, kRamCapacity);
  bdev_create.Wait();
  REQUIRE(bdev_create->GetReturnCode() == 0);
  std::this_thread::sleep_for(50ms);
  auto reg_task = cte_client->AsyncRegisterTarget(
      "tensor_map_ram", clio::run::bdev::BdevType::kRam, kRamCapacity,
      clio::run::PoolQuery::Local(), bdev_pool_id);
  reg_task.Wait();
  REQUIRE(reg_task->GetReturnCode() == 0);
  std::this_thread::sleep_for(50ms);

  g_initialized = true;
#endif
}

}  // namespace

/** Read the whole mapped tensor [0,total) as raw bytes via read_range. */
__global__ void TensorReadKernel(clio::run::IpcManagerGpuInfo info,
                                 gv::DeviceView<uint8_t> view,
                                 uint8_t *out, clio::run::u64 total) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  dev::vector<uint8_t> v(view, g_ipc_manager_ptr);
  if (blockIdx.x != 0) return;
  v.read_range(0, total, [out](clio::run::u64 i, uint8_t b) { out[i] = b; });
  (void)g_ipc_manager;
}

/** Typed (scalar-format) mapping: read the tensor as float elements. */
__global__ void TensorReadF32Kernel(clio::run::IpcManagerGpuInfo info,
                                    gv::DeviceView<float> view,
                                    float *out, clio::run::u64 nelem) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  dev::vector<float> v(view, g_ipc_manager_ptr);
  if (blockIdx.x != 0) return;
  v.read_range(0, nelem, [out](clio::run::u64 i, float f) { out[i] = f; });
  (void)g_ipc_manager;
}

#if !CTP_IS_DEVICE_PASS

namespace {

/** A ggml tensor type's block geometry (from ggml-common.h static_asserts). */
struct GgmlFormat {
  const char *name;
  clio::run::u32 block_elems;  ///< weights per block (1 for scalar types)
  clio::run::u32 block_bytes;  ///< encoded bytes per block
};

// K_SCALE_SIZE=12, QK_K=256, QK4_0=QK8_0=32.
constexpr GgmlFormat kFormats[] = {
    {"F32",  1,   4},    // scalar
    {"F16",  1,   2},    // scalar
    {"BF16", 1,   2},    // scalar
    {"Q4_0", 32,  18},   // half + 16 nibble bytes
    {"Q8_0", 32,  34},   // half + 32 int8
    {"Q4_K", 256, 144},  // 2 half + 12 scales + 128
    {"Q5_K", 256, 176},  // 2 half + 12 scales + 128 + 32
    {"Q6_K", 256, 210},  // half + 16 scales + 192
};

/** Bytes ggml would use for `nelem` weights in `f`. */
clio::run::u64 FormatBytes(const GgmlFormat &f, clio::run::u64 nelem) {
  REQUIRE(nelem % f.block_elems == 0);
  return (nelem / f.block_elems) * f.block_bytes;
}

/** Deterministic, format-agnostic byte pattern for a tensor's payload. */
uint8_t TensorByte(clio::run::u64 i) {
  return static_cast<uint8_t>((i * 131u + 7u) & 0xFFu);
}

/**
 * Write `nbytes` of tensor payload into `tag` as "<tag>_b0_pi<page>" blobs --
 * the exact naming/geometry gpu_vector resolves against, and what the CAE
 * ModelWeightsAssimilator produces. The final page is zero-padded.
 */
void PutTensorPages(const std::string &tag, clio::run::u64 nbytes,
                    clio::run::u32 npages) {
  gv::Vector<uint8_t> vec(tag, /*nblocks=*/1, /*gpu_id=*/0,
                          /*gpu_pages_per_block=*/kCachePages,
                          /*host_pages_per_block=*/0, kPageSizeBytes,
                          /*cache_period_us=*/20000, gv::CacheMode::kLegacy,
                          /*manager_threads_per_block=*/32,
                          /*allow_cold_miss_fault=*/true);
  auto tag_id = vec.TagId();
  auto *cte = CLIO_CTE_CLIENT;

  for (clio::run::u32 p = 0; p < npages; ++p) {
    auto buf = CLIO_IPC->AllocateBuffer(kPageSizeBytes);
    auto *bytes = reinterpret_cast<uint8_t *>(buf.ptr_);
    std::memset(bytes, 0, kPageSizeBytes);
    const clio::run::u64 base = static_cast<clio::run::u64>(p) * kPageSizeBytes;
    const clio::run::u64 n =
        (base + kPageSizeBytes <= nbytes) ? kPageSizeBytes : (nbytes - base);
    for (clio::run::u64 j = 0; j < n; ++j) bytes[j] = TensorByte(base + j);
    std::string blob_name = tag + "_b0_pi" + std::to_string(p);
    auto task = cte->AsyncPutBlob(tag_id, blob_name, 0, kPageSizeBytes,
                                  buf.shm_.template Cast<void>(), 1.0f,
                                  clio::cte::core::Context(), 0);
    task.Wait();
    REQUIRE(task->return_code_ == 0);
  }
}

/**
 * Map `tag` as a byte vector with a cache SMALLER than the tensor and read the
 * whole thing back in one kernel. Returns after asserting bit-exactness and
 * that every page really arrived through a device fault.
 */
void MapAndVerifyTensor(const std::string &label, const std::string &tag,
                        clio::run::u64 nbytes, clio::run::u32 npages,
                        clio::run::IpcManagerGpuInfo gpu_info) {
  auto *out = ctp::GpuApi::MallocHost<uint8_t>(nbytes);
  REQUIRE(out != nullptr);
  std::memset(out, 0, nbytes);

  std::fprintf(stderr,
               "[%s] %llu B, %u pages, cache=%u pages (%.2fx oversubscribed)\n",
               label.c_str(), (unsigned long long) nbytes, npages, kCachePages,
               static_cast<double>(npages) / kCachePages);
  {
    gv::Vector<uint8_t> vec(tag, /*nblocks=*/1, /*gpu_id=*/0, kCachePages,
                            /*host_pages_per_block=*/0, kPageSizeBytes,
                            /*cache_period_us=*/20000, gv::CacheMode::kLegacy,
                            /*manager_threads_per_block=*/32,
                            /*allow_cold_miss_fault=*/true);
    TensorReadKernel<<<1, 32>>>(gpu_info, vec.Device(), out, nbytes);
    ctp::GpuApi::Synchronize();

    auto s = vec.StatsSnapshot();
    std::fprintf(stderr,
                 "[%s] resolve=%llu cold=%llu fault=%llu hits=%llu\n",
                 label.c_str(), s.resolve_total, s.resolve_cold_miss,
                 s.resolve_fault_get, s.resolve_hits);
    // Every page of the tensor must have been demand-faulted.
    REQUIRE(s.resolve_fault_get == npages);
    REQUIRE(s.resolve_cold_miss == npages);
    REQUIRE(s.resolve_hits == 0);
  }

  clio::run::u64 bad = 0;
  for (clio::run::u64 i = 0; i < nbytes; ++i) {
    if (out[i] != TensorByte(i)) {
      if (bad < 4) {
        std::fprintf(stderr, "[%s] mismatch @%llu: got %u want %u\n",
                     label.c_str(), (unsigned long long) i, out[i],
                     TensorByte(i));
      }
      ++bad;
    }
  }
  std::fprintf(stderr, "[%s] mismatches=%llu / %llu\n", label.c_str(),
               (unsigned long long) bad, (unsigned long long) nbytes);
  REQUIRE(bad == 0);
  ctp::GpuApi::FreeHost(out);
}

}  // namespace

/**
 * Every ggml format, mapped and demand-faulted. Shape is a real qwen2.5-0.5b
 * ffn_down row count x a hidden size, chosen so each format lands in the
 * few-MB range -- i.e. tens of pages against an 8-page cache.
 */
TEST_CASE("gpu_vector: map tensors of every ggml format and fault them in",
          "[gpu_vector][cte][tensor][fault]") {
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->GetGpuIpcManager() != nullptr);
  auto gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);

  // 4864 x 896 == qwen2.5-0.5b blk.N.ffn_down.weight. Divisible by 256 so
  // every K-quant super-block is whole.
  const clio::run::u64 nelem = 4864ULL * 896ULL;

  for (const auto &f : kFormats) {
    const clio::run::u64 nbytes = FormatBytes(f, nelem);
    const clio::run::u32 npages =
        static_cast<clio::run::u32>((nbytes + kPageSizeBytes - 1) /
                                    kPageSizeBytes);
    std::string tag = std::string("tensor_map_") + f.name;
    std::fprintf(stderr, "\n=== format %s: %llu elems -> %llu B (%u B/%u elem)\n",
                 f.name, (unsigned long long) nelem,
                 (unsigned long long) nbytes, f.block_bytes, f.block_elems);
    PutTensorPages(tag, nbytes, npages);
    MapAndVerifyTensor(f.name, tag, nbytes, npages, gpu_info);
  }
}

/**
 * Scalar-format tensors can also be mapped with their natural element type
 * rather than as bytes. This checks Vector<float> element indexing lines up
 * with the byte layout (i.e. mapping is not silently off by a stride).
 */
TEST_CASE("gpu_vector: map an F32 tensor with its natural element type",
          "[gpu_vector][cte][tensor][fault]") {
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;
  auto gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);

  const clio::run::u64 nelem = 4864ULL * 896ULL / 8;  // keep it quick
  const clio::run::u64 nbytes = nelem * sizeof(float);
  const clio::run::u32 npages =
      static_cast<clio::run::u32>((nbytes + kPageSizeBytes - 1) /
                                  kPageSizeBytes);
  const std::string tag = "tensor_map_f32_typed";
  PutTensorPages(tag, nbytes, npages);

  auto *out = ctp::GpuApi::MallocHost<float>(nbytes);
  REQUIRE(out != nullptr);
  std::memset(out, 0, nbytes);
  {
    gv::Vector<float> vec(tag, /*nblocks=*/1, /*gpu_id=*/0, kCachePages,
                          /*host_pages_per_block=*/0, kPageSizeBytes,
                          /*cache_period_us=*/20000, gv::CacheMode::kLegacy,
                          /*manager_threads_per_block=*/32,
                          /*allow_cold_miss_fault=*/true);
    TensorReadF32Kernel<<<1, 32>>>(gpu_info, vec.Device(), out, nelem);
    ctp::GpuApi::Synchronize();
    auto s = vec.StatsSnapshot();
    std::fprintf(stderr, "[F32-TYPED] fault=%llu of %u pages, hits=%llu\n",
                 s.resolve_fault_get, npages, s.resolve_hits);
    REQUIRE(s.resolve_fault_get == npages);
    REQUIRE(s.resolve_hits == 0);
  }

  // Each float must equal the 4 payload bytes at its element offset.
  clio::run::u64 bad = 0;
  for (clio::run::u64 e = 0; e < nelem; ++e) {
    uint8_t want[4];
    for (int b = 0; b < 4; ++b) want[b] = TensorByte(e * 4 + b);
    float wf;
    std::memcpy(&wf, want, 4);
    float got = out[e];
    if (std::memcmp(&got, &wf, 4) != 0) ++bad;
  }
  std::fprintf(stderr, "[F32-TYPED] element mismatches=%llu / %llu\n",
               (unsigned long long) bad, (unsigned long long) nelem);
  REQUIRE(bad == 0);
  ctp::GpuApi::FreeHost(out);
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

#else

int main() { return 0; }

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
