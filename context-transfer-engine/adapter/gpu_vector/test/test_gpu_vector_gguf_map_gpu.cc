/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * MAPPING REAL MODEL TENSORS (qwen2.5 .gguf) AS GPU VECTORS.
 *
 * The sibling test_gpu_vector_tensor_map covers every ggml block layout using
 * synthetic tensors with ggml-exact geometry. This one closes the remaining
 * gap: an ACTUAL model file. It reads the GGUF tensor table, and for a
 * selection of real tensors (one per distinct ggml type) it
 *
 *   file bytes -> CTE page blobs -> gpu_vector -> read_range in a GPU kernel
 *
 * with the cache deliberately far smaller than the tensor, so every page is
 * demand-faulted, and verifies the bytes the GPU saw are bit-identical to the
 * bytes in the file.
 *
 * OPT-IN. Multi-GB models cannot be a CI dependency, so the test is skipped
 * (passing, with a printed reason) unless CLIO_TEST_GGUF names a readable file.
 * Point it at e.g. qwen2.5-0.5b-instruct-fp16.gguf.
 *
 * The GGUF parser (gguf_lite.h) was validated against llama.cpp's own gguf-py
 * reader: identical tensor count and byte-identical offsets/sizes.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include "simple_test.h"
#include "gguf_lite.h"

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <clio_ctp/util/gpu_api.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace gv  = clio::cte::gpu_vector;
namespace dev = cte::gpu::dev;

namespace {

bool g_initialized = false;

constexpr clio::run::u64 kPageSizeBytes = 64 * 1024;
constexpr clio::run::u32 kCachePages = 8;
// Keep a unit test quick: skip tensors bigger than this (qwen2.5-0.5b's
// token_embd is 272 MB, which measures PutBlob throughput, not faulting).
constexpr clio::run::u64 kMaxTensorBytes = 32ull << 20;
// ...and big enough that the cache is genuinely oversubscribed.
constexpr clio::run::u64 kMinTensorBytes = 4ull * kPageSizeBytes;

void EnsureInit() {
#if !CTP_IS_DEVICE_PASS
  if (g_initialized) return;
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
  clio::run::PoolId bdev_pool_id(974, 0);
  clio::run::bdev::Client bdev_client(bdev_pool_id);
  auto bdev_create = bdev_client.AsyncCreate(
      clio::run::PoolQuery::Dynamic(), std::string("gguf_map_ram"),
      bdev_pool_id, clio::run::bdev::BdevType::kRam, kRamCapacity);
  bdev_create.Wait();
  REQUIRE(bdev_create->GetReturnCode() == 0);
  std::this_thread::sleep_for(50ms);
  auto reg_task = cte_client->AsyncRegisterTarget(
      "gguf_map_ram", clio::run::bdev::BdevType::kRam, kRamCapacity,
      clio::run::PoolQuery::Local(), bdev_pool_id);
  reg_task.Wait();
  REQUIRE(reg_task->GetReturnCode() == 0);
  std::this_thread::sleep_for(50ms);
  g_initialized = true;
#endif
}

}  // namespace

/** Read the whole mapped tensor back as raw bytes via read_range. */
__global__ void GgufReadKernel(clio::run::IpcManagerGpuInfo info,
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

/** Store `bytes` as "<tag>_b0_pi<page>" blobs; last page zero-padded. */
void PutTensorBytes(const std::string &tag, const std::vector<uint8_t> &bytes,
                    clio::run::u32 npages) {
  gv::Vector<uint8_t> vec(tag, /*nblocks=*/1, /*gpu_id=*/0, kCachePages,
                          /*host_pages_per_block=*/0, kPageSizeBytes,
                          /*cache_period_us=*/20000, gv::CacheMode::kLegacy,
                          /*manager_threads_per_block=*/32,
                          /*allow_cold_miss_fault=*/true);
  auto tag_id = vec.TagId();
  auto *cte = CLIO_CTE_CLIENT;
  const clio::run::u64 nbytes = bytes.size();

  for (clio::run::u32 p = 0; p < npages; ++p) {
    auto buf = CLIO_IPC->AllocateBuffer(kPageSizeBytes);
    auto *dst = reinterpret_cast<uint8_t *>(buf.ptr_);
    std::memset(dst, 0, kPageSizeBytes);
    const clio::run::u64 base = static_cast<clio::run::u64>(p) * kPageSizeBytes;
    const clio::run::u64 n =
        (base + kPageSizeBytes <= nbytes) ? kPageSizeBytes : (nbytes - base);
    std::memcpy(dst, bytes.data() + base, (size_t) n);
    std::string blob_name = tag + "_b0_pi" + std::to_string(p);
    auto task = cte->AsyncPutBlob(tag_id, blob_name, 0, kPageSizeBytes,
                                  buf.shm_.template Cast<void>(), 1.0f,
                                  clio::cte::core::Context(), 0);
    task.Wait();
    REQUIRE(task->return_code_ == 0);
  }
}

}  // namespace

TEST_CASE("gpu_vector: map real .gguf model tensors and demand-fault them",
          "[gpu_vector][cte][tensor][gguf][fault]") {
  const char *path_env = std::getenv("CLIO_TEST_GGUF");
  if (path_env == nullptr || path_env[0] == '\0') {
    std::fprintf(stderr,
                 "[GGUF] SKIP: set CLIO_TEST_GGUF=<model.gguf> to run this "
                 "(multi-GB models are not a CI dependency)\n");
    return;
  }
  const std::string path = path_env;

  std::vector<gguf_lite::TensorInfo> tensors;
  std::string err;
  if (!gguf_lite::ReadTensorTable(path, &tensors, &err)) {
    std::fprintf(stderr, "[GGUF] SKIP: cannot parse '%s': %s\n", path.c_str(),
                 err.c_str());
    return;
  }
  std::fprintf(stderr, "[GGUF] %s: %zu tensors\n", path.c_str(),
               tensors.size());

  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->GetGpuIpcManager() != nullptr);
  auto gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);

  // One tensor per distinct ggml type, within the size window.
  std::set<uint32_t> seen_types;
  std::vector<const gguf_lite::TensorInfo *> picked;
  for (const auto &t : tensors) {
    if (t.nbytes < kMinTensorBytes || t.nbytes > kMaxTensorBytes) continue;
    if (gguf_lite::FormatForGgmlType(t.ggml_type)->block_bytes == 0) continue;
    if (!seen_types.insert(t.ggml_type).second) continue;
    picked.push_back(&t);
  }
  std::fprintf(stderr, "[GGUF] selected %zu tensors covering %zu ggml types\n",
               picked.size(), seen_types.size());
  // A model with no tensor in the window would make this vacuous.
  REQUIRE(!picked.empty());

  std::ifstream f(path, std::ios::binary);
  REQUIRE(f.good());

  int idx = 0;
  for (const auto *t : picked) {
    const char *tname = gguf_lite::FormatForGgmlType(t->ggml_type)->name;
    std::vector<uint8_t> src((size_t) t->nbytes);
    f.seekg((std::streamoff) t->file_offset);
    REQUIRE(f.good());
    f.read(reinterpret_cast<char *>(src.data()), (std::streamsize) t->nbytes);
    REQUIRE(f.good());

    const clio::run::u32 npages = (clio::run::u32) ((t->nbytes +
        kPageSizeBytes - 1) / kPageSizeBytes);
    const std::string tag = "gguf_map_" + std::to_string(idx++);
    std::fprintf(stderr,
                 "\n[GGUF] '%s' type=%s %llu B, %u pages, cache=%u "
                 "(%.2fx oversubscribed)\n",
                 t->name.c_str(), tname, (unsigned long long) t->nbytes,
                 npages, kCachePages,
                 (double) npages / kCachePages);

    PutTensorBytes(tag, src, npages);

    auto *out = ctp::GpuApi::MallocHost<uint8_t>(t->nbytes);
    REQUIRE(out != nullptr);
    std::memset(out, 0, (size_t) t->nbytes);
    {
      gv::Vector<uint8_t> vec(tag, /*nblocks=*/1, /*gpu_id=*/0, kCachePages,
                              /*host_pages_per_block=*/0, kPageSizeBytes,
                              /*cache_period_us=*/20000, gv::CacheMode::kLegacy,
                              /*manager_threads_per_block=*/32,
                              /*allow_cold_miss_fault=*/true);
      GgufReadKernel<<<1, 32>>>(gpu_info, vec.Device(), out, t->nbytes);
      ctp::GpuApi::Synchronize();
      auto s = vec.StatsSnapshot();
      std::fprintf(stderr, "[GGUF] fault=%llu/%u cold=%llu hits=%llu\n",
                   s.resolve_fault_get, npages, s.resolve_cold_miss,
                   s.resolve_hits);
      // Every page arrived by demand fault -- nothing was resident, nothing
      // was prefetched.
      REQUIRE(s.resolve_fault_get == npages);
      REQUIRE(s.resolve_hits == 0);
    }

    clio::run::u64 bad = 0;
    for (clio::run::u64 i = 0; i < t->nbytes; ++i) {
      if (out[i] != src[i]) ++bad;
    }
    std::fprintf(stderr, "[GGUF] '%s' mismatches=%llu / %llu\n",
                 t->name.c_str(), (unsigned long long) bad,
                 (unsigned long long) t->nbytes);
    REQUIRE(bad == 0);
    ctp::GpuApi::FreeHost(out);
  }
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

#else

int main() { return 0; }

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
