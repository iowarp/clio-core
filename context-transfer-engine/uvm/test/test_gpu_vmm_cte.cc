/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * GpuVmm with use_cte=true (issue #1190): evicted pages become CTE blobs and
 * come back intact through both the synchronous and the asynchronous touch,
 * and destroy() removes the blobs. Before the fix the CTE path was never
 * compiled, so use_cte was silently ignored, and the async restore freed its
 * staging buffer while the copy was still reading it.
 *
 * Host-only: the page contents are written and read with cuMemcpy, so no
 * kernel (and no device architecture) is involved.
 */

#include <cuda.h>

#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

#include "../../../context-runtime/test/simple_test.h"
#include "clio_cte/core/core_client.h"
#include "clio_cte/uvm/gpu_vmm.h"
#include "clio_runtime/bdev/bdev_client.h"
#include "clio_runtime/clio_runtime.h"

namespace {

constexpr size_t kVaBytes = 64ULL * 1024 * 1024;
const char *kTag = "gpu_vmm_cte_test";

/** Bring up an in-process runtime with a RAM target for the CTE. */
bool InitCte() {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    return false;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) return false;
  auto *cte = CLIO_CTE_CLIENT;
  clio::run::PoolId bdev_id(953, 0);
  clio::run::bdev::Client bdev(bdev_id);
  auto create = bdev.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                 "ram::gpu_vmm_cte_test", bdev_id,
                                 clio::run::bdev::BdevType::kRam);
  create.Wait();
  auto reg = cte->AsyncRegisterTarget(
      "ram::gpu_vmm_cte_test", clio::run::bdev::BdevType::kRam,
      256ULL * 1024 * 1024, clio::run::PoolQuery::Local(), bdev_id);
  reg.Wait();
  return reg->GetReturnCode() == 0;
}

/** Fill a page with a recognisable pattern. */
std::vector<unsigned> Pattern(size_t page_bytes, unsigned seed) {
  std::vector<unsigned> v(page_bytes / sizeof(unsigned));
  for (size_t i = 0; i < v.size(); ++i) v[i] = seed * 2654435761u + i;
  return v;
}

/** Read a mapped page back to the host. */
std::vector<unsigned> ReadPage(clio::cte::uvm::GpuVirtualMemoryManager &vmm,
                               size_t page) {
  std::vector<unsigned> v(vmm.getPageSize() / sizeof(unsigned));
  REQUIRE(cuMemcpyDtoH(v.data(), vmm.getPagePtr(page), vmm.getPageSize()) ==
          CUDA_SUCCESS);
  return v;
}

/** Size of a page blob in the test tag (0 when absent). */
clio::run::u64 BlobSize(const clio::cte::core::TagId &tag, size_t page) {
  auto *cte = CLIO_CTE_CLIENT;
  auto t = cte->AsyncGetBlobSize(tag, "page_" + std::to_string(page));
  t.Wait();
  return t->GetReturnCode() == 0 ? t->size_ : 0;
}

}  // namespace

TEST_CASE("GpuVmm use_cte: pages round-trip through CTE", "[uvm][cte]") {
  REQUIRE(InitCte());

  clio::cte::uvm::GpuVmmConfig cfg;
  cfg.va_size_bytes = kVaBytes;
  cfg.prefetch_window = 0;  // only the pages this test touches
  cfg.use_cte = true;
  cfg.cte_tag_name = kTag;
  clio::cte::uvm::GpuVirtualMemoryManager vmm;
  REQUIRE(vmm.init(cfg) == CUDA_SUCCESS);
  const size_t page = vmm.getPageSize();
  clio::cte::core::Tag tag(kTag);

  // Write two pages and evict them: one synchronously, one asynchronously.
  auto p0 = Pattern(page, 1);
  auto p1 = Pattern(page, 2);
  REQUIRE(vmm.touchPage(0) == CUDA_SUCCESS);
  REQUIRE(vmm.touchPage(1) == CUDA_SUCCESS);
  REQUIRE(cuMemcpyHtoD(vmm.getPagePtr(0), p0.data(), page) == CUDA_SUCCESS);
  REQUIRE(cuMemcpyHtoD(vmm.getPagePtr(1), p1.data(), page) == CUDA_SUCCESS);
  REQUIRE(vmm.evictPage(0) == CUDA_SUCCESS);
  REQUIRE(vmm.evictPageAsync(1) == CUDA_SUCCESS);
  REQUIRE(vmm.isEvictedToHost(0));
  REQUIRE(vmm.isEvictedToHost(1));

  // The bytes now live in CTE, not in host RAM.
  REQUIRE(BlobSize(tag.GetTagId(), 0) == page);
  REQUIRE(BlobSize(tag.GetTagId(), 1) == page);

  // Synchronous restore.
  REQUIRE(vmm.touchPage(0) == CUDA_SUCCESS);
  REQUIRE(ReadPage(vmm, 0) == p0);

  // Asynchronous restore: the staging buffer must stay valid until the copy
  // ran. Reuse the freed staging space at once, so a premature free shows
  // up as corrupted page bytes.
  REQUIRE(vmm.touchPageAsync(1) == CUDA_SUCCESS);
  auto *ipc = CLIO_CPU_IPC;
  std::vector<ctp::ipc::FullPtr<char>> scribble;
  for (int i = 0; i < 4; ++i) {
    auto b = ipc->AllocateBuffer(page);
    if (b.IsNull()) break;
    std::memset(b.ptr_, 0xEE, page);
    scribble.push_back(b);
  }
  vmm.syncTransfer();
  for (auto &b : scribble) ipc->FreeBuffer(b);
  REQUIRE(ReadPage(vmm, 1) == p1);

  // destroy() removes the page blobs.
  vmm.destroy();
  REQUIRE(BlobSize(tag.GetTagId(), 0) == 0);
  REQUIRE(BlobSize(tag.GetTagId(), 1) == 0);
}

SIMPLE_TEST_MAIN()
