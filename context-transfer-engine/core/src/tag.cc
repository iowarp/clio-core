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

#include <clio_cte/core/core_client.h>
#include <clio_ctp/util/logging.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace clio::cte::core {

// AggregateOut timing for Tag::PutBlob(const char*) — broken down into the
// four steps that path performs per call (alloc shm, memcpy, sync RPC,
// free). Counters are flushed by FlushPutBlobTiming(); the POSIX adapter
// calls that at the end of each Filesystem::Write so we get one summary
// line per file-write operation.
namespace {
std::atomic<uint64_t> g_pb_calls{0};
std::atomic<uint64_t> g_pb_bytes{0};
std::atomic<uint64_t> g_pb_alloc_ns{0};
std::atomic<uint64_t> g_pb_memcpy_ns{0};
std::atomic<uint64_t> g_pb_rpc_ns{0};
std::atomic<uint64_t> g_pb_free_ns{0};
}  // namespace

void FlushPutBlobTiming(const char *label) {
  uint64_t n = g_pb_calls.exchange(0);
  if (n == 0) return;
  uint64_t bytes = g_pb_bytes.exchange(0);
  uint64_t alloc_ns = g_pb_alloc_ns.exchange(0);
  uint64_t memcpy_ns = g_pb_memcpy_ns.exchange(0);
  uint64_t rpc_ns = g_pb_rpc_ns.exchange(0);
  uint64_t free_ns = g_pb_free_ns.exchange(0);
  uint64_t total_ns = alloc_ns + memcpy_ns + rpc_ns + free_ns;
  double total_ms = total_ns / 1e6;
  double bw_mbs = (total_ns > 0) ? (bytes * 1e3 / total_ns) : 0.0;
  HLOG(kInfo,
       "[PutBlob breakdown {}] calls={} bytes={} total={:.2f}ms ({:.1f} MB/s) "
       "alloc={:.2f}ms ({:.1f}%) memcpy={:.2f}ms ({:.1f}%) "
       "rpc={:.2f}ms ({:.1f}%) free={:.2f}ms ({:.1f}%) | "
       "per-call alloc={:.1f}us memcpy={:.1f}us rpc={:.1f}us free={:.1f}us",
       label, n, bytes, total_ms, bw_mbs,
       alloc_ns / 1e6, 100.0 * alloc_ns / total_ns,
       memcpy_ns / 1e6, 100.0 * memcpy_ns / total_ns,
       rpc_ns / 1e6, 100.0 * rpc_ns / total_ns,
       free_ns / 1e6, 100.0 * free_ns / total_ns,
       alloc_ns / 1e3 / n, memcpy_ns / 1e3 / n,
       rpc_ns / 1e3 / n, free_ns / 1e3 / n);
}

Tag::Tag(const std::string &tag_name) : tag_name_(tag_name) {
  auto *cte_client = CLIO_CTE_CLIENT;
  auto task = cte_client->AsyncGetOrCreateTag(tag_name);
  task.Wait();

  if (task->GetReturnCode() != 0) {
    throw std::runtime_error("GetOrCreateTag operation failed");
  }

  tag_id_ = task->tag_id_;
}

Tag::Tag(const TagId &tag_id) : tag_id_(tag_id), tag_name_("") {}

void Tag::PutBlob(const std::string &blob_name, const char *data, size_t data_size,
                  size_t off, float score, const Context &context) {
  using clk = std::chrono::steady_clock;
  auto t0 = clk::now();

  // Private-memory path (issue #830): hand the caller's buffer straight to the
  // client, which picks the runtime-mode no-copy path or client-mode SHM
  // staging (TASK_DATA_OWNER) itself — no manual allocate/copy/free at this
  // layer. Write-side analog of the #823 private GetBlob.
  auto fut = AsyncPutBlob(blob_name, data, data_size, off, score, context);
  fut.Wait();

  auto t3 = clk::now();

  // Empty future -> client-mode SHM staging could not be allocated.
  if (fut.get() == nullptr) {
    throw std::runtime_error("Failed to allocate shared memory for PutBlob");
  }
  if (fut.get()->GetReturnCode() != 0) {
    // Log the actual return code so callers (e.g. the POSIX adapter, IOR,
    // wfbench) can correlate "PutBlob operation failed" with the failure mode.
    HLOG(kError,
         "PutBlob failed: tag_id={},{} blob='{}' off={} size={} rc={}",
         tag_id_.major_, tag_id_.minor_, blob_name, off, data_size,
         fut.get()->GetReturnCode());
    throw std::runtime_error(
        std::string("PutBlob operation failed (rc=") +
        std::to_string(fut.get()->GetReturnCode()) + ")");
  }

  auto ns = [](clk::time_point a, clk::time_point b) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
  };
  // The allocate/copy/free now happen inside the client (or not at all, in
  // runtime mode), so only the aggregate call time is meaningful here.
  g_pb_calls.fetch_add(1, std::memory_order_relaxed);
  g_pb_bytes.fetch_add(data_size, std::memory_order_relaxed);
  g_pb_rpc_ns.fetch_add(ns(t0, t3), std::memory_order_relaxed);
}

void Tag::PutBlob(const std::string &blob_name, const ctp::ipc::ShmPtr<> &data, size_t data_size,
                  size_t off, float score, const Context &context) {
  auto *cte_client = CLIO_CTE_CLIENT;
  auto task = cte_client->AsyncPutBlob(tag_id_, blob_name,
                                       off, data_size, data, score, context, 0,
                                       clio::run::PoolQuery::Dynamic());
  task.Wait();

  if (task->GetReturnCode() != 0) {
    // Log the actual return code so callers (e.g. the POSIX adapter,
    // IOR, wfbench) can correlate "PutBlob operation failed" exceptions
    // with the specific failure mode on the runtime side.
    HLOG(kError,
         "PutBlob failed: tag_id={},{} blob='{}' off={} size={} rc={}",
         tag_id_.major_, tag_id_.minor_, blob_name, off, data_size,
         task->GetReturnCode());
    throw std::runtime_error(
        std::string("PutBlob operation failed (rc=") +
        std::to_string(task->GetReturnCode()) + ")");
  }
}

// NOTE: AsyncPutBlob(const char*) overload removed due to memory management issues.
// For async operations, the caller must manage shared memory lifecycle by:
// 1. Allocating: ctp::ipc::FullPtr<char> shm_ptr = CLIO_IPC->AllocateBuffer(data_size);
// 2. Copying data: memcpy(shm_ptr.ptr_, data, data_size);
// 3. Calling: AsyncPutBlob(blob_name, shm_ptr.shm_, data_size, off, score);
// 4. Keeping shm_ptr alive until task completes

clio::run::Future<PutBlobTask> Tag::AsyncPutBlob(const std::string &blob_name, const ctp::ipc::ShmPtr<> &data,
                                             size_t data_size, size_t off, float score,
                                             const Context &context) {
  auto *cte_client = CLIO_CTE_CLIENT;
  return cte_client->AsyncPutBlob(tag_id_, blob_name,
                                  off, data_size, data, score, context);
}

clio::run::Future<PutBlobTask> Tag::AsyncPutBlob(const std::string &blob_name,
                                                 const char *data, size_t data_size,
                                                 size_t off, float score,
                                                 const Context &context) {
  if (data == nullptr && data_size > 0) {
    throw std::invalid_argument("data buffer must not be null for AsyncPutBlob");
  }
  // issue #830: write straight from the caller's PRIVATE buffer. The client
  // picks runtime-direct no-copy or client-staging (TASK_DATA_OWNER); the
  // returned Future is readable after Wait() (empty only if client-mode
  // staging alloc failed).
  auto *cte_client = CLIO_CTE_CLIENT;
  return cte_client->AsyncPutBlob(tag_id_, blob_name, off, data_size, data,
                                  score, context);
}

void Tag::GetBlob(const std::string &blob_name, char *data, size_t data_size, size_t off) {
  // Validate input parameters
  if (data_size == 0) {
    throw std::invalid_argument("data_size must be specified for GetBlob");
  }

  if (data == nullptr) {
    throw std::invalid_argument("data buffer must be pre-allocated by caller");
  }

  // The zero-IPC fast path is NATIVE to CoreClient::AsyncGetBlob now (issue
  // #783/#817 via TryShmGet) — delegating to the private-buffer overload gets
  // shared-cache hit, runtime-mode direct read, and client-mode staging in one
  // place instead of duplicating the cache probe here.
  auto fut = CLIO_CTE_CLIENT->AsyncGetBlob(tag_id_, blob_name, off, data_size,
                                           0, data);
  if (fut.get() == nullptr) {
    // Only reachable when client-mode SHM staging could not be allocated.
    throw std::runtime_error("Failed to allocate shared memory for GetBlob");
  }
  fut.Wait();
  if (fut->GetReturnCode() != 0) {
    throw std::runtime_error("GetBlob failed with code " +
                             std::to_string(fut->GetReturnCode()));
  }
}

clio::run::Future<GetBlobTask> Tag::AsyncGetBlob(const std::string &blob_name,
                                                 char *data, size_t data_size,
                                                 size_t off) {
  // Validate input parameters
  if (data_size == 0) {
    throw std::invalid_argument("data_size must be specified for AsyncGetBlob");
  }
  if (data == nullptr) {
    throw std::invalid_argument("data buffer must be pre-allocated by caller");
  }

  // issue #823: read straight into the caller's PRIVATE buffer. The client
  // picks the fastest of shared-cache / runtime-direct / client-staging; the
  // returned Future is empty on a cache hit and readable otherwise.
  auto *cte_client = CLIO_CTE_CLIENT;
  return cte_client->AsyncGetBlob(tag_id_, blob_name, off, data_size,
                                  /*flags=*/0, data);
}

void Tag::GetBlob(const std::string &blob_name, ctp::ipc::ShmPtr<> data, size_t data_size, size_t off) {
  // Validate input parameters
  if (data_size == 0) {
    throw std::invalid_argument("data_size must be specified for GetBlob");
  }

  if (data.IsNull()) {
    throw std::invalid_argument("data pointer must be pre-allocated by caller. "
                               "Use CLIO_IPC->AllocateBuffer(data_size) to allocate shared memory.");
  }

  auto *cte_client = CLIO_CTE_CLIENT;
  auto task = cte_client->AsyncGetBlob(tag_id_, blob_name,
                                       off, data_size, 0, data);
  task.Wait();

  if (task->GetReturnCode() != 0) {
    throw std::runtime_error("GetBlob operation failed");
  }

}

float Tag::GetBlobScore(const std::string &blob_name) {
  auto *cte_client = CLIO_CTE_CLIENT;
  auto task = cte_client->AsyncGetBlobScore(tag_id_, blob_name);
  task.Wait();

  float score = task->score_;
  return score;
}

clio::run::u64 Tag::GetBlobSize(const std::string &blob_name) {
  auto *cte_client = CLIO_CTE_CLIENT;

  // issue #783 fast path: answer from the shared-memory metadata cache with
  // ZERO IPC when the blob is cached. A miss (or an inconsistent read) simply
  // falls through to the RPC below -- the cache is never treated as
  // authoritative, and "not cached" must never be reported as "size 0".
  //
  // A cached record with total_size_ == 0 is ALSO untrustworthy (issue #862):
  // a freshly created blob's record can be visible in the mirror before the
  // completing put's size is republished, and callers cannot tell "empty
  // blob" from "absent" from a zero return. Confirm zero via the RPC below --
  // genuinely empty blobs are degenerate and rare, so the extra round trip is
  // irrelevant; the stale zero showed up as read-after-write NOT_FOUNDs on
  // freshly inserted keys (YCSB workload D).
  if (cte_client->HasShmCache()) {
    ShmBlobRecord rec;
    if (cte_client->TryGetBlobRecordShm(tag_id_, blob_name, &rec) &&
        rec.total_size_ != 0) {
      return rec.total_size_;
    }
  }

  auto task = cte_client->AsyncGetBlobSize(tag_id_, blob_name);
  task.Wait();

  clio::run::u64 size = task->size_;
  return size;
}

std::vector<std::string> Tag::GetContainedBlobs() {
  auto *cte_client = CLIO_CTE_CLIENT;
  auto task = cte_client->AsyncGetContainedBlobs(tag_id_);
  task.Wait();

  std::vector<std::string> blobs = task->blob_names_;
  return blobs;
}

void Tag::ReorganizeBlob(const std::string &blob_name, float new_score) {
  auto *cte_client = CLIO_CTE_CLIENT;
  auto task = cte_client->AsyncReorganizeBlob(tag_id_, blob_name, new_score);
  task.Wait();

  if (task->GetReturnCode() != 0) {
    throw std::runtime_error("ReorganizeBlob operation failed");
  }
}

} // namespace clio::cte::core
