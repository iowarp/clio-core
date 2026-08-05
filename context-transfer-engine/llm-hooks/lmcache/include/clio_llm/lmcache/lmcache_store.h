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

#ifndef CLIO_LLM_LMCACHE_LMCACHE_STORE_H_
#define CLIO_LLM_LMCACHE_LMCACHE_STORE_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "clio_cte/core/core_client.h"

namespace clio_llm::lmcache {

struct LMCacheStoreTestAccess;

/**
 * CTE-backed opaque byte store for the LMCache storage plugin.
 *
 * LMCacheStore owns no runtime daemon. It initializes the CTE client, opens a
 * configured tag, and maps LMCache blob names to CTE blobs in that tag.
 */
class LMCacheStore {
 public:
  // Async GET/size operations submitted before the window is drained; it also
  // gives the runtime's task batching (CLIO_BATCH_LANE/CLIO_CTE_BATCHING) a
  // burst deep enough to actually merge. Puts no longer use a store window:
  // they go through the CTE client's deferred-put registry (issue #862),
  // which flow-controls on real shared-memory capacity.
  static constexpr std::size_t kDefaultMaxInflight = 256;

  /**
   * Parsed metadata for a CLIOKV1 LMCache record.
   */
  struct RecordInfo {
    std::string metadata_json;
    std::uint64_t payload_offset = 0;
    std::uint64_t payload_size = 0;
  };

  /**
   * Construct an unopened store.
   */
  LMCacheStore() = default;

  /**
   * Release local CTE tag state.
   */
  ~LMCacheStore();

  LMCacheStore(const LMCacheStore &) = delete;
  LMCacheStore &operator=(const LMCacheStore &) = delete;
  LMCacheStore(LMCacheStore &&) noexcept;
  LMCacheStore &operator=(LMCacheStore &&) noexcept;

  /**
   * Initialize the CTE client and open or create the backing tag.
   *
   * @param config_path Optional CTE config path. Current CTE initialization
   *     receives configuration through the runtime compose environment, so this
   *     value is forwarded for compatibility.
   * @param tag_name CTE tag name used for LMCache KV blobs.
   * @param pool_query_mode Routing mode: "local", "dynamic", or "broadcast".
   * @return True when the CTE client and tag are ready.
   */
  bool Init(const std::string &config_path,
            const std::string &tag_name = "lmcache_kv",
            const std::string &pool_query_mode = "local");

  /**
   * Store an opaque byte blob.
   *
   * @param blob_name CTE blob name.
   * @param data Blob bytes to write.
   * @return True on committed write, false on CTE failure.
   */
  bool PutBytes(const std::string &blob_name, std::string_view data);

  /**
   * Store multiple opaque byte blobs through the CTE deferred-put pipeline.
   *
   * Every put is submitted via AsyncPutBlobDefer (issue #862); the CTE
   * client's registry owns the futures, stages its own copy of the payload at
   * submit (so caller buffers are free on return in every mode), batches puts
   * into MultiPutBlob tasks, and self-throttles on shared-memory capacity.
   * Puts stay deferred past the call; reads through this store are
   * read-after-write consistent (served from the pending puts when needed).
   *
   * @param blob_names CTE blob names.
   * @param payloads Blob bytes to write, one per blob name.
   * @param max_inflight Unused for puts; retained for API compatibility.
   * @return Per-blob ACCEPTED status. Completion failures are counted by
   *     clio::cte::core::Client::DeferErrorCount and leave the blob absent.
   */
  std::vector<bool> PutMany(const std::vector<std::string> &blob_names,
                            const std::vector<std::string> &payloads,
                            std::size_t max_inflight = kDefaultMaxInflight);

  /**
   * Store multiple LMCache records through the CTE deferred-put pipeline.
   *
   * Each stored blob uses the stable CLIOKV1 record layout: fixed header,
   * metadata JSON bytes, then payload bytes, submitted as one deferred
   * vectored put (issues #830/#862): the record is assembled directly into
   * the CTE's accumulating deferred batch at submit, caller buffers are free
   * on return in every mode, and the puts stay deferred past the call.
   *
   * @param blob_names CTE blob names.
   * @param metadata_jsons Metadata JSON bytes, one per blob name.
   * @param payloads Caller-owned payload buffers, one per blob name.
   * @param payload_sizes Payload sizes in bytes.
   * @param max_inflight Unused for puts; retained for API compatibility.
   * @return Per-blob ACCEPTED status. Completion failures are counted by
   *     clio::cte::core::Client::DeferErrorCount and leave the blob absent.
   */
  std::vector<bool> PutManyRecords(
      const std::vector<std::string> &blob_names,
      const std::vector<std::string> &metadata_jsons,
      const std::vector<const void *> &payloads,
      const std::vector<std::size_t> &payload_sizes,
      std::size_t max_inflight = kDefaultMaxInflight);

  /**
   * Fetch a blob as bytes.
   *
   * @param blob_name CTE blob name.
   * @return Blob bytes, or std::nullopt on miss or failed read.
   */
  std::optional<std::vector<std::uint8_t>> GetBytes(
      const std::string &blob_name);

  /**
   * Fetch multiple blobs as byte vectors with windowed async CTE submission.
   *
   * @param blob_names CTE blob names.
   * @param max_inflight Maximum number of CTE tasks submitted before waiting.
   *     Values less than one are treated as one.
   * @return Per-blob bytes, or std::nullopt on miss or failed read.
   */
  std::vector<std::optional<std::vector<std::uint8_t>>> GetMany(
      const std::vector<std::string> &blob_names,
      std::size_t max_inflight = kDefaultMaxInflight);

  /**
   * Fetch a blob into a caller-owned byte buffer.
   *
   * @param blob_name CTE blob name.
   * @param destination Writable destination buffer.
   * @param destination_size Destination buffer size in bytes.
   * @param known_blob_size Optional blob size from a previous CTE lookup.
   * @return True only when the blob exists, fits, and was copied completely.
   */
  bool GetBytesInto(
      const std::string &blob_name, void *destination,
      std::size_t destination_size,
      std::optional<std::uint64_t> known_blob_size = std::nullopt);

  /**
   * Fetch multiple blobs into caller-owned byte buffers.
   *
   * @param blob_names CTE blob names.
   * @param destinations Writable destination buffers, one per blob name.
   * @param destination_sizes Destination buffer sizes in bytes.
   * @param known_blob_sizes Optional per-blob sizes from previous CTE lookups.
   *     When omitted, sizes are queried in a batched first phase.
   * @param max_inflight Maximum number of CTE tasks submitted before waiting.
   *     Values less than one are treated as one.
   * @return Per-blob success status.
   */
  std::vector<bool> GetManyInto(
      const std::vector<std::string> &blob_names,
      const std::vector<void *> &destinations,
      const std::vector<std::size_t> &destination_sizes,
      const std::vector<std::optional<std::uint64_t>> &known_blob_sizes = {},
      std::size_t max_inflight = kDefaultMaxInflight);

  /**
   * Fetch byte ranges from multiple blobs into caller-owned byte buffers.
   *
   * Uses the CTE private-memory get (issue #823) on every path: a co-located
   * runtime reads directly into the destination buffers (and shared-cache
   * hits are served with zero IPC); a separate-process client is staged
   * internally by the CTE. CLIO_LMCACHE_DIRECT_READ is no longer consulted —
   * the direct behavior is always on where it is safe.
   *
   * @param blob_names CTE blob names.
   * @param offsets Per-blob byte offsets.
   * @param sizes Per-blob read sizes in bytes.
   * @param destinations Writable destination buffers, one per blob name.
   * @param destination_sizes Destination buffer sizes in bytes.
   * @param max_inflight Maximum number of CTE tasks submitted before waiting.
   *     Values less than one are treated as one.
   * @return Per-blob success status.
   */
  std::vector<bool> GetManyRangesInto(
      const std::vector<std::string> &blob_names,
      const std::vector<std::uint64_t> &offsets,
      const std::vector<std::uint64_t> &sizes,
      const std::vector<void *> &destinations,
      const std::vector<std::size_t> &destination_sizes,
      std::size_t max_inflight = kDefaultMaxInflight);

  /**
   * Read CLIOKV1 metadata and payload location without reading payload bytes.
   *
   * @param blob_names CTE blob names.
   * @param max_inflight Maximum number of CTE tasks submitted before waiting.
   *     Values less than one are treated as one.
   * @return Per-blob record info, or std::nullopt on miss or invalid record.
   */
  std::vector<std::optional<RecordInfo>> ReadRecordInfos(
      const std::vector<std::string> &blob_names,
      std::size_t max_inflight = kDefaultMaxInflight);

  /**
   * Check whether a blob exists.
   *
   * @param blob_name CTE blob name.
   * @return True only when CTE confirms the blob exists.
   */
  bool Exists(const std::string &blob_name);

  /**
   * Check whether multiple blobs exist.
   *
   * @param blob_names CTE blob names.
   * @param max_inflight Maximum number of CTE tasks submitted before waiting.
   *     Values less than one are treated as one.
   * @return Per-blob existence status.
   */
  std::vector<bool> ExistsMany(const std::vector<std::string> &blob_names,
                               std::size_t max_inflight = kDefaultMaxInflight);

  /**
   * Return the blob size.
   *
   * @param blob_name CTE blob name.
   * @return Size in bytes, or std::nullopt on miss or CTE failure.
   */
  std::optional<std::uint64_t> Size(const std::string &blob_name);

  /**
   * Return sizes for multiple blobs.
   *
   * @param blob_names CTE blob names.
   * @param max_inflight Maximum number of CTE tasks submitted before waiting.
   *     Values less than one are treated as one.
   * @return Per-blob size, or std::nullopt on miss or CTE failure.
   */
  std::vector<std::optional<std::uint64_t>> SizeMany(
      const std::vector<std::string> &blob_names,
      std::size_t max_inflight = kDefaultMaxInflight);

  /**
   * Delete a blob from the backing tag.
   *
   * @param blob_name CTE blob name.
   * @return True when CTE reports successful deletion.
   */
  bool Delete(const std::string &blob_name);

  /**
   * Release local tag state. Safe to call more than once.
   */
  void Close();

  /**
   * Report whether Init has completed successfully.
   *
   * @return True when operations can be submitted.
   */
  bool IsReady() const { return ready_; }

 private:
  friend struct LMCacheStoreTestAccess;

  /**
   * Read the opt-in direct-read setting from the process environment.
   *
   * @return True only when CLIO_LMCACHE_DIRECT_READ is "1" or "true",
   *     matched case-insensitively.
   */
  static bool DirectReadEnabledFromEnv();

  /**
   * Translate a textual pool query mode into a non-null CTE pool query.
   *
   * @param mode User-provided routing mode.
   * @return A named pool query, defaulting to local for unknown values.
   */
  static clio::run::PoolQuery ParsePoolQueryMode(const std::string &mode);

  bool ready_ = false;
  /** Whether this initialized store opted into direct caller-buffer reads. */
  bool direct_read_enabled_ = false;
  std::string tag_name_ = "lmcache_kv";
  clio::cte::core::TagId tag_id_;
  clio::run::PoolQuery pool_query_ = clio::run::PoolQuery::Local();
};

}  // namespace clio_llm::lmcache

#endif  // CLIO_LLM_LMCACHE_LMCACHE_STORE_H_
