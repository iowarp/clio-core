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

#include "shim/shim.h"

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/data_organizer/frecency_organizer.h>
#include <clio_cte/core/data_organizer/data_organizer.h>

#include <cstring>

namespace cte_ffi {

// Maximum blob size (16 GB) - must match Rust constant
constexpr uint64_t MAX_BLOB_SIZE = 16ULL * 1024ULL * 1024ULL * 1024ULL;

// Thread-safe initialization globals
std::once_flag g_init_flag;
bool g_init_done = false;

/// Initialize the CTE client with the given config path.
/// Thread-safe: only initializes once even if called multiple times.
/// @param config_path Path to the CTE configuration file
/// @return 0 on success, -1 on failure
int32_t cte_init(rust::Str config_path) {
  std::call_once(g_init_flag, [&]() {
    std::string path(config_path.data(), config_path.size());
    bool ok = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    if (!ok) {
      g_init_done = false;
      return;
    }
    // CLIO_CTE_CLIENT_INIT is in clio::cte::core namespace (inside the namespace
    // block)
    g_init_done = ::clio::cte::core::CLIO_CTE_CLIENT_INIT(path);
  });
  return g_init_done ? 0 : -1;
}

/// Create a new CTE client instance.
/// Uses the globally initialized CTE client pool_id.
/// @return Unique pointer to a new Client instance
std::unique_ptr<Client> client_new() {
  // Get the global CTE client that was initialized by CLIO_CTE_CLIENT_INIT
  auto* global_client = clio::cte::core::g_cte_client;
  if (global_client == nullptr) {
    // Fallback: create a client (will fail later when used)
    return std::make_unique<Client>();
  }
  // Create a client with the proper pool_id from the global client
  auto client = std::make_unique<Client>();
  client->inner.pool_id_ = global_client->pool_id_;
  return client;
}

/// Create a new Tag from a name string.
/// @param name The tag name
/// @return Unique pointer to a new Tag instance
std::unique_ptr<Tag> tag_new(rust::Str name) {
  std::string n(name.data(), name.size());
  return std::make_unique<Tag>(n);
}

/// Create a new Tag from a unique ID (major, minor).
/// @param major Major component of the unique ID
/// @param minor Minor component of the unique ID
/// @return Unique pointer to a new Tag instance
std::unique_ptr<Tag> tag_from_id(uint32_t major, uint32_t minor) {
  clio::run::UniqueId id(major, minor);
  return std::make_unique<Tag>(id);
}

/// Get the major component of a tag's unique ID.
/// @param tag The tag to query
/// @return The major component
uint32_t tag_get_id_major(const Tag& tag) {
  return tag.inner.GetTagId().major_;
}

/// Get the minor component of a tag's unique ID.
/// @param tag The tag to query
/// @return The minor component
uint32_t tag_get_id_minor(const Tag& tag) {
  return tag.inner.GetTagId().minor_;
}

/// Get the score associated with a blob in this tag.
/// @param tag The tag containing the blob
/// @param name The blob name
/// @return The blob's score
float tag_get_blob_score(const Tag& tag, rust::Str name) {
  std::string n(name.data(), name.size());
  return tag.inner.GetBlobScore(n);
}

/// Reorganize a blob within this tag with the given score.
/// @param tag The tag containing the blob
/// @param name The blob name
/// @param score The new score for the blob
/// @return 0 on success (synchronous operation, no error detection)
int32_t tag_reorganize_blob(const Tag& tag, rust::Str name, float score) {
  std::string n(name.data(), name.size());
  tag.inner.ReorganizeBlob(n, score);
  // Tag::ReorganizeBlob is synchronous and returns void
  // Return 0 for success (no error detection available from sync API)
  return 0;
}

/// Get the size of a blob in this tag.
/// @param tag The tag containing the blob
/// @param name The blob name
/// @return The blob size in bytes
uint64_t tag_get_blob_size(const Tag& tag, rust::Str name) {
  std::string n(name.data(), name.size());
  return tag.inner.GetBlobSize(n);
}

/// Reorganize a blob via the client (async operation).
/// @param client The client to use
/// @param major Tag ID major component
/// @param minor Tag ID minor component
/// @param name The blob name
/// @param score The new score for the blob
/// @return Task return code (0 = success)
int32_t client_reorganize_blob(const Client& client, uint32_t major,
                               uint32_t minor, rust::Str name, float score) {
  clio::run::UniqueId tag_id(major, minor);
  std::string blob_name(name.data(), name.size());
  auto task = client.inner.AsyncReorganizeBlob(tag_id, blob_name, score);
  task.Wait();
  return task->GetReturnCode();
}

/// Delete a blob via the client (async operation).
/// @param client The client to use
/// @param major Tag ID major component
/// @param minor Tag ID minor component
/// @param name The blob name
/// @return Task return code (0 = success)
int32_t client_del_blob(const Client& client, uint32_t major, uint32_t minor,
                        rust::Str name) {
  clio::run::UniqueId tag_id(major, minor);
  std::string blob_name(name.data(), name.size());
  auto task = client.inner.AsyncDelBlob(tag_id, blob_name);
  task.Wait();
  return task->GetReturnCode();
}

/// Put a blob into the tag at the given offset with a score.
/// @param tag The tag to put the blob into
/// @param name The blob name
/// @param data The blob data bytes
/// @param offset The offset within the blob
/// @param score The score for the blob
/// @return 0 on success, -1 if data too large, -2 if offset overflow
int32_t tag_put_blob(const Tag& tag, rust::Str name,
                     rust::Slice<const uint8_t> data, uint64_t offset,
                     float score) {
  std::string n(name.data(), name.size());

  // Validate blob size
  uint64_t data_size = data.size();
  if (data_size > MAX_BLOB_SIZE) {
    return -1;  // Error: data too large
  }

  // Check for offset overflow
  if (offset > MAX_BLOB_SIZE - data_size) {
    return -2;  // Error: offset + size overflow
  }

  tag.inner.PutBlob(n, reinterpret_cast<const char*>(data.data()), data.size(),
                    static_cast<size_t>(offset), score);
  return 0;  // Success
}

/// Get a blob from the tag into an output buffer.
/// @param tag The tag containing the blob
/// @param name The blob name
/// @param size Number of bytes to read
/// @param offset Offset within the blob
/// @param out Output vector to store the data
void tag_get_blob(const Tag& tag, rust::Str name, uint64_t size,
                  uint64_t offset, rust::Vec<uint8_t>& out) {
  std::string n(name.data(), name.size());
  auto buf = std::vector<uint8_t>(size);
  tag.inner.GetBlob(n, reinterpret_cast<char*>(buf.data()),
                    static_cast<size_t>(size), static_cast<size_t>(offset));
  out.clear();
  out.reserve(buf.size());
  std::memcpy(out.data(), buf.data(), buf.size());
  out.truncate(buf.size());
}

/// Get all blob names contained in this tag.
/// @param tag The tag to query
/// @param out Output vector of blob names
void tag_get_contained_blobs(const Tag& tag, rust::Vec<rust::String>& out) {
  auto blobs = tag.inner.GetContainedBlobs();
  out.clear();
  out.reserve(blobs.size());
  for (const auto& b : blobs) {
    out.push_back(rust::String(b));
  }
}

/// Poll the telemetry log and return raw encoded entries.
/// Each entry: op(u32) + off(u64) + size(u64) + tag_major(u32) + tag_minor(u32) +
///             mod_time_nanos(u64) + read_time_nanos(u64) + logical_time(u64)
///             = 4 + 8 + 8 + 4 + 4 + 8 + 8 + 8 = 52 bytes per entry
/// @param client The client to poll
/// @param min_time Minimum timestamp to fetch entries from
/// @param timeout_sec Timeout in seconds (0 = no timeout)
/// @param out Output vector of raw bytes
/// @return 0 on success, 1 on timeout, 2 on error
int32_t client_poll_telemetry_raw(const Client& client, uint64_t min_time,
                                  float timeout_sec, rust::Vec<uint8_t>& out) {
  auto task = client.inner.AsyncPollTelemetryLog(min_time);

  // Wait with timeout (0 means no timeout, but we use passed timeout_sec)
  bool completed = task.Wait(timeout_sec);

  if (!completed) {
    // Timeout occurred
    return 1;
  }

  // Check for errors
  if (task->GetReturnCode() != 0) {
    return 2;
  }

  // PERFORMANCE: Use std::vector as intermediate buffer for memcpy serialization
  // rust::Vec lacks resize(), so we serialize to std::vector first, then copy
  const size_t kRecordSize = 52;
  const size_t total_size = task->entries_.size() * kRecordSize;
  std::vector<uint8_t> buf(total_size);
  uint8_t* w = buf.data();

  for (const auto& entry : task->entries_) {
    // op (u32)
    uint32_t op = static_cast<uint32_t>(entry.op_);
    std::memcpy(w, &op, 4); w += 4;

    // off (u64)
    std::memcpy(w, &entry.off_, 8); w += 8;

    // size (u64)
    std::memcpy(w, &entry.size_, 8); w += 8;

    // tag_major (u32)
    std::memcpy(w, &entry.tag_id_.major_, 4); w += 4;

    // tag_minor (u32)
    std::memcpy(w, &entry.tag_id_.minor_, 4); w += 4;

    // mod_time_nanos (u64) - Timestamp is now clio::run::u64 typedef
    std::memcpy(w, &entry.mod_time_, 8); w += 8;

    // read_time_nanos (u64) - Timestamp is now clio::run::u64 typedef
    std::memcpy(w, &entry.read_time_, 8); w += 8;

    // logical_time (u64)
    std::memcpy(w, &entry.logical_time_, 8); w += 8;
  }

  // ONE bulk copy into the rust::Vec
  out.clear();
  out.reserve(buf.size());
  std::memcpy(out.data(), buf.data(), buf.size());
  out.truncate(buf.size());

  return 0;  // Success
}

/// Get blob info with performance-critical serialization.
/// Format: score(f32) + total_size(u64) + blocks_count(u32) + blocks[...]
/// Each block: target_pool_id(u64) + block_size(u64) + block_offset(u64) = 24 bytes
/// @param client The client to query
/// @param major Tag ID major component
/// @param minor Tag ID minor component
/// @param name The blob name
/// @param out Output vector of raw bytes
/// @return Task return code (0 = success)
int32_t client_get_blob_info_raw(const Client& client, uint32_t major,
                                 uint32_t minor, rust::Str name,
                                 rust::Vec<uint8_t>& out) {
  clio::run::UniqueId tag_id(major, minor);
  std::string blob_name(name.data(), name.size());

  auto task = client.inner.AsyncGetBlobInfo(tag_id, blob_name);
  task.Wait();

  if (task->GetReturnCode() != 0) {
    return task->GetReturnCode();
  }

  // PERFORMANCE: Serialize into std::vector first, then ONE bulk memcpy to rust::Vec
  const size_t kHeaderSize = 16;
  const size_t kBlockSize = 24;
  const size_t buf_total_size = kHeaderSize + task->blocks_.size() * kBlockSize;
  std::vector<uint8_t> buf(buf_total_size);
  uint8_t* w = buf.data();

  // score (f32) via uint32_t bits
  uint32_t score_bits;
  static_assert(sizeof(score_bits) == sizeof(task->score_), "Size mismatch");
  std::memcpy(&score_bits, &task->score_, sizeof(float));
  std::memcpy(w, &score_bits, 4); w += 4;

  // total_size (u64)
  std::memcpy(w, &task->total_size_, 8); w += 8;

  // blocks_count (u32)
  uint32_t blocks_count = static_cast<uint32_t>(task->blocks_.size());
  std::memcpy(w, &blocks_count, 4); w += 4;

  // each block
  for (const auto& block : task->blocks_) {
    // target_pool_id (u64) - PoolId is a class, convert to u64 first
    uint64_t pool_id =
        block.target_pool_id_.IsNull() ? 0 : block.target_pool_id_.ToU64();
    std::memcpy(w, &pool_id, 8); w += 8;

    // block_size (u64)
    std::memcpy(w, &block.block_size_, 8); w += 8;

    // block_offset (u64)
    std::memcpy(w, &block.block_offset_, 8); w += 8;
  }

  // ONE bulk copy into the rust::Vec
  out.clear();
  out.reserve(buf.size());
  std::memcpy(out.data(), buf.data(), buf.size());
  out.truncate(buf.size());

  return 0;
}

/// Trigger the CTE's internal DynamicReorganize task for one replica.
/// This delegates reorganization to the built-in data organizer (issue #738)
/// instead of Aneris running its own per-blob reorganization loop.
/// @param replica_id Which replica (0-based) to fire
/// @param period_us   Period in microseconds for the periodic task
/// @return 0 on success, non-zero on error
int32_t client_dynamic_reorganize(uint32_t replica_id, double period_us) {
  auto* global_client = ::clio::cte::core::g_cte_client;
  if (global_client == nullptr) {
    return -1;  // CTE client not initialized
  }
  auto task = global_client->AsyncDynamicReorganize(
      clio::run::PoolQuery::Local(), replica_id, period_us);
  task.Wait();
  return task->GetReturnCode();
}

/// Compute the frecency score using the C++ FrecencyDataOrganizer formula.
/// Lets Rust validate its own frecency implementation against the C++ source
/// of truth (formula: 0.5 * 2^(-age/600) + 0.5 * (n/(n+10)), clamped to [0,1]).
/// @param access_count      Number of data ops on the blob (Put+Get)
/// @param last_access_nanos Last access time (steady-clock ns) = max(last_read, last_modified)
/// @param now_nanos         Current steady-clock time (ns)
/// @return Score as float in [0, 1]
float cte_frecency_compute_score(uint64_t access_count,
                                 uint64_t last_access_nanos,
                                 uint64_t now_nanos) {
  clio::cte::core::OrganizerBlobStat stat;
  stat.access_count_ = static_cast<clio::run::u64>(access_count);
  stat.last_read_ = static_cast<clio::cte::core::Timestamp>(last_access_nanos);
  stat.last_modified_ = static_cast<clio::cte::core::Timestamp>(last_access_nanos);
  return clio::cte::core::FrecencyDataOrganizer::ComputeScore(
      stat, static_cast<clio::cte::core::Timestamp>(now_nanos));
}

}  // namespace cte_ffi
