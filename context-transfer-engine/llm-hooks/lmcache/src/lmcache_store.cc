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

#include "clio_llm/lmcache/lmcache_store.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include "clio_ctp/data_structures/ipc/ring_buffer.h"
#include "clio_ctp/memory/allocator/malloc_allocator.h"
#include "clio_ctp/util/logging.h"
#include "clio_runtime/manager.h"

namespace clio_llm::lmcache {
namespace {

using Clock = std::chrono::steady_clock;

constexpr char kRecordMagic[8] = {'C', 'L', 'I', 'O', 'K', 'V', '1', '\0'};
constexpr std::uint32_t kRecordVersion = 1;
constexpr std::size_t kRecordHeaderSize = 24;
constexpr float kWaitTimeoutSec = 30.0f;

/**
 * Wait for a CTE future with the LMCache operation timeout.
 *
 * A timeout does not cancel the underlying CTE task. Callers that own a shared
 * memory buffer must therefore propagate the exception without freeing that
 * buffer, because the task may still access it.
 *
 * @tparam FutureT CTE future type.
 * @param future Future to wait for.
 * @param operation Human-readable operation name used in diagnostics.
 * @throws std::runtime_error when the future does not complete in time.
 */
template <typename FutureT>
void WaitForFuture(FutureT *future, const char *operation) {
  if (future->Wait(kWaitTimeoutSec)) {
    return;
  }
  HLOG(kError, "LMCacheStore::{} timed out after {} ms", operation,
       static_cast<int>(kWaitTimeoutSec * 1000.0f));
  throw std::runtime_error(std::string("LMCacheStore::") + operation +
                           " timed out after 30000 ms");
}

/**
 * Return whether opt-in LMCache CTE profiling is enabled.
 *
 * @return True when CLIO_LMCACHE_PROFILE is set to a non-zero value.
 */
bool ProfileEnabled() {
  const char *value = std::getenv("CLIO_LMCACHE_PROFILE");
  return value != nullptr && std::string(value) != "0";
}

/**
 * Convert a steady-clock duration to milliseconds.
 *
 * @param duration Duration to convert.
 * @return Duration in milliseconds.
 */
double ToMs(Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

struct BatchProfile {
  Clock::time_point start = Clock::now();
  Clock::duration alloc = Clock::duration::zero();
  Clock::duration copy_in = Clock::duration::zero();
  Clock::duration submit = Clock::duration::zero();
  Clock::duration wait = Clock::duration::zero();
  Clock::duration copy_out = Clock::duration::zero();
  Clock::duration free = Clock::duration::zero();
  std::size_t requested = 0;
  std::size_t submitted = 0;
  std::size_t succeeded = 0;
  std::uint64_t bytes = 0;
};

/**
 * Emit a one-line LMCacheStore batch profile summary.
 *
 * @param operation Operation name.
 * @param profile Aggregated profile data.
 * @param max_inflight Window size used by this call.
 */
void LogBatchProfile(const char *operation, const BatchProfile &profile,
                     std::size_t max_inflight) {
  const double total_ms = ToMs(Clock::now() - profile.start);
  HLOG(kInfo,
       "CLIO_LMCACHE_PROFILE cxx op={} requested={} submitted={} succeeded={} "
       "bytes={} max_inflight={} total_ms={:.3f} alloc_ms={:.3f} "
       "copy_in_ms={:.3f} submit_ms={:.3f} wait_ms={:.3f} "
       "copy_out_ms={:.3f} free_ms={:.3f}",
       operation, profile.requested, profile.submitted, profile.succeeded,
       profile.bytes, max_inflight, total_ms, ToMs(profile.alloc),
       ToMs(profile.copy_in), ToMs(profile.submit), ToMs(profile.wait),
       ToMs(profile.copy_out), ToMs(profile.free));
}

/**
 * Lowercase an ASCII string.
 *
 * @param value ASCII string to normalize.
 * @return Lowercase string.
 */
std::string LowerAscii(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

/**
 * Normalize a user-provided inflight limit.
 *
 * @param max_inflight Requested maximum in-flight operation count.
 * @return At least one in-flight slot.
 */
std::size_t NormalizeMaxInflight(std::size_t max_inflight) {
  return std::max<std::size_t>(max_inflight, 1);
}

/**
 * Write a big-endian unsigned 32-bit integer.
 *
 * @param value Integer value to write.
 * @param destination Destination byte pointer.
 */
void WriteBigEndian32(std::uint32_t value, char *destination) {
  destination[0] = static_cast<char>((value >> 24) & 0xff);
  destination[1] = static_cast<char>((value >> 16) & 0xff);
  destination[2] = static_cast<char>((value >> 8) & 0xff);
  destination[3] = static_cast<char>(value & 0xff);
}

/**
 * Write a big-endian unsigned 64-bit integer.
 *
 * @param value Integer value to write.
 * @param destination Destination byte pointer.
 */
void WriteBigEndian64(std::uint64_t value, char *destination) {
  destination[0] = static_cast<char>((value >> 56) & 0xff);
  destination[1] = static_cast<char>((value >> 48) & 0xff);
  destination[2] = static_cast<char>((value >> 40) & 0xff);
  destination[3] = static_cast<char>((value >> 32) & 0xff);
  destination[4] = static_cast<char>((value >> 24) & 0xff);
  destination[5] = static_cast<char>((value >> 16) & 0xff);
  destination[6] = static_cast<char>((value >> 8) & 0xff);
  destination[7] = static_cast<char>(value & 0xff);
}

/**
 * Read a big-endian unsigned 32-bit integer.
 *
 * @param source Source byte pointer.
 * @return Parsed integer value.
 */
std::uint32_t ReadBigEndian32(const char *source) {
  return (static_cast<std::uint32_t>(
              static_cast<unsigned char>(source[0]))
          << 24) |
         (static_cast<std::uint32_t>(
              static_cast<unsigned char>(source[1]))
          << 16) |
         (static_cast<std::uint32_t>(
              static_cast<unsigned char>(source[2]))
          << 8) |
         static_cast<std::uint32_t>(static_cast<unsigned char>(source[3]));
}

/**
 * Read a big-endian unsigned 64-bit integer.
 *
 * @param source Source byte pointer.
 * @return Parsed integer value.
 */
std::uint64_t ReadBigEndian64(const char *source) {
  std::uint64_t value = 0;
  for (std::size_t i = 0; i < 8; ++i) {
    value = (value << 8) |
            static_cast<std::uint64_t>(static_cast<unsigned char>(source[i]));
  }
  return value;
}

/**
 * Compute the full LMCache record size, rejecting overflow.
 *
 * @param metadata_size Metadata JSON size in bytes.
 * @param payload_size Payload size in bytes.
 * @return Full record size, or nullopt when it cannot be represented.
 */
std::optional<std::size_t> ComputeRecordSize(std::size_t metadata_size,
                                             std::size_t payload_size) {
  if (metadata_size > std::numeric_limits<std::uint32_t>::max()) {
    return std::nullopt;
  }
  const std::size_t max_size = std::numeric_limits<std::size_t>::max();
  if (metadata_size > max_size - kRecordHeaderSize ||
      payload_size > max_size - kRecordHeaderSize - metadata_size) {
    return std::nullopt;
  }
  return kRecordHeaderSize + metadata_size + payload_size;
}

/**
 * Encode the fixed CLIOKV1 record header.
 *
 * @param metadata_size Metadata JSON size in bytes.
 * @param payload_size Payload size in bytes.
 * @param destination Destination buffer with kRecordHeaderSize capacity.
 */
void WriteRecordHeader(std::size_t metadata_size, std::size_t payload_size,
                       char *destination) {
  std::memcpy(destination, kRecordMagic, sizeof(kRecordMagic));
  WriteBigEndian32(kRecordVersion, destination + 8);
  WriteBigEndian32(static_cast<std::uint32_t>(metadata_size), destination + 12);
  WriteBigEndian64(static_cast<std::uint64_t>(payload_size), destination + 16);
}

/**
 * Assemble a CLIOKV1 record into a caller-provided buffer.
 *
 * @param metadata Metadata JSON bytes.
 * @param payload Payload bytes.
 * @param payload_size Payload size in bytes.
 * @param destination Destination buffer with full record capacity.
 */
void WriteRecord(std::string_view metadata, const void *payload,
                 std::size_t payload_size, char *destination) {
  WriteRecordHeader(metadata.size(), payload_size, destination);
  std::memcpy(destination + kRecordHeaderSize, metadata.data(),
              metadata.size());
  std::memcpy(destination + kRecordHeaderSize + metadata.size(), payload,
              payload_size);
}

/**
 * Parse the fixed CLIOKV1 record header.
 *
 * @param header Header bytes with kRecordHeaderSize capacity.
 * @return Parsed record info with metadata offset implied, or nullopt.
 */
std::optional<LMCacheStore::RecordInfo> ParseRecordHeader(
    const char *header) {
  if (std::memcmp(header, kRecordMagic, sizeof(kRecordMagic)) != 0) {
    return std::nullopt;
  }
  if (ReadBigEndian32(header + 8) != kRecordVersion) {
    return std::nullopt;
  }
  const std::uint32_t metadata_size = ReadBigEndian32(header + 12);
  const std::uint64_t payload_size = ReadBigEndian64(header + 16);
  LMCacheStore::RecordInfo info;
  info.payload_offset = kRecordHeaderSize + metadata_size;
  info.payload_size = payload_size;
  return info;
}

struct SizeInFlight {
  std::size_t index = 0;
  clio::run::Future<clio::cte::core::GetBlobSizeTask> future;
};

struct GetInFlight {
  std::size_t index = 0;
  clio::run::Future<clio::cte::core::GetBlobTask> future;
};

/**
 * Runtime-mode end-of-batch barrier (issue #862).
 *
 * Small values are copy-at-submit (the CTE's accumulating batch), but LARGE
 * values (>= the CTE batch-chunk threshold) ship zero-copy with a co-located
 * runtime and read the caller's buffers until awaited — and every buffer this
 * store hands the CTE lives only for the duration of the call. Await this
 * batch's keys (not a global drain: other threads keep submitting into the
 * process-wide registry) before returning. Client mode stages everything at
 * submit, so there the puts ride past the call.
 *
 * @param tag_id Tag the batch wrote to.
 * @param blob_names Batch blob names.
 * @param accepted Per-blob submit status; only accepted puts are awaited.
 * @param profile Optional profile accumulator (wait time).
 */
void AwaitBatchPuts(const clio::cte::core::TagId &tag_id,
                    const std::vector<std::string> &blob_names,
                    const std::vector<bool> &accepted, BatchProfile *profile) {
  if (!CLIO_RUNTIME_MANAGER->IsRuntime()) {
    return;
  }
  const auto begin = Clock::now();
  for (std::size_t i = 0; i < blob_names.size(); ++i) {
    if (accepted[i]) {
      clio::cte::core::Client::AwaitPendingPuts(tag_id, blob_names[i]);
    }
  }
  if (profile != nullptr) {
    profile->wait += Clock::now() - begin;
  }
}

/**
 * Log deferred-put completion failures observed since `errors_before`.
 *
 * Deferred puts report completion failures through the CTE client's sticky
 * error counter, not per operation; a failed put simply never publishes its
 * blob, so later lookups miss and LMCache recomputes.
 *
 * @param operation Operation name.
 * @param errors_before DeferErrorCount() snapshot from before the batch.
 */
void WarnDeferErrors(const char *operation, std::uint64_t errors_before) {
  const std::uint64_t delta =
      clio::cte::core::Client::DeferErrorCount() - errors_before;
  if (delta != 0) {
    HLOG(kWarning, "LMCacheStore::{}: {} deferred puts failed on completion",
         operation, delta);
  }
}

/**
 * Unbounded in-flight window: a growable single-thread SPSC ring
 * (ctp::ipc::ext_ring_buffer — Push doubles the ring instead of failing or
 * blocking). The submitting thread is both producer and consumer, so no
 * locking is involved. Submission NEVER stalls on a window boundary; instead
 * completed futures are harvested opportunistically from the front after each
 * submit (ReapInFlight) and the remainder is waited at the end of the batch
 * (DrainInFlight).
 */
template <typename FlightT>
using FlightQueue =
    ctp::ipc::ext_ring_buffer<FlightT, ctp::ipc::MallocAllocator>;

/**
 * Pop every already-complete future from the FRONT of the window, stopping at
 * the first incomplete one. Wait() is still invoked on each completed future
 * before it is handed to `done` — on an already-complete future it returns
 * immediately, but it is what runs the task's PostWait (the client-mode
 * private get copies its staged bytes into the caller buffer there).
 *
 * @param window In-flight queue (oldest submission at the front).
 * @param operation Operation name for Wait diagnostics.
 * @param done Callback receiving each completed flight entry.
 */
template <typename FlightT, typename DoneFn>
void ReapInFlight(FlightQueue<FlightT> *window, const char *operation,
                  DoneFn &&done) {
  FlightT probe;
  while (window->Peek(window->GetHead(), probe)) {
    if (!probe.future.IsComplete()) {
      return;  // head still running: everything behind it stays queued
    }
    FlightT item;
    if (!window->Pop(item)) {
      return;
    }
    WaitForFuture(&item.future, operation);
    done(item);
  }
}

/**
 * Blocking end-of-batch drain: wait every remaining in-flight future in
 * submission order and hand each to `done`.
 *
 * @param window In-flight queue.
 * @param operation Operation name for Wait diagnostics.
 * @param done Callback receiving each completed flight entry.
 * @param profile Optional profile accumulator (wait time).
 */
template <typename FlightT, typename DoneFn>
void DrainInFlight(FlightQueue<FlightT> *window, const char *operation,
                   DoneFn &&done, BatchProfile *profile) {
  FlightT item;
  while (window->Pop(item)) {
    const auto begin = Clock::now();
    WaitForFuture(&item.future, operation);
    if (profile != nullptr) {
      profile->wait += Clock::now() - begin;
    }
    done(item);
  }
}

/**
 * Build private-memory segments for one CLIOKV1 record (issue #830).
 *
 * The segments reference caller-owned private buffers directly; the CTE's
 * private vectored put reads from them zero-copy when co-located with the
 * runtime and stages them through ONE internal SHM buffer otherwise.
 *
 * @param header Stable header storage retained until task completion.
 * @param metadata Stable caller-owned metadata bytes.
 * @param payload Stable caller-owned payload bytes.
 * @param payload_size Payload size in bytes.
 * @return Header, metadata, and payload segments in record order.
 */
std::vector<clio::cte::core::Client::PrivBlobSegment>
MakeRecordSegments(std::array<char, kRecordHeaderSize> *header,
                   const std::string &metadata, const void *payload,
                   std::size_t payload_size) {
  std::vector<clio::cte::core::Client::PrivBlobSegment> segments;
  segments.reserve(3);
  segments.emplace_back(0, kRecordHeaderSize, header->data());
  segments.emplace_back(kRecordHeaderSize, metadata.size(),
                        const_cast<char *>(metadata.data()));
  segments.emplace_back(kRecordHeaderSize + metadata.size(), payload_size,
                        const_cast<char *>(static_cast<const char *>(payload)));
  return segments;
}

}  // namespace

bool LMCacheStore::DirectReadEnabledFromEnv() {
  const char *value = std::getenv("CLIO_LMCACHE_DIRECT_READ");
  if (value == nullptr) {
    return false;
  }
  const std::string normalized = LowerAscii(value);
  return normalized == "1" || normalized == "true";
}

LMCacheStore::~LMCacheStore() { Close(); }

LMCacheStore::LMCacheStore(LMCacheStore &&other) noexcept {
  *this = std::move(other);
}

LMCacheStore &LMCacheStore::operator=(LMCacheStore &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  Close();
  ready_ = other.ready_;
  tag_name_ = std::move(other.tag_name_);
  tag_id_ = other.tag_id_;
  pool_query_ = other.pool_query_;
  direct_read_enabled_ = other.direct_read_enabled_;
  other.ready_ = false;
  other.direct_read_enabled_ = false;
  return *this;
}

bool LMCacheStore::Init(const std::string &config_path,
                        const std::string &tag_name,
                        const std::string &pool_query_mode) {
  if (ready_) {
    return true;
  }

  direct_read_enabled_ = DirectReadEnabledFromEnv();
  try {
    pool_query_ = ParsePoolQueryMode(pool_query_mode);
    if (!clio::cte::core::CLIO_CTE_CLIENT_INIT(config_path, pool_query_)) {
      HLOG(kError, "LMCacheStore: CLIO_CTE_CLIENT_INIT failed");
      return false;
    }

    auto *cte_client = CLIO_CTE_CLIENT;
    auto task = cte_client->AsyncGetOrCreateTag(
        tag_name, clio::cte::core::TagId::GetNull(), pool_query_);
    WaitForFuture(&task, "GetOrCreateTag");
    if (task->GetReturnCode() != 0) {
      HLOG(kError, "LMCacheStore: GetOrCreateTag failed tag={} rc={}", tag_name,
           task->GetReturnCode());
      return false;
    }

    tag_name_ = tag_name;
    tag_id_ = task->tag_id_;
    ready_ = true;
    return true;
  } catch (const std::exception &e) {
    HLOG(kError, "LMCacheStore: Init failed: {}", e.what());
    ready_ = false;
    return false;
  }
}

bool LMCacheStore::PutBytes(const std::string &blob_name,
                            std::string_view data) {
  if (!ready_ || blob_name.empty() || data.empty()) {
    return false;
  }

  // Deferred private-memory put (issue #862): the CTE client's registry owns
  // the future and awaits its own oldest deferred puts when client-mode SHM
  // staging runs out — the store keeps no window and no buffers.
  auto *cte_client = CLIO_CTE_CLIENT;
  const std::uint64_t errors_before =
      clio::cte::core::Client::DeferErrorCount();
  const int rc = cte_client->AsyncPutBlobDefer(
      tag_id_, blob_name, 0, data.size(), data.data(), 1.0f,
      clio::cte::core::Context(), 0, pool_query_);
  if (rc != 0) {
    return false;
  }
  // Single-blob put: await it so the return value is a real commit status.
  // (The client staged its own copy at submit; `data` is already free.)
  clio::cte::core::Client::AwaitPendingPuts(tag_id_, blob_name);
  return clio::cte::core::Client::DeferErrorCount() == errors_before;
}

std::vector<bool> LMCacheStore::PutMany(
    const std::vector<std::string> &blob_names,
    const std::vector<std::string> &payloads, std::size_t max_inflight) {
  std::vector<bool> results(blob_names.size(), false);
  if (!ready_ || blob_names.size() != payloads.size()) {
    return results;
  }

  const bool profile_enabled = ProfileEnabled();
  BatchProfile profile;
  profile.requested = blob_names.size();
  // Deferred private-memory puts (issue #862): every put is submitted through
  // the CTE client's defer registry, which owns the futures and self-throttles
  // on real SHM capacity (client-mode staging exhaustion awaits the oldest
  // deferred put and retries instead of failing the batch). The store keeps no
  // put window; a result of true means the put was ACCEPTED — completion
  // failures surface through DeferErrorCount and a missing blob.
  auto *cte_client = CLIO_CTE_CLIENT;
  const std::uint64_t errors_before =
      clio::cte::core::Client::DeferErrorCount();
  for (std::size_t i = 0; i < blob_names.size(); ++i) {
    if (blob_names[i].empty() || payloads[i].empty()) {
      continue;
    }

    const auto begin = Clock::now();
    const int rc = cte_client->AsyncPutBlobDefer(
        tag_id_, blob_names[i], 0, payloads[i].size(), payloads[i].data(),
        1.0f, clio::cte::core::Context(), 0, pool_query_);
    if (profile_enabled) {
      profile.submit += Clock::now() - begin;
      profile.bytes += payloads[i].size();
      ++profile.submitted;
    }
    results[i] = rc == 0;
    if (profile_enabled && rc == 0) {
      ++profile.succeeded;
    }
  }

  AwaitBatchPuts(tag_id_, blob_names, results,
                 profile_enabled ? &profile : nullptr);
  WarnDeferErrors("PutMany", errors_before);
  if (profile_enabled) {
    LogBatchProfile("PutMany", profile, max_inflight);
  }
  return results;
}

std::vector<bool> LMCacheStore::PutManyRecords(
    const std::vector<std::string> &blob_names,
    const std::vector<std::string> &metadata_jsons,
    const std::vector<const void *> &payloads,
    const std::vector<std::size_t> &payload_sizes, std::size_t max_inflight) {
  std::vector<bool> results(blob_names.size(), false);
  if (!ready_ || metadata_jsons.size() != blob_names.size() ||
      payloads.size() != blob_names.size() ||
      payload_sizes.size() != blob_names.size()) {
    return results;
  }

  // One DEFERRED vectored put per record (issues #830/#862). Small records
  // bump-copy into the CTE's accumulating 64-put batch; records at or above
  // the CTE batch-chunk threshold ship zero-copy with a co-located runtime,
  // so the caller buffers (and the headers vector below) must live until
  // AwaitBatchPuts lands the batch.
  const bool profile_enabled = ProfileEnabled();
  BatchProfile profile;
  profile.requested = blob_names.size();
  auto *cte_client = CLIO_CTE_CLIENT;
  const std::uint64_t errors_before =
      clio::cte::core::Client::DeferErrorCount();
  std::vector<std::array<char, kRecordHeaderSize>> headers(blob_names.size());
  for (std::size_t i = 0; i < blob_names.size(); ++i) {
    if (blob_names[i].empty() || metadata_jsons[i].empty() ||
        payloads[i] == nullptr || payload_sizes[i] == 0) {
      continue;
    }
    const auto record_size =
        ComputeRecordSize(metadata_jsons[i].size(), payload_sizes[i]);
    if (!record_size.has_value()) {
      continue;
    }
    WriteRecordHeader(metadata_jsons[i].size(), payload_sizes[i],
                      headers[i].data());
    auto segments = MakeRecordSegments(&headers[i], metadata_jsons[i],
                                       payloads[i], payload_sizes[i]);
    const auto begin = Clock::now();
    const int rc =
        cte_client->AsyncPutBlobVectoredDefer(tag_id_, blob_names[i], segments);
    if (profile_enabled) {
      profile.submit += Clock::now() - begin;
      profile.bytes += *record_size;
      ++profile.submitted;
    }
    results[i] = rc == 0;
    if (profile_enabled && rc == 0) {
      ++profile.succeeded;
    }
  }

  AwaitBatchPuts(tag_id_, blob_names, results,
                 profile_enabled ? &profile : nullptr);
  WarnDeferErrors("PutManyRecords", errors_before);
  if (profile_enabled) {
    LogBatchProfile("PutManyRecords", profile, max_inflight);
  }
  return results;
}

std::optional<std::vector<std::uint8_t>> LMCacheStore::GetBytes(
    const std::string &blob_name) {
  const std::optional<std::uint64_t> blob_size = Size(blob_name);
  if (!blob_size.has_value() || *blob_size == 0) {
    return std::nullopt;
  }

  // Deferred private-memory get (issues #823/#862): read-after-write safe —
  // any deferred put still in flight for this blob is awaited first — with
  // the bytes landing straight in the result vector.
  std::vector<std::uint8_t> result(*blob_size);
  auto *cte_client = CLIO_CTE_CLIENT;
  auto task = cte_client->AsyncGetBlobDefer(
      tag_id_, blob_name, 0, *blob_size,
      reinterpret_cast<char *>(result.data()), 0, pool_query_);
  if (task.IsNull()) {
    return std::nullopt;  // staging-allocation failure in client mode
  }
  WaitForFuture(&task, "GetBlob");

  if (task->GetReturnCode() != 0) {
    return std::nullopt;
  }
  return result;
}

std::vector<std::optional<std::vector<std::uint8_t>>> LMCacheStore::GetMany(
    const std::vector<std::string> &blob_names, std::size_t max_inflight) {
  std::vector<std::optional<std::vector<std::uint8_t>>> results(
      blob_names.size());
  if (!ready_) {
    return results;
  }

  const bool profile_enabled = ProfileEnabled();
  BatchProfile profile;
  profile.requested = blob_names.size();
  const auto sizes = SizeMany(blob_names, max_inflight);
  FlightQueue<GetInFlight> window(CTP_MALLOC,
                                  NormalizeMaxInflight(max_inflight));
  auto *cte_client = CLIO_CTE_CLIENT;
  auto on_done = [&](const GetInFlight &item) {
    if (item.future->GetReturnCode() == 0) {
      if (profile_enabled) {
        ++profile.succeeded;
      }
    } else {
      results[item.index].reset();
    }
  };

  for (std::size_t i = 0; i < blob_names.size(); ++i) {
    if (blob_names[i].empty() || !sizes[i].has_value() || *sizes[i] == 0) {
      continue;
    }

    // Materialize the result vector up front and read straight into it
    // (issue #823) — no SHM bounce buffer, no copy-out. The outer results
    // vector is pre-sized, so the element's storage is stable across the
    // in-flight waits; failed reads drop it in on_done.
    auto begin = Clock::now();
    results[i].emplace(*sizes[i]);
    if (profile_enabled) {
      profile.alloc += Clock::now() - begin;
    }
    begin = Clock::now();
    auto future = cte_client->AsyncGetBlobDefer(
        tag_id_, blob_names[i], 0, *sizes[i],
        reinterpret_cast<char *>(results[i]->data()), 0, pool_query_);
    if (profile_enabled) {
      profile.submit += Clock::now() - begin;
      profile.bytes += *sizes[i];
      ++profile.submitted;
    }
    if (future.IsNull()) {
      results[i].reset();
      continue;
    }
    window.Emplace(GetInFlight{i, std::move(future)});
    ReapInFlight(&window, "GetBlob", on_done);
  }

  DrainInFlight(&window, "GetBlob", on_done,
                profile_enabled ? &profile : nullptr);
  if (profile_enabled) {
    LogBatchProfile("GetMany", profile, max_inflight);
  }
  return results;
}

bool LMCacheStore::GetBytesInto(const std::string &blob_name, void *destination,
                                std::size_t destination_size,
                                std::optional<std::uint64_t> known_blob_size) {
  if (destination == nullptr) {
    return false;
  }

  const std::optional<std::uint64_t> blob_size =
      known_blob_size.has_value() ? known_blob_size : Size(blob_name);
  if (!blob_size.has_value() || *blob_size == 0 ||
      *blob_size > destination_size) {
    return false;
  }

  // Deferred private-memory get (issues #823/#862): read straight into the
  // caller's buffer, after any deferred put for this blob has landed.
  auto *cte_client = CLIO_CTE_CLIENT;
  auto task = cte_client->AsyncGetBlobDefer(tag_id_, blob_name, 0, *blob_size,
                                            static_cast<char *>(destination),
                                            0, pool_query_);
  if (task.IsNull()) {
    return false;  // staging-allocation failure in client mode
  }
  WaitForFuture(&task, "GetBlob");
  return task->GetReturnCode() == 0;
}

std::vector<bool> LMCacheStore::GetManyInto(
    const std::vector<std::string> &blob_names,
    const std::vector<void *> &destinations,
    const std::vector<std::size_t> &destination_sizes,
    const std::vector<std::optional<std::uint64_t>> &known_blob_sizes,
    std::size_t max_inflight) {
  std::vector<bool> results(blob_names.size(), false);
  if (!ready_ || destinations.size() != blob_names.size() ||
      destination_sizes.size() != blob_names.size()) {
    return results;
  }
  if (!known_blob_sizes.empty() &&
      known_blob_sizes.size() != blob_names.size()) {
    return results;
  }

  max_inflight = NormalizeMaxInflight(max_inflight);
  const bool profile_enabled = ProfileEnabled();
  BatchProfile profile;
  profile.requested = blob_names.size();
  const std::vector<std::optional<std::uint64_t>> sizes =
      known_blob_sizes.empty() ? SizeMany(blob_names, max_inflight)
                               : known_blob_sizes;
  FlightQueue<GetInFlight> window(CTP_MALLOC,
                                  NormalizeMaxInflight(max_inflight));
  auto *cte_client = CLIO_CTE_CLIENT;
  auto on_done = [&](const GetInFlight &item) {
    const bool ok = item.future->GetReturnCode() == 0;
    results[item.index] = ok;
    if (profile_enabled && ok) {
      ++profile.succeeded;
    }
  };

  for (std::size_t i = 0; i < blob_names.size(); ++i) {
    if (blob_names[i].empty() || destinations[i] == nullptr ||
        !sizes[i].has_value() || *sizes[i] == 0 ||
        *sizes[i] > destination_sizes[i]) {
      continue;
    }

    // Deferred private-memory get (issues #823/#862): read straight into the
    // caller buffer.
    auto begin = Clock::now();
    auto future = cte_client->AsyncGetBlobDefer(
        tag_id_, blob_names[i], 0, *sizes[i],
        static_cast<char *>(destinations[i]), 0, pool_query_);
    if (profile_enabled) {
      profile.submit += Clock::now() - begin;
      profile.bytes += *sizes[i];
      ++profile.submitted;
    }
    if (future.IsNull()) {
      continue;
    }
    window.Emplace(GetInFlight{i, std::move(future)});
    ReapInFlight(&window, "GetBlob", on_done);
  }

  DrainInFlight(&window, "GetBlob", on_done,
                profile_enabled ? &profile : nullptr);
  if (profile_enabled) {
    LogBatchProfile("GetManyInto", profile, max_inflight);
  }
  return results;
}

std::vector<bool> LMCacheStore::GetManyRangesInto(
    const std::vector<std::string> &blob_names,
    const std::vector<std::uint64_t> &offsets,
    const std::vector<std::uint64_t> &sizes,
    const std::vector<void *> &destinations,
    const std::vector<std::size_t> &destination_sizes,
    std::size_t max_inflight) {
  std::vector<bool> results(blob_names.size(), false);
  if (!ready_ || offsets.size() != blob_names.size() ||
      sizes.size() != blob_names.size() ||
      destinations.size() != blob_names.size() ||
      destination_sizes.size() != blob_names.size()) {
    return results;
  }
  // Private-memory ranged gets (issue #823): read straight into the caller
  // buffers on every path. This replaces both the old CLIO_LMCACHE_DIRECT_READ
  // hand-rolled pointer path (safe only for a co-located single-host runtime)
  // and the SHM bounce-buffer fallback.
  const bool profile_enabled = ProfileEnabled();
  BatchProfile profile;
  profile.requested = blob_names.size();
  FlightQueue<GetInFlight> window(CTP_MALLOC,
                                  NormalizeMaxInflight(max_inflight));
  auto *cte_client = CLIO_CTE_CLIENT;
  auto on_done = [&](const GetInFlight &item) {
    const bool ok = item.future->GetReturnCode() == 0;
    results[item.index] = ok;
    if (profile_enabled && ok) {
      ++profile.succeeded;
    }
  };

  for (std::size_t i = 0; i < blob_names.size(); ++i) {
    if (blob_names[i].empty() || destinations[i] == nullptr || sizes[i] == 0 ||
        sizes[i] > destination_sizes[i]) {
      continue;
    }

    auto begin = Clock::now();
    auto future = cte_client->AsyncGetBlobDefer(
        tag_id_, blob_names[i], offsets[i], sizes[i],
        static_cast<char *>(destinations[i]), 0, pool_query_);
    if (profile_enabled) {
      profile.submit += Clock::now() - begin;
      profile.bytes += sizes[i];
      ++profile.submitted;
    }
    if (future.IsNull()) {
      continue;
    }
    window.Emplace(GetInFlight{i, std::move(future)});
    ReapInFlight(&window, "GetBlob", on_done);
  }

  DrainInFlight(&window, "GetBlob", on_done,
                profile_enabled ? &profile : nullptr);
  if (profile_enabled) {
    LogBatchProfile("GetManyRangesInto", profile, max_inflight);
  }
  return results;
}

std::vector<std::optional<LMCacheStore::RecordInfo>>
LMCacheStore::ReadRecordInfos(const std::vector<std::string> &blob_names,
                              std::size_t max_inflight) {
  std::vector<std::optional<RecordInfo>> results(blob_names.size());
  if (!ready_) {
    return results;
  }

  max_inflight = NormalizeMaxInflight(max_inflight);
  const bool profile_enabled = ProfileEnabled();
  BatchProfile profile;
  profile.requested = blob_names.size();
  auto begin = Clock::now();
  std::vector<std::array<char, kRecordHeaderSize>> headers(blob_names.size());
  std::vector<void *> header_destinations;
  std::vector<std::size_t> header_destination_sizes;
  std::vector<std::uint64_t> header_offsets(blob_names.size(), 0);
  std::vector<std::uint64_t> header_sizes(blob_names.size(),
                                          kRecordHeaderSize);
  header_destinations.reserve(blob_names.size());
  header_destination_sizes.reserve(blob_names.size());
  for (auto &header : headers) {
    header_destinations.push_back(header.data());
    header_destination_sizes.push_back(header.size());
  }
  if (profile_enabled) {
    profile.alloc += Clock::now() - begin;
  }

  begin = Clock::now();
  const auto header_statuses =
      GetManyRangesInto(blob_names, header_offsets, header_sizes,
                        header_destinations, header_destination_sizes,
                        max_inflight);
  if (profile_enabled) {
    profile.wait += Clock::now() - begin;
  }

  std::vector<std::string> metadata_buffers(blob_names.size());
  std::vector<std::string> metadata_names;
  std::vector<void *> metadata_destinations;
  std::vector<std::size_t> metadata_destination_sizes;
  std::vector<std::uint64_t> metadata_offsets;
  std::vector<std::uint64_t> metadata_sizes;
  std::vector<std::size_t> metadata_indexes;

  begin = Clock::now();
  for (std::size_t i = 0; i < blob_names.size(); ++i) {
    if (!header_statuses[i]) {
      continue;
    }
    auto info = ParseRecordHeader(headers[i].data());
    if (!info.has_value() || info->payload_offset < kRecordHeaderSize) {
      continue;
    }
    const std::uint64_t metadata_size = info->payload_offset - kRecordHeaderSize;
    if (metadata_size == 0 ||
        metadata_size > std::numeric_limits<std::size_t>::max()) {
      continue;
    }
    metadata_buffers[i].resize(static_cast<std::size_t>(metadata_size));
    metadata_names.push_back(blob_names[i]);
    metadata_destinations.push_back(metadata_buffers[i].data());
    metadata_destination_sizes.push_back(metadata_buffers[i].size());
    metadata_offsets.push_back(kRecordHeaderSize);
    metadata_sizes.push_back(metadata_size);
    metadata_indexes.push_back(i);
    results[i] = *info;
  }
  if (profile_enabled) {
    profile.copy_out += Clock::now() - begin;
  }

  begin = Clock::now();
  const auto metadata_statuses = GetManyRangesInto(
      metadata_names, metadata_offsets, metadata_sizes, metadata_destinations,
      metadata_destination_sizes, max_inflight);
  if (profile_enabled) {
    profile.wait += Clock::now() - begin;
  }

  begin = Clock::now();
  for (std::size_t i = 0; i < metadata_indexes.size(); ++i) {
    const std::size_t result_index = metadata_indexes[i];
    if (!metadata_statuses[i] || !results[result_index].has_value()) {
      results[result_index] = std::nullopt;
      continue;
    }
    results[result_index]->metadata_json = std::move(
        metadata_buffers[result_index]);
    ++profile.succeeded;
    profile.bytes += results[result_index]->metadata_json.size();
  }
  if (profile_enabled) {
    profile.copy_out += Clock::now() - begin;
    LogBatchProfile("ReadRecordInfos", profile, max_inflight);
  }
  return results;
}

bool LMCacheStore::Exists(const std::string &blob_name) {
  return Size(blob_name).has_value();
}

std::vector<bool> LMCacheStore::ExistsMany(
    const std::vector<std::string> &blob_names, std::size_t max_inflight) {
  const auto sizes = SizeMany(blob_names, max_inflight);
  std::vector<bool> results(blob_names.size(), false);
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    results[i] = sizes[i].has_value();
  }
  return results;
}

std::optional<std::uint64_t> LMCacheStore::Size(const std::string &blob_name) {
  if (!ready_ || blob_name.empty()) {
    return std::nullopt;
  }

  // A still-deferred put has published no metadata yet; land it first
  // (issue #862 — free when nothing is pending: one relaxed load).
  clio::cte::core::Client::AwaitPendingPuts(tag_id_, blob_name);
  auto *cte_client = CLIO_CTE_CLIENT;
  auto task = cte_client->AsyncGetBlobSize(tag_id_, blob_name, pool_query_);
  WaitForFuture(&task, "GetBlobSize");
  if (task->GetReturnCode() != 0) {
    return std::nullopt;
  }
  return static_cast<std::uint64_t>(task->size_);
}

std::vector<std::optional<std::uint64_t>> LMCacheStore::SizeMany(
    const std::vector<std::string> &blob_names, std::size_t max_inflight) {
  std::vector<std::optional<std::uint64_t>> results(blob_names.size());
  if (!ready_) {
    return results;
  }

  const bool profile_enabled = ProfileEnabled();
  BatchProfile profile;
  profile.requested = blob_names.size();
  FlightQueue<SizeInFlight> window(CTP_MALLOC,
                                   NormalizeMaxInflight(max_inflight));
  auto *cte_client = CLIO_CTE_CLIENT;
  auto on_done = [&](const SizeInFlight &item) {
    if (item.future->GetReturnCode() == 0) {
      results[item.index] = static_cast<std::uint64_t>(item.future->size_);
      if (profile_enabled) {
        ++profile.succeeded;
      }
    }
  };

  for (std::size_t i = 0; i < blob_names.size(); ++i) {
    if (blob_names[i].empty()) {
      continue;
    }

    // Deferred puts publish metadata only on completion; land this key's
    // puts before asking for its size (issue #862).
    clio::cte::core::Client::AwaitPendingPuts(tag_id_, blob_names[i]);
    auto begin = Clock::now();
    auto future =
        cte_client->AsyncGetBlobSize(tag_id_, blob_names[i], pool_query_);
    if (profile_enabled) {
      profile.submit += Clock::now() - begin;
      ++profile.submitted;
    }
    window.Emplace(SizeInFlight{i, std::move(future)});
    ReapInFlight(&window, "GetBlobSize", on_done);
  }

  DrainInFlight(&window, "GetBlobSize", on_done,
                profile_enabled ? &profile : nullptr);
  if (profile_enabled) {
    LogBatchProfile("SizeMany", profile, max_inflight);
  }
  return results;
}

bool LMCacheStore::Delete(const std::string &blob_name) {
  if (!ready_ || blob_name.empty()) {
    return false;
  }

  // Keep put→delete ordering for a key whose put is still deferred.
  clio::cte::core::Client::AwaitPendingPuts(tag_id_, blob_name);
  auto *cte_client = CLIO_CTE_CLIENT;
  auto task = cte_client->AsyncDelBlob(tag_id_, blob_name, pool_query_);
  WaitForFuture(&task, "DeleteBlob");
  return task->GetReturnCode() == 0;
}

void LMCacheStore::Close() {
  if (ready_) {
    // Land any client-mode deferred puts before the client goes away.
    clio::cte::core::Client::AwaitPutsUntilSpace(0);
  }
  ready_ = false;
  direct_read_enabled_ = false;
  tag_id_ = clio::cte::core::TagId();
  pool_query_ = clio::run::PoolQuery::Local();
}

clio::run::PoolQuery LMCacheStore::ParsePoolQueryMode(const std::string &mode) {
  const std::string lower_mode = LowerAscii(mode);
  if (lower_mode == "dynamic") {
    return clio::run::PoolQuery::Dynamic();
  }
  if (lower_mode == "broadcast") {
    return clio::run::PoolQuery::Broadcast();
  }
  return clio::run::PoolQuery::Local();
}

}  // namespace clio_llm::lmcache
