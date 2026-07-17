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

#pragma once

#include <clio_cte/core/core_client.h>
#include <clio_runtime/clio_runtime.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rust/cxx.h"

// ---- CteShmHandle: FFI-safe SHM buffer handle (global scope) ----
// This is a SHARED struct. We define it here; cxx will use this definition
// (cxx checks #ifndef CXXBRIDGE1_STRUCT_CteShmHandle before defining).

#ifndef CXXBRIDGE1_STRUCT_CteShmHandle
#define CXXBRIDGE1_STRUCT_CteShmHandle
/**
 * FFI-safe SHM buffer handle, bit-identical to ctp::ipc::ShmPtr<char>
 * (ShmPtrBase<char, false>) layout: {AllocatorId {u32,u32}, OffsetPtr {u64}}.
 * 16 bytes. Passed across the Rust<->C++ boundary as the zero-copy buffer
 * handle. Rust mirrors this as #[repr(C)] CteShmHandle.
 */
struct CteShmHandle {
  uint32_t alloc_id_major;
  uint32_t alloc_id_minor;
  uint64_t off;

  // Comparison operators (cxx generates implementations for PartialEq)
  bool operator==(const CteShmHandle& rhs) const noexcept;
  bool operator!=(const CteShmHandle& rhs) const noexcept;
};
static_assert(sizeof(CteShmHandle) == 16,
              "CteShmHandle must be 16 bytes to match ShmPtr<char>");
#endif  // CXXBRIDGE1_STRUCT_CteShmHandle

// ---- T23: FFI-internal shared structs for metadata bulk transfer ----
// These are defined here so cxx uses our definitions (cxx checks guards
// before defining). Rust mirrors these as #[repr(C)] shared structs.

#ifndef CXXBRIDGE1_STRUCT_CteTelemetryEntry
#define CXXBRIDGE1_STRUCT_CteTelemetryEntry
/**
 * FFI-internal telemetry record. C++ fills fields directly (no byte
 * serialization). Rust reads fields directly — no from_le_bytes re-parse.
 * 56 bytes with C padding (op_ has 4 bytes padding before off_).
 */
struct CteTelemetryEntry {
  uint32_t op;           // CteOp as u32 (4 bytes)
  uint64_t off;          // 8 bytes
  uint64_t size;         // 8 bytes
  uint32_t tag_major;    // 4 bytes
  uint32_t tag_minor;    // 4 bytes
  uint64_t mod_time;     // 8 bytes
  uint64_t read_time;    // 8 bytes
  uint64_t logical_time; // 8 bytes

  // Comparison operators (cxx generates implementations for PartialEq)
  bool operator==(const CteTelemetryEntry& rhs) const noexcept;
  bool operator!=(const CteTelemetryEntry& rhs) const noexcept;
};
#endif  // CXXBRIDGE1_STRUCT_CteTelemetryEntry

#ifndef CXXBRIDGE1_STRUCT_BlobBlock
#define CXXBRIDGE1_STRUCT_BlobBlock
/**
 * FFI-internal block placement info. target_pool_id is u64 (the C++ PoolId
 * class is converted via IsNull()?0:ToU64() in the bulk fn).
 */
struct BlobBlock {
  uint64_t pool_id;
  uint64_t block_size;
  uint64_t block_offset;

  // Comparison operators (cxx generates implementations for PartialEq)
  bool operator==(const BlobBlock& rhs) const noexcept;
  bool operator!=(const BlobBlock& rhs) const noexcept;
};
#endif  // CXXBRIDGE1_STRUCT_BlobBlock

#ifndef CXXBRIDGE1_STRUCT_BlobInfoBulk
#define CXXBRIDGE1_STRUCT_BlobInfoBulk
/**
 * FFI-internal blob_info header (score + total_size). The blocks Vec is
 * returned as a separate out-param (cxx shared structs cannot contain Vec).
 */
struct BlobInfoBulk {
  float score;
  uint64_t total_size;

  // Comparison operators (cxx generates implementations for PartialEq)
  bool operator==(const BlobInfoBulk& rhs) const noexcept;
  bool operator!=(const BlobInfoBulk& rhs) const noexcept;
};
#endif  // CXXBRIDGE1_STRUCT_BlobInfoBulk

namespace cte_ffi {

// Thread-safe initialization
extern std::once_flag g_init_flag;
extern bool g_init_done;

// Opaque wrapper types - shared across FFI boundary
// Tag wraps clio::cte::core::Tag. Mutable inner allows cxx to pass
// const Tag& while Tag methods remain non-const.
struct Tag {
  mutable clio::cte::core::Tag inner;

  explicit Tag(const std::string& name) : inner(name) {}
  explicit Tag(const clio::cte::core::TagId& id) : inner(id) {}
};

// Client wraps clio::cte::core::Client for FFI boundary
struct Client {
  mutable clio::cte::core::Client inner;
};

// Forward-declared: defined by cxx-generated code (shared struct)
struct CteTagId;

// Initialization
int32_t cte_init(rust::Str config_path);

// Client operations
std::unique_ptr<Client> client_new();

// Tag factory functions
std::unique_ptr<Tag> tag_new(rust::Str name);
std::unique_ptr<Tag> tag_from_id(uint32_t major, uint32_t minor);

// Tag blob operations - simple scalar returns only
float tag_get_blob_score(const Tag& tag, rust::Str name);
int32_t tag_reorganize_blob(const Tag& tag, rust::Str name, float score);
uint64_t tag_get_blob_size(const Tag& tag, rust::Str name);

// Operations with buffers - avoid shared struct returns
// Returns 0 on success, negative on error
// -1 = size limit exceeded, -2 = offset overflow
int32_t client_reorganize_blob(const Client& client, uint32_t major,
                               uint32_t minor, rust::Str name, float score);

int32_t client_del_blob(const Client& client, uint32_t major, uint32_t minor,
                        rust::Str name);

int32_t tag_put_blob(const Tag& tag, rust::Str name,
                     rust::Slice<const uint8_t> data, uint64_t offset,
                     float score);

void tag_get_blob(const Tag& tag, rust::Str name, uint64_t size,
                  uint64_t offset, rust::Vec<uint8_t>& out);

void tag_get_contained_blobs(const Tag& tag, rust::Vec<rust::String>& out);

// Telemetry - returns flat array: each entry is (op:u32, off:u64, size:u64,
// tag_major:u32, tag_minor:u32, mod_time_nanos:i64, read_time_nanos:i64,
// logical_time:u64) Total 52 bytes per entry. Caller interprets the byte
// buffer.
// Returns: 0 on success with data, 1 on timeout, 2 on error
int32_t client_poll_telemetry_raw(const Client& client, uint64_t min_time,
                                  float timeout_sec, rust::Vec<uint8_t>& out);

// T23: Bulk telemetry - returns typed CteTelemetryEntry records directly
// (no byte buffer serialization). The CteTelemetryEntry struct is defined
// by cxx from the Rust bridge (shared struct).
// Returns: 0 success, 1 timeout, 2 error (same codes as client_poll_telemetry_raw)
int32_t client_poll_telemetry_bulk(const Client& client, uint64_t min_time,
                                   float timeout_sec,
                                   rust::Vec<CteTelemetryEntry>& out_entries);

// GetBlobInfo - returns blob metadata with block placement
// Format: score(f32) + total_size(u64) + blocks_count(u32) + blocks[...]
// Each block: pool_id(u64) + block_size(u64) + block_offset(u64) = 24 bytes
int32_t client_get_blob_info_raw(const Client& client, uint32_t major,
                                 uint32_t minor, rust::Str name,
                                 rust::Vec<uint8_t>& out);

// T23: Bulk blob_info - returns typed BlobInfoBulk + blocks Vec directly (no
// byte buffer). cxx shared structs cannot contain Vec, so blocks is a separate
// out-param. The BlobInfoBulk/BlobBlock structs are defined by cxx from the
// Rust bridge (shared structs).
// Returns: Task return code (0 = success)
int32_t client_get_blob_info_bulk(const Client& client, uint32_t major,
                                  uint32_t minor, rust::Str name,
                                  BlobInfoBulk& out_info,
                                  rust::Vec<BlobBlock>& out_blocks);

// Tag ID helpers (exposed for Rust-side conversion)
uint32_t tag_get_id_major(const Tag& tag);
uint32_t tag_get_id_minor(const Tag& tag);

// Client query operations
int32_t client_register_target(rust::Str target_path, uint64_t size);
int32_t client_del_tag(rust::Str name);
void client_tag_query(rust::Str regex, uint32_t max_tags,
                      rust::Vec<rust::String>& out);
void client_blob_query(rust::Str tag_re, rust::Str blob_re, uint32_t max_results,
                       rust::Vec<rust::String>& out);

// NEW: Dynamic reorganization trigger
/// Trigger the CTE's internal DynamicReorganize task for one replica.
/// @param replica_id Which replica (0-based) to fire
/// @param period_us   Period in microseconds for the periodic task
/// @return 0 on success, non-zero on error
int32_t client_dynamic_reorganize(uint32_t replica_id, double period_us);

// NEW: Frecency score computation
/// Compute the frecency score using the C++ FrecencyDataOrganizer formula.
/// @param access_count      Number of data ops on the blob
/// @param last_access_nanos Last access time (steady-clock ns)
/// @param now_nanos         Current steady-clock time (ns)
/// @return Score as float in [0, 1]
float cte_frecency_compute_score(uint64_t access_count,
                                 uint64_t last_access_nanos,
                                 uint64_t now_nanos);

// ---- Zero-copy SHM buffer + future handle (migrated from raw extern "C" into
// the cxx bridge). CteShmHandle is a cxx shared struct (generated by the cxx
// bridge from src/ffi.rs); FutureHandle is a cxx opaque type whose full
// definition lives here. ----

/**
 * Type-erased handle to a C++ Future<PutBlobTask> or Future<GetBlobTask>.
 * cxx sees this as an opaque type; Rust holds UniquePtr<FutureHandle>. The
 * destructor calls destroy_fn_ (deletes the boxed concrete Future) —
 * UniquePtr<FutureHandle>::drop runs ~FutureHandle(), replacing the old
 * cte_future_destroy. Rust callers MUST still drop(future) explicitly before
 * freeing the SHM buffer (wait-before-free invariant).
 */
struct FutureHandle {
  /// Wait for completion. Returns 0 on completion, 1 timeout, <0 error.
  int32_t (*wait_fn_)(void *future, float timeout_sec, int32_t *out_rc);
  /// Destroy the boxed concrete Future<...> (called by ~FutureHandle).
  void (*destroy_fn_)(void *future);
  /// The boxed concrete Future<PutBlobTask>/<GetBlobTask> (owned).
  void *future_;

  /// Wait for the future to complete (dispatches to wait_fn_).
  int32_t wait(float timeout_sec, int32_t *out_rc) const {
    if (wait_fn_ && future_) return wait_fn_(future_, timeout_sec, out_rc);
    if (out_rc) *out_rc = 0;
    return -1;
  }

  /// Destructor: destroys the boxed Future. Replaces cte_future_destroy.
  /// (UniquePtr<FutureHandle>::drop calls this; the FutureHandle struct itself
  ///  is then deleted by the UniquePtr — two different allocations, no double-free.)
  ~FutureHandle() {
    if (destroy_fn_ && future_) {
      destroy_fn_(future_);
      future_ = nullptr;  // idempotent guard
    }
  }
};

/// Allocate a SHM buffer of `bytes` bytes. Returns a null handle on failure.
/// Resolve the write pointer via cte_shm_handle_to_ptr(handle) separately.
CteShmHandle cte_alloc_shm_buffer(uint64_t bytes);

/// Free a SHM buffer. No-op on a null handle. The runtime must have completed
/// any async op using it (call future_wait + drop the future first).
void cte_free_shm_buffer(CteShmHandle handle);

/// Resolve a CteShmHandle to a raw address (as size_t; Rust casts to *mut u8).
/// Returns 0 on a null/unresolvable handle.
size_t cte_shm_handle_to_ptr(CteShmHandle handle);

/// Block until the future completes (dispatches to FutureHandle::wait).
int32_t future_wait(const FutureHandle& handle, float timeout_sec, int32_t& out_rc);

/// Submit a zero-copy AsyncPutBlob. Returns a FutureHandle (null on failure).
/// LIFETIME: caller must future_wait() then drop the handle BEFORE cte_free_shm_buffer.
std::unique_ptr<FutureHandle> async_put_shm(const Tag& tag, rust::Str blob_name,
                                             uint64_t offset, uint64_t size,
                                             CteShmHandle data, float score);

/// Submit a zero-copy AsyncGetBlob into a caller SHM buffer. Returns FutureHandle (null on failure).
/// LIFETIME: caller must future_wait then read via cte_shm_handle_to_ptr then drop the handle then cte_free_shm_buffer.
std::unique_ptr<FutureHandle> async_get_shm(const Tag& tag, rust::Str blob_name,
                                             uint64_t offset, uint64_t size,
                                             CteShmHandle out_buf);

}  // namespace cte_ffi
