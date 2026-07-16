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

// GetBlobInfo - returns blob metadata with block placement
// Format: score(f32) + total_size(u64) + blocks_count(u32) + blocks[...]
// Each block: pool_id(u64) + block_size(u64) + block_offset(u64) = 24 bytes
int32_t client_get_blob_info_raw(const Client& client, uint32_t major,
                                 uint32_t minor, rust::Str name,
                                 rust::Vec<uint8_t>& out);

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

}  // namespace cte_ffi

// ---- Zero-copy SHM buffer C-ABI (parallel to the cxx bridge above) ----

#ifdef __cplusplus
extern "C" {
#endif

// Forward-declare Tag for C-ABI functions (Tag is defined in cte_ffi namespace)
// We use an opaque struct here since the C-ABI only needs a pointer
struct Tag;

/**
 * FFI-safe SHM buffer handle, bit-identical to ctp::ipc::ShmPtr<char>
 * (ShmPtrBase<char, false>) layout: {AllocatorId {u32,u32}, OffsetPtr {u64}}.
 * 16 bytes. Passed across the Rust<->C++ boundary as the zero-copy buffer
 * handle. Rust mirrors this as #[repr(C)] CteShmHandle.
 *
 * NEVER construct this manually on the Rust side — always obtain it from
 * cte_alloc_shm_buffer, and release it with cte_free_shm_buffer. The
 * alloc_id + off pair is meaningless without the IpcManager allocator map.
 */
struct CteShmHandle {
  uint32_t alloc_id_major;  ///< AllocatorId.major_ (e.g. PID)
  uint32_t alloc_id_minor;  ///< AllocatorId.minor_
  uint64_t off;             ///< Offset within the allocator's region
};
static_assert(sizeof(CteShmHandle) == 16,
              "CteShmHandle must be 16 bytes to match ShmPtr<char>");

/**
 * Opaque handle to a C++ clio::run::Future<...> (PutBlob/GetBlob task future).
 * Rust treats this as *mut c_void. The C++ side wraps the concrete Future in
 * a type-erased holder (CteFutureInner) so the C-ABI can wait/destroy it
 * without templating the boundary.
 */
typedef void *CteFutureHandle;

/**
 * Allocate a shared-memory buffer of `bytes` bytes.
 * @param bytes Size to allocate
 * @param out_ptr Set to a writable raw pointer into the SHM segment (the
 *        pointer the CTE runtime will read from). Set to nullptr on failure.
 * @return A CteShmHandle identifying the buffer (use cte_free_shm_buffer to
 *         release). A null handle (alloc_id_major==0 && alloc_id_minor==0 &&
 *         off==0) means allocation failed.
 */
CteShmHandle cte_alloc_shm_buffer(uint64_t bytes, uint8_t **out_ptr);

/**
 * Free a SHM buffer obtained from cte_alloc_shm_buffer.
 * No-op on a null handle (off==0 && alloc_id all zero). After this call the
 * handle is invalid and must not be reused — the runtime must have completed
 * any AsyncPutBlob/AsyncGetBlob using it (call cte_future_wait first).
 */
void cte_free_shm_buffer(CteShmHandle handle);

/**
 * Resolve a CteShmHandle back to a raw pointer (e.g. to read data the runtime
 * wrote into a get_blob buffer). Returns nullptr if the handle is null or
 * unresolvable.
 */
uint8_t *cte_shm_handle_to_ptr(CteShmHandle handle);

/**
 * Block until the future completes or timeout elapses.
 * @param future The opaque future handle from cte_tag_async_put_shm/get_shm
 * @param timeout_sec Max seconds to wait (<=0 = wait forever)
 * @param out_rc Set to the task's return code on completion (0 = success)
 * @return 0 on completion, 1 on timeout, negative on error
 */
int32_t cte_future_wait(CteFutureHandle future, float timeout_sec,
                        int32_t *out_rc);

/**
 * Destroy a future handle (releases the C++ Future wrapper). No-op on null.
 */
void cte_future_destroy(CteFutureHandle future);

/**
 * Submit a zero-copy AsyncPutBlob using a caller-allocated SHM buffer.
 * Caller must cte_future_wait(out_future) THEN cte_free_shm_buffer(data) —
 * freeing the buffer before the future completes is use-after-free.
 * @param tag The CTE tag to put into (opaque pointer to cte_ffi::Tag)
 * @param blob_name Null-terminated blob name
 * @param offset Offset within the blob
 * @param size Number of bytes in the SHM buffer
 * @param data CteShmHandle of the caller-allocated, data-filled SHM buffer
 * @param score Placement score (-1.0 = auto, 0.0-1.0 = explicit)
 * @param out_future Set to an opaque CteFutureHandle on success (wait on it,
 *        then destroy it, then free the buffer). Set to nullptr on failure.
 * @return 0 on success, negative on error
 */
int32_t cte_tag_async_put_shm(const void *tag, const char *blob_name,
                              uint64_t offset, uint64_t size,
                              CteShmHandle data, float score,
                              CteFutureHandle *out_future);

#ifdef __cplusplus
}
#endif
