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

//! Zero-copy SHM buffer wrapper for the CTE Rust bindings.
//!
//! `ShmBuffer` owns a shared-memory buffer allocated by the C++ `IpcManager`
//! (via the `cte_alloc_shm_buffer` C-ABI). Rust writes into / reads from the
//! raw pointer directly — no host-side copy. The CTE runtime reads/writes the
//! same SHM region, giving true zero-copy on the data path (in SHM transport
//! mode; in TCP/IPC mode only the 16-byte handle is serialized, the data
//! stays in SHM).
//!
//! ## Lifetime (CRITICAL)
//! A `ShmBuffer` passed to an async put/get (Phases 04/05) must NOT be
//! dropped before the corresponding `CteFutureHandle` completes — the runtime
//! reads/writes the SHM region asynchronously. `Drop` calls
//! `cte_free_shm_buffer`; the async methods (when added) enforce
//! wait-before-free by holding the buffer until `cte_future_wait` returns.

use crate::error::{CteError, CteResult};
use crate::ffi_c::{
    cte_alloc_shm_buffer, cte_free_shm_buffer, cte_shm_handle_to_ptr, CteShmHandle,
};

/// A zero-copy SHM buffer. Owns a `CteShmHandle` + the raw pointer + length.
///
/// Not `Send`/`Sync` by default (the raw pointer is non-Send); the async
/// wrappers (Phases 04/05) move it into a `spawn_blocking` closure where
/// single-threaded access is guaranteed.
pub struct ShmBuffer {
    handle: CteShmHandle,
    ptr: *mut u8,
    len: usize,
    // !Send marker — keep the buffer pinned to the constructing thread by
    // default. The async path explicitly moves it across via spawn_blocking
    // only when the closure owns it exclusively.
    _not_send: std::marker::PhantomData<*mut u8>,
}

impl ShmBuffer {
    /// Allocate a SHM buffer of `len` bytes.
    /// Returns `Err(CteError::ShmAllocFailed)` if the C++ allocator fails.
    pub fn alloc(len: usize) -> CteResult<Self> {
        let mut out_ptr: *mut u8 = std::ptr::null_mut();
        // SAFETY: cte_alloc_shm_buffer writes the handle + out_ptr. The
        // handle is POD; out_ptr is a valid out-param.
        let handle = unsafe { cte_alloc_shm_buffer(len as u64, &mut out_ptr) };
        if handle.is_null() || out_ptr.is_null() {
            return Err(CteError::ShmAllocFailed);
        }
        Ok(Self {
            handle,
            ptr: out_ptr,
            len,
            _not_send: std::marker::PhantomData,
        })
    }

    /// The underlying handle (for passing to async put/get FFI in Phases 04/05).
    pub fn handle(&self) -> CteShmHandle {
        self.handle
    }

    /// Buffer length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if len == 0.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Read-only access to the SHM region (for reading data the runtime wrote,
    /// e.g. after a get_blob). Valid until `Drop`.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr + len are valid for the buffer's lifetime (until Drop
        // calls cte_free_shm_buffer). The runtime has completed any async op
        // by the time the caller reads (enforced by the async wrappers).
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Read/write access to the SHM region (for writing data before a put_blob).
    /// Valid until `Drop`.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Raw write pointer (for FFI into Rust-written data, e.g. memcpy from
    /// another buffer). Valid until `Drop`.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// Resolve a `CteShmHandle` (e.g. one returned from a get_blob) to a raw
    /// pointer, without taking ownership. The caller must NOT free the handle
    /// via this — use `ShmBuffer`/`cte_free_shm_buffer` for that.
    pub fn ptr_for_handle(handle: CteShmHandle) -> *mut u8 {
        // SAFETY: cte_shm_handle_to_ptr is a pure lookup; returns null on
        // null/unresolvable handle.
        unsafe { cte_shm_handle_to_ptr(handle) }
    }
}

impl Drop for ShmBuffer {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: the handle was obtained from cte_alloc_shm_buffer and
            // not yet freed. The async wrappers guarantee any runtime op
            // using this buffer has completed before Drop runs.
            unsafe { cte_free_shm_buffer(self.handle) };
            self.handle = CteShmHandle::null();
            self.ptr = std::ptr::null_mut();
        }
    }
}

// SAFETY: ShmBuffer owns a single SHM region; the C++ IpcManager allocator is
// thread-safe. We keep the default !Send via PhantomData<*mut u8>; do NOT add
// Send without a concurrent-use audit.
