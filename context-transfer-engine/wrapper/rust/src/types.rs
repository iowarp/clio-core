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
 * INTERRUPTION) HOWEVER CAUSED AND ON THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

//! Core types for CTE Rust bindings
//!
//! These types MUST match the C++ layout exactly for safe FFI.

/// Operation types for CTE
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CteOp {
    PutBlob = 0,
    GetBlob = 1,
    DelBlob = 2,
    GetOrCreateTag = 3,
    DelTag = 4,
    GetTagSize = 5,
    ReorganizeBlob = 6,
}

impl From<u32> for CteOp {
    fn from(value: u32) -> Self {
        match value {
            0 => CteOp::PutBlob,
            1 => CteOp::GetBlob,
            2 => CteOp::DelBlob,
            3 => CteOp::GetOrCreateTag,
            4 => CteOp::DelTag,
            5 => CteOp::GetTagSize,
            6 => CteOp::ReorganizeBlob,
            _ => CteOp::PutBlob,
        }
    }
}

/// Block device types
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BdevType {
    File = 0,
    Ram = 1,
}

/// Chimaera runtime modes
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeMode {
    Client = 0,
    Server = 1,
    Runtime = 2,
}

/// Unique ID for tags, blobs, and pools
///
/// **Layout Critical**: This MUST match clio::run::UniqueId (8 bytes):
/// - major: u32 - Major identifier
/// - minor: u32 - Minor identifier
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CteTagId {
    pub major: u32,
    pub minor: u32,
}

impl CteTagId {
    /// Create a new CteTagId with major and minor components
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// Create a null (invalid) CteTagId
    pub const fn null() -> Self {
        Self { major: 0, minor: 0 }
    }

    /// Check if this is a null/invalid ID
    pub fn is_null(&self) -> bool {
        self.major == 0 && self.minor == 0
    }

    /// Convert to u64 for storage/serialization
    pub fn to_u64(&self) -> u64 {
        ((self.major as u64) << 32) | (self.minor as u64)
    }

    /// Convert from u64
    pub fn from_u64(v: u64) -> Self {
        Self {
            major: (v >> 32) as u32,
            minor: v as u32,
        }
    }
}

impl Default for CteTagId {
    fn default() -> Self {
        Self::null()
    }
}

/// Steady clock time point (nanosecond precision, monotonic)
///
/// Represents C++ `std::chrono::steady_clock::time_point`.
/// This is a duration since an arbitrary epoch, NOT convertible to wall-clock time.
/// Use `duration_since()` for computing time intervals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SteadyTime {
    /// Nanoseconds since steady_clock epoch
    pub nanos: i64,
}

impl SteadyTime {
    /// Create a new SteadyTime from nanoseconds
    pub const fn from_nanos(nanos: i64) -> Self {
        Self { nanos }
    }

    /// Compute duration between two SteadyTime points
    pub fn duration_since(&self, earlier: &SteadyTime) -> std::time::Duration {
        std::time::Duration::from_nanos((self.nanos - earlier.nanos) as u64)
    }

    /// Get elapsed time from reference point
    ///
    /// # Panics
    /// Panics if `earlier` is later than self
    pub fn elapsed_from(&self, earlier: &SteadyTime) -> std::time::Duration {
        assert!(
            self.nanos >= earlier.nanos,
            "SteadyTime::elapsed_from: earlier time is later than self"
        );
        self.duration_since(earlier)
    }
}

impl Default for SteadyTime {
    fn default() -> Self {
        Self { nanos: 0 }
    }
}

/// Telemetry entry for CTE operations
///
/// Contains metadata about a CTE operation for monitoring and debugging.
/// Timestamp fields are clio::run::u64 (nanoseconds since steady_clock epoch).
#[derive(Debug, Clone)]
pub struct CteTelemetry {
    /// Operation type (as u32 for FFI compatibility)
    pub op: CteOp,
    /// Offset in the blob
    pub off: u64,
    /// Size of the operation
    pub size: u64,
    /// Tag ID associated with the operation
    pub tag_id: CteTagId,
    /// Modification time (nanoseconds since steady_clock epoch)
    pub mod_time: u64,
    /// Read time (nanoseconds since steady_clock epoch)
    pub read_time: u64,
    /// Logical time counter
    pub logical_time: u64,
}

/// Pool query routing variants
///
/// Defines how tasks are routed to CTE pools:
/// - Local: Execute on current node only
/// - Dynamic: Automatic optimization based on load
/// - Broadcast: Send to all nodes
#[derive(Debug, Clone, Copy)]
pub enum PoolQuery {
    /// Broadcast to all nodes with timeout
    Broadcast { net_timeout: f32 },
    /// Dynamic routing with automatic optimization
    Dynamic { net_timeout: f32 },
    /// Local node only
    Local,
}

impl PoolQuery {
    /// Create a Broadcast query with specified timeout
    pub fn broadcast(timeout: f32) -> Self {
        Self::Broadcast {
            net_timeout: timeout,
        }
    }

    /// Create a Dynamic query with specified timeout
    pub fn dynamic(timeout: f32) -> Self {
        Self::Dynamic {
            net_timeout: timeout,
        }
    }

    /// Create a Local query
    pub fn local() -> Self {
        Self::Local
    }

    /// Get the network timeout for this query variant
    pub fn net_timeout(&self) -> f32 {
        match self {
            Self::Broadcast { net_timeout } => *net_timeout,
            Self::Dynamic { net_timeout } => *net_timeout,
            Self::Local => 0.0,
        }
    }
}

impl Default for PoolQuery {
    fn default() -> Self {
        Self::Local
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cte_tag_id_layout() {
        // Verify 8-byte layout
        assert_eq!(std::mem::size_of::<CteTagId>(), 8);

        let id = CteTagId::new(1, 2);
        assert_eq!(id.major, 1);
        assert_eq!(id.minor, 2);

        assert!(!id.is_null());
        assert!(CteTagId::null().is_null());
    }

    #[test]
    fn test_steady_time() {
        let t1 = SteadyTime::from_nanos(1000);
        let t2 = SteadyTime::from_nanos(2000);

        let duration = t2.duration_since(&t1);
        assert_eq!(duration.as_nanos(), 1000);
    }

    #[test]
    fn test_pool_query() {
        let local = PoolQuery::local();
        let dynamic = PoolQuery::dynamic(30.0);
        let broadcast = PoolQuery::broadcast(60.0);

        assert_eq!(local.net_timeout(), 0.0);
        assert_eq!(dynamic.net_timeout(), 30.0);
        assert_eq!(broadcast.net_timeout(), 60.0);
    }
}
