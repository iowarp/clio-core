// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — context-runtime Rust adaptation (issue #756).
//
//! The context-runtime core POD types — the first layer of the task ABI
//! (see `context-runtime/rust/TASK_ABI.md`).
//!
//! Every type here is `#[repr(C)]`, POD, and frozen: they appear inside
//! shared-memory task records and in serialized archives, so their layout
//! and their *behavior* (routing decisions, null semantics, string forms)
//! are a contract with the C++ side, not an implementation detail.
//!
//! # C++ → Rust mapping
//!
//! | C++ (`clio_runtime/types.h`, `pool_query.h`) | Rust |
//! |---|---|
//! | `UniqueId` | [`UniqueId`] |
//! | `PoolId` (= `UniqueId`) | [`PoolId`] (alias) |
//! | `TaskId` | [`TaskId`] |
//! | `RoutingMode` | [`RoutingMode`] |
//! | `PoolQuery` | [`PoolQuery`] |
//! | `TASK_*` flag bits | [`task_flags`] |
//!
//! # Divergences
//!
//! - C++ `TaskId::node_id_` is declared `u32` but its ctor takes `u64`; the
//!   port keeps the **field** as `u32` (the layout is what matters) and
//!   takes `u32` in the constructor, dropping the silent narrowing.
//! - `PoolQuery::FromString`/`ToString` parse/print the same forms as the
//!   C++; unknown strings yield `None` instead of the C++ fatal-log path.
//! - C++ `ContainerId` is a `u32` alias; represented directly as `u32`.

#![deny(unsafe_op_in_unsafe_fn)]

use std::fmt;

// ---------------------------------------------------------------------------
// UniqueId / PoolId
// ---------------------------------------------------------------------------

/// `{ major, minor }` identity (C++ `UniqueId`). Null is `(0, 0)`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct UniqueId {
    pub major: u32,
    pub minor: u32,
}

/// Pool identifier (C++ `using PoolId = UniqueId`).
pub type PoolId = UniqueId;

impl UniqueId {
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// C++ `GetNull()`.
    pub const fn null() -> Self {
        Self { major: 0, minor: 0 }
    }

    /// C++ `IsNull()`.
    pub const fn is_null(&self) -> bool {
        self.major == 0 && self.minor == 0
    }

    /// C++ `ToU64()` — major in the high half.
    pub const fn to_u64(self) -> u64 {
        ((self.major as u64) << 32) | (self.minor as u64)
    }

    /// C++ `FromU64()`.
    pub const fn from_u64(v: u64) -> Self {
        Self {
            major: (v >> 32) as u32,
            minor: (v & 0xFFFF_FFFF) as u32,
        }
    }

    /// C++ `FromString("major.minor")`. `None` if malformed (the C++ logs
    /// fatal instead — documented divergence). Also available as the
    /// idiomatic [`std::str::FromStr`] impl: `"512.0".parse::<UniqueId>()`.
    pub fn parse_id(s: &str) -> Option<Self> {
        let (a, b) = s.split_once('.')?;
        Some(Self::new(a.trim().parse().ok()?, b.trim().parse().ok()?))
    }
}

/// Malformed `"major.minor"` string (C++ `FromString` logs fatal instead).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseUniqueIdError;

impl fmt::Display for ParseUniqueIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "expected \"major.minor\"")
    }
}
impl std::error::Error for ParseUniqueIdError {}

impl std::str::FromStr for UniqueId {
    type Err = ParseUniqueIdError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        UniqueId::parse_id(s).ok_or(ParseUniqueIdError)
    }
}

impl fmt::Display for UniqueId {
    /// C++ `ToString()` — "major.minor" (NOT the ostream `PoolId(major:..)`
    /// debug form, which is only used for logging).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

// ---------------------------------------------------------------------------
// TaskId
// ---------------------------------------------------------------------------

/// Task identity (C++ `TaskId`) — 32 bytes, frozen by TASK_ABI.md §3.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TaskId {
    /// Process id of the creator.
    pub pid: u32,
    /// Thread id of the creator.
    pub tid: u32,
    /// Monotonic sequence number per thread.
    pub major: u32,
    /// Replica index for replicated tasks (0 = origin).
    pub replica_id: u32,
    /// Unique counter, incremented for root tasks AND subtasks.
    pub unique: u32,
    /// Node id for distributed execution.
    pub node_id: u32,
    /// Network key for the send/recv map (pointer-derived on the C++ side).
    pub net_key: u64,
}

impl TaskId {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        pid: u32,
        tid: u32,
        major: u32,
        replica_id: u32,
        unique: u32,
        node_id: u32,
        net_key: u64,
    ) -> Self {
        Self {
            pid,
            tid,
            major,
            replica_id,
            unique,
            node_id,
            net_key,
        }
    }

    /// A null/default task id (all zero), as C++ default-construction gives.
    pub const fn null() -> Self {
        Self::new(0, 0, 0, 0, 0, 0, 0)
    }

    pub const fn is_null(&self) -> bool {
        self.pid == 0 && self.tid == 0 && self.major == 0 && self.unique == 0
    }
}

// ---------------------------------------------------------------------------
// Task flags (TASK_* bits)
// ---------------------------------------------------------------------------

/// Task property bits (C++ `TASK_*`, stored in `task_flags_`).
///
/// These are **append-only** wire values: they travel in serialized task
/// archives (TASK_ABI.md §4 rationale applies equally here).
pub mod task_flags {
    /// The task repeats on a period (`period_ns`).
    pub const PERIODIC: u32 = 1 << 0;
    /// This task instance owns its data buffer and must free it at teardown
    /// (the conditional-free rule; TASK_ABI.md §6).
    pub const DATA_OWNER: u32 = 1 << 1;
    /// The task executes on a remote node.
    pub const REMOTE: u32 = 1 << 2;
}

// ---------------------------------------------------------------------------
// PoolQuery
// ---------------------------------------------------------------------------

/// Routing mode (C++ `RoutingMode`). Discriminants are frozen: they are
/// serialized inside `PoolQuery`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RoutingMode {
    /// Route to the local node only.
    #[default]
    Local = 0,
    /// Route to a specific container by id.
    DirectId = 1,
    /// Route by hash (load balancing).
    DirectHash = 2,
    /// Route to a range of containers.
    Range = 3,
    /// Broadcast to all containers.
    Broadcast = 4,
    /// Route to a specific physical node.
    Physical = 5,
    /// Dynamic routing with cache optimization (routes to Monitor).
    Dynamic = 6,
    /// GPU → CPU direction (the only GPU-related mode).
    ToLocalCpu = 7,
    /// Do nothing.
    Null = 8,
    /// Batch + aggregate matching tasks at the neighborhood leader.
    ManyToOne = 9,
    /// Like ManyToOne, but the aggregate runs only once tasks from ALL
    /// containers have arrived (a collective barrier).
    AllToOne = 10,
}

/// Container identity within a pool (C++ `ContainerId`, a `u32` alias).
pub type ContainerId = u32;

/// Where and how a task should execute (C++ `PoolQuery`).
///
/// Layout is frozen (TASK_ABI.md §5). Note that parity here is
/// **behavioral** as well as structural: both languages must make identical
/// routing decisions for identical inputs.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoolQuery {
    pub routing_mode: RoutingMode,
    /// Hash value for `DirectHash`.
    pub hash_value: u32,
    /// Container id for `DirectId`.
    pub container_id: ContainerId,
    /// Starting offset for `Range`.
    pub range_offset: u32,
    /// Container count for `Range`.
    pub range_count: u32,
    /// Node id for `Physical`.
    pub node_id: u32,
    /// Return node id for distributed responses.
    pub ret_node: u32,
    /// Per-task network timeout in seconds (-1 = use default).
    pub net_timeout: f32,
    /// Task TTL in seconds (#628); < 0 = infinite.
    pub ttl: f32,
    /// GPU parallelism: 1 = lane 0, 32 = full warp, > 32 = multi-warp.
    pub parallelism: u32,
    /// `ManyToOne` batch sub-key.
    pub batch_key: u64,
    /// `ManyToOne` batch window, nanoseconds.
    pub batch_for_ns: u64,
}

impl Default for PoolQuery {
    fn default() -> Self {
        Self {
            routing_mode: RoutingMode::Local,
            hash_value: 0,
            container_id: 0,
            range_offset: 0,
            range_count: 0,
            node_id: 0,
            ret_node: 0,
            net_timeout: -1.0,
            ttl: -1.0,
            parallelism: 1,
            batch_key: 0,
            batch_for_ns: 0,
        }
    }
}

impl PoolQuery {
    /// C++ `PoolQuery::Local()`.
    pub fn local() -> Self {
        Self::default()
    }

    /// C++ `PoolQuery::DirectId(container_id, net_timeout)`.
    pub fn direct_id(container_id: ContainerId, net_timeout: f32) -> Self {
        Self {
            routing_mode: RoutingMode::DirectId,
            container_id,
            net_timeout,
            ..Self::default()
        }
    }

    /// C++ `PoolQuery::DirectHash(hash, net_timeout)`.
    pub fn direct_hash(hash: u32, net_timeout: f32) -> Self {
        Self {
            routing_mode: RoutingMode::DirectHash,
            hash_value: hash,
            net_timeout,
            ..Self::default()
        }
    }

    /// C++ `PoolQuery::Range(offset, count, net_timeout)`.
    pub fn range(offset: u32, count: u32, net_timeout: f32) -> Self {
        Self {
            routing_mode: RoutingMode::Range,
            range_offset: offset,
            range_count: count,
            net_timeout,
            ..Self::default()
        }
    }

    /// C++ `PoolQuery::Broadcast(net_timeout)`.
    pub fn broadcast(net_timeout: f32) -> Self {
        Self {
            routing_mode: RoutingMode::Broadcast,
            net_timeout,
            ..Self::default()
        }
    }

    /// C++ `PoolQuery::Physical(node_id, net_timeout)`.
    pub fn physical(node_id: u32, net_timeout: f32) -> Self {
        Self {
            routing_mode: RoutingMode::Physical,
            node_id,
            net_timeout,
            ..Self::default()
        }
    }

    /// C++ `PoolQuery::Dynamic(net_timeout)`.
    pub fn dynamic(net_timeout: f32) -> Self {
        Self {
            routing_mode: RoutingMode::Dynamic,
            net_timeout,
            ..Self::default()
        }
    }

    /// C++ `PoolQuery::ManyToOne(container_hash, batch_key, batch_for_ns)`.
    pub fn many_to_one(container_hash: u32, batch_key: u64, batch_for_ns: u64) -> Self {
        Self {
            routing_mode: RoutingMode::ManyToOne,
            hash_value: container_hash,
            batch_key,
            batch_for_ns,
            ..Self::default()
        }
    }

    /// C++ `PoolQuery::AllToOne(container_hash, batch_key)`.
    pub fn all_to_one(container_hash: u32, batch_key: u64) -> Self {
        Self {
            routing_mode: RoutingMode::AllToOne,
            hash_value: container_hash,
            batch_key,
            ..Self::default()
        }
    }

    /// True when the query targets more than one container.
    pub fn is_collective(&self) -> bool {
        matches!(
            self.routing_mode,
            RoutingMode::Broadcast | RoutingMode::Range | RoutingMode::AllToOne
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_layouts_are_frozen() {
        // TASK_ABI.md §3: TaskId is 32 B, PoolId 8 B.
        assert_eq!(std::mem::size_of::<UniqueId>(), 8);
        assert_eq!(std::mem::size_of::<TaskId>(), 32);
        assert_eq!(std::mem::align_of::<TaskId>(), 8);
        // PoolQuery: mode + 6×u32 + 2×f32 + u32 + 2×u64, 8-aligned.
        assert_eq!(std::mem::align_of::<PoolQuery>(), 8);
        assert_eq!(std::mem::size_of::<PoolQuery>(), 56);
        assert_eq!(std::mem::size_of::<RoutingMode>(), 4);
    }

    #[test]
    fn unique_id_u64_roundtrip_and_null() {
        let id = UniqueId::new(512, 7);
        assert_eq!(id.to_u64(), (512u64 << 32) | 7);
        assert_eq!(UniqueId::from_u64(id.to_u64()), id);
        assert!(UniqueId::null().is_null());
        assert!(!id.is_null());
        // Ordering: major first, then minor (C++ operator<).
        assert!(UniqueId::new(1, 9) < UniqueId::new(2, 0));
        assert!(UniqueId::new(1, 1) < UniqueId::new(1, 2));
    }

    #[test]
    fn unique_id_string_forms() {
        assert_eq!(UniqueId::new(512, 0).to_string(), "512.0");
        assert_eq!(UniqueId::parse_id("512.0"), Some(UniqueId::new(512, 0)));
        assert_eq!(UniqueId::parse_id(" 400 . 2 "), Some(UniqueId::new(400, 2)));
        assert_eq!(UniqueId::parse_id("512"), None);
        assert_eq!(UniqueId::parse_id("a.b"), None);
        assert_eq!(UniqueId::parse_id(""), None);
        // Idiomatic FromStr surface (round-trips ToString).
        assert_eq!("512.0".parse::<UniqueId>(), Ok(UniqueId::new(512, 0)));
        assert!("bogus".parse::<UniqueId>().is_err());
        let id = UniqueId::new(400, 2);
        assert_eq!(id.to_string().parse::<UniqueId>(), Ok(id));
    }

    #[test]
    fn routing_mode_discriminants_are_frozen() {
        // Serialized inside PoolQuery — renumbering breaks in-flight tasks.
        assert_eq!(RoutingMode::Local as u32, 0);
        assert_eq!(RoutingMode::DirectId as u32, 1);
        assert_eq!(RoutingMode::DirectHash as u32, 2);
        assert_eq!(RoutingMode::Range as u32, 3);
        assert_eq!(RoutingMode::Broadcast as u32, 4);
        assert_eq!(RoutingMode::Physical as u32, 5);
        assert_eq!(RoutingMode::Dynamic as u32, 6);
        assert_eq!(RoutingMode::ToLocalCpu as u32, 7);
        assert_eq!(RoutingMode::Null as u32, 8);
        assert_eq!(RoutingMode::ManyToOne as u32, 9);
        assert_eq!(RoutingMode::AllToOne as u32, 10);
    }

    #[test]
    fn pool_query_constructors_match_cpp_defaults() {
        let l = PoolQuery::local();
        assert_eq!(l.routing_mode, RoutingMode::Local);
        assert_eq!(l.net_timeout, -1.0); // "use default"
        assert_eq!(l.ttl, -1.0); // infinite
        assert_eq!(l.parallelism, 1); // lane 0

        assert_eq!(PoolQuery::direct_hash(42, -1.0).hash_value, 42);
        assert_eq!(PoolQuery::direct_id(3, -1.0).container_id, 3);
        let r = PoolQuery::range(2, 5, -1.0);
        assert_eq!((r.range_offset, r.range_count), (2, 5));
        assert_eq!(PoolQuery::physical(9, -1.0).node_id, 9);
        let m = PoolQuery::many_to_one(7, 11, 1_000);
        assert_eq!((m.hash_value, m.batch_key, m.batch_for_ns), (7, 11, 1_000));

        assert!(PoolQuery::broadcast(-1.0).is_collective());
        assert!(PoolQuery::range(0, 4, -1.0).is_collective());
        assert!(PoolQuery::all_to_one(1, 0).is_collective());
        assert!(!PoolQuery::local().is_collective());
        assert!(!PoolQuery::direct_hash(1, -1.0).is_collective());
    }

    #[test]
    fn task_flags_are_stable_bits() {
        assert_eq!(task_flags::PERIODIC, 1);
        assert_eq!(task_flags::DATA_OWNER, 2);
        assert_eq!(task_flags::REMOTE, 4);
    }

    #[test]
    fn task_id_null_semantics() {
        assert!(TaskId::null().is_null());
        assert!(TaskId::default().is_null());
        assert!(!TaskId::new(1, 2, 3, 0, 4, 0, 0).is_null());
    }
}
