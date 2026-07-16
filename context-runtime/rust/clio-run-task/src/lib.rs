// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — context-runtime Rust adaptation (issue #756).
//
//! The context-runtime `Task` — the base task record, the trait surface that
//! stands in for the C++ virtuals, and the per-execution [`RunContext`].
//!
//! Mirrors `context-runtime/include/clio_runtime/task.h`.
//!
//! # Task is deliberately NOT a POD
//!
//! `TASK_ABI.md` argued for splitting `Task` into a frozen POD record plus a
//! process-local side table, because a C++ `Task` carries a vtable and a
//! `unique_ptr`, and neither can be handed to Rust. That whole argument is a
//! consequence of *cross-language* interop. The port now assumes an
//! all-Rust runtime, with C++ bindings deferred, so the constraint is gone
//! and `Task` stays a normal, non-POD Rust type — see TASK_ABI.md §0.
//!
//! What that buys, concretely:
//!
//! | C++ mechanism | Rust here | Notes |
//! |---|---|---|
//! | `virtual ~Task()` | `Box<dyn Task>` drop glue | the vtable is the drop table |
//! | per-method teardown dispatch (§6) | `Drop` | unnecessary: the compiler writes it |
//! | `shared_ptr<Task>` | [`TaskHandle`] (`Arc<dyn Task>`) | |
//! | `static_cast<PutBlobTask*>` in the method-id switch | [`Task::as_any`] downcast | checked, not assumed |
//! | `TASK_DATA_OWNER` conditional free | the owning field's `Drop` | the leak rule stops being a rule |
//!
//! The last row is the real prize. The conditional free that §6 wanted to
//! promote from destructor code into a written spec clause — the rule behind
//! the ZMQ leak fix — is just ownership. A field that owns its buffer frees
//! it; one that borrows does not. Nothing to specify, and nothing to get
//! wrong in one language and not the other.
//!
//! # Structure follows the C++
//!
//! Field names, grouping, and the IN/OUT/TEMP annotations are kept as-is so
//! the two implementations stay diffable while both exist:
//!
//! - **IN** — travels client → runtime (serialized into the task archive).
//! - **OUT** — travels runtime → client.
//! - **TEMP** — process-local; never serialized and never copied.
//!
//! In C++ these are empty macros, i.e. comments the compiler ignores. Here
//! TEMP state is enforced by construction instead: it lives in [`RunContext`]
//! behind [`TaskBase`]'s accessors, and a fresh task simply has none.

#![deny(unsafe_op_in_unsafe_fn)]

use std::any::Any;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

use clio_run_types::{task_flags, ContainerId, PoolId, PoolQuery, TaskId};
use ctp_types::Bitfield32;

/// Method identifier (C++ `using MethodId = u32`). Method ids are an
/// append-only registry: they appear in serialized archives and the WAL.
pub type MethodId = u32;

/// Shared owning handle to a task (C++ `clio::run::shared_ptr<Task>`).
///
/// `Arc` gives the same shared ownership, and dropping the last handle runs
/// the concrete task's destructor through the vtable — which is exactly what
/// the C++ virtual destructor does, without needing a teardown dispatch table.
///
/// Mutation through a handle is intentionally left open: in the C++ only the
/// executing worker touches a task's mutable state, and the borrow checker
/// wants that invariant made explicit rather than assumed. The choice
/// (worker-exclusive `&mut` vs. interior mutability) belongs to the
/// worker/scheduler port, where the access pattern is actually visible. The
/// completion-signalling fields that *are* shared today ([`FutureInfo`],
/// `return_code`, `completer`, `is_new_data`) are atomics, so they already
/// work through a shared handle.
pub type TaskHandle = Arc<dyn Task>;

// ---------------------------------------------------------------------------
// TaskGroup / TaskStat
// ---------------------------------------------------------------------------

/// Scheduling affinity group (C++ `TaskGroup`). Tasks sharing a group are
/// pinned to the same worker once routed. `-1` is the null group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskGroup {
    id: i64,
}

impl Default for TaskGroup {
    /// The null group, as the C++ default ctor gives.
    fn default() -> Self {
        Self::null()
    }
}

impl TaskGroup {
    /// The null group (C++ `TaskGroup()`, `id_ == -1`).
    pub const fn null() -> Self {
        Self { id: -1 }
    }

    /// C++ `explicit TaskGroup(int64_t)`.
    pub const fn new(id: i64) -> Self {
        Self { id }
    }

    /// C++ `IsNull()`.
    pub const fn is_null(&self) -> bool {
        self.id == -1
    }

    pub const fn id(&self) -> i64 {
        self.id
    }
}

/// Group-level I/O and compute characteristics (C++ `TaskStat`).
///
/// When a task belongs to a [`TaskGroup`], these describe the whole group,
/// not the individual task, and route every group member consistently.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TaskStat {
    /// I/O size in bytes.
    pub io_size: usize,
    /// Normalized compute time in microseconds.
    pub compute: usize,
    /// Normalized wall time input for `InferWallClockTime`.
    pub wall_time: f32,
}

// ---------------------------------------------------------------------------
// FutureInfo
// ---------------------------------------------------------------------------

/// Self-contained completion state embedded in every task (C++ `FutureInfo`).
///
/// The task is its own future record: this replaces the separate
/// `(gpu::)FutureShm`. Per-process (the enclosing member is TEMP) — not the
/// wire format.
#[derive(Debug, Default)]
pub struct FutureInfo {
    /// Completion signal, set by the completing worker (or the client recv
    /// thread when the response lands) and polled by the waiter.
    is_complete: AtomicU32,
    /// `size_of` the concrete task — the GPU worker needs it to move the task
    /// across the device boundary. (C++ `task_size_`, formerly `pod_size_`.)
    pub task_size: u32,
    /// PID of the thread awaiting completion (to signal on completion).
    pub waiter_pid: u32,
    /// TID of the thread awaiting completion.
    pub waiter_tid: u32,
}

impl FutureInfo {
    /// True once the task has completed.
    ///
    /// Acquire, pairing with [`Self::set_complete`]'s release: a waiter that
    /// observes completion must also observe every write the completing
    /// worker made to the task's OUT fields beforehand.
    pub fn is_complete(&self) -> bool {
        self.is_complete.load(Ordering::Acquire) != 0
    }

    /// Publish completion (release; see [`Self::is_complete`]).
    pub fn set_complete(&self) {
        self.is_complete.store(1, Ordering::Release);
    }

    /// Back to not-complete (C++ `is_complete_.store(0)` in `SetNull`).
    pub fn clear_complete(&self) {
        self.is_complete.store(0, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// RunContext
// ---------------------------------------------------------------------------

/// Per-execution lifecycle flags (C++ `RunContext::RunCtxFlag`), packed into
/// one word rather than four bools.
pub mod run_ctx_flags {
    /// Task is waiting for completion.
    pub const YIELDED: u32 = 1 << 0;
    /// Task did work in its last execution.
    pub const DID_WORK: u32 = 1 << 1;
    /// `RouteTask` has placed this task.
    pub const ROUTED: u32 = 1 << 2;
    /// Execution has begun.
    pub const STARTED: u32 = 1 << 3;
}

/// A task's private execution-state extension (C++ `RunContext`).
///
/// Exists only while the task is executing, and is never serialized or
/// copied. As in the C++, every field is private and reached through
/// [`TaskBase`]'s accessors — there is deliberately no accessor handing out
/// the `RunContext` itself, so no caller can dereference one that isn't there.
///
/// # Not yet ported
///
/// The C++ `RunContext` also carries state whose types are still C++-only,
/// held back until those modules land (MIGRATION.md orders memory → ds →
/// lightbeam → worker):
///
/// - `coro_handle_` / `fiber_state_` — awaits the `ctp-coroutine` executor
/// - `container_` (`DynamicContainer`), `lane_`, `event_queue_` — awaits the
///   container/worker port
/// - `future_` — awaits the Future port
/// - `origin_`, `client_pid_`, `client_net_key_`, `input_`/`output_`,
///   `response_*` — awaits the lightbeam transport port
///
/// Adding them here before their types exist would mean inventing the types,
/// which is how the ABI drift in `clio-run-types` happened.
#[derive(Default)]
pub struct RunContext {
    /// Per-execution lifecycle flags (`run_ctx_flags`).
    flags: Bitfield32,
    /// Worker executing this task.
    worker_id: u32,
    /// Time in microseconds for the task to yield.
    yield_time_us: f64,
    /// Number of times the task has yielded.
    yield_count: u32,
    /// When the task was blocked (real time).
    block_start: Option<Instant>,
    /// Pool queries for task distribution.
    pool_queries: Vec<PoolQuery>,
    /// Replica tasks for this execution (owning handles).
    subtasks: Vec<TaskHandle>,
    /// Completed replica count.
    ///
    /// Atomic because the net send/recv workers and stale-state flushing all
    /// bump it concurrently; a lost increment leaves `completed < subtasks`
    /// forever, the origin task's future never fires, and it surfaces as a
    /// writer hang under heavy load. (The C++ carries the same warning.)
    completed_replicas: AtomicU32,
}

impl std::fmt::Debug for RunContext {
    /// Hand-written because `dyn Task` is not `Debug` — and recursing into
    /// every replica would be a poor way to print a context anyway. Subtasks
    /// show as a count.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunContext")
            .field("flags", &self.flags)
            .field("worker_id", &self.worker_id)
            .field("yield_time_us", &self.yield_time_us)
            .field("yield_count", &self.yield_count)
            .field("block_start", &self.block_start)
            .field("pool_queries", &self.pool_queries.len())
            .field("subtasks", &self.subtasks.len())
            .field("completed_replicas", &self.completed_replicas)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// TaskBase
// ---------------------------------------------------------------------------

/// The base task record (the data members of the C++ `Task`).
///
/// Concrete tasks embed this as their first field and implement [`Task`],
/// which is this port's spelling of "inherits from `Task`". Public fields
/// mirror the C++, which also has them public; only the [`RunContext`] is
/// private, exactly as there.
#[derive(Debug)]
pub struct TaskBase {
    // --- IN: client → runtime ---
    /// Pool identifier for task execution.
    pub pool_id: PoolId,
    /// Task identifier for routing.
    pub task_id: TaskId,
    /// Where and how the task should execute.
    pub pool_query: PoolQuery,
    /// Method identifier for the task type.
    pub method: MethodId,
    /// Task properties (`clio_run_types::task_flags`).
    pub task_flags: Bitfield32,
    /// Period in nanoseconds for periodic tasks.
    pub period_ns: f64,
    /// Scheduling affinity group (null = no affinity).
    pub task_group: TaskGroup,

    // --- OUT: runtime → client ---
    /// Task return code (0 = success).
    pub return_code: AtomicU32,
    /// Container that completed this task.
    pub completer: AtomicU32,

    // --- TEMP: process-local, never serialized or copied ---
    /// Completion state; the task is its own future record.
    pub fut: FutureInfo,
    /// "New streaming data available" signal (CPU streaming).
    pub is_new_data: AtomicU32,
    /// Owned execution state; `None` whenever the task is not executing.
    run_ctx: Option<Box<RunContext>>,
}

impl Default for TaskBase {
    fn default() -> Self {
        Self {
            pool_id: PoolId::null(),
            task_id: TaskId::null(),
            pool_query: PoolQuery::default(),
            method: 0,
            task_flags: Bitfield32::default(),
            period_ns: 0.0,
            task_group: TaskGroup::null(),
            return_code: AtomicU32::new(0),
            completer: AtomicU32::new(0),
            fut: FutureInfo::default(),
            is_new_data: AtomicU32::new(0),
            run_ctx: None,
        }
    }
}

impl TaskBase {
    /// A task record addressed at `pool_id` / `method` via `pool_query`.
    pub fn new(pool_id: PoolId, method: MethodId, pool_query: PoolQuery) -> Self {
        Self {
            pool_id,
            method,
            pool_query,
            ..Self::default()
        }
    }

    /// Reset every field to its null state (C++ `SetNull`).
    ///
    /// Note `completer` resets to 0, which the C++ comments call "null (0 is
    /// invalid container ID)" — that disagrees with `kInvalidContainerId`
    /// (`-1`), the sentinel `PoolQuery` uses. This mirrors the C++ rather
    /// than quietly picking a side; the two want reconciling, but a lone
    /// change here would just desync the implementations.
    pub fn set_null(&mut self) {
        self.pool_id = PoolId::null();
        self.task_id = TaskId::null();
        self.pool_query = PoolQuery::default();
        self.method = 0;
        self.task_flags.clear();
        self.period_ns = 0.0;
        self.run_ctx = None;
        self.return_code.store(0, Ordering::Relaxed);
        self.completer.store(0, Ordering::Relaxed);
        self.fut.clear_complete();
        self.is_new_data.store(0, Ordering::Relaxed);
        self.task_group = TaskGroup::null();
    }

    /// Copy the IN/OUT fields from another task record (C++ `CopyStart`).
    ///
    /// `run_ctx` is deliberately not copied: each task owns its own.
    pub fn copy_start(&mut self, other: &TaskBase) {
        self.pool_id = other.pool_id;
        self.task_id = other.task_id;
        self.pool_query = other.pool_query;
        self.method = other.method;
        self.task_flags = other.task_flags;
        self.period_ns = other.period_ns;
        self.return_code
            .store(other.return_code.load(Ordering::Relaxed), Ordering::Relaxed);
        self.completer
            .store(other.completer.load(Ordering::Relaxed), Ordering::Relaxed);
        self.task_group = other.task_group;
    }

    // --- flag accessors (C++ IsPeriodic / IsDataOwner / IsRemote) ---

    /// C++ `IsPeriodic()`.
    pub fn is_periodic(&self) -> bool {
        self.task_flags.any(task_flags::PERIODIC)
    }

    /// C++ `IsDataOwner()` — the task owns its data buffer.
    pub fn is_data_owner(&self) -> bool {
        self.task_flags.any(task_flags::DATA_OWNER)
    }

    /// C++ `IsRemote()` — received from another node.
    pub fn is_remote(&self) -> bool {
        self.task_flags.any(task_flags::REMOTE)
    }

    /// The task needs no response.
    pub fn is_fire_and_forget(&self) -> bool {
        self.task_flags.any(task_flags::FIRE_AND_FORGET)
    }

    /// The synthetic aggregate task a leader runs for a `ManyToOne` batch.
    pub fn is_batch_aggregate(&self) -> bool {
        self.task_flags.any(task_flags::BATCH_AGGREGATE)
    }

    /// The task ingressed from an external user client.
    pub fn is_external_client(&self) -> bool {
        self.task_flags.any(task_flags::EXTERNAL_CLIENT)
    }

    // --- period (C++ GetPeriod/SetPeriod take a unit multiplier) ---

    /// C++ `GetPeriod(unit)` — period in `unit` nanoseconds-per-unit.
    pub fn period(&self, unit: f64) -> f64 {
        self.period_ns / unit
    }

    /// C++ `SetPeriod(period, unit)`.
    pub fn set_period(&mut self, period: f64, unit: f64) {
        self.period_ns = period * unit;
    }

    // --- return code / completer ---

    /// True when the task completed successfully (return code 0).
    pub fn succeeded(&self) -> bool {
        self.return_code.load(Ordering::Relaxed) == 0
    }

    /// Record the container that completed this task and its return code,
    /// then publish completion. Ordered so a waiter that sees completion also
    /// sees the OUT fields (see [`FutureInfo::is_complete`]).
    pub fn complete(&self, completer: ContainerId, return_code: u32) {
        self.return_code.store(return_code, Ordering::Relaxed);
        self.completer.store(completer, Ordering::Relaxed);
        self.fut.set_complete();
    }

    // --- RunContext lifecycle (C++ BeginRunContext / EnsureRunCtx / ResetRunCtx) ---

    /// Begin executing: install a fresh [`RunContext`], dropping any
    /// previously held one (C++ `BeginRunContext`).
    pub fn begin_run_context(&mut self) {
        self.run_ctx = Some(Box::default());
    }

    /// Allocate the [`RunContext`] only if absent (C++ `EnsureRunCtx`).
    pub fn ensure_run_ctx(&mut self) {
        if self.run_ctx.is_none() {
            self.begin_run_context();
        }
    }

    /// Back to not-executing (C++ `ResetRunCtx`). The context's `Drop` does
    /// the teardown the C++ hangs off `unique_ptr`.
    pub fn reset_run_ctx(&mut self) {
        self.run_ctx = None;
    }

    /// Whether the task is currently executing (has a [`RunContext`]).
    pub fn is_executing(&self) -> bool {
        self.run_ctx.is_some()
    }

    /// The execution state, or `None` when the task is not executing.
    ///
    /// Crate-private: outside code goes through the accessors below, mirroring
    /// the C++ rule that `RunContext` is Task's private extension.
    fn ctx(&self) -> Option<&RunContext> {
        self.run_ctx.as_deref()
    }

    fn ctx_mut(&mut self) -> Option<&mut RunContext> {
        self.run_ctx.as_deref_mut()
    }

    // --- RunContext accessors ---
    //
    // The C++ equivalents throw when run_ctx_ is null. These return Option
    // instead: "not executing" is an ordinary state (it is every task's
    // state before BeginRunContext), so it belongs in the type rather than
    // in an exception. Callers that genuinely require execution can unwrap.

    /// C++ `RunWorkerId()`.
    pub fn run_worker_id(&self) -> Option<u32> {
        self.ctx().map(|c| c.worker_id)
    }

    /// C++ `SetRunWorkerId(v)`. No-op when not executing.
    pub fn set_run_worker_id(&mut self, v: u32) {
        if let Some(c) = self.ctx_mut() {
            c.worker_id = v;
        }
    }

    /// C++ `IsYielded()` — false when not executing.
    pub fn is_yielded(&self) -> bool {
        self.ctx()
            .is_some_and(|c| c.flags.any(run_ctx_flags::YIELDED))
    }

    /// C++ `SetYielded(v)`.
    pub fn set_yielded(&mut self, v: bool) {
        self.set_run_ctx_flag(run_ctx_flags::YIELDED, v);
    }

    /// Whether execution has begun (`RCTX_STARTED`).
    pub fn is_started(&self) -> bool {
        self.ctx()
            .is_some_and(|c| c.flags.any(run_ctx_flags::STARTED))
    }

    /// Mark execution as begun.
    pub fn set_started(&mut self, v: bool) {
        self.set_run_ctx_flag(run_ctx_flags::STARTED, v);
    }

    /// Whether `RouteTask` has placed this task (`RCTX_ROUTED`).
    ///
    /// This is the retired `TASK_ROUTED` bit: it moved out of `task_flags`
    /// (where it was serialized onto the wire despite being execution-local)
    /// and into the RunContext. Its old bit 1 stays reserved forever.
    pub fn is_routed(&self) -> bool {
        self.ctx().is_some_and(|c| c.flags.any(run_ctx_flags::ROUTED))
    }

    /// Mark the task as routed.
    pub fn set_routed(&mut self, v: bool) {
        self.set_run_ctx_flag(run_ctx_flags::ROUTED, v);
    }

    /// Whether the task did work in its last execution (`RCTX_DID_WORK`).
    pub fn did_work(&self) -> bool {
        self.ctx()
            .is_some_and(|c| c.flags.any(run_ctx_flags::DID_WORK))
    }

    /// Record whether the task did work.
    pub fn set_did_work(&mut self, v: bool) {
        self.set_run_ctx_flag(run_ctx_flags::DID_WORK, v);
    }

    fn set_run_ctx_flag(&mut self, mask: u32, v: bool) {
        if let Some(c) = self.ctx_mut() {
            if v {
                c.flags.set_bits(mask);
            } else {
                c.flags.unset_bits(mask);
            }
        }
    }

    /// C++ `YieldTimeUs()`.
    pub fn yield_time_us(&self) -> Option<f64> {
        self.ctx().map(|c| c.yield_time_us)
    }

    /// C++ `SetYieldTimeUs(v)`.
    pub fn set_yield_time_us(&mut self, v: f64) {
        if let Some(c) = self.ctx_mut() {
            c.yield_time_us = v;
        }
    }

    /// How many times the task has yielded.
    pub fn yield_count(&self) -> Option<u32> {
        self.ctx().map(|c| c.yield_count)
    }

    /// Count another yield.
    pub fn incr_yield_count(&mut self) {
        if let Some(c) = self.ctx_mut() {
            c.yield_count += 1;
        }
    }

    /// When the task blocked (C++ `BlockStart()`).
    pub fn block_start(&self) -> Option<Instant> {
        self.ctx().and_then(|c| c.block_start)
    }

    /// Stamp the block-start time.
    pub fn set_block_start(&mut self, t: Instant) {
        if let Some(c) = self.ctx_mut() {
            c.block_start = Some(t);
        }
    }

    /// Pool queries for task distribution.
    pub fn pool_queries(&self) -> &[PoolQuery] {
        self.ctx().map(|c| c.pool_queries.as_slice()).unwrap_or(&[])
    }

    /// Push a pool query onto the distribution list.
    pub fn push_pool_query(&mut self, q: PoolQuery) {
        if let Some(c) = self.ctx_mut() {
            c.pool_queries.push(q);
        }
    }

    /// Replica subtasks of this execution (owning handles).
    pub fn subtasks(&self) -> &[TaskHandle] {
        self.ctx().map(|c| c.subtasks.as_slice()).unwrap_or(&[])
    }

    /// Add a replica subtask.
    pub fn push_subtask(&mut self, t: TaskHandle) {
        if let Some(c) = self.ctx_mut() {
            c.subtasks.push(t);
        }
    }

    /// Completed replica count.
    pub fn completed_replicas(&self) -> u32 {
        self.ctx()
            .map(|c| c.completed_replicas.load(Ordering::Acquire))
            .unwrap_or(0)
    }

    /// Count a completed replica, returning the count *after* the bump.
    ///
    /// `fetch_add` rather than load/store: the net send and recv workers race
    /// here, and a lost increment strands the origin task's future forever.
    pub fn incr_completed_replicas(&self) -> u32 {
        self.ctx()
            .map(|c| c.completed_replicas.fetch_add(1, Ordering::AcqRel) + 1)
            .unwrap_or(0)
    }

    /// True once every replica has reported completion.
    pub fn all_replicas_complete(&self) -> bool {
        match self.ctx() {
            Some(c) => c.completed_replicas.load(Ordering::Acquire) as usize >= c.subtasks.len(),
            None => false,
        }
    }
}

// ---------------------------------------------------------------------------
// The Task trait
// ---------------------------------------------------------------------------

/// A runtime task (C++: deriving from `class Task`).
///
/// The C++ base class contributes data plus exactly one virtual — the
/// destructor. Dispatch of the real behavior (`Run`, `SaveTask`, `LoadTask`,
/// `NewCopyTask`, …) is a method-id switch in the generated container code,
/// not a Task virtual, so this trait deliberately does not declare `run`:
/// that belongs to the container port, and putting it here would restructure
/// the design rather than port it.
///
/// So the trait surface is: reach the base record, and recover the concrete
/// type. `Box<dyn Task>`'s drop glue supplies the virtual destructor for free.
///
/// Implement it with [`impl_task!`] rather than by hand.
pub trait Task: Any + Send + Sync {
    /// The base task record.
    fn base(&self) -> &TaskBase;
    /// The base task record, mutably.
    fn base_mut(&mut self) -> &mut TaskBase;
    /// Downcast hook — the checked form of the C++ switch's
    /// `static_cast<ConcreteTask*>(task)`.
    fn as_any(&self) -> &dyn Any;
    /// Mutable downcast hook.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl dyn Task {
    /// Recover the concrete task type, or `None` if it isn't a `T`.
    ///
    /// The C++ `static_cast` on the wrong method id is undefined behavior;
    /// this returns `None`.
    pub fn downcast_ref<T: Task>(&self) -> Option<&T> {
        self.as_any().downcast_ref::<T>()
    }

    /// Mutable [`Self::downcast_ref`].
    pub fn downcast_mut<T: Task>(&mut self) -> Option<&mut T> {
        self.as_any_mut().downcast_mut::<T>()
    }
}

/// Implement [`Task`] for a struct whose base record field is named `base`
/// (or a named field, given a second argument).
///
/// This is the counterpart of the C++ `CLASS_NAME` / `CLASS_NEW_ARGS` macro
/// dance in `task.h`: boilerplate that exists because the language cannot
/// express "this struct is a Task" any more directly.
///
/// ```
/// use clio_run_task::{impl_task, Task, TaskBase};
///
/// struct PingTask {
///     base: TaskBase,
///     payload: String, // owns its data: freed by Drop, no TASK_DATA_OWNER rule
/// }
/// impl_task!(PingTask);
/// ```
#[macro_export]
macro_rules! impl_task {
    ($t:ty) => {
        $crate::impl_task!($t, base);
    };
    ($t:ty, $field:ident) => {
        impl $crate::Task for $t {
            fn base(&self) -> &$crate::TaskBase {
                &self.$field
            }
            fn base_mut(&mut self) -> &mut $crate::TaskBase {
                &mut self.$field
            }
            fn as_any(&self) -> &dyn ::std::any::Any {
                self
            }
            fn as_any_mut(&mut self) -> &mut dyn ::std::any::Any {
                self
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    // Two concrete task types, standing in for the generated ones.
    struct PingTask {
        base: TaskBase,
        payload: String,
    }
    impl_task!(PingTask);

    struct PongTask {
        base: TaskBase,
    }
    impl_task!(PongTask);

    fn ping(method: MethodId) -> PingTask {
        PingTask {
            base: TaskBase::new(PoolId::new(1, 0), method, PoolQuery::local()),
            payload: "hello".into(),
        }
    }

    #[test]
    fn task_group_null_semantics() {
        assert!(TaskGroup::null().is_null());
        assert!(TaskGroup::default().is_null());
        assert_eq!(TaskGroup::null().id(), -1);
        assert!(!TaskGroup::new(0).is_null()); // 0 IS a real group
        assert!(!TaskGroup::new(7).is_null());
    }

    #[test]
    fn base_defaults_mirror_cpp_ctor() {
        let t = ping(15);
        let b = t.base();
        assert_eq!(b.pool_id, PoolId::new(1, 0));
        assert_eq!(b.method, 15);
        assert_eq!(b.period_ns, 0.0);
        assert!(b.task_group.is_null());
        assert!(b.succeeded());
        assert!(!b.fut.is_complete());
        assert!(!b.is_executing());
        // The PoolQuery defaults fixed in clio-run-types flow through.
        assert_eq!(b.pool_query.parallelism, 32);
        assert!(!b.pool_query.has_container_id());
    }

    #[test]
    fn flag_accessors_read_the_right_bits() {
        let mut t = ping(1);
        let b = t.base_mut();
        assert!(!b.is_periodic() && !b.is_data_owner() && !b.is_remote());

        b.task_flags.set_bits(task_flags::DATA_OWNER);
        assert!(b.is_data_owner());
        // The aliasing regression: REMOTE must not read as DATA_OWNER.
        assert!(!b.is_remote());
        assert!(!b.is_periodic());

        b.task_flags.set_bits(task_flags::REMOTE);
        assert!(b.is_remote() && b.is_data_owner());

        b.task_flags.unset_bits(task_flags::DATA_OWNER);
        assert!(b.is_remote() && !b.is_data_owner());

        b.task_flags.set_bits(task_flags::FIRE_AND_FORGET);
        assert!(b.is_fire_and_forget());
        assert!(!b.is_batch_aggregate() && !b.is_external_client());
    }

    #[test]
    fn period_round_trips_through_a_unit() {
        const K_MILLI: f64 = 1_000_000.0; // ns per ms
        let mut t = ping(1);
        let b = t.base_mut();
        b.set_period(5.0, K_MILLI);
        assert_eq!(b.period_ns, 5_000_000.0);
        assert_eq!(b.period(K_MILLI), 5.0);
    }

    #[test]
    fn run_context_lifecycle() {
        let mut t = ping(1);
        let b = t.base_mut();

        // Not executing: accessors report absence rather than exploding.
        assert!(!b.is_executing());
        assert_eq!(b.run_worker_id(), None);
        assert!(!b.is_yielded() && !b.is_started() && !b.is_routed());
        assert_eq!(b.pool_queries().len(), 0);
        assert_eq!(b.subtasks().len(), 0);
        b.set_run_worker_id(3); // no-op, must not panic
        assert_eq!(b.run_worker_id(), None);

        b.begin_run_context();
        assert!(b.is_executing());
        b.set_run_worker_id(3);
        assert_eq!(b.run_worker_id(), Some(3));

        b.set_yielded(true);
        b.set_started(true);
        assert!(b.is_yielded() && b.is_started());
        assert!(!b.is_routed() && !b.did_work()); // independent bits
        b.set_yielded(false);
        assert!(!b.is_yielded() && b.is_started());

        // begin_ replaces the context; ensure_ keeps it.
        b.begin_run_context();
        assert_eq!(b.run_worker_id(), Some(0), "fresh context");
        b.set_run_worker_id(9);
        b.ensure_run_ctx();
        assert_eq!(b.run_worker_id(), Some(9), "ensure must not clobber");

        b.reset_run_ctx();
        assert!(!b.is_executing());
        assert_eq!(b.run_worker_id(), None);
    }

    #[test]
    fn replica_completion_counts_and_all_complete() {
        let mut t = ping(1);
        let b = t.base_mut();
        b.begin_run_context();

        // No subtasks: vacuously complete, matching `completed >= size()`.
        assert!(b.all_replicas_complete());

        for _ in 0..2 {
            b.push_subtask(Arc::new(PongTask {
                base: TaskBase::default(),
            }));
        }
        assert_eq!(b.subtasks().len(), 2);
        assert!(!b.all_replicas_complete());

        assert_eq!(b.incr_completed_replicas(), 1);
        assert!(!b.all_replicas_complete());
        assert_eq!(b.incr_completed_replicas(), 2);
        assert!(b.all_replicas_complete());
    }

    #[test]
    fn completion_publishes_out_fields() {
        let t = ping(1);
        let b = t.base();
        assert!(!b.fut.is_complete());
        b.complete(4, 0);
        assert!(b.fut.is_complete());
        assert!(b.succeeded());
        assert_eq!(b.completer.load(Ordering::Relaxed), 4);

        let t2 = ping(1);
        t2.base().complete(2, 22);
        assert!(!t2.base().succeeded());
        assert_eq!(t2.base().return_code.load(Ordering::Relaxed), 22);
    }

    #[test]
    fn set_null_clears_everything() {
        let mut t = ping(15);
        let b = t.base_mut();
        b.task_flags.set_bits(task_flags::REMOTE);
        b.task_group = TaskGroup::new(4);
        b.set_period(1.0, 1000.0);
        b.begin_run_context();
        b.complete(3, 9);

        b.set_null();
        assert!(b.pool_id.is_null());
        assert!(b.task_id.is_null());
        assert_eq!(b.method, 0);
        assert!(!b.is_remote());
        assert_eq!(b.period_ns, 0.0);
        assert!(b.task_group.is_null());
        assert!(b.succeeded());
        assert!(!b.fut.is_complete());
        assert!(!b.is_executing(), "SetNull drops the RunContext");
    }

    #[test]
    fn copy_start_copies_in_out_but_not_run_ctx() {
        let mut src = ping(15);
        let sb = src.base_mut();
        sb.task_flags.set_bits(task_flags::DATA_OWNER);
        sb.task_group = TaskGroup::new(2);
        sb.pool_query = PoolQuery::direct_id(7, -1.0);
        sb.begin_run_context();
        sb.complete(5, 3);

        let mut dst = PongTask {
            base: TaskBase::default(),
        };
        dst.base_mut().copy_start(src.base());
        let db = dst.base();
        assert_eq!(db.pool_id, PoolId::new(1, 0));
        assert_eq!(db.method, 15);
        assert!(db.is_data_owner());
        assert_eq!(db.task_group, TaskGroup::new(2));
        assert_eq!(db.pool_query.container_id, 7);
        assert_eq!(db.return_code.load(Ordering::Relaxed), 3);
        assert_eq!(db.completer.load(Ordering::Relaxed), 5);
        assert!(!db.is_executing(), "each task owns its own RunContext");
    }

    #[test]
    fn dyn_task_downcasts_to_the_concrete_type() {
        let t: Box<dyn Task> = Box::new(ping(15));
        assert_eq!(t.base().method, 15);
        assert_eq!(t.downcast_ref::<PingTask>().unwrap().payload, "hello");
        // The wrong type is None, not the C++ static_cast's UB.
        assert!(t.downcast_ref::<PongTask>().is_none());

        let mut t: Box<dyn Task> = Box::new(ping(1));
        t.downcast_mut::<PingTask>().unwrap().payload = "bye".into();
        assert_eq!(t.downcast_ref::<PingTask>().unwrap().payload, "bye");
    }

    #[test]
    fn dropping_a_boxed_task_runs_the_concrete_destructor() {
        // The point of keeping Task non-POD: `Box<dyn Task>` drop glue IS the
        // virtual destructor, so owned fields are freed with no teardown
        // table and no TASK_DATA_OWNER conditional-free rule.
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        struct OwnsBuffer {
            base: TaskBase,
            _buf: Vec<u8>,
        }
        impl_task!(OwnsBuffer);
        impl Drop for OwnsBuffer {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::Relaxed);
            }
        }

        {
            let _t: Box<dyn Task> = Box::new(OwnsBuffer {
                base: TaskBase::default(),
                _buf: vec![0u8; 64],
            });
            assert_eq!(DROPS.load(Ordering::Relaxed), 0);
        }
        assert_eq!(DROPS.load(Ordering::Relaxed), 1, "dropped through the vtable");
    }

    #[test]
    fn task_handles_share_ownership_like_shared_ptr() {
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        struct Counted {
            base: TaskBase,
        }
        impl_task!(Counted);
        impl Drop for Counted {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::Relaxed);
            }
        }

        let a: TaskHandle = Arc::new(Counted {
            base: TaskBase::default(),
        });
        let b = Arc::clone(&a);
        assert_eq!(Arc::strong_count(&a), 2);
        drop(a);
        assert_eq!(DROPS.load(Ordering::Relaxed), 0, "still one handle alive");
        drop(b);
        assert_eq!(DROPS.load(Ordering::Relaxed), 1, "last handle frees it");
    }
}
