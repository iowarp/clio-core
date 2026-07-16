# Task ABI: the C++/Rust contract for context-runtime (issue #756)

This is the gating design document for porting context-runtime. It defines
what a *task* is as a **frozen memory/wire format** rather than as a C++
class, so that C++ and Rust can both create, execute, and destroy tasks —
and so the C++ APIs survive the migration as facades.

Read alongside `context-transport-primitives/rust/MEMORY_DESIGN.md` (the
segment/pointer contract this builds on).

## 1. Why the C++ `Task` cannot simply move

`clio::run::Task` (context-runtime/include/clio_runtime/task.h) has three
properties that are C++-only by construction:

1. **Polymorphism.** `Task` has a virtual destructor (issue #556 / the
   shared_ptr RAII work), so every task object carries a **vtable pointer**.
   Vtables do not cross languages: a Rust-constructed object can never be a
   valid C++ `Task*`.
2. **Templates.** `Admin::CreateTask` is
   `GetOrCreatePoolTask<CreateParamsT, MethodId, ...>` with variadic emplace
   ctors and member-template `SerializeIn<Archive>`. Templates are
   compile-time C++; there is no artifact a Rust crate can instantiate.
3. **Allocator-backed members.** `priv::string blob_name_`, `ShmPtr<>
   blob_data_`, whose invariants live in C++ code.

Hence the cardinal rule:

> **Whoever constructs an object destroys it — unless the type's layout AND
> teardown are specified as an ABI rather than as code.**

## 2. The enabling observation: the C++ `Task` conflates two things

Reading `task.h`, the fields fall into two disjoint groups:

| Group | Fields | Shareable? |
|---|---|---|
| **Task record** | `pool_id_`, `task_id_`, `pool_query_`, `method_`, `task_flags_`, `period_ns_`, `task_group_`, `return_code_`, `completer_`, `fut_.{is_complete_, task_size_, waiter_pid_, waiter_tid_}` | ✅ POD, meaningful in any process |
| **Execution state** | `run_ctx_` (a `unique_ptr` to process-local heap!), `coro_handle_`/`fiber_state_`, `worker_id_`, `container_` (`DynamicContainer`), `is_new_data_` | ❌ process-local; a pointer into *this* process's heap |

The second group is already meaningless in another process — today's C++ SHM
tasks carry it across the boundary anyway and simply never touch it there
(a latent hazard: a task record in SHM contains a `unique_ptr` and a vptr
that are garbage to every other process).

**The split is the design.** `TaskPodBase` is group 1 only. Group 2 moves to
a **process-local side table** in the runtime, keyed by task id. This is
what makes tasks language-neutral, and it removes the vptr/`unique_ptr`-in-
shared-memory hazard as a side effect.

## 3. Frozen layout: `TaskPodBase`

```rust
#[repr(C)]
pub struct TaskPodBase {
    // --- identity / routing ---
    pub task_id: TaskId,        // 32 B (see below)
    pub pool_id: PoolId,        // 8 B  { major: u32, minor: u32 }
    pub pool_query: PoolQuery,  // POD, see §5
    pub method: u32,            // method id (see §4)
    pub task_flags: u32,        // TASK_* bits
    pub period_ns: f64,         // periodic tasks
    pub task_group: i64,        // scheduling affinity (0 = none)
    // --- lifecycle (in-band, so either language can be last releaser) ---
    pub refcount: AtomicU32,    // replaces the C++ shared_ptr control block
    pub is_complete: AtomicU32, // completion signal (was fut_.is_complete_)
    pub return_code: u32,
    pub completer: u32,         // container id that completed it
    pub task_size: u32,         // sizeof(concrete task) — teardown/copy need it
    pub waiter_pid: u32,
    pub waiter_tid: u32,
    pub owner_impl: u32,        // 1 = C++ constructed, 2 = Rust constructed
}
```

`TaskId` is frozen as today's C++ layout (`types.h`):
`{ pid: u32, tid: u32, major: u32, replica_id: u32, unique: u32, node_id: u32, net_key: u64 }` = 32 B.
`PoolId` = `UniqueId` = `{ major: u32, minor: u32 }`; null is `(0,0)`.

Rules:
- The concrete task struct **starts with** `TaskPodBase` (`#[repr(C)]`, C++
  side: standard-layout struct with `TaskPodBase` as first member, **no
  virtuals**).
- No process-local pointers anywhere in a task record. Payload references
  are `ShmPtr` (MEMORY_DESIGN.md), never raw pointers.
- `owner_impl` records the constructing implementation — the transition
  single-owner rule (§7) is enforced against it.

## 4. Method-id registry

Each module owns a `u32` method space (already true: `Method::kPutBlob = 15`
etc. in the autogen `*_methods.h`). The registry is **append-only** —
method ids appear in serialized task archives and in the WAL, so renumbering
breaks stored/in-flight data, exactly like the compression `LibraryId` wire
values.

Dispatch is *already* method-id switch-based in the autogen code
(`Run`, `SaveTask`, `LoadTask`, `NewCopyTask`, `AggregateOut`,
`LocalSaveTask`, ...) — the virtual destructor is the **lone holdout**.

## 5. `PoolQuery`

`PoolQuery` is POD-shaped already (routing mode + a few u32/u64 params +
net_timeout). It is frozen field-for-field; modes: `DirectId`, `DirectHash`,
`Range`, `Broadcast`, `Physical`, `Dynamic`, `Local`, `ManyToOne`,
`AllToOne`. Both languages must produce identical routing decisions for
identical inputs — this is a *behavioral* ABI, not just a layout one, so the
port owes a shared conformance test vector set.

## 6. Teardown: from virtual dtor to dispatch table

Replace `~Task()` with a per-method teardown specification, mirroring how
`SaveTask`/`NewCopyTask` already work:

```
destroy_task(method, ptr, alloc):
  match method:
    kPutBlob => {
      if flags & TASK_DATA_OWNER { alloc.free(blob_data) }   // the ZMQ-leak-fix rule, now a SPEC clause
      shm_string_destroy(&mut blob_name, alloc)
      // ...
    }
    kGetOrCreatePool => { shm_string_destroy(&mut pool_name, alloc); ... }
    ...
```

Consequences:
- Rust task types are `#[repr(C)]` with **no `Drop` impl** (a `Drop` on a
  shared-memory view is a double-free machine). Owning handles call
  `destroy_in(alloc)` exactly once.
- The `TASK_DATA_OWNER` conditional free stops being dtor code and becomes a
  written rule — reviewable, and identical in both languages.
- Refcount lives in `TaskPodBase.refcount`, so "last dropper frees" works
  regardless of which language drops last.

## 7. Two adoption patterns

**Pattern A — migrate the handler, keep the task C++ (zero API change).**
The C++ task struct and all client-side API stay. The runtime-side handler
moves to Rust and receives the task through FFI as an opaque handle +
accessors (or as its serialized archive bytes). C++ allocates and destroys;
Rust only decides *when*. Correct during the transition, awkward for
data-heavy tasks.

**Pattern B — task defined in Rust `#[repr(C)]`, C++ gets a mirrored POD +
inline facade.** The destination. The codebase already invented the bridge:
the **Pod task family from #556** (`PodPutBlobTask`: `fixed_string<32>`, no
SSO fixup, method-id dispatch, no virtuals) *is* a language-neutral task.
Generalize it:

```rust
// Rust — authoritative
#[repr(C)]
pub struct CreateTaskPod {
    pub base: TaskPodBase,
    pub pool_name: FixedString<64>,
    pub params: SerializedBytes,   // CreateParams as archive bytes, not a template
}
```
```cpp
// C++ — mirrored layout, no virtuals; the "API" is inline sugar
struct CreateTaskPod { /* identical layout */ };
inline Future<CreateTaskPod> AsyncCreatePool(...) { /* fill + send */ }
```

The C++ API never disappears; it degrades from *the implementation* to *an
inline veneer over the shared ABI*.

## 8. Worked example: `Admin::CreateTask` (the easy one)

`GetOrCreatePoolTask<CreateParams>` already carries its params as
**serialized bytes** (`chimod_params_`, a `priv::string`; `GetParams()`
deserializes on demand). So de-templating costs almost nothing:

| C++ today | Pattern B |
|---|---|
| `GetOrCreatePoolTask<CreateParamsT>` template | `CreateTaskPod` (one type) |
| `CreateParamsT` compile-time param | `params: SerializedBytes` (already the case on the wire) |
| `GetParams()` template method | `params.deserialize::<T>()` on either side |
| virtual dtor | teardown table entry: destroy `pool_name`, free `params` |

## 9. Worked example: `PutBlobTask` (the hard one)

Needs: `blob_name` (`priv::string` → `ShmString`, ABI-specified layout incl.
the SSO threshold — the `ctp-ds/src/shm_string.rs` port), `blob_data`
(`ShmPtr`, already ABI), and the **conditional owner free** (spec clause
above). Once those three exist, `PutBlobTask` is Pattern-B-able.

## 10. What can never be shared

Some current OUT fields are raw process-local C++ heap: e.g.
`ListTargetsTask::target_names_` is `std::vector<std::string>`, and
`SemanticSearchTask::results_` is a `std::vector<SemanticSearchResult>`.
These cannot be ABI-shared **even between two C++ processes** — they work
today only because those paths either share an address space or re-serialize.

The migration audit must flag every such member. Each one either becomes
`ShmVec`/`ShmString`, or the task stays behind a serialization boundary
permanently (which is fine — that is what `SaveTask`/`LoadTask` are for).

## 11. Risk register

| Risk | Mitigation |
|---|---|
| Cross-language allocator races | MEMORY_DESIGN.md pillar 1: one segment, one owner. Tasks are allocated by the owner of their segment. |
| Method-id or layout drift between the languages | Both sides generated/checked from this spec; layout asserted by `size_of`/`offset_of` tests on BOTH sides; CI conformance vectors. |
| `PoolQuery` routing divergence | Shared conformance test vectors (§5), not just layout tests. |
| A task record still containing a vptr | Pattern-B types are asserted virtual-free on the C++ side (`std::is_standard_layout` + explicit size check). |
