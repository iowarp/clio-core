# CTP Rust port audit — type fragmentation and layering (issue #756)

Findings from auditing the agent-generated CTP tranche (commit `d969ac0b`)
while trying to build the context-runtime port on top of it.

**The tranche compiles and its 1210 tests pass. That is not the same as it
being a coherent port**, and the gap is the subject of this document. Each
module was ported in isolation by a separate agent, and where two C++
headers shared a type, each agent re-declared that type locally instead of
sharing one. Every module therefore type-checks against *itself* and tests
*itself*, which is exactly why the suite is green. The types they exchange
are nevertheless incompatible, so the modules cannot yet be wired together —
and wiring them together is what porting context-runtime requires.

Nothing here is a code-generation slip to be patched. These are structural
and want deliberate consolidation.

## 1. `ctp-lightbeam` declares the same C++ types three times

C++ `clio_ctp/lightbeam/lightbeam.h` defines **one** `Bulk`, `LbmMeta`,
`ClientInfo`, `TransportType`, `TransportMode` and `LbmContext`.
`shm_transport.h` and `socket_transport.h` `#include` it.

The Rust port has three of each — one per module file:

| Type | `transport.rs` | `shm_transport.rs` | `socket_transport.rs` |
|---|---|---|---|
| `Bulk` | :281 | :276 | :285 |
| `LbmMeta` | :414 | :317 | :323 |
| `ClientInfo` | :362 | :296 | :300 |
| `TransportType` | :691 | :254 | :264 |

They are not even the same shape:

```rust
// transport.rs — faithful to lightbeam.h
pub struct Bulk { data: FullPtr, size: usize, flags: Bitfield32, desc: usize, mr: usize }
// shm_transport.rs — invented
pub struct Bulk { data: Vec<u8>, shm: ShmPtr<u8>, size: usize, flags: Bitfield32 }
// socket_transport.rs — invented
pub struct Bulk { data: Vec<u8>, size: usize, flags: Bitfield32, owned: bool }
```

The C++ is `{ FullPtr<char> data; size_t size; bitfield32_t flags; void* desc;
void* mr; }`, so **`transport.rs` is the correct one** and the other two are
divergent local inventions.

Consequence: a `Bulk` produced by the SHM transport cannot be handed to the
socket transport, and neither can be handed to the generic `Transport`
surface. `LbmMeta` — the archive/transport hand-off type — is likewise three
unrelated types. In C++ this is one polymorphic surface.

**Fix:** `transport.rs` becomes the canonical `lightbeam.h` (it already is,
in content); `shm_transport.rs` and `socket_transport.rs` import from it and
drop their copies. This is a real port fix rather than a rename: each
transport's logic is written against its own `Bulk`'s shape, so the SHM
transport's `shm: ShmPtr<u8>` and the socket transport's `owned: bool` have
to be re-expressed in terms of the canonical `FullPtr`/`desc`/`mr` (which is
how the C++ carries exactly this information).

## 2. `ctp-ds/src/global_serialize.rs` contains context-runtime, not CTP

CTP is the bottom layer; context-runtime sits on top. The C++ respects this:
`clio_ctp/.../global_serialize.h` contains **no** runtime types, and no CTP
header references `clio::run` domain types. Serialization stays generic
through templates.

The Rust file (2715 lines) is roughly one third generic byte engine and two
thirds context-runtime, and its own module doc says so — it declares that it
ports "the archive shape layered on top of them in
`clio_runtime/task_archives.h`". It defines:

| Symbol | :line | Belongs in |
|---|---|---|
| `UniqueId`, `PoolId` | 751 | `clio-run-types` (already there) |
| `TaskId` | 808 | `clio-run-types` (already there) |
| `TaskInfo` | 866 | context-runtime archives |
| `MsgType` | ~920 | context-runtime archives |
| `Bulk` | 948 | `ctp-lightbeam` (a 5th copy; see §1) |
| `NetTaskArchive` | 1051 | context-runtime archives |
| `SaveTaskArchive` | 1136 | context-runtime archives |
| `LoadTaskArchive` | 1266 | context-runtime archives |

Why it happened is legible and almost sympathetic: C++ serializes generically
via templates, and Rust cannot. The agent needed *concrete* `GlobalSave`/
`GlobalLoad` impls to have anything to test, so it defined the domain types
next to the traits — and having done that, the archives followed.

Two consequences:

1. **`TaskId`/`UniqueId` now exist twice**, and only the `clio-run-types`
   copies are checked against the C++ headers by
   `clio-run-types/tests/cpp_abi_conformance.rs`. The `ctp-ds` copies are
   invisible to it and free to drift — which matters, because that harness
   exists precisely because four hand-transcription bugs got through review.
2. **`task_archives.h` is already ported**, just in the wrong crate. So this
   is mostly a move, not a rewrite.

**Fix:** keep the generic engine (`GlobalSerialize`, `GlobalDeserialize`,
`GlobalSave`/`GlobalLoad` + primitive/`String` impls) in `ctp-ds`. Move the
archives to a new context-runtime crate. Delete the duplicate domain types
and use `clio-run-types`.

The orphan rule decides where the trait impls live: `impl ctp_ds::GlobalSave
for clio_run_types::UniqueId` is legal only inside `clio-run-types` (local
type, foreign trait) — not in a third crate. So `clio-run-types` takes a
dependency on `ctp-ds` and implements the traits for its own types. That is
the correct direction: runtime depends on CTP, never the reverse.

## 3. Lower-severity duplicates, for triage

- `RingBufferEntry` in both `ctp-ds/ring_buffer.rs:330` and
  `multi_ring_buffer.rs:329` — same crate, likely one C++ type.
- `Gpu`, `DeviceBuffer`, `Module`, `Kernel`, `GpuError` in both
  `ctp-gpu/cuda.rs` and `ctp-gpu-backends/hip.rs`. Defensible — they are
  genuinely different backends — but the C++ reaches them through one
  `GpuApi` surface, so they should end up behind a shared trait rather than
  as unrelated look-alikes.
- `ctp-net/thallium.rs:577` `Bulk<'a>` is thallium's own and probably fine.

## 4. What this means for the context-runtime port

context-runtime sits directly on lightbeam and the archives, so §1 and §2 are
prerequisites rather than cleanup-someday. Porting more of the runtime onto
three incompatible `LbmMeta`s would mean choosing one arbitrarily and
discovering the mistake at integration, which is the expensive end to find it.

Recommended order:

1. §2 — mostly a move; unblocks the archives, which is how tasks actually
   travel (tasks are serialized into a buffer that is copied through shared
   memory; they are not stored in it).
2. §1 — a real fix; unblocks every transport.
3. Then resume the runtime port (container → module/pool manager → worker →
   ipc_manager).

## 5. The generalisable lesson

Parallel agents ported one C++ header each. Nothing owned the *shared* types
between them, and a green test suite could not see it: isolation is what made
the modules independently testable and is also what let them diverge.

For the remaining work, shared types should be ported **first and once**, and
the module agents made to import them. Where a C++ header is `#include`d by
two others, that header is a contract, and it needs an owner before its
consumers are parallelised.
