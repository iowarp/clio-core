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

## 1a. `ctp-lightbeam`'s abstraction is implemented only by its test doubles

Found while scoping the fix in §1, and it is the more serious half.

`ctp-lightbeam` has **zero `use crate::`** statements. `transport.rs`,
`shm_transport.rs` and `socket_transport.rs` never reference each other.
They are three parallel ports that share a crate and nothing else.

`transport.rs` defines `pub trait Transport` (:647) and `TransportFactory`
(:835), a runtime registry keyed by `TransportType`.

> **Correction (first draft of this section was wrong).** It claimed "in C++,
> `Transport` is a base class and `ShmMpscTransport`/`SocketTransport` derive
> from it". Half of that is false and the mechanism is not what it says.
>
> C++ `Transport` has **no virtual functions at all** — `~Transport()` is
> `= default`, not virtual, and `Send`/`Recv` are `template <typename MetaT>`,
> which cannot be virtual. It is a tagged base carrying `type_`/`mode_`, and
> `transport_factory_impl.h` dispatches by hand: `Transport::Expose` does
> `switch (type_) { case …: return static_cast<ZeroMqTransport*>(this)->Expose(…); }`,
> and destruction is the same switch over `delete static_cast<Concrete*>(t)`.
> It is the same vtable-free dispatch the runtime uses for task method ids.
> (`static_cast` base→derived does still require inheritance, which is why
> those classes derive.)
>
> And who derives is not what the draft said:
>
> | C++ class | derives `Transport`? |
> |---|---|
> | `ShmTransport` (`shm_transport.h`) | yes, `#if CTP_IS_HOST` |
> | `SocketTransport`, `ZeroMqTransport`, `ThalliumTransport`, `NixlTransport` | yes |
> | **`ShmMpscTransport`** (`shm_mpsc_transport.h`) | **no — standalone** |
>
> `ShmMpscTransport` is a different class from `ShmTransport`: the named MPSC
> segment transport (issue #642), used directly by `ipc_cpu2cpu.cc` rather
> than through the factory. Rust's `shm_transport.rs` ports *both* headers,
> which is what made them easy to conflate.
>
> The finding survives the correction, but state it properly: **`ShmTransport`
> and `SocketTransport` should be reachable through the abstraction and are
> not.** `ShmMpscTransport` standing alone is faithful, not a defect.

In Rust today:

- The only `impl Transport for` types are `MinimalTransport` (:983) and
  `EmTransport` (:1011) — **both inside `transport.rs`'s own `#[cfg(test)]`
  module**, which starts at :948.
- `ShmTransport` (`shm_transport.rs:958`) and `SocketTransport`
  (`socket_transport.rs:902`) do **not** implement `Transport`, though their
  C++ counterparts derive from it. `transport.rs` never names them.
- `ShmMpscTransport` (`shm_transport.rs:1247`) correctly does not — its C++
  counterpart has no base either.
- Every `TransportFactory::register` call is at :1658 or later — also inside
  `#[cfg(test)]` — and registers `make_minimal`/`make_nothing`.

Note the Rust trait is itself a deliberate divergence, and a good one: it
replaces the C++ `switch` + `static_cast` with something the compiler checks.
Worth keeping — but it has to be *implemented* to mean anything.

So at runtime `TransportFactory::get(TransportType::Shm, …)` returns `None`:
no real backend is registered, and none could be, because the real transports
do not implement the trait the registry stores. The abstraction is exercised
exclusively by mocks defined next to it.

This is why the 140 tests pass while nothing is connected. The trait's tests
test the trait against doubles; each transport's tests test that transport
against itself. Both halves are green and there is no test that could notice
the gap, because a test that used a real transport *through* the factory is
exactly the test nobody wrote.

Consequence for the estimate: §1 is not "unify some type declarations". It is
"connect the port to its own abstraction" — make `ShmTransport` and
`SocketTransport` implement `Transport`, unify the types they exchange, and
register them with the factory. That is the largest single item in this audit.

### How widespread is this? Bounded — and mostly good news

Running the §5 grep across every trait in the tranche (counting implementors
outside `#[cfg(test)]` against implementors inside it):

| Trait | Crate | Real impls | Test-only impls |
|---|---|---|---|
| `Transport` | ctp-lightbeam | **0** | 2 |
| `EventManager` | ctp-lightbeam | **0** | 1 |
| `BulkAllocator` | ctp-ds | **0** | 1 |
| `Compressor` | ctp-compress | 11 | 0 |
| `Distribution` | ctp-util | 5 | 0 |
| `Coroutine` | ctp-coroutine | 2 | 0 |
| `RegionSource` | ctp-memory | 2 | 0 |
| `AsyncIo`, `ThreadModel`, `Aes256BlockCipher`, `RegexMatcher`, `RegexCompiler`, `RingLane` | various | 1 each | — |

So the hollowness is **not** systemic. It is three abstractions, two of them
in `ctp-lightbeam` and one in `ctp-ds`. Everything else — the compression
backends, the coroutine executors, the thread models, the allocator region
sources — has real implementors.

That fits the §5 story rather than contradicting it. `Transport` and
`EventManager` are precisely the two places where C++ puts a base class in
one header and its derived classes in *other* headers, so they are what got
split across agents. Traits whose implementors live in the same header as the
trait (`Compressor`'s backends, `Distribution`'s variants) came through
intact, because no split ran through them.

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

### Progress and the ownership decision

Done: `TransportType`, `TransportMode` and `ClientInfo` are unified
(`734369a2`). `LbmMeta` needs no work of its own — all three declarations are
already field-for-field identical, and differ *only* because their `Bulk` and
`ClientInfo` differ. So **`Bulk` is the single keystone**: unify it and
`LbmMeta` follows.

The decision `Bulk` turns on is where the payload's storage lives.

- The two invented `Bulk`s own a `Vec<u8>`. Safe, and it works today.
- The canonical `Bulk` (and the C++) holds a `FullPtr` — a *location* —
  and the buffer is owned elsewhere.

**Keep the canonical `FullPtr` form.** A `Vec<u8>` cannot express "this
payload lives in the shared segment at `(alloc_id, off)`", which is the whole
point of bulk transfer; owning the bytes forces a copy exactly where the
design exists to avoid one. The `Vec` form only looks adequate because no
shared-segment payload has reached these transports yet.

That costs safety at one seam — turning a `FullPtr` into bytes needs
`unsafe` — but the cost is inherent, not a porting artifact: a cross-process
pointer cannot be a safe slice. Contain it in one helper per transport.

Ownership of *received* buffers rides on the allocator id, as in the C++, via
the new `RECV_ALLOCATED_ID` sentinel (`AllocatorId(u32::MAX-1, u32::MAX-1)`)
and `Bulk::is_recv_allocated`. This is the part `owned: bool` gets wrong, and
it is worth being precise about: the sentinel does not record *that* a buffer
is owned, it records *which allocator* owns it. A recv `Bulk` may hold either
a buffer the transport allocated on the system allocator, or one belonging to
a CTP allocator that the task archive swapped in for the `BULK_EXPOSE`/copy
routes. Freeing the second as if it were the first is a **crash, not a leak**
— a CTP `MallocAllocator`'s user pointer sits 16 bytes inside the real malloc
region, so `free` on it yields glibc "free(): invalid pointer". The C++
carries a long comment about this; it is a bug someone already paid for.
`owned: bool` flattens the distinction away and is only survivable today
because the archives are not wired up yet — i.e. it would fail exactly when
§2 lands.

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

The sharper version, from §1a: **inheritance across headers did not survive
the split.** Every place the C++ has a base class in one header and its
derived classes in others, the port produced the base (with mock implementors
beside it) and the deriveds (standing alone), and nothing joined them. The
tests cannot catch it, because both halves are independently testable and the
only test that would fail — a real implementation reached through the
abstraction — spans exactly the boundary the agents were split along.

So the test count was never evidence of integration. It measured what each
agent could see. When resuming, the useful question about any ported
abstraction is not "do its tests pass" but **"what non-test type implements
it?"** — `grep` for `impl <Trait> for` and check whether the answers live
outside `#[cfg(test)]`. That one grep is what turned this audit from
"duplicated types" into "unimplemented abstraction".

## 6. Status

- [x] `FullPtr` → `ctp-memory` (commit `a77f2a9f`). One definition again.
- [ ] §1a — real transports implement `Transport`; factory registers them.
- [ ] §1 — one `Bulk`/`LbmMeta`/`ClientInfo`/`TransportType`.
- [ ] §2 — archives out of `ctp-ds` into context-runtime.
- [ ] §3 — triage `RingBufferEntry`, the GPU look-alikes.
- [ ] Re-audit the other crates the same way (`impl … for` outside tests).
