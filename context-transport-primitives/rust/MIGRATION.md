# CTP → Rust Migration Plan (issue #756)

Goal: migrate the ENTIRE system to Rust — context-exploration-engine,
context-runtime, context-transfer-engine, context-assimilation-engine, and
context-transport-primitives — tracking how compilation times and code
quality are affected at every step (see Metrics below). Measured scope of
the C++ codebase (2026-07-16, excluding build dirs and external/):

| Component | Files | Lines |
|---|---|---|
| context-transport-primitives | 248 | 72,012 |
| context-runtime | 185 | 83,085 |
| context-transfer-engine | 179 | 76,894 |
| context-assimilation-engine | 48 | 12,739 |
| context-exploration-engine | 8 | 2,146 |
| **Total** | **668** | **~247k** |

Some components are unavoidably C++ and stay behind wrapper crates:

- **GPU kernels** (CUDA/ROCm/SYCL device code) — see "GPU extensions" below
- **libthallium transport** (Mochi/Argobots C++ API)
- Anything else with a C++-template-shaped API surface consumed by device code

## GPU extensions (ctp-gpu)

Kernels remain CUDA C++ (`ctp-gpu/kernels/*.cu`); the `ctp-gpu` crate wraps
them for Rust callers, mirroring the C++ `GpuApi` surface (device/managed
alloc, memcpy, launch, synchronize). Mechanism: kernel source is embedded
at build time and **JIT-compiled by NVRTC at runtime for the local device's
exact SM (CUBIN)** — no nvcc/host-C++-compiler needed at build time, and no
driver-JIT PTX-version skew (a 13.x NVRTC emits PTX a 12.x-era driver can't
load; compiling to the device's own SM sidesteps it). `--features cuda`
links the driver API + NVRTC via CUDA_PATH/CUDA_HOME//usr/local/cuda.
Verified on RTX 5080 (sm_120, CUDA 13.3) and buildable against the
devcontainer's CUDA 12.6. ROCm/SYCL backends follow the same pattern when
their modules migrate.

## Metrics: compile time + code quality

`metrics/collect.py` appends one row per crate per run to
`metrics/history.csv`: LOC (incl. wrapped .cc/.cu), `unsafe` count, clippy
warnings, cold/warm build seconds, tests passed. `--cpp-cmd` times an
equivalent C++ build as the `cpp-baseline` row. First datapoint
(2026-07-16): the 5-crate Rust workspace cold-builds in **~6.5 s** and
warm-rebuilds in **<1 s per crate** vs **83 s** for a clean `clio_ctp_host`
C++ build in the devcontainer — NOT yet apples-to-apples (2.1k Rust lines
vs the full 72k-line C++ lib); the honest comparison emerges as Rust
coverage approaches the C++ module's scope, which is exactly what the
history file tracks. Quality snapshot: 0 clippy warnings; 82 `unsafe`
tokens, all confined to the FFI/GPU boundary crates (ctp-types has zero).

## Strategy: parallel crate + C ABI

The `ctp-rs` cargo workspace (this directory) implements CTP modules in Rust.
Each module exposes a **C ABI** (`ctp-ffi` crate → `include/ctp_rs.h`) so the
C++ consumers (context-runtime, CTE) can adopt modules one at a time. The C++
implementation of a module is retained until its Rust replacement reaches
parity (same tests passing), then the C++ side becomes a thin shim over the
FFI and is eventually deleted.

Builds are integrated behind the CMake option `CLIO_CTP_ENABLE_RUST`
(default **OFF**) so nothing changes for existing builds/CI until opted in.

## Coroutine model: both, behind one trait

Mirroring the C++ side's stackless (C++20) / stackful
(`CLIO_CORE_ENABLE_BOOST_COROUTINES`) duality, `ctp-coroutine` defines one
`TaskExecutor` abstraction with two backends:

| Backend | Mechanism | Feature flag |
|---|---|---|
| `stackless` (default) | native Rust `async`/`await` on a minimal cooperative single-thread executor with `yield_now()` | always on |
| `boost-fibers` | stackful fibers wrapping **Boost.Context** via a small C++ shim (`shim/boost_fiber_shim.cc`, built with the `cc` crate) | `boost-fibers` |

The stackful backend exists for interop with the C++ worker during migration:
C++ boost-fiber tasks and Rust tasks can share one scheduler. New pure-Rust
code should use the stackless backend.

## Module order

| Phase | Module | Crate | Notes |
|---|---|---|---|
| 1 (this PR) | `introspect/SystemInfo`, core `types` | `ctp-introspect`, `ctp-types` | OS abstraction; proves FFI + build end-to-end |
| 1 (this PR) | coroutine abstraction | `ctp-coroutine` | trait + async backend; boost backend feature-gated |
| 2 | `util` (logging, config parse), `serialize` | | serde ecosystem |
| 3 | `memory` (backends, allocators, smart_ptr) | | unsafe-heavy SHM core; needs careful parity tests + miri where possible |
| 4 | `data_structures` (ipc ring buffers, priv maps) | | depends on 3 |
| 5 | `lightbeam` (zmq/socket/shm transports) | | thallium stays C++ behind a wrapper crate; zmq via `zmq` crate or FFI to the vendored libzmq |
| 6 | `thread` (CTP_THREAD_MODEL), `io`, `compress`, `search`, `solver` | | compress backends largely stay C/C++ libs behind FFI |

## FFI conventions

- All exported symbols prefixed `ctp_rs_`.
- Strings cross the boundary as UTF-8 `const char*` (NUL-terminated); Rust
  copies immediately, never retains the pointer. Returned strings are
  allocated by Rust and freed with `ctp_rs_string_free`.
- No Rust panics may cross the boundary: every `extern "C"` fn is wrapped in
  `catch_unwind` and reports failure via return codes.
- Header `ctp-ffi/include/ctp_rs.h` is **hand-maintained** (reviewed like any
  API); a CI check comparing it against the exported symbols is future work.

## Project-rule carryovers

- **Never `thread_local`** (per project rule / Windows per-DLL duplication):
  Rust `thread_local!` is similarly banned in crates that may be delivered as
  multiple cdylibs; thread-affine state goes through the executor context.
- Timing values reported in **ms**.
- Tests: pure-Rust tests use `cargo test`; anything touching the Clio runtime
  uses the C++ `simple_test.h` harness through the FFI.
