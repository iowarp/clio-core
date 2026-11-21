# Building and testing

Paths are relative to the repository root. `file:line` references were checked
against branch `cte-shm-cache`; line numbers drift, so search for the quoted
name if a line does not match.

## 1. The build-directory rules (non-negotiable)

- **Never build in-source.** No `CMakeFiles/`, `CMakeCache.txt`, `Makefile` or
  `cmake_install.cmake` may appear in a source directory. If you create one by
  accident: stop, delete every such artifact from the source tree, and rebuild
  from a proper build directory.
- **Never hardcode absolute paths in a CMakeLists.txt.** Use `find_package()`
  for every dependency and the module functions in section 3.
- **Never `rm -rf` a build directory** (and never `rm -rf build/*` followed by
  `cmake ..` -- that is an in-source build waiting to happen). To start clean,
  configure a NEW directory next to the old one:

  ```bash
  cmake -S . -B build/fresh --preset=debug   # or -DCMAKE_BUILD_TYPE=... options
  cmake --build build/fresh -j "$(nproc)"
  ```

  To clean objects inside one tree: `cmake --build <dir> --target clean`.
- **Third-party libraries** are built in a directory completely outside the
  repository (e.g. `~/builds/<lib>`), installed, and the temporary build
  directory is removed afterwards. Never leave `/tmp/build_*` trees behind.
- **A build directory that cluster jobs load binaries from is LIVE** while any
  job using it is queued or running (check the batch queue, e.g. `qstat -u
  $USER`, first). Rebuilding it mid-run swaps libraries under the daemons and
  invalidates the run. Compile-check in a separate build directory and rebuild
  the live one only between jobs. The Aurora harness says the same:
  "Without [patchelf] the daemons load libraries from the build directory, so
  do not rebuild during a run" (`CI/aurora/README.md`, section "What is
  different from Ares").
- **Install before re-running.** Never re-run tasks/tests against an old
  install after changing code: build (and install, if the thing you run uses
  the installed tree) first. In the devcontainer `sudo` is needed to install
  into system prefixes.
- Libraries use **RPATHs** (section 5), so `LD_LIBRARY_PATH` does NOT override
  which library a binary loads. A stale library next to the binary wins.

## 2. Configure presets and options

`CMakePresets.json` defines the presets (`debug`, `release`, `asan`, `ubsan`,
`msan`, `sanitize`, `cuda-release`, `rocm-debug`, `sycl-debug`, `vfd`,
`release-adapter`, `release-fuse`, `boost`, `leak-check`, ...). **Use the
`debug` preset when compiling in this repo** unless a task needs another. The
`debug` preset (`CMakePresets.json:10-37`) uses Unix Makefiles,
`binaryDir=${sourceDir}/build`, `CMAKE_BUILD_TYPE=Debug`, and turns on runtime,
CTE, CAE, CEE, tests, compression, ADIOS2 adapter, Python, coverage, ZMQ,
cereal and io_uring.

Component switches (`CMakeLists.txt:32-35`), all default ON:
`CLIO_CORE_ENABLE_RUNTIME`, `CLIO_CORE_ENABLE_CTE`, `CLIO_CORE_ENABLE_CAE`,
`CLIO_CORE_ENABLE_CEE`. Example:
`cmake --preset=debug -DCLIO_CORE_ENABLE_CTE=ON -DCLIO_CORE_ENABLE_CAE=OFF`.

Other options you will meet (`CMakeLists.txt:40-203`):

| option | default | effect |
|---|---|---|
| `CLIO_CORE_ENABLE_TESTS` | OFF | master switch for all unit tests (the `debug` preset sets ON) |
| `CLIO_CORE_ENABLE_BENCHMARKS` | ON | benchmarks |
| `CLIO_CORE_ENABLE_RPATH` | ON | relative RPATHs (section 5) |
| `CLIO_CORE_ENABLE_BOOST_COROUTINES` | OFF | stackful Boost.Context fibers instead of C++20 stackless coroutines (`CMakeLists.txt:126`) |
| `CLIO_CORE_ENABLE_CUDA` / `_ROCM` / `_SYCL` | OFF | GPU backends (see `gpu.md`) |
| `CLIO_CORE_ENABLE_ZMQ` | ON | ZeroMQ transport |
| `CLIO_CORE_ENABLE_THALLIUM`, `CLIO_CORE_ENABLE_LIBFABRIC` | OFF | other transports |
| `CLIO_CORE_ENABLE_NIXL`, `CLIO_CORE_ENABLE_NVSHMEM` | OFF | NIXL / NVSHMEM (see `module-anatomy.md` section "Linking") |
| `CLIO_CORE_ENABLE_ELF` | OFF | ELF interception, required by CTE adapters |
| `CLIO_CTE_ENABLE_COMPRESS` | OFF | compressor module + tests |
| `CLIO_CORE_ENABLE_IO_URING` | OFF | io_uring bdev backend |
| `CLIO_ENABLE_DAOS` | OFF | native DAOS bdev |
| `CLIO_CORE_ENABLE_HDF5` | OFF | HDF5 **in the CAE only** (`CMakeLists.txt:203`) |
| `CLIO_CTE_ENABLE_HDF5_VOL` | OFF | builds `clio_vol.cc` and the `cte_hdf5_vol_*` tests |
| `CLIO_CTE_ENABLE_VFD` | OFF | builds `H5FDclio.cc` and the VFD suites |
| `CLIO_CTE_ENABLE_FUSE_ADAPTER`, `_POSIX_ADAPTER` (ON), `_STDIO_`, `_MPIIO_`, `_ADIOS2_`, `_NVIDIA_GDS_` | | CTE adapters |

**Adapter tests need their adapter enabled.** Without the option, a change to
a connector compiles nothing and its tests silently do not exist in the tree.
Check that the file was actually compiled before claiming a build is clean. For
how the HDF5 connectors fit together see
`context-transfer-engine/adapter/HDF5_QUICKSTART.md`. Set
`CLIO_REQUIRE_RUNTIME=1` in any measurement so that a run that never reached
the runtime fails instead of reporting a clean zero.

## 3. Building

```bash
cmake --build <dir> -j "$(nproc)"              # everything
cmake --build <dir> -j 16 -- -k                # keep going past errors (-k goes AFTER --)
cmake --build <dir> -j 16 --target <target>    # one target -- NOT after a task-struct change
```

- **Confirm the targets you care about were built**: look for
  `Built target <name>` (or check the `.so`/binary timestamp). A quiet log or a
  log tail is not proof.
- **Rebuild everything after touching a task struct.** Task structs
  (`PutBlobTask`, `EvictTask`, ...) cross the client/runtime boundary through
  shared memory and the network, so their layout is ABI. Add or reorder a
  field, build only one target, and the other side reads stale offsets --
  which surfaces as plausible garbage, not a crash. (Real case: adding
  `EvictTask::droppable_only_` and rebuilding only the test binary made
  `cte_evict` report `bytes_evicted=0 blobs_evicted=2097152` -- the byte
  budget landing in the wrong field.) Build with no `--target`, including
  benchmarks and adapters that compile runtime headers into themselves.
- **Always verify that code still compiles after a change.** Never comment
  code out to make a build pass.
- All code must compile **without warnings** (the tree is warning-clean): use
  `size_t` for container sizes (no sign-compare warnings), mark intentionally
  unused variables `(void)x;`, avoid implicit narrowing conversions.

## 4. Local build hygiene: two traps that look alike

Both look like "whole families of tests red locally while CI is green", and
both can be present at once.

| symptom | cause | fix |
|---|---|---|
| `PutBlob` rc=11, `available_targets` empty, `GetCapacity` 0 forever even though bdev pools compose and `RegisterTarget` reports success; `cte_evict`, `cte_tiered_storage_all`, `cte_reorganize_all`, HDF5 VOL/VFD suites all red | a **stale build tree** incrementally rebuilt across many commits | configure a NEW build directory next to the old one (`cmake -S . -B build/fresh -DBUILD_TESTING=ON -DCLIO_CORE_ENABLE_TESTS=ON` then build). Do not restart the container or rebuild the devcontainer image -- neither is the cause. |
| tests fail within seconds; `RegisterTarget` rc=1; log shows `Permission denied` on a `/tmp` path, e.g. `Failed to open bdev file: /tmp/clio_vol_io_backend.dat (Permission denied)` or `PermissionError: ... '/tmp/hdf5compat/clio_suite.yaml'` | **root-owned leftovers in /tmp** from sessions that ran the container as root (the container now runs as `iowarp`) | `sudo chown -R iowarp:iowarp /tmp/hdf5compat /tmp/hdf5vfdcompat` and `sudo find /tmp -maxdepth 1 -user root \( -name 'clio*' -o -name '*.dat' -o -name '*.h5' \) -exec chown iowarp:iowarp {} +` |

The second trap hits the adapter suites hardest (`cte_hdf5_vol_io_*`, VOL and
VFD compat suites) because each keeps a fixed backing file or work directory in
`/tmp`.

**CI green is weaker evidence than it looks.** `ci-linux.yml` runs ctest with
`--repeat until-pass:3` and excludes labels `ollama|force_net|flaky`
(`.github/workflows/ci-linux.yml:340`); other workflows exclude tests by name
(`-E`, e.g. `ci-adapters.yml:180`). A test can be flaky, or skipped outright,
and still show green.

## 5. RPATH

Relative RPATHs are on by default (`CLIO_CORE_ENABLE_RPATH=ON`,
`CMakeLists.txt:55`): Linux uses `$ORIGIN` and `$ORIGIN/../lib`; macOS uses
`@loader_path` and `@loader_path/../lib`. Binaries are relocatable (system
installs, pip wheels, conda packages). Disable with
`-DCLIO_CORE_ENABLE_RPATH=OFF` if you want `LD_LIBRARY_PATH` to decide.

## 6. Dependencies

- **Devcontainer**: dependencies come from apt or are built from source in
  `docker/deps-cpu.Dockerfile`; a Python venv is auto-activated at
  `/home/iowarp/venv`; CMake finds packages under `/usr/local` and `/usr`. Do
  NOT use conda in the devcontainer.
- **Outside**: `CI/ci-deps.sh` (recommended, builds from source), system apt
  packages, or the conda recipes in `installers/conda/` (conda is otherwise
  reserved for CI installer paths).
- **ADIOS2** is built from source (v2.11.0, C++20-compatible, x86_64 and
  ARM64) with MPI, HDF5 and ZeroMQ; SST is disabled because of an ARM64 Linux
  problem in the DILL library.

## 7. Unit tests

### Framework: `simple_test.h`, never Catch2, for anything that starts the runtime

Catch2's runner conflicts with runtime initialization (identical code
segfaulted under Catch2 and passed under `simple_test.h`). Use
`context-runtime/test/simple_test.h`:

- Add `${CLIO_RUN_ROOT}/test` to the test's include directories and
  `#include "simple_test.h"` (e.g.
  `context-runtime/modules/MOD_NAME/test/CMakeLists.txt` and
  `test_collectives.cc:35`), or use a relative include such as
  `#include "../../../context-runtime/test/simple_test.h"`.
- Macros: `TEST_CASE(name, tags)`, `SECTION`, `REQUIRE`, `REQUIRE_FALSE`,
  `REQUIRE_NOTHROW`, `INFO`, `FAIL` (`simple_test.h:210-431`). There is **no
  `CHECK`**; use `REQUIRE`.
- End the file with `SIMPLE_TEST_MAIN()` (`simple_test.h:495`). It sets
  `CLIO_UNIT_TEST=1` (so `CLIO_RUN_INLINE` keeps the normal task path, see
  `scheduling-and-performance.md`), runs the tests whose name matches
  `argv[1]` (a filter), then calls `SimpleTest::g_test_finalize` if set.

### Runtime initialization

The entry point is the inline function `clio::run::CLIO_INIT(mode,
default_with_runtime = false, is_restart = false)`
(`context-runtime/include/clio_runtime/clio_runtime.h:120-123`), wrapping
`ClioInitImpl`. Do not call private or legacy initializers (`initializeBoth`, `initializeRuntime`,
`initializeClient`).

- Mode: `clio::run::RuntimeMode::kClient` for unit tests.
- `default_with_runtime = true` for unit tests: the process tries to become the
  runtime and, if the port is already bound, attaches as a client
  (`clio_runtime.h:85-100`).
- Env override: `CLIO_WITH_RUNTIME=1` (ensure a runtime) / `=0` (client only);
  unset means use `default_with_runtime` (`clio_runtime.h:78-84`). It is not
  `CLIO_IPC_MODE` -- that variable selects the client transport (see
  `architecture.md`). (Older notes call the transport variable `CLIO_X`; no
  code reads that name.)
- Register the finalizer so ZMQ shuts down cleanly:
  `SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;`.

Canonical fixture (from
`context-runtime/modules/MOD_NAME/test/test_collectives.cc:66-92`):

```cpp
class Fixture {
 public:
  Fixture() : pool_id_(MakePoolId()) {
    if (!g_initialized) {
      bool ok = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
      REQUIRE(ok);
      g_initialized = true;
      SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
      std::this_thread::sleep_for(100ms);
      REQUIRE(CLIO_IPC != nullptr);
      REQUIRE(CLIO_POOL_MANAGER != nullptr);
    }
    client_ = clio::run::MOD_NAME::Client(pool_id_);
    auto create = client_.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                      "my_test_pool", pool_id_);
    create.Wait();
    REQUIRE(create->return_code_ == 0);       // ALWAYS check Create
    client_.pool_id_ = create->new_pool_id_;
  }
  ...
};
```

### Always check that Create succeeded

Many test failures are a failed Create followed by tests running against an
uninitialized pool. After every Create (admin, bdev, CTE, your module):
`REQUIRE(create->GetReturnCode() == 0)` (or `return_code_`). 0 is success.

### Registering with ctest

Register each binary once with `add_test(NAME ... COMMAND <target>
WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin)` and give it a `TIMEOUT`
(`context-runtime/modules/MOD_NAME/test/CMakeLists.txt:215-260`). Running a
whole binary in one runtime init/finalize cycle is deliberate: many filtered
entries over the same binary each pay an init/finalize cycle and exposed a
shutdown race. Link the module's `_client` and `_runtime` targets plus
`clio_admin_client`/`clio_admin_runtime`.

Run tests from the build directory: `ctest --test-dir <dir> -R <regex>
--output-on-failure`, or run the binary directly with a name filter as
`argv[1]`.

### Test what the policy promises

Under a write-back cache, a read before a flush is not expected to be coherent:
verify after `fsync`. Under write-through, verify immediately (see
`cte-data-path.md`).

## 8. Multi-node validation (Aurora)

After cache, routing or consistency changes, run the clio-fs suites on Aurora
with `CI/aurora/run.sh` (`CI/aurora/README.md`):

```bash
CI/aurora/run.sh --build build/aurora --suite smoke --nodes 2 --queue debug --time 00:30:00 --wait   # harness check
CI/aurora/run.sh --build build/aurora --suite cache --nodes 32 --wait     # policy-cache tests, a few minutes
CI/aurora/run.sh --build build/aurora --suite posix --nodes 32 --wait     # posix, dist, dirs, apps, cache
```

Suites: `smoke`, `cache`, `posix`, `faults`, `stress`, `safe`, `perf`,
`io500`; `all` = every one but `smoke` and `cache`. One PBS job per suite
(`debug-scaling` allows one running and one queued job per user). Results go
to `~/clio_aurora_runs/<date>-<sha>-<nodes>n/report.md`; `--baseline <dir>`
enables regression detection, `--resume` continues an interrupted run. The
tests themselves live in
`context-transfer-engine/test/integration/clio_fs_suite`. For IO500 comparisons
use `--repeat 3` at least: one run is not a result.

## 9. Docker deployment (for multi-node testing on one host)

`docker/quickstart/docker-compose.yml` brings up a cluster with the server
config at `/etc/iowarp/clio.yaml` (`CLIO_SERVER_CONF`). Other compose files
live under `docker/*/docker-compose.yml`. Rules:

- Container `shm_size` must be >= the sum of all shared-memory segments the
  runtime creates (main, client data (default 512 MB), queue and metadata
  segments; see `architecture.md` section 5), e.g. `shm_size: 2gb`.
- Provide a hostfile (one node IP per line) mounted read-only, e.g.
  `./hostfile:/etc/iowarp/hostfile:ro`, and name it in the server YAML under
  `networking.hostfile`. The hostfile is read only from YAML; there is no
  `CLIO_HOSTFILE` environment variable.
- Environment overrides that exist in code: `CLIO_SERVER_CONF` (config path),
  `CLIO_PORT` (default 9413), `CLIO_SERVER_ADDR` (default 127.0.0.1),
  `CLIO_NUM_THREADS`, `CLIO_MAIN_SEGMENT_SIZE`, `CLIO_IPC_MODE`
  (SHM/TCP/IPC), `CTP_LOG_LEVEL`. The full list is in `architecture.md`
  section 6. Variables older notes list -- `CLIO_SCHED_WORKERS`,
  `CLIO_CLIENT_DATA_SEGMENT_SIZE`, `CLIO_RUNTIME_DATA_SEGMENT_SIZE`,
  `CLIO_SHM_SIZE`, `CLIO_HOSTFILE`, `CLIO_LOG_LEVEL`, `CLIO_X` -- are not read
  anywhere.

Typical use: `cd docker/quickstart && docker compose up -d`, `docker compose
ps`, `docker compose logs -f <service>`, `docker compose down`.

## 10. Deployment scripts (Jarvis packages)

When a deployment package configures devices: `mkdir` every parent directory
named in the devices configuration during the configure phase, and remove each
of those directories in the clean phase.

## 11. Python wheels

Wheels are built by scikit-build-core from `pyproject.toml`
(`build-backend = "scikit_build_core.build"`, Release build, CMake defines in
`[tool.scikit-build.cmake.define]`): `python -m build --wheel`. The intent is a
self-contained, relocatable wheel (IOWarp libraries, the command-line tools
such as `clio-run`/`clio_cte`, headers and CMake config files, `$ORIGIN`
RPATHs, no `LD_LIBRARY_PATH`). `.github/workflows/build-pip.yml` is the
reference recipe. Older notes mention an `IOWARP_BUNDLE_BINARIES=ON` switch,
`BUILD_WHEEL.md` and `build_wheel.sh`; none of these exist in the tree any
more -- follow `pyproject.toml` and the workflow instead.

## 12. Cleaning a polluted source tree

If CMake artifacts landed in the source tree, remove exactly those (never a
real build directory you still need):

```bash
find . -path ./build -prune -o \( -name CMakeCache.txt -o -name cmake_install.cmake \
     -o -name CTestTestfile.cmake -o -name Makefile -o -name '*.make' \
     -o -name DartConfiguration.tcl -o -name CPackConfig.cmake \
     -o -name CPackSourceConfig.cmake -o -name build.ninja -o -name rules.ninja \
     -o -name '.ninja_*' \) -print            # inspect first, then add -delete
find . -path ./build -prune -o -type d \( -name CMakeFiles -o -name Testing \) -print
```

Inspect the list before deleting: `Makefile`s and `*.cmake` files that are
checked in (`git ls-files`) must stay.
