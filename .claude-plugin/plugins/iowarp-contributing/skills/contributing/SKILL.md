---
description: "IOWarp/Clio ecosystem contributing guidelines — git workflow, code standards, project architecture, and development practices. Use when making any code contribution."
---

# IOWarp Clio Ecosystem — Contributing Guidelines

You are helping a developer contribute to the IOWarp ecosystem. Follow these standards strictly for all code contributions. These guidelines apply to **clio-core** and all related repositories in the IOWarp organization.

---

## Git Workflow

### Branch Naming

All feature branches MUST reference a GitHub issue number:
```
<issue-number>-<short-kebab-description>
```

Examples:
- `318-per-container-qtable-load-assessment`
- `249-build-dashboard-context-runtime`
- `42-fix-zmq-reconnect-timeout`

### Commit Messages

Write commit messages that explain **why**, not what. The diff shows what changed.

**Format:**
```
<imperative verb> <what changed>

<optional body explaining why, constraints, tradeoffs>
```

**Good examples:**
```
Add per-container load prediction model with online learning

The existing global load model couldn't capture per-container variance
in task execution times. Each container now maintains its own
exponential moving average for both CPU and wall-clock time.
```

```
Reset CPU timer and predicted load for periodic task rescheduling
```

```
Fix ZMQ reconnect race condition on node failure

The PUSH socket was reconnecting before the PULL socket had
re-bound, causing silent message drops for ~200ms.
```

**Bad examples:**
- `fixed stuff`
- `update code`
- `WIP`
- `changes to runtime`
- `Merge branch 'main' into feature` (squash merge instead)

### Pull Request Standards

**Title:** Under 70 characters, imperative voice, references the feature/fix.

**Body format:**
```markdown
## Summary
- <1-3 bullet points describing the change>

## Motivation
<Why this change is needed — link to issue>

## Test plan
- [ ] Unit tests pass: `ctest -R <relevant_tests>`
- [ ] No new compiler warnings
- [ ] Code formatted: `clang-format` check passes
- [ ] Headers present on all new files

## Breaking changes
<None, or describe what breaks and migration path>
```

**PR rules:**
- One logical change per PR. Don't bundle unrelated fixes.
- Squash-merge to main. Keep main's history clean.
- All CI checks must pass before merge.
- Request review from at least one team member.

### Merge Strategy

- **Feature → main**: Squash merge (clean linear history)
- **Hotfix → main**: Regular merge (preserves fix attribution)
- **Never force-push to main.**
- Rebase feature branches on main before creating the PR: `git rebase origin/main`

---

## Code Standards

### C++ Style

- **Style guide:** Google C++ Style Guide
- **Formatter:** clang-format (config in `.clang-format`, 80-column limit, no tabs)
- **Linter:** cpplint (config in `CPPLINT.cfg`)
- **Max function length:** 100 lines. Extract helpers logically.
- **Docstrings:** Every function gets a Doxygen-compatible docstring documenting parameters, return value, and purpose.

### File Headers

Every C/C++ source file (.h, .hpp, .cc, .cpp) MUST have the BSD 3-Clause license header. Use the template from `.header_template`. Run `CI/update_headers.py` to auto-apply.

### Critical Rules

1. **NEVER hardcode absolute paths in CMakeLists.txt files.**
2. **NEVER build outside `/workspace/build`.** No in-source builds. No `/tmp/build_*`.
3. **NEVER use raw GPU macros** (`__CUDACC__`, `__HIPCC__`, etc.) — use `HSHM_IS_GPU`, `HSHM_IS_HOST`, `HSHM_IS_GPU_COMPILER`, etc. from `context-transport-primitives/include/hermes_shm/constants/macros.h`.
4. **NEVER write mock/stub code** unless explicitly requested. All implementations must be real and working.
5. **NEVER use Catch2 with Chimaera runtime.** Use `simple_test.h` for unit tests.
6. **NEVER use null pool queries.** Always use `local` if unsure.
7. **Always store singleton pointers** before dereferencing: `auto *x = Singleton<T>::GetInstance(); x->var_;`
8. **Name all QueueIds and priorities semantically.** Never use raw integers.
9. **All timing output MUST be in milliseconds (ms).**

### Naming Conventions

| Entity | Convention | Example |
|--------|-----------|---------|
| Classes | PascalCase | `PoolManager`, `TaskLane` |
| Functions | PascalCase | `CreatePool()`, `GetNodeId()` |
| Variables | snake_case with trailing `_` for members | `pool_id_`, `container_id_` |
| Constants | kPascalCase | `kAdminPoolId`, `kCtePoolName` |
| Macros | UPPER_SNAKE | `HSHM_IS_GPU`, `CHI_IPC` |
| CMake targets | namespace::component | `chimaera::admin_client` |
| ChiMod names | lowercase underscore | `wrp_cte_core`, `chimaera_admin` |

---

## Project Architecture — Codemap

### Repository Structure

```
clio-core/
├── context-transport-primitives/   # Shared memory data structures, IPC, GPU support
│   ├── include/hermes_shm/         # Public headers (hshm:: namespace)
│   ├── src/                        # Implementation
│   └── docs/MODULE_DEVELOPMENT_GUIDE.md  # ChiMod dev guide
│
├── context-runtime/                # Chimaera modular runtime (chi:: namespace)
│   ├── include/chimaera/           # Runtime headers
│   ├── src/scheduler/              # Worker, scheduler implementation
│   ├── modules/                    # Built-in ChiMods (admin, bdev)
│   ├── config/                     # Default YAML configs
│   └── test/                       # Unit + integration tests
│
├── context-transfer-engine/        # I/O buffering engine (wrp_cte:: namespace)
│   ├── core/                       # CTE ChiMod
│   ├── adapter/                    # I/O pathway adapters
│   ├── compressor/                 # Compression ChiMod
│   └── config/                     # CTE configurations
│
├── context-assimilation-engine/    # Data ingestion (wrp_cae:: namespace)
│   ├── core/                       # CAE ChiMod
│   └── data/                       # OMNI format definitions
│
├── context-exploration-engine/     # Data exploration tools
│   ├── api/                        # Python CEE API
│   └── iowarp-cei-mcp/            # MCP server for HDF5
│
├── docker/                         # Dockerfiles and deployment configs
├── .devcontainer/                  # VS Code devcontainer configs
├── CI/                             # CI scripts, sanitizers, header updater
├── cmake/                          # CMake helper modules
├── installers/                     # conda, pip, spack, vcpkg recipes
├── external/                       # Git submodules (hindsight, graphiti, jarvis-cd)
├── AGENTS.md                       # Development guide and coding standards
└── CMakePresets.json               # Build presets
```

### Component Dependency Flow

```
Applications
    │
    ├── context-exploration-engine (CEE)
    ├── context-assimilation-engine (CAE)
    └── context-transfer-engine (CTE)
            │
            └── context-runtime (Chimaera)
                    │
                    └── context-transport-primitives (HSHM)
```

Every component depends on the ones below it. Never create upward dependencies.

### Key Abstractions

**ChiMod (Chimaera Module):** The fundamental extensibility unit. Each engine (CTE, CAE) is implemented as a ChiMod with a client library and a runtime library.

**Task:** The unit of work in Chimaera. Tasks are coroutine-based, support cooperative yielding, and are scheduled across worker threads.

**Pool:** A logical grouping of resources managed by a ChiMod. Each pool has a unique ID and name.

**PoolQuery:** Determines task routing. Types: `Local`, `Dynamic`, `Broadcast`, `DirectHash`.

**Block Device (bdev):** Storage backend abstraction. Types: `kRam` (memory), `kFile` (file-backed).

### The Clio Ecosystem

| Repository | Purpose |
|-----------|---------|
| **clio-core** | Core platform — runtime, transport, engines |
| **clio-kit** | Developer toolkit and examples |
| **clio-agent** | Science agent for AI-driven data management |
| **docs** | Documentation site (Docusaurus) |
| **iowarp** | Platform installation scripts |
| **iowarp-agents** | Collection of scientific AI agents |
| **memorybench** | Memory/RAG benchmark suite |

When making changes, consider cross-repo impact:
- **clio-core API changes** affect clio-kit, clio-agent, and downstream users
- **Config format changes** must update `docs/docs/deployment/configuration.md`
- **ChiMod interface changes** must update `MODULE_DEVELOPMENT_GUIDE.md`

---

## Development Workflow

### The Full Issue-to-PR Cycle

This is the standard workflow for every bug fix and feature. Follow it step by step.

#### Step 1: Document the Problem — Create a GitHub Issue

Before writing any code, create an issue that describes the problem or feature:

```bash
gh issue create \
  --title "Fix: <concise description of the bug or feature>" \
  --body "## Description
<What is broken or what needs to be added>

## Steps to Reproduce
1. <step>
2. <step>

## Expected vs Actual Behavior
- **Expected:** <what should happen>
- **Actual:** <what happens instead>

## Acceptance Criteria
- [ ] <testable criterion>
- [ ] <testable criterion>"
```

For bugs found during exploration, include the error message, stack trace, and the file/line where it occurs. The issue becomes the single source of truth for the work.

#### Step 2: Create a Branch from the Issue

```bash
git checkout main && git pull origin main
git checkout -b <issue-number>-<short-description>
```

The branch name MUST start with the issue number. Example: `325-validate-marketplace`.

#### Step 3: Write a Failing Test First

Before implementing any fix, write a test that reproduces the problem:

```bash
# Write the test
# Build
cmake --build build -j$(nproc)
# Install (RPATHs require installation)
sudo cmake --install build
# Run and confirm it FAILS
cd build && ctest -R <your_test> -VV
```

This is critical — a test that fails proves you understand the bug. If you can't write a failing test, you don't understand the problem well enough yet.

For Chimaera runtime tests, use `simple_test.h` (NOT Catch2):
```cpp
#include "../../../context-runtime/test/simple_test.h"

TEST_CASE("Reproduce issue #<N>", "[regression]") {
  bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
  REQUIRE(success);
  // ... code that triggers the bug ...
  REQUIRE(expected_condition);
}

SIMPLE_TEST_MAIN()
```

#### Step 4: Implement the Fix

Now implement the fix. Build and verify:

```bash
# Build with your changes
cmake --build build -j$(nproc)
sudo cmake --install build

# Verify YOUR test now passes
cd build && ctest -R <your_test> -VV

# Run ALL tests to check for regressions
ctest -VV
```

Both must succeed: your new test passes AND no existing tests break.

#### Step 5: Push and Create PR

```bash
# Stage specific files (never `git add -A`)
git add <changed files>
git commit -m "Fix <description>

Closes #<issue-number>"

# Push and create PR
git push -u origin <issue-number>-<description>
gh pr create \
  --title "Fix <concise description>" \
  --body "## Summary
- <what changed and why>

## Motivation
Fixes #<issue-number>

## Test plan
- [ ] New regression test passes: \`ctest -R <test_name>\`
- [ ] All existing tests pass: \`ctest -VV\`
- [ ] No new compiler warnings
- [ ] Code formatted with clang-format
- [ ] BSD 3-Clause headers on all new files

## Breaking changes
None"
```

### Before Starting Work

1. Ensure you're on a feature branch from latest main:
   ```bash
   git checkout main && git pull origin main
   git checkout -b <issue-number>-<description>
   ```
2. Verify the build is clean:
   ```bash
   cmake --preset=debug && cmake --build build -j$(nproc)
   cd build && ctest -VV
   ```

### While Working

- **Build incrementally:** `cmake --build build -j$(nproc)`
- **Always install before testing:** `sudo cmake --install build` (RPATHs, not LD_LIBRARY_PATH)
- **Run relevant tests after every change:** `ctest -R <component>`
- **Format before committing:** Ensure clang-format compliance
- **Update docs** if you change configs, APIs, or ChiMod interfaces

### Agent Usage

Use the right agent for the job:
- `incremental-logic-builder` — for code changes
- `code-compilation-reviewer` — after making CMake or code changes
- `unit-test-generator` — for new test coverage
- `cpp-debug-analyzer` — for runtime crashes or memory issues
- `dockerfile-ci-expert` — for Docker or CI changes

### Testing Requirements

- All new code must have unit tests
- Use `simple_test.h` (NOT Catch2) for Chimaera runtime tests
- Initialize with `chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true)`
- Always `ASSERT_EQ(client.GetReturnCode(), 0)` after Create operations
- Run sanitizers before submitting: `bash CI/run_sanitizers.sh`

### CI Pipeline

Every PR triggers:
1. **build-and-test** — Full build + CTest on ubuntu-24.04 (x86_64 + ARM)
2. **cpplint** — Style check
3. **sanitizers** — ASan, MSan
4. **install-conda** — Conda path validation

All must pass before merge.

---

## Quick Reference

```bash
# 1. Create issue
gh issue create --title "Fix: ZMQ reconnect timeout" --body "..."

# 2. Branch from issue
git checkout main && git pull origin main
git checkout -b 42-fix-zmq-timeout

# 3. Write failing test, verify it fails
cmake --build build -j$(nproc) && sudo cmake --install build
cd build && ctest -R my_new_test -VV   # Should FAIL

# 4. Implement fix, verify all pass
cmake --build build -j$(nproc) && sudo cmake --install build
cd build && ctest -VV                  # ALL should PASS

# 5. Commit and PR
git add <specific files>
git commit -m "Fix ZMQ reconnect timeout

Closes #42"
git push -u origin 42-fix-zmq-timeout
gh pr create --title "Fix ZMQ reconnect timeout" --body "Fixes #42 ..."

# Other useful commands
clang-format -style=file -i <file>     # Format check
python3 CI/update_headers.py           # Header update
cpplint --filter=-legal/copyright <f>  # Lint
bash CI/run_sanitizers.sh --asan       # Sanitizers
```
