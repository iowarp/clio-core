# gsbench_driver — C++ self-supervising Gray-Scott benchmark driver (spec)

**Audience:** the implementing agent. **Goal:** replace the two duplicated shell orchestrators
(`../run_threeway_bench.sh` and `../scaling_campaign/run_campaign.sh`) with ONE typed C++ driver that
fork/execs the existing benchmark binary per arm, with a single arm/study registry as the source of
truth. The compute (the Gray-Scott sim + all arm "sinks") stays exactly where it is —
`../gray_scott_threeway_bench.cu` — and is NOT touched. We only replace the shell orchestration.

---

## 0. Non-negotiable constraints

1. **Per-arm process isolation is mandatory.** Each arm MUST run in its own OS process: the CLIO
   runtime + CUDA context + CTE server keep per-process global state, and running two arms in one
   process corrupts the second. So the driver is a *multi-process supervisor*: it `fork()`s +
   `execvp()`s a child per (arm, config, rep). This is exactly what the shell scripts do; do not try
   to run arms in-process.
2. **GPU-run safety (READ THIS before running anything).** The bench arms drive the real GPU, and the
   user's GPU also runs their desktop compositor. Large configs / full sweeps can hang their desktop.
   You may BUILD freely (host CPU work). For RUNNING: only a MINIMAL smoke test at tiny size is
   allowed (see §10) — e.g. 2 cheap arms (`sync`,`async`) at `GSBENCH_N=128 GSBENCH_SNAPS=2
   GSBENCH_STEPS_PER=2`, 1 rep. DO NOT run the full campaign, large N, or long step counts yourself —
   those are for the user to run headless.
3. **Do not touch** `gray_scott_threeway_bench.cu`, the reuse work (`snapshot_reuse_test.cu`,
   `gpu_cte_dataset.h`, `gpu_dataset_handle.h`), or `context-runtime`. Only add the new driver + wire
   its build.

---

## 1. Confirmed design decisions

1. **Standalone binary** `gsbench_run` (a new CMake target), NOT a Catch2 test and NOT a mode inside
   `kvhdf5_e2e_tests`. The supervisor is not a test.
2. **Child = exec the existing test binary with the arm's Catch2 selector.** `fork()` →
   `setenv(GSBENCH_*)` for the arm+config → `execvp(<path to kvhdf5_e2e_tests>, {selector})`. This
   reuses ALL proven arm/sim code unchanged and gives perfect isolation. (A later refactor could pull
   the arms into a callable library; NOT now — do the easy exec-based version first and verify.)
3. **State reset between children = native `/dev/shm` unlink + a targeted `pkill`.** Port whatever
   `hard_reset`/`reset_state` in the shell scripts does (kill leftover runtime/server procs, remove
   `/dev/shm/clio*` and any other shm the runtime leaves). Calling `pkill`/`system()` for the
   process-kill part is acceptable; do the shm unlink natively (glob + `unlink`) or via `system("rm
   -f /dev/shm/clio* ...")` — match the scripts.
4. **Scope = everything now:** the single-config multi-arm run (replacing `run_threeway_bench.sh`,
   including its median/spread/vs-baseline table) AND the full campaign studies (C/K/B/W/P) with reps
   and resume (replacing `run_campaign.sh`).

**Output:** results go to **stdout** by default (a table for a single-config run; tagged result lines
for a sweep). `--out <file>` optionally ALSO persists results to a file for resume. Resume = on start,
read `--out` file, skip any (study,cfg,rep,arm) already present (match the shell's resume semantics,
which key on a completed result line existing).

---

## 2. Authoritative sources to PORT (read these fully before writing the registry)

The arm→selector→env map and the study sweeps ALREADY EXIST in the shell scripts. Port them; they are
the source of truth. Read, in full:

- **`../run_threeway_bench.sh`** — the single-config runner. Note especially: `tag_of()` (arm →
  Catch2 selector), the per-arm env `case` block, `hard_reset`/state-reset, the async-VOL / HDF5 VOL
  env setup (`LD_PRELOAD`, `env -u HDF5_VOL_CONNECTOR`, etc.), and the inline **Python block** that
  parses `GSBENCH_RESULT` lines and prints the median/spread/vs-raw table (port this table logic to
  C++). Also how it locates the binary (`GSBENCH_BUILD_DIR`).
- **`../scaling_campaign/run_campaign.sh`** — the campaign. Note: `tag_of()` and the arm `case` block
  (must MATCH the runner's), the STUDY definitions (C=compute/steps_per, K=chunks, B=I/O size/N,
  W=weak scaling, P=pooled M-sweep) with their value lists and per-study arm sets
  (`ARMS_C`/`ARMS_SCALE`), the tiers (`ram`/`file` via `set_tier`), reps (3, or 5 on HEADLINE
  configs), the resume logic (keys on `GSBENCH_RESULT` present in a per-run log), and how it writes
  `raw/master_raw.log` (the `SWEEP=… CFG=… REP=… ARM=… <result line>` tagging).
- **`../scaling_campaign/README.md`** — the canonical description of the studies and what each sweeps.
- **`../gray_scott_threeway_bench.cu`** — READ ONLY these parts: the arm `TEST_CASE`s (search
  `TEST_CASE(` near the bottom, ~lines 2300+) to get the EXACT Catch2 selector strings; the `Cfg`
  struct + the `EnvU`/`EnvS` env knobs (search `struct Cfg`, `GSBENCH_`); and `PrintResult` (search
  `GSBENCH_RESULT`) to get the EXACT result-line format you must parse. Do NOT modify this file.

**Result line format** (confirm against `PrintResult`): one line to **stderr** per child, e.g.
`GSBENCH_RESULT arm=<s> N=<u> chunks=<u> blocks=<u> snaps=<u> steps=<u> bdev=<s> pinned=<u>
durable=<u> MB=<f> ms=<f> MBps=<f> checksum=<llu>`. The driver captures the child's combined
stdout+stderr and greps this line (checksum is used to verify all arms computed identical bytes).

---

## 3. File layout (split for maintainability)

Put everything under `context-transfer-engine/adapter/kvhdf5/test/e2e/gsbench_driver/`:

```
gsbench_driver/
  SPEC.md              # this file
  CMakeLists.txt       # builds the gsbench_run target; add_subdirectory'd from ../CMakeLists.txt
  main.cpp             # CLI parse -> dispatch to single-config run or campaign
  arms.h / arms.cpp    # the arm REGISTRY: {name, catch2_selector, env_overrides[]}; the single
                       #   source of truth ported from the two shell scripts' tag_of()+case blocks
  studies.h / studies.cpp  # the campaign study definitions (C/K/B/W/P): value lists, per-study arm
                           #   sets, tiers, reps, HEADLINE configs
  runner.h / runner.cpp    # the supervisor core: fork/setenv/execvp one child, capture output,
                           #   extract the GSBENCH_RESULT line, state-reset between children
  results.h / results.cpp  # parse a GSBENCH_RESULT line into a struct; aggregate (median/spread);
                           #   render the stdout table; write/read the resume file
  cli.h / cli.cpp      # minimal hand-rolled arg parser (no external CLI lib; a handful of flags)
```

Split arms into their own file(s) as above. If the registry gets large you may further split (e.g.
`arms_clio.cpp`, `arms_hdf5.cpp`), but one `arms.cpp` is fine to start. Keep each file focused.

**The driver is pure host C++ (no CUDA).** It only fork/execs the CUDA binary. So it compiles without
nvcc — but it lives under the CUDA-gated e2e dir, so gate the `add_subdirectory(gsbench_driver)` under
the same `CLIO_CORE_ENABLE_CUDA` guard the e2e CMakeLists already uses (the driver is useless without
the CUDA bench binary it drives).

---

## 4. CLI shape (minimal, hand-rolled)

Design a small, clear CLI. Suggested (adjust as sensible):

```
gsbench_run [--bin <path>] [--out <file>] [--reps N] [--tier ram|file] [common GSBENCH knobs...]
  # single-config mode (default): run the given arms once at one config, print a table
  --arms raw,sync,async,hdf5,...        # default: a sensible core set (raw,sync,async,hostclio,hdf5)
  --n 512 --chunks 4 --snaps 4 --steps 8 --bdev ram ...   # config knobs -> GSBENCH_* env

gsbench_run --campaign [--studies C,K,B,W,P] [--tiers ram,file] [--out master_raw.log] [--resume]
  # campaign mode: run the full study sweeps with reps; emit tagged result lines; resume from --out
```

- `--bin` defaults to locating `kvhdf5_e2e_tests` next to the driver (same `bin/` dir) or via an env
  var; make it robust (the shell used `GSBENCH_BUILD_DIR`).
- Unknown/extra `GSBENCH_*` may be passed through the environment untouched (don't clobber the
  caller's env except for the knobs an arm/config sets).
- Print `--help`.

---

## 5. Supervisor mechanics (runner.cpp)

For each unit of work `(arm, config, rep)`:
1. State reset (kill leftover procs + shm unlink) — BEFORE each child, matching the scripts.
2. `fork()`. In the child: apply the arm's env overrides + the config's `GSBENCH_*` env
   (`setenv(...,1)`), redirect so the child's stdout+stderr are captured (a pipe), then
   `execvp(bin, {bin, selector, nullptr})` where `selector` is the arm's Catch2 selector. (Catch2
   runs the one hidden `[.]` test case that selector names.)
3. In the parent: read the child's captured output, `waitpid`, extract the `GSBENCH_RESULT` line
   (warn if missing / child crashed / nonzero exit), record it.
4. Enforce a per-child timeout (a child can hang); kill + record a failure on timeout. Make the
   timeout generous but present.

Preserve the scripts' special env handling (HDF5 async VOL `LD_PRELOAD`/plugin path, `env -u
HDF5_VOL_CONNECTOR` isolation for non-async arms, `--shm-size` is a container thing not ours) — port
whatever they set per arm.

Checksum check: across the arms of one config, assert (warn loudly) if checksums differ — that means
the arms did NOT compute identical bytes, which invalidates the comparison (the scripts do this).

---

## 6. Output (results.cpp)

- **Single-config table:** median + spread (min/max or stddev) of `ms`/`MBps` per arm over reps, and
  a "vs baseline" ratio (the scripts compare vs `raw`). Port the columns from the runner's Python
  block. Print to stdout. Also print the shared checksum + a PASS/FAIL on checksum agreement.
- **Campaign:** for each (study,cfg,rep,arm) print a tagged line to stdout (and to `--out` if given),
  in a stable, machine-parseable form (keep the scripts' `SWEEP=… CFG=… REP=… ARM=… <result line>`
  convention so existing offline CSV/plot tooling still works). The user builds CSV/plots offline
  from this, exactly as today.
- **Resume:** if `--out` exists and `--resume`, load it, and skip any work unit whose result line is
  already present.

---

## 7. Build wiring

- Add `add_subdirectory(gsbench_driver)` to `../CMakeLists.txt` (guarded by `CLIO_CORE_ENABLE_CUDA`,
  near where the e2e target is defined — the driver depends on `kvhdf5_e2e_tests` existing to exec).
- `gsbench_driver/CMakeLists.txt`: define `add_executable(gsbench_run ...)` from the `.cpp` files;
  it's plain C++17/20 host code (link whatever std libs you need; no CUDA, no CLIO libs required — it
  only fork/execs a binary and parses text). Add `add_dependencies(gsbench_run kvhdf5_e2e_tests)` so
  the child binary is built first.
- Optionally add a convenience custom target (like the existing `threeway_bench` target) that runs
  `gsbench_run` — but do NOT wire it into `ctest` (same reason the scripts aren't: it pkills/wipes
  shm).

---

## 8. Environment (build & run)

- Everything builds INSIDE the running devcontainer `iowarp-kvhdf5-gpu` (repo mounted at `/workspace`;
  the host has no nvcc, but the DRIVER is host C++ so it also builds there fine). Build dir:
  `/workspace/build-bench`.
- Build: `docker exec iowarp-kvhdf5-gpu bash -lc 'cmake -S /workspace -B /workspace/build-bench >/dev/null; cmake --build /workspace/build-bench --target gsbench_run -j 2>&1 | tail -40'`
  (a fresh source dir needs a reconfigure so CMake picks up the new subdirectory).
- The built driver: `/workspace/build-bench/bin/gsbench_run` (or wherever the target lands — confirm).
  The child binary it execs: `/workspace/build-bench/bin/kvhdf5_e2e_tests`.

---

## 9. Reference: the arms (from an earlier survey — CONFIRM against the scripts)

~11 arm labels over 8 Catch2 cases. Approximate mapping (VERIFY exact selectors + env in the scripts):

| label         | Catch2 selector (verify) | env overrides (verify)            |
|---------------|--------------------------|-----------------------------------|
| raw_inline    | gsbench_raw              | GSBENCH_RAW_INLINE=1               |
| raw_threaded  | gsbench_raw              | GSBENCH_RAW_INLINE=0               |
| sync          | gsbench_sync             | —                                 |
| async         | gsbench_async            | (GSBENCH_DATA_PINNED=0)            |
| async_pinned  | gsbench_async            | GSBENCH_DATA_PINNED=1              |
| pooled        | gsbench_pooled           | GSBENCH_POOL=<M>                   |
| hostclio      | gsbench_hostclio         | —                                 |
| hdf5_inline   | gsbench_hdf5             | GSBENCH_RAW_INLINE=1               |
| hdf5_threaded | gsbench_hdf5             | GSBENCH_RAW_INLINE=0               |
| hdf5_naive    | gsbench_hdf5_naive       | —                                 |
| hdf5_async    | gsbench_hdf5_async       | (async VOL LD_PRELOAD/plugin env) |

The Catch2 selectors above are the TAG forms the scripts pass; confirm the exact string each script's
`tag_of()` emits and use THAT verbatim.

---

## 10. Verification (do this; respect §0.2 GPU safety)

1. **Build** `gsbench_run` cleanly (fix compile errors).
2. **`--help`** prints and CLI parses (host-only, safe).
3. **Dry-run listing** (add a `--dry-run` that prints each planned child's selector+env WITHOUT
   forking a GPU child) — verify the registry + a small study expand to the right commands. Safe.
4. **ONE tiny smoke run** (only if it stays tiny): `gsbench_run --arms sync,async --n 128 --snaps 2
   --steps 2 --reps 1 --bdev ram` — verifies fork/exec/capture/parse/table + state-reset end to end.
   This is comparable to the existing e2e tests' GPU load. Do NOT go bigger, and do NOT run
   `--campaign` (it's a long GPU sweep — the user runs that headless).
5. Report: what was built, the file layout, the `--dry-run` expansion for a couple of arms + one
   study, and the tiny smoke's stdout table (or why you couldn't run it). Note anything in the scripts
   you couldn't cleanly port.

Keep the existing shell scripts in place (do not delete them) until the user confirms the C++ driver
reaches parity.
