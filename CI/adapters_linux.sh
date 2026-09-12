#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Linux adapter matrix: build all five CTE I/O adapters and run their tests.
#
# Runs INSIDE the iowarp/deps-cpu image; ci-adapters.yml's linux-adapters job is
# a thin `docker run ... bash /workspace/CI/adapters_linux.sh` wrapper around it.
#
# WHY THIS IS A FILE AND NOT AN INLINE `bash -c '...'`, which is what it was:
# a single apostrophe inside one of the comments -- "the VFD C suite's tool
# matrix" -- closed the outer shell's quoting. Everything after it left the
# container script and was run by the RUNNER's shell instead, on a host with no
# HDF5 anywhere. The visible symptom was `required HDF5 tool missing from
# deps-cpu: h5dump` on an image that demonstrably ships h5dump, with the export
# that would have found it sitting right there in the echoed script. The whole
# test phase -- ctest, both compat suites, the mount smoke, xfstests -- silently
# never ran; the job failed on the first line after the build and looked like an
# image regression for three CI rounds. Quoting is not the place to be clever:
# in a file, an apostrophe is just an apostrophe.
#
# Usage:  CI/adapters_linux.sh [build-dir]      (default: build-adapters)
# ---------------------------------------------------------------------------
set -e

BUILD_DIR="${1:-build-adapters}"

# /usr/local/bin is where deps-cpu puts HDF5 (2.1.1) and its CLI tools. Both lib
# and bin go on the path so the adapters link the right libhdf5 AND the compat
# suites can actually run their native-tool oracle rather than skipping it.
export PATH=/usr/local/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH
git config --global --add safe.directory /workspace

# FUSE backend + ELF interception headers (libelf/HDF5 already in image)
apt-get update -qq
apt-get install -y -qq libfuse3-dev fuse3 pkg-config libelf-dev

# --------------------------------------------------------------- HDF5 tool gate
# The HDF5 CLI tools are the differential oracle for both compat suites and for
# the VFD C suite's tool matrix. deps-cpu ships them, so a missing one is an
# image regression: fail rather than let the suites take their (locally correct)
# skip path and report green without the evidence. h5stat/h5clear are required
# too -- they are what the §1.1(b)/(c) gates are measured with.
#
# Self-diagnosing on failure. "tool absent", "PATH is not what we think" and
# "this is not even the shell you think it is" all produce the identical
# one-line symptom -- and the last of those was the actual cause, undetected
# through two speculative fixes because nothing in the output could separate
# them. `id` and `$PATH` alone would have named it in one run. Print what
# distinguishes them before exiting.
check_hdf5_tools() {
  local when="$1" t missing=""
  for t in h5dump h5ls h5diff h5stat h5clear h5repack h5cc; do
    command -v "$t" >/dev/null 2>&1 || missing="$missing $t"
  done
  if [ -z "$missing" ]; then
    echo "== HDF5 tools ($when): $(command -v h5dump) =="
    return 0
  fi
  echo "::error::required HDF5 tools missing from deps-cpu ($when):$missing"
  echo "-- PATH --"; echo "$PATH"
  echo "-- identity --"; id; echo "shell=$0"
  echo "-- ls /usr/local/bin/h5* --"; ls -la /usr/local/bin/h5* || true
  echo "-- ls /usr/bin/h5* --";       ls -la /usr/bin/h5* || true
  echo "-- type -a per tool --"
  for t in h5dump h5ls h5diff h5stat h5clear h5repack h5cc; do
    echo "  $t: $(type -a "$t" 2>&1 | tr '\n' ' ')"
  done
  echo "-- find --"; find /usr/local /opt /usr -name h5dump 2>/dev/null || true
  return 1
}

# Discovery, not assumption: the install location has moved across deps-cpu
# revisions, so locate the tools rather than hardcode where they live. This is
# what ci-vfd.yml does. Absence stays a HARD failure.
h5bin="$(command -v h5dump 2>/dev/null \
         || find /usr/local /opt /usr -name h5dump -type f 2>/dev/null | head -1)"
[ -n "$h5bin" ] && export PATH="$(dirname "$h5bin"):$PATH"

# Gate BEFORE the build. This is an image-sanity question and is answerable
# without compiling a single object file; asked here it costs seconds, asked
# after the build (where it originally sat) it costs nine minutes to say the
# same thing. Not a fix for anything -- the failure was the quoting bug in the
# header -- just the right place for the question.
check_hdf5_tools "pre-build" || exit 1

# h5py for the VOL compat suite, SOURCE-built (--no-binary) against the image
# HDF5 (/usr/local, 2.1.1) so the connector loads into h5py. A PyPI wheel
# bundles its own libhdf5 and cannot load a VOL built against the system HDF5.
# Pin the suite to this interpreter via -DPython3_EXECUTABLE below.
HDF5_DIR=/usr/local /usr/bin/python3 -m pip install \
  --break-system-packages --no-binary=h5py h5py numpy

cmake -B "$BUILD_DIR" \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCLIO_CORE_ENABLE_RUNTIME=ON \
  -DCLIO_CORE_ENABLE_CTE=ON \
  -DCLIO_CORE_ENABLE_CAE=OFF \
  -DCLIO_CORE_ENABLE_CEE=OFF \
  -DCLIO_CORE_ENABLE_TESTS=ON \
  -DCLIO_CORE_ENABLE_BENCHMARKS=OFF \
  -DCLIO_CORE_ENABLE_PYTHON=OFF \
  -DCLIO_CORE_ENABLE_CONDA=OFF \
  -DCLIO_CORE_ENABLE_ELF=ON \
  -DCLIO_CTE_ENABLE_POSIX_ADAPTER=ON \
  -DCLIO_CTE_ENABLE_STDIO_ADAPTER=ON \
  -DCLIO_CTE_ENABLE_VFD=ON \
  -DCLIO_CTE_ENABLE_HDF5_VOL=ON \
  -DCLIO_CTE_ENABLE_FUSE_ADAPTER=ON \
  -DCLIO_CTE_ENABLE_ADIOS2_ADAPTER=OFF \
  -DCLIO_CTE_ENABLE_COMPRESS=OFF \
  -DCLIO_CORE_ENABLE_GRAY_SCOTT=OFF \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCLIO_CTP_LOG_LEVEL=1
cmake --build "$BUILD_DIR" -j"$(nproc)"

# Re-assert at the point of USE. The suites below are what consume these tools,
# so the gate protecting them has to hold here and not merely have held before
# the build. A second `command -v` loop costs nothing and keeps the pre-build
# move from narrowing what the gate covers.
check_hdf5_tools "post-build" || exit 1

# Make "tool absent -> skip" a failure inside the VFD C suite.
export CLIO_REQUIRE_HDF5_TOOLS=1

# Adapter unit/smoke tests. Exclude the docker-in-docker FUSE integration test
# (LABELS include "docker").
#
# compat_suite is excluded HERE and run explicitly below. Both suites match this
# -R by name -- cte_hdf5_vol_compat_suite on "hdf5_vol", cte_hdf5_vfd_compat_suite
# on "vfd" -- so without this they run twice each: once here and once in the
# skip-detection loop. They are RUN_SERIAL with TIMEOUT 900, and
# --repeat until-pass:2 can double a failing one again, so the duplicate is
# minutes of wall clock for no extra evidence. The windows job already excludes
# them for the same reason.
ctest --test-dir "$BUILD_DIR" --output-on-failure \
  --timeout 180 --repeat until-pass:2 \
  -R "posix|stdio|vfd|hdf5_vol|fuse" \
  -E "copy_workspace|compat_suite" -LE "docker|ollama"

# --repeat until-pass:2 above keeps a one-off flake from blocking a merge, and
# in doing so makes the flake INVISIBLE: the check goes green and the only trace
# is in LastTest.log. That is how cte_hdf5_vol_io_dataset failed its first
# attempt in 5 of 146 runs (a read-back DATA MISMATCH, not a timeout) without
# anyone noticing until the logs were mined by hand for #1022. Every other CI
# job already runs this reporter; the adapters job was the one that did not.
# Reports only -- never gates, never changes the exit status.
bash CI/report_flaky_tests.sh \
  "$BUILD_DIR/Testing/Temporary/LastTest.log" "adapters (linux, all 5)" || true

# The compat suites self-skip (exit 125 -> CTest "Skipped") when the toolchain is
# missing, which is right locally and WRONG here: a skipped differential suite is
# no evidence at all, and ctest above counts a skip as success. Assert both
# actually ran.
for t in cte_hdf5_vol_compat_suite cte_hdf5_vfd_compat_suite; do
  if ctest --test-dir "$BUILD_DIR" -R "^${t}$" -N | grep -q "$t"; then
    ctest --test-dir "$BUILD_DIR" -R "^${t}$" --output-on-failure \
      --timeout 900 2>&1 | tee "/tmp/${t}.log"
    if grep -qiE "Skipped|SKIP:" "/tmp/${t}.log"; then
      echo "::error::${t} was SKIPPED; the differential compat evidence did not run"
      exit 1
    fi
  else
    echo "::error::${t} is not registered; expected it in this build"
    exit 1
  fi
done

# Live end-to-end mount smoke (real libfuse3 mount + I/O).
chmod +x CI/fuse_mount_smoke.sh
CI/fuse_mount_smoke.sh "$BUILD_DIR"

# xfstests conformance gate (issue #677): install xfstests + its runtime deps,
# then run ONLY the tests known to pass against the clio FUSE fs
# (scripts/xfstests/ci_baseline_pass.txt -- the max passing set). Every baseline
# test must pass or the job fails. Currently-failing tests are tracked in a
# GitHub issue, not run.
bash scripts/xfstests/install_deps.sh
CLIO_BUILD_DIR="$PWD/$BUILD_DIR" bash scripts/xfstests/run_ci_xfstests.sh

# SCRATCH xfstests gate (issue #677): the tests above use only TEST_DIR; these
# need a second, isolated scratch device via two persistent clio runtimes +
# xfstests native fuse mount-helper.
# NON-BLOCKING for now: this driver is branch-only (not on dev) and has never
# passed in the deps-cpu docker -- the keeper runtimes come up but every test
# fails instantly at _scratch_mount (the native fuse mount-helper is not wired up
# in the container). It was silently masked until the embedded gate above started
# passing (set -e stopped the step first). Run it for visibility but do not fail
# the job on it until the mount-helper setup is made CI-ready; the embedded gate
# is the enforced signal. See scripts/xfstests/REMAINING_WORK.md.
CLIO_BUILD_DIR="$PWD/$BUILD_DIR" \
  bash scripts/xfstests/run_ci_scratch_xfstests.sh \
  || echo "::warning::scratch xfstests gate not yet CI-ready (branch-only, mount-helper unset in docker); non-blocking"
