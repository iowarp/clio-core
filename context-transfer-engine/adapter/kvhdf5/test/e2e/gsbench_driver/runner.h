// The supervisor core: fork/setenv/execvp one child, capture its combined stdout+stderr,
// extract the GSBENCH_RESULT line, and native state-reset between children. Ports run_arm() /
// reset_state() / hard_reset() from run_threeway_bench.sh and scaling_campaign/run_campaign.sh.
#pragma once

#include <string>
#include <vector>

#include "arms.h"
#include "results.h"

namespace gsbench {

// The fully-resolved environment for one child, split into sets (setenv) and unsets
// (unsetenv) -- e.g. every non-hdf5_async arm unsets HDF5_VOL_CONNECTOR, matching the scripts'
// `env -u HDF5_VOL_CONNECTOR` isolation (it is process-global and would otherwise leak the
// async VOL onto the plain hdf5 arm's file).
struct ChildEnvPlan {
    std::vector<EnvOverride> set;
    std::vector<std::string> unset;
};

// Builds the full per-child environment: async-VOL isolation (LD_LIBRARY_PATH /
// HDF5_PLUGIN_PATH / HDF5_VOL_CONNECTOR, ported from run_campaign.sh's VOL_LD/VOL_PLUGIN/
// VOL_CONNECTOR + per-arm case block), the arm's own selection env (arms.cpp registry), then
// `extra_env` (the work-unit's GSBENCH_* config knobs) applied last. Pure function -- does not
// touch the calling process's real environment. `bin_dir` = dirname(bin_path).
ChildEnvPlan ComputeChildEnv(const Arm& arm, const std::vector<EnvOverride>& extra_env,
                              const std::string& bin_dir);

struct RunnerOptions {
    std::string bin_path;        // path to kvhdf5_e2e_tests
    unsigned timeout_sec = 600;  // per-child wall-clock guard
    std::string log_dir;         // if non-empty, write each child's captured output here
};

// Runs one child: fork() -> apply `plan` via setenv/unsetenv -> chdir(bin_dir) -> setsid() ->
// execvp(bin_path, {bin_path, arm.selector, nullptr}). Parent captures combined stdout+stderr
// via a pipe, waits with a wall-clock deadline (SIGKILL to the child's process group on
// expiry -- the CTE runtime may itself fork, so a plain kill(pid) is not enough), and extracts
// the GSBENCH_RESULT line from the captured text.
ChildOutcome RunOneChild(const RunnerOptions& opts, const Arm& arm, const ChildEnvPlan& plan,
                          const std::string& log_path);

// Kills any leftover kvhdf5_e2e_test process (comm-name truncates to 15 chars, so match on
// that exact truncated name, zombie-aware: a container's pid 1 often does not reap children,
// leaving defunct entries that must be ignored, not waited on) and unlinks /dev/shm/chi_*.
// `hard`, when true, ALSO wipes `scratch_dir`'s bdev file and raw_out/hdf5_out dirs (matching
// run_campaign.sh's hard_reset; the lighter single-config reset_state does not do this).
void StateReset(bool hard, const std::string& scratch_dir);

}  // namespace gsbench
