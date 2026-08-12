// The arm registry: single source of truth for {name -> Catch2 selector, env overrides}.
// Ported from run_threeway_bench.sh's per-arm `run_arm` calls and
// scaling_campaign/run_campaign.sh's tag_of()+case block, cross-checked against the
// TEST_CASE selector strings and Cfg env knobs in gray_scott_threeway_bench.cu.
#pragma once

#include <optional>
#include <string>
#include <vector>

namespace gsbench {

struct EnvOverride {
    std::string key;
    std::string value;
};

// One arm: a label, the Catch2 tag selector that runs it (exec'd as
// `kvhdf5_e2e_tests <selector>`), and the env overrides that select its variant of the
// underlying TEST_CASE (e.g. GSBENCH_RAW_INLINE picks hdf5_inline vs hdf5_threaded).
struct Arm {
    std::string name;
    std::string selector;              // e.g. "[gsbench_hdf5]"
    std::vector<EnvOverride> env;       // always-applied overrides for this arm
    bool is_hdf5_async = false;         // needs the async-VOL LD_LIBRARY_PATH/plugin/connector env
    bool is_pooled = false;             // needs GSBENCH_POOL set per-config (caller supplies value)
    // True only for the arm that a --hdf5-async-pool override is allowed to reach. Unlike
    // GSBENCH_POOL (read by exactly one arm, `pooled`), GSBENCH_HDF5_ASYNC_POOL is read by the
    // ONE function backing BOTH async_VOL and async_VOL_reuse, so a global override would
    // silently pool the unbounded baseline whenever the two arms run in the same invocation.
    // Callers must therefore gate the override on this flag rather than on the env var's name.
    bool reads_hdf5_async_pool = false;
};

// The full 11-arm registry (order = declaration order below; stable for iteration).
const std::vector<Arm>& AllArms();

// Look up by name; nullptr if unknown.
const Arm* FindArm(const std::string& name);

// The single-config mode's default arm set when --arms is not given. A superset of
// run_threeway_bench.sh's default sweep (raw/hdf5_inline/hdf5_threaded/hostclio/sync/async),
// with `raw` split into its two explicit variants (raw_inline, raw_threaded) since the
// original script's unlabeled "raw" arm ran under whatever GSBENCH_RAW_INLINE happened to be
// set to -- the driver makes that choice explicit instead of implicit.
std::vector<std::string> DefaultArmNames();

// Parse a comma-separated --arms list into names, validating each exists in the registry.
// Returns std::nullopt and sets `err` if any name is unknown.
std::optional<std::vector<std::string>> ParseArmList(const std::string& csv, std::string& err);

}  // namespace gsbench
