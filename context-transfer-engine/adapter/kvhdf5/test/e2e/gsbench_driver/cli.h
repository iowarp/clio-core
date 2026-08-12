// Minimal hand-rolled CLI parser (no external CLI library; a handful of flags).
#pragma once

#include <string>
#include <vector>

namespace gsbench {

struct DriverConfig {
    // mode
    bool campaign = false;
    bool dry_run = false;
    bool help = false;
    bool resume = false;

    // locations
    std::string bin;         // path to kvhdf5_e2e_tests; empty = auto-detect
    std::string build_dir;   // GSBENCH_BUILD_DIR equivalent, used to locate `bin` + default scratch
    std::string scratch;     // scratch dir for bdev/raw/hdf5 outputs; empty = derived from build_dir
    std::string out;         // --out file (results + campaign resume log)

    // single-config mode knobs
    std::vector<std::string> arms;        // --arms a,b,c (empty = DefaultArmNames())
    std::string baseline = "raw_inline";  // --baseline: arm the "vs" column is relative to
    unsigned reps = 1;
    std::string tier = "file";  // --tier ram|file (single-config convenience)

    unsigned n = 6400, chunks = 4, snaps = 12, steps_per = 48;
    unsigned submit_blocks = 0;  // 0 = default to chunks, matching the campaign's convention
    unsigned pool = 0;           // GSBENCH_POOL, only consulted for the `pooled` arm
    // GSBENCH_HDF5_ASYNC_POOL override for async_VOL_reuse. 0 = not given: leave the registry's
    // per-arm default (2) alone. NOT injected globally -- async_VOL reads the same env var, so
    // the driver applies this only to arms with Arm::reads_hdf5_async_pool.
    unsigned hdf5_async_pool = 0;

    // explicit overrides of tier-derived storage env (empty = tier default)
    std::string bdev;
    unsigned bdev_cap_mb = 0;  // 0 = auto (BdevCap(n,snaps))
    std::string bdev_path, disk_dir, hdf5_dir;

    unsigned timeout_sec = 600;  // per-child wall-clock guard

    // campaign mode knobs
    std::vector<std::string> studies;  // --studies C,K,B,W,P (empty = all five)
    std::vector<std::string> tiers;    // --tiers ram,file (empty = both)
};

// Parses argv[1..argc). Returns false and sets `err` on a bad flag/value. --help alone is
// always accepted (sets cfg.help and returns true without validating the rest).
bool ParseArgs(int argc, char** argv, DriverConfig& cfg, std::string& err);

void PrintHelp();

}  // namespace gsbench
