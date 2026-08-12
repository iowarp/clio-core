#include "cli.h"

#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace gsbench {

namespace {

std::vector<std::string> SplitCsv(const std::string& csv) {
    std::vector<std::string> out;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(tok);
    }
    return out;
}

bool ParseUnsigned(const std::string& s, unsigned& out) {
    if (s.empty()) return false;
    char* end = nullptr;
    long v = std::strtol(s.c_str(), &end, 10);
    if (*end != '\0' || v < 0) return false;
    out = static_cast<unsigned>(v);
    return true;
}

}  // namespace

bool ParseArgs(int argc, char** argv, DriverConfig& cfg, std::string& err) {
    std::vector<std::string> args(argv + 1, argv + argc);

    auto need_value = [&](size_t i, const std::string& flag) -> const std::string* {
        if (i + 1 >= args.size()) {
            err = flag + " needs a value";
            return nullptr;
        }
        return &args[i + 1];
    };

    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (a == "--help" || a == "-h") {
            cfg.help = true;
            return true;
        } else if (a == "--campaign") {
            cfg.campaign = true;
        } else if (a == "--dry-run") {
            cfg.dry_run = true;
        } else if (a == "--resume") {
            cfg.resume = true;
        } else if (a == "--bin") {
            auto v = need_value(i, a); if (!v) return false; cfg.bin = *v; ++i;
        } else if (a == "--build-dir") {
            auto v = need_value(i, a); if (!v) return false; cfg.build_dir = *v; ++i;
        } else if (a == "--scratch") {
            auto v = need_value(i, a); if (!v) return false; cfg.scratch = *v; ++i;
        } else if (a == "--out") {
            auto v = need_value(i, a); if (!v) return false; cfg.out = *v; ++i;
        } else if (a == "--arms") {
            auto v = need_value(i, a); if (!v) return false; cfg.arms = SplitCsv(*v); ++i;
        } else if (a == "--baseline") {
            auto v = need_value(i, a); if (!v) return false; cfg.baseline = *v; ++i;
        } else if (a == "--reps") {
            auto v = need_value(i, a); if (!v) return false;
            if (!ParseUnsigned(*v, cfg.reps)) { err = "--reps: bad value " + *v; return false; }
            ++i;
        } else if (a == "--tier") {
            auto v = need_value(i, a); if (!v) return false;
            if (*v != "ram" && *v != "file") { err = "--tier must be ram or file"; return false; }
            cfg.tier = *v; ++i;
        } else if (a == "--n") {
            auto v = need_value(i, a); if (!v) return false;
            if (!ParseUnsigned(*v, cfg.n)) { err = "--n: bad value"; return false; }
            ++i;
        } else if (a == "--chunks") {
            auto v = need_value(i, a); if (!v) return false;
            if (!ParseUnsigned(*v, cfg.chunks)) { err = "--chunks: bad value"; return false; }
            ++i;
        } else if (a == "--snaps") {
            auto v = need_value(i, a); if (!v) return false;
            if (!ParseUnsigned(*v, cfg.snaps)) { err = "--snaps: bad value"; return false; }
            ++i;
        } else if (a == "--steps") {
            auto v = need_value(i, a); if (!v) return false;
            if (!ParseUnsigned(*v, cfg.steps_per)) { err = "--steps: bad value"; return false; }
            ++i;
        } else if (a == "--submit-blocks") {
            auto v = need_value(i, a); if (!v) return false;
            if (!ParseUnsigned(*v, cfg.submit_blocks)) { err = "--submit-blocks: bad value"; return false; }
            ++i;
        } else if (a == "--pool") {
            auto v = need_value(i, a); if (!v) return false;
            if (!ParseUnsigned(*v, cfg.pool)) { err = "--pool: bad value"; return false; }
            ++i;
        } else if (a == "--hdf5-async-pool") {
            auto v = need_value(i, a); if (!v) return false;
            if (!ParseUnsigned(*v, cfg.hdf5_async_pool)) {
                err = "--hdf5-async-pool: bad value " + *v; return false;
            }
            ++i;
        } else if (a == "--bdev") {
            auto v = need_value(i, a); if (!v) return false; cfg.bdev = *v; ++i;
        } else if (a == "--bdev-cap-mb") {
            auto v = need_value(i, a); if (!v) return false;
            if (!ParseUnsigned(*v, cfg.bdev_cap_mb)) { err = "--bdev-cap-mb: bad value"; return false; }
            ++i;
        } else if (a == "--bdev-path") {
            auto v = need_value(i, a); if (!v) return false; cfg.bdev_path = *v; ++i;
        } else if (a == "--disk-dir") {
            auto v = need_value(i, a); if (!v) return false; cfg.disk_dir = *v; ++i;
        } else if (a == "--hdf5-dir") {
            auto v = need_value(i, a); if (!v) return false; cfg.hdf5_dir = *v; ++i;
        } else if (a == "--timeout") {
            auto v = need_value(i, a); if (!v) return false;
            if (!ParseUnsigned(*v, cfg.timeout_sec)) { err = "--timeout: bad value"; return false; }
            ++i;
        } else if (a == "--studies") {
            auto v = need_value(i, a); if (!v) return false; cfg.studies = SplitCsv(*v); ++i;
        } else if (a == "--tiers") {
            auto v = need_value(i, a); if (!v) return false; cfg.tiers = SplitCsv(*v); ++i;
        } else {
            err = "unknown flag: " + a;
            return false;
        }
    }
    return true;
}

void PrintHelp() {
    std::puts(R"HELP(gsbench_run - self-supervising Gray-Scott I/O benchmark driver

Fork/execs kvhdf5_e2e_tests once per (arm, config, rep), in its own process (CLIO
runtime + CUDA context state is per-process). Replaces run_threeway_bench.sh and
scaling_campaign/run_campaign.sh with one typed C++ driver over a shared arm/study
registry.

SINGLE-CONFIG MODE (default): run the given arms once at one config, print a table.

  gsbench_run [options] [config knobs]

  --arms a,b,c            arms to run (default: raw_inline,raw_threaded,hostclio,
                           sync,async,hdf5_inline,hdf5_threaded)
  --baseline <arm>        arm the "vs" column is relative to (default: raw_inline)
  --reps N                repeats per arm (default: 1); table reports the median
  --tier ram|file         storage tier convenience (default: file)
  --n --chunks --snaps --steps --submit-blocks --pool
                           GSBENCH_N/CHUNKS/SNAPS/STEPS_PER/SUBMIT_BLOCKS/POOL
  --hdf5-async-pool M     reused pinned buffers for the async_VOL_reuse arm
                           (GSBENCH_HDF5_ASYNC_POOL; default 2 when omitted). Applied to
                           async_VOL_reuse ONLY -- async_VOL stays unbounded (0) even when
                           both arms run in the same invocation.
  --bdev --bdev-cap-mb --bdev-path --disk-dir --hdf5-dir
                           override tier-derived storage env directly

CAMPAIGN MODE: run the full study sweeps (C/K/B/W/P) with reps; emit tagged
"SWEEP=... CFG=... REP=... ARM=..." result lines; resumable via --out.

  gsbench_run --campaign [--studies C,K,B,W,P] [--tiers ram,file] [--out FILE] [--resume]

COMMON:
  --bin <path>       path to kvhdf5_e2e_tests (default: auto-detect next to this binary)
  --build-dir <dir>  GSBENCH_BUILD_DIR equivalent, used to locate --bin and scratch
  --scratch <dir>    scratch dir for bdev/raw/hdf5 outputs (default: <build-dir>/gsbench_scratch)
  --out <file>       also persist results here (single-config: table's raw lines;
                     campaign: every tagged result line, and the resume source)
  --timeout <sec>    per-child wall-clock guard (default 600)
  --dry-run          print each planned child's selector+env WITHOUT running it
  --help             this message
)HELP");
}

}  // namespace gsbench
