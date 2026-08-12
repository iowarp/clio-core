// gsbench_run: CLI parse -> dispatch to single-config run or campaign. See SPEC.md.
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include <unistd.h>

#include "arms.h"
#include "cli.h"
#include "results.h"
#include "runner.h"
#include "studies.h"

namespace fs = std::filesystem;
using namespace gsbench;

namespace {

std::string SelfExeDir() {
    char buf[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return ".";
    buf[n] = '\0';
    return fs::path(std::string(buf)).parent_path().string();
}

// --bin, else GSBENCH_BUILD_DIR/--build-dir's bin/, else next to this binary, else $GSBENCH_BIN.
std::string ResolveBin(const DriverConfig& cfg) {
    if (!cfg.bin.empty()) return cfg.bin;
    std::string build_dir = cfg.build_dir;
    if (build_dir.empty()) {
        if (const char* e = std::getenv("GSBENCH_BUILD_DIR")) build_dir = e;
    }
    if (!build_dir.empty()) {
        std::string candidate = build_dir + "/bin/kvhdf5_e2e_tests";
        if (fs::exists(candidate)) return candidate;
    }
    std::string next_to_self = SelfExeDir() + "/kvhdf5_e2e_tests";
    if (fs::exists(next_to_self)) return next_to_self;
    if (const char* e = std::getenv("GSBENCH_BIN")) {
        if (fs::exists(e)) return e;
    }
    return "";
}

std::string DefaultBuildDir(const DriverConfig& cli, const std::string& bin_dir) {
    if (!cli.build_dir.empty()) return cli.build_dir;
    // bin lives at <build_dir>/bin/kvhdf5_e2e_tests
    return fs::path(bin_dir).parent_path().string();
}

void PrintDryRunUnit(const std::string& tag, const std::vector<EnvOverride>& cfg_env, int rep,
                      const std::string& arm_name, const std::string& bin_dir) {
    const Arm* arm = FindArm(arm_name);
    if (!arm) {
        std::printf("%s rep=%d arm=%s  ! unknown arm\n", tag.c_str(), rep, arm_name.c_str());
        return;
    }
    ChildEnvPlan plan = ComputeChildEnv(*arm, cfg_env, bin_dir);
    std::printf("%s rep=%d arm=%-14s selector=%-24s", tag.c_str(), rep, arm_name.c_str(),
                arm->selector.c_str());
    for (const auto& u : plan.unset) std::printf(" -u %s", u.c_str());
    for (const auto& e : plan.set) std::printf(" %s=%s", e.key.c_str(), e.value.c_str());
    std::printf("\n");
}

// The single-config env vector is shared by every arm in a --arms run, and ComputeChildEnv
// appends it AFTER the per-arm registry env (setenv overwrite => last wins), so anything put in
// it overrides every arm's registry default. That is fine for GSBENCH_POOL (only `pooled` reads
// it) but NOT for GSBENCH_HDF5_ASYNC_POOL: async_VOL and async_VOL_reuse are the same
// TEST_CASE reading the same var, so a global --hdf5-async-pool would silently pool the
// unbounded baseline and invalidate the comparison. Hence this per-arm view: the override is
// appended only for arms that opt in via Arm::reads_hdf5_async_pool (async_VOL_reuse alone).
std::vector<EnvOverride> ArmEnv(const Arm& arm, const std::vector<EnvOverride>& shared,
                                 const DriverConfig& cli) {
    if (!cli.hdf5_async_pool || !arm.reads_hdf5_async_pool) return shared;
    std::vector<EnvOverride> e = shared;
    e.push_back({"GSBENCH_HDF5_ASYNC_POOL", std::to_string(cli.hdf5_async_pool)});
    return e;
}

int RunSingleConfig(const DriverConfig& cli, const std::string& bin_path) {
    std::vector<std::string> arm_names = cli.arms.empty() ? DefaultArmNames() : cli.arms;
    for (const auto& a : arm_names) {
        if (!FindArm(a)) {
            std::fprintf(stderr, "gsbench_run: unknown arm '%s'\n", a.c_str());
            return 2;
        }
    }

    const std::string bin_dir = fs::path(bin_path).parent_path().string();
    const std::string build_dir = DefaultBuildDir(cli, bin_dir);
    const std::string scratch =
        !cli.scratch.empty() ? cli.scratch
        : (build_dir.empty() ? "gsbench_scratch" : build_dir + "/gsbench_scratch");

    const unsigned n = cli.n, chunks = cli.chunks, snaps = cli.snaps, steps = cli.steps_per;
    const unsigned submit_blocks = cli.submit_blocks ? cli.submit_blocks : chunks;

    std::vector<EnvOverride> env = {
        {"GSBENCH_N", std::to_string(n)},
        {"GSBENCH_CHUNKS", std::to_string(chunks)},
        {"GSBENCH_SUBMIT_BLOCKS", std::to_string(submit_blocks)},
        {"GSBENCH_SNAPS", std::to_string(snaps)},
        {"GSBENCH_STEPS_PER", std::to_string(steps)},
        // Both source scripts default this to 0 (buffered writes); O_DIRECT is a measured wash
        // on the reference NVMe (see scaling_campaign/README.md finding 8).
        {"GSBENCH_RAW_ODIRECT", "0"},
    };

    // Tier-derived storage env, matching run_threeway_bench.sh's defaults (scratch-relative
    // for `file`; /dev/shm for `ram`).
    if (cli.tier == "ram") {
        env.push_back({"GSBENCH_BDEV", "ram"});
        env.push_back({"GSBENCH_DISK_DIR", "/dev/shm/gsbench_raw"});
        env.push_back({"GSBENCH_HDF5_DIR", "/dev/shm/gsbench_hdf5"});
    } else {
        env.push_back({"GSBENCH_BDEV", "file"});
        env.push_back({"GSBENCH_DISK_DIR", scratch + "/raw_out"});
        env.push_back({"GSBENCH_HDF5_DIR", scratch + "/hdf5_out"});
    }
    env.push_back({"GSBENCH_BDEV_PATH", scratch + "/clio_bdev.dat"});
    const unsigned cap = cli.bdev_cap_mb ? cli.bdev_cap_mb : static_cast<unsigned>(BdevCap(n, snaps));
    env.push_back({"GSBENCH_BDEV_CAP_MB", std::to_string(cap)});
    if (cli.pool) env.push_back({"GSBENCH_POOL", std::to_string(cli.pool)});

    // explicit overrides win over the tier defaults
    if (!cli.bdev.empty()) env.push_back({"GSBENCH_BDEV", cli.bdev});
    if (!cli.bdev_path.empty()) env.push_back({"GSBENCH_BDEV_PATH", cli.bdev_path});
    if (!cli.disk_dir.empty()) env.push_back({"GSBENCH_DISK_DIR", cli.disk_dir});
    if (!cli.hdf5_dir.empty()) env.push_back({"GSBENCH_HDF5_DIR", cli.hdf5_dir});

    if (cli.dry_run) {
        std::printf("-- single-config dry-run: %zu arm(s) x %u rep(s) --\n", arm_names.size(),
                    cli.reps);
        for (unsigned rep = 1; rep <= cli.reps; ++rep) {
            for (const auto& a : arm_names) {
                const Arm* arm = FindArm(a);  // validated non-null at the top of this function
                PrintDryRunUnit("[single]", ArmEnv(*arm, env, cli), rep, a, bin_dir);
            }
        }
        return 0;
    }

    fs::create_directories(scratch);
    fs::create_directories(scratch + "/raw_out");
    fs::create_directories(scratch + "/hdf5_out");
    fs::create_directories("/dev/shm/gsbench_raw");
    fs::create_directories("/dev/shm/gsbench_hdf5");

    RunnerOptions ropts;
    ropts.bin_path = bin_path;
    ropts.timeout_sec = cli.timeout_sec;

    std::vector<ChildOutcome> outcomes;
    std::ofstream out_file;
    if (!cli.out.empty()) out_file.open(cli.out, std::ios::app);

    for (unsigned rep = 1; rep <= cli.reps; ++rep) {
        if (cli.reps > 1) {
            std::printf("================ REPEAT %u/%u ================\n", rep, cli.reps);
        }
        for (const auto& a : arm_names) {
            const Arm* arm = FindArm(a);
            StateReset(/*hard=*/false, "");
            std::printf(">>> arm: %s\n", a.c_str());
            ChildEnvPlan plan = ComputeChildEnv(*arm, ArmEnv(*arm, env, cli), bin_dir);
            ChildOutcome oc = RunOneChild(ropts, *arm, plan, "");
            if (oc.result) {
                std::printf("    %s\n", oc.result->raw_line.c_str());
                if (out_file) out_file << oc.result->raw_line << "\n";
            } else {
                std::printf("    (no GSBENCH_RESULT line; exit=%d timed_out=%d)\n", oc.exit_code,
                            oc.timed_out ? 1 : 0);
            }
            outcomes.push_back(std::move(oc));
        }
    }
    StateReset(/*hard=*/false, "");

    auto rows = Aggregate(outcomes, arm_names);
    std::printf("\n================ BENCHMARK RESULT ================\n");
    PrintSingleConfigTable(std::cout, rows, cli.baseline);
    return 0;
}

int RunCampaign(const DriverConfig& cli, const std::string& bin_path) {
    for (const auto& t : cli.tiers) {
        if (t != "ram" && t != "file") {
            std::fprintf(stderr, "gsbench_run: --tiers: unknown tier '%s'\n", t.c_str());
            return 2;
        }
    }
    for (const auto& s : cli.studies) {
        if (s != "C" && s != "K" && s != "B" && s != "W" && s != "P") {
            std::fprintf(stderr, "gsbench_run: --studies: unknown study '%s'\n", s.c_str());
            return 2;
        }
    }

    const std::string bin_dir = fs::path(bin_path).parent_path().string();
    const std::string build_dir = DefaultBuildDir(cli, bin_dir);
    const std::string scratch =
        !cli.scratch.empty() ? cli.scratch
        : (build_dir.empty() ? "gsbench_campaign_scratch"
                              : build_dir + "/gsbench_campaign_scratch");

    std::vector<WorkConfig> units = BuildCampaign(cli.studies, cli.tiers, scratch);

    if (cli.dry_run) {
        std::printf("-- campaign dry-run: %zu config(s) --\n", units.size());
        for (const auto& wc : units) {
            std::printf("==== %s ====\n", wc.description.c_str());
            const std::string tag = "  [" + wc.study + "/" + wc.cfg_key + "]";
            for (int rep = 1; rep <= wc.reps; ++rep) {
                for (const auto& a : wc.arms) PrintDryRunUnit(tag, wc.env, rep, a, bin_dir);
            }
        }
        return 0;
    }

    fs::create_directories(scratch);
    fs::create_directories(scratch + "/raw_out");
    fs::create_directories(scratch + "/hdf5_out");
    fs::create_directories("/dev/shm/gsbench_raw");
    fs::create_directories("/dev/shm/gsbench_hdf5");

    std::map<ResumeKey, std::string> done;
    if (cli.resume && !cli.out.empty()) done = LoadResumeFile(cli.out);

    std::ofstream out_file;
    if (!cli.out.empty()) out_file.open(cli.out, std::ios::app);

    RunnerOptions ropts;
    ropts.bin_path = bin_path;
    ropts.timeout_sec = cli.timeout_sec;

    size_t total = 0, skipped = 0, failed = 0;
    for (const auto& wc : units) {
        std::printf("==== %s ====\n", wc.description.c_str());
        for (int rep = 1; rep <= wc.reps; ++rep) {
            for (const auto& a : wc.arms) {
                ++total;
                ResumeKey key{wc.study, wc.cfg_key, a, rep};
                auto it = done.find(key);
                if (it != done.end()) {
                    ++skipped;
                    std::printf("SKIP  %s/%s rep%d %s (cached)\n", wc.study.c_str(),
                                wc.cfg_key.c_str(), rep, a.c_str());
                    std::cout << it->second << "\n";
                    continue;
                }

                const Arm* arm = FindArm(a);
                ChildEnvPlan plan = ComputeChildEnv(*arm, wc.env, bin_dir);

                std::string result_txt;
                for (int attempt = 1; attempt <= 2; ++attempt) {
                    StateReset(/*hard=*/true, scratch);
                    ChildOutcome oc = RunOneChild(ropts, *arm, plan, "");
                    if (oc.result) {
                        result_txt = oc.result->raw_line;
                        break;
                    }
                    std::printf("RETRY %s/%s rep%d %s (empty, attempt %d)\n", wc.study.c_str(),
                                wc.cfg_key.c_str(), rep, a.c_str(), attempt);
                }
                StateReset(/*hard=*/true, scratch);

                std::string line;
                if (result_txt.empty()) {
                    ++failed;
                    std::printf("!!!! FAIL  %s/%s rep%d %s\n", wc.study.c_str(),
                                wc.cfg_key.c_str(), rep, a.c_str());
                    line = CampaignTagLine(wc.study, wc.cfg_key, rep, a, "FAILED");
                } else {
                    std::printf("ok    %s/%s rep%d %s :: %s\n", wc.study.c_str(),
                                wc.cfg_key.c_str(), rep, a.c_str(), result_txt.c_str());
                    line = CampaignTagLine(wc.study, wc.cfg_key, rep, a, result_txt);
                }
                std::cout << line << "\n";
                if (out_file) {
                    out_file << line << "\n";
                    out_file.flush();
                }
            }
        }
    }
    StateReset(/*hard=*/true, scratch);

    std::printf("################ CAMPAIGN DONE ################\n");
    std::printf("total=%zu skipped=%zu failed=%zu\n", total, skipped, failed);
    return failed > 0 ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    DriverConfig cli;
    std::string err;
    if (!ParseArgs(argc, argv, cli, err)) {
        std::fprintf(stderr, "gsbench_run: %s\n\n", err.c_str());
        PrintHelp();
        return 2;
    }
    if (cli.help) {
        PrintHelp();
        return 0;
    }

    std::string bin_path = ResolveBin(cli);
    if (bin_path.empty()) {
        if (!cli.dry_run) {
            std::fprintf(stderr,
                         "gsbench_run: could not locate kvhdf5_e2e_tests "
                         "(pass --bin, --build-dir, or set GSBENCH_BUILD_DIR)\n");
            return 1;
        }
        bin_path = "kvhdf5_e2e_tests";  // dry-run only: used just for display of selectors/env
    }

    // Process-wide runtime env, set ONCE for the whole driver run -- these are identical for
    // every child (not per-arm), matching the scripts' top-level `export`s.
    setenv("CLIO_BIND_ADDR", "127.0.0.1", 0);
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);
    const std::string bin_dir = fs::path(bin_path).parent_path().string();
    if (!bin_dir.empty()) setenv("CHI_REPO_PATH", bin_dir.c_str(), 1);

    // The CLIO arms need a single-worker server config (concurrent-put-safe substrate for
    // multi-chunk async snapshots); generate one unless the caller already supplied
    // CLIO_SERVER_CONF, matching both scripts.
    std::string conf_tmp;
    if (!cli.dry_run && !std::getenv("CLIO_SERVER_CONF")) {
        char tmpl[] = "/tmp/gsbench_conf_XXXXXX";
        int fd = mkstemp(tmpl);
        if (fd >= 0) {
            const char* content = "runtime:\n  num_threads: 1\n";
            if (write(fd, content, std::strlen(content)) < 0) { /* best-effort */ }
            close(fd);
            conf_tmp = tmpl;
            setenv("CLIO_SERVER_CONF", conf_tmp.c_str(), 1);
        }
    }

    const int rc = cli.campaign ? RunCampaign(cli, bin_path) : RunSingleConfig(cli, bin_path);

    if (!conf_tmp.empty()) unlink(conf_tmp.c_str());
    return rc;
}
