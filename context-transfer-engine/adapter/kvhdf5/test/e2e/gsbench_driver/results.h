// Parses the child binary's `GSBENCH_RESULT ...` line, aggregates repeats (median/spread),
// and renders the single-config table / campaign tagged lines / resume file.
//
// Result-line format (must match PrintResult() in gray_scott_threeway_bench.cu verbatim):
//   GSBENCH_RESULT arm=<s> N=<u> chunks=<u> blocks=<u> snaps=<u> steps=<u> bdev=<s>
//     pinned=<u> durable=<u> MB=<f> ms=<f> MBps=<f> checksum=<llu>
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

namespace gsbench {

struct ResultLine {
    std::string arm;
    unsigned N = 0, chunks = 0, blocks = 0, snaps = 0, steps = 0;
    std::string bdev;
    unsigned pinned = 0, durable = 0;
    double MB = 0, ms = 0, MBps = 0;
    uint64_t checksum = 0;
    std::string raw_line;  // the original "GSBENCH_RESULT ..." text, kept verbatim for logging
};

// Scans `text` (a child's captured stdout+stderr) for the LAST line beginning with
// "GSBENCH_RESULT " (the scripts use `tail -1` / `grep | tee`, and a hung/retried child could
// in principle emit more than one; last-one-wins matches their semantics) and parses it.
std::optional<ResultLine> ExtractResultLine(const std::string& text);

// The outcome of running one child process (one arm at one config, one rep).
struct ChildOutcome {
    std::string arm;
    bool exited_ok = false;      // process exited (didn't crash/timeout); NOT the same as "has result"
    bool timed_out = false;
    int exit_code = -1;
    std::optional<ResultLine> result;
    std::string log_path;        // where the full captured output was written, if any
};

// ---- single-config table (ports run_threeway_bench.sh's inline python block) -------------

struct ArmAggregate {
    std::string arm;
    double mb = 0;
    double median_ms = 0;
    double min_ms = 0, max_ms = 0;
    int n = 0;
    std::vector<uint64_t> checksums;  // distinct checksums seen; >1 means DISAGREEMENT
};

// Aggregates `outcomes` (all reps, all arms, one config) into one row per arm, in the order
// `arm_order` lists them (skips arms with zero successful results).
std::vector<ArmAggregate> Aggregate(const std::vector<ChildOutcome>& outcomes,
                                     const std::vector<std::string>& arm_order);

// Prints the single-config table: arm / MB / ms(med) / MB/s / vs-baseline / spread / n /
// checksum, plus a checksum PASS/FAIL summary line. Mirrors the python block's columns.
void PrintSingleConfigTable(std::ostream& os, const std::vector<ArmAggregate>& rows,
                             const std::string& baseline_arm);

// ---- campaign tagged lines + resume -------------------------------------------------------

// Renders one campaign result line in the scripts' `SWEEP=... CFG=... REP=... ARM=... <line>`
// convention (so existing offline CSV/plot tooling still works unchanged).
std::string CampaignTagLine(const std::string& study, const std::string& cfg_key, int rep,
                             const std::string& arm, const std::string& result_or_failed);

// A resume key identifies one work unit: (study, cfg_key, rep, arm).
struct ResumeKey {
    std::string study, cfg_key, arm;
    int rep = 0;
    bool operator<(const ResumeKey& o) const {
        if (study != o.study) return study < o.study;
        if (cfg_key != o.cfg_key) return cfg_key < o.cfg_key;
        if (rep != o.rep) return rep < o.rep;
        return arm < o.arm;
    }
};

// Loads a campaign --out file (if it exists) and returns the set of (study,cfg,rep,arm) keys
// that already have a completed (non-FAILED) result line, for --resume to skip.
std::map<ResumeKey, std::string> LoadResumeFile(const std::string& path);

}  // namespace gsbench
