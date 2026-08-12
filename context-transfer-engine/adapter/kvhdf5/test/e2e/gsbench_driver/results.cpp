#include "results.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace gsbench {

namespace {

// Pulls "key=value" out of a GSBENCH_RESULT line. Values never contain spaces (see
// PrintResult's format string), so splitting on whitespace is safe.
std::optional<std::string> FieldOf(const std::vector<std::string>& tokens, const std::string& key) {
    const std::string prefix = key + "=";
    for (const auto& t : tokens) {
        if (t.rfind(prefix, 0) == 0) return t.substr(prefix.size());
    }
    return std::nullopt;
}

std::vector<std::string> Split(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string tok;
    while (ss >> tok) out.push_back(tok);
    return out;
}

bool ParseOneLine(const std::string& line, ResultLine& r) {
    auto tokens = Split(line);
    if (tokens.empty() || tokens[0] != "GSBENCH_RESULT") return false;
    auto get = [&](const char* k) { return FieldOf(tokens, k); };
    auto arm = get("arm");
    auto N = get("N");
    auto chunks = get("chunks");
    auto blocks = get("blocks");
    auto snaps = get("snaps");
    auto steps = get("steps");
    auto bdev = get("bdev");
    auto pinned = get("pinned");
    auto durable = get("durable");
    auto mb = get("MB");
    auto ms = get("ms");
    auto mbps = get("MBps");
    auto checksum = get("checksum");
    if (!arm || !N || !chunks || !blocks || !snaps || !steps || !bdev || !pinned ||
        !durable || !mb || !ms || !mbps || !checksum) {
        return false;
    }
    try {
        r.arm = *arm;
        r.N = static_cast<unsigned>(std::stoul(*N));
        r.chunks = static_cast<unsigned>(std::stoul(*chunks));
        r.blocks = static_cast<unsigned>(std::stoul(*blocks));
        r.snaps = static_cast<unsigned>(std::stoul(*snaps));
        r.steps = static_cast<unsigned>(std::stoul(*steps));
        r.bdev = *bdev;
        r.pinned = static_cast<unsigned>(std::stoul(*pinned));
        r.durable = static_cast<unsigned>(std::stoul(*durable));
        r.MB = std::stod(*mb);
        r.ms = std::stod(*ms);
        r.MBps = std::stod(*mbps);
        r.checksum = std::stoull(*checksum);
    } catch (const std::exception&) {
        return false;
    }
    r.raw_line = line;
    return true;
}

}  // namespace

std::optional<ResultLine> ExtractResultLine(const std::string& text) {
    std::istringstream iss(text);
    std::string line;
    std::optional<ResultLine> best;
    while (std::getline(iss, line)) {
        // Strip a trailing '\r' in case the child's output has CRLF line endings.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        ResultLine r;
        if (ParseOneLine(line, r)) best = r;  // last one wins, matching the scripts' `tail -1`
    }
    return best;
}

std::vector<ArmAggregate> Aggregate(const std::vector<ChildOutcome>& outcomes,
                                     const std::vector<std::string>& arm_order) {
    std::vector<ArmAggregate> rows;
    for (const auto& arm : arm_order) {
        ArmAggregate agg;
        agg.arm = arm;
        std::vector<double> samples;
        for (const auto& o : outcomes) {
            if (o.arm != arm || !o.result) continue;
            samples.push_back(o.result->ms);
            agg.mb = o.result->MB;
            agg.checksums.push_back(o.result->checksum);
        }
        if (samples.empty()) continue;
        std::sort(samples.begin(), samples.end());
        agg.n = static_cast<int>(samples.size());
        agg.min_ms = samples.front();
        agg.max_ms = samples.back();
        const size_t mid = samples.size() / 2;
        agg.median_ms = (samples.size() % 2 == 0)
                             ? (samples[mid - 1] + samples[mid]) / 2.0
                             : samples[mid];
        std::sort(agg.checksums.begin(), agg.checksums.end());
        agg.checksums.erase(std::unique(agg.checksums.begin(), agg.checksums.end()),
                             agg.checksums.end());
        rows.push_back(std::move(agg));
    }
    return rows;
}

void PrintSingleConfigTable(std::ostream& os, const std::vector<ArmAggregate>& rows,
                             const std::string& baseline_arm) {
    if (rows.empty()) {
        os << "(no results to report)\n";
        return;
    }
    double base_ms = 0;
    for (const auto& r : rows) {
        if (r.arm == baseline_arm) base_ms = r.median_ms;
    }

    // The "vs <baseline>" header can be longer than the original script's fixed "vs raw" (the
    // driver's baseline names are longer, e.g. "raw_inline") -- size the column to fit it so the
    // header never runs into its neighbor.
    const std::string vs_label = "vs " + baseline_arm;
    const int vs_width = std::max<int>(9, static_cast<int>(vs_label.size()) + 1);

    char buf[256];
    std::snprintf(buf, sizeof(buf), "%-15s%9s%11s%10s%*s%9s%4s  checksum", "arm", "MB",
                  "ms(med)", "MB/s", vs_width, vs_label.c_str(), "spread", "n");
    os << "\n" << buf << "\n";

    std::unordered_set<uint64_t> all_checksums;
    for (const auto& r : rows) {
        const double mbps = r.mb / (r.median_ms / 1000.0);
        std::string rel = "-";
        if (base_ms > 0) {
            std::snprintf(buf, sizeof(buf), "%.2fx", base_ms / r.median_ms);
            rel = buf;
        }
        std::string spread = "-";
        if (r.n > 1 && r.median_ms > 0) {
            std::snprintf(buf, sizeof(buf), "%.1f%%",
                          100.0 * (r.max_ms - r.min_ms) / r.median_ms);
            spread = buf;
        }
        std::string cksum = r.checksums.size() == 1 ? std::to_string(r.checksums[0]) : "VARIES!";
        std::snprintf(buf, sizeof(buf), "%-15s%9.1f%11.2f%10.1f%*s%9s%4d  %s", r.arm.c_str(),
                      r.mb, r.median_ms, mbps, vs_width, rel.c_str(), spread.c_str(), r.n,
                      cksum.c_str());
        os << buf << "\n";
        for (auto c : r.checksums) all_checksums.insert(c);
    }

    os << "\nchecksums "
       << (all_checksums.size() == 1 ? "MATCH (identical computation)"
                                      : "DIFFER  <-- TABLE IS INVALID")
       << "\n";
}

std::string CampaignTagLine(const std::string& study, const std::string& cfg_key, int rep,
                             const std::string& arm, const std::string& result_or_failed) {
    return "SWEEP=" + study + " CFG=" + cfg_key + " REP=" + std::to_string(rep) +
           " ARM=" + arm + " " + result_or_failed;
}

std::map<ResumeKey, std::string> LoadResumeFile(const std::string& path) {
    std::map<ResumeKey, std::string> done;
    std::ifstream in(path);
    if (!in) return done;
    std::string line;
    while (std::getline(in, line)) {
        // Expected shape: "SWEEP=<s> CFG=<s> REP=<n> ARM=<s> GSBENCH_RESULT ..."
        auto tokens = Split(line);
        std::optional<std::string> sweep, cfg, rep, arm;
        for (const auto& t : tokens) {
            if (t.rfind("SWEEP=", 0) == 0) sweep = t.substr(6);
            else if (t.rfind("CFG=", 0) == 0) cfg = t.substr(4);
            else if (t.rfind("REP=", 0) == 0) rep = t.substr(4);
            else if (t.rfind("ARM=", 0) == 0) arm = t.substr(4);
        }
        if (!sweep || !cfg || !rep || !arm) continue;
        // Only a real (non-FAILED) result line counts as "already done", matching
        // run_campaign.sh's resume check (it greps for the GSBENCH_RESULT pattern, not FAILED).
        if (line.find("GSBENCH_RESULT") == std::string::npos) continue;
        ResumeKey key{*sweep, *cfg, *arm, std::stoi(*rep)};
        done[key] = line;
    }
    return done;
}

}  // namespace gsbench
