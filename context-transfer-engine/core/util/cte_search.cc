/**
 * cte_search — search the CTE from the command line.
 *
 * Three modes:
 *   regex     (default) — BlobQuery: list blobs whose tag and blob names
 *                         match the given regexes.
 *   semantic            — SemanticSearch: BM25 keyword ranking over blob bytes.
 *   temporal            — TemporalSearch: filter by last_modified timestamp.
 *
 * Usage examples
 * --------------
 *   cte_search ".*" ".*"
 *   cte_search "exp_.*" ".*"
 *   cte_search ".*" ".*" --semantic "plasma temperature gradient" --max 5
 *   cte_search ".*" ".*" --since 1h
 *   cte_search ".*" ".*" --since 30m --max 20
 *   cte_search ".*" ".*" --time-begin 1700000000000000000 \
 *                        --time-end   1700003600000000000
 *
 * Environment:
 *   CLIO_SERVER_CONF — path to runtime YAML config (required unless the
 *                      runtime is already running in another process)
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_ctp/util/logging.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Duration parsing  (e.g. "30s", "5m", "2h30m", "1d")
// ---------------------------------------------------------------------------

static uint64_t ParseDurationNs(const std::string &s) {
  const char *p = s.c_str();
  uint64_t total_ns = 0;
  bool consumed = false;

  auto read_number = [](const char *&p, double &out) -> bool {
    char *end;
    out = std::strtod(p, &end);
    if (end == p) { return false; }
    p = end;
    return true;
  };

  while (*p != '\0') {
    double n;
    if (!read_number(p, n)) { break; }
    consumed = true;
    uint64_t unit_ns;
    if (*p == 'd') { unit_ns = 86400ULL * 1'000'000'000ULL; ++p; }
    else if (*p == 'h') { unit_ns = 3600ULL * 1'000'000'000ULL; ++p; }
    else if (*p == 'm') { unit_ns = 60ULL * 1'000'000'000ULL; ++p; }
    else if (*p == 's' || *p == '\0') {
      unit_ns = 1'000'000'000ULL;
      if (*p == 's') { ++p; }
    } else {
      throw std::invalid_argument(
          std::string("Unknown unit '") + *p + "' in duration: " + s);
    }
    total_ns += static_cast<uint64_t>(n * static_cast<double>(unit_ns));
  }
  if (!consumed) {
    throw std::invalid_argument("Cannot parse duration: " + s);
  }
  return total_ns;
}

// CLOCK_MONOTONIC — same clock CTE uses for last_modified.
static uint64_t MonotonicNowNs() {
  return static_cast<uint64_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

struct Args {
  std::string tag_re;
  std::string blob_re;

  // mode
  std::string semantic;    // non-empty → semantic mode
  std::string since;       // non-empty → temporal via duration shorthand
  uint64_t time_begin = 0; // temporal explicit lower bound
  uint64_t time_end   = 0; // temporal explicit upper bound

  uint32_t max_results = 0;
  std::string format = "names";  // names | table | json
  bool tags_only = false;        // print only unique tag names
};

static void PrintUsage(const char *prog) {
  std::cerr <<
    "Usage: " << prog << " <tag_re> [blob_re] [options]\n"
    "\n"
    "Positional arguments:\n"
    "  tag_re    Full-string regex matched against tag names\n"
    "  blob_re   Full-string regex matched against blob names (default: .*)\n"
    "\n"
    "Mode flags (mutually exclusive):\n"
    "  --semantic QUERY     BM25 keyword query text\n"
    "  --since DURATION     Temporal: blobs modified within DURATION of now\n"
    "                       (e.g. 30s, 5m, 2h, 1d, 1h30m)\n"
    "  --time-begin NS      Temporal: lower bound, nanoseconds (CLOCK_MONOTONIC)\n"
    "\n"
    "Other options:\n"
    "  --time-end NS        Upper bound for --time-begin (0 = no upper bound)\n"
    "  --max N              Maximum results (0 = unlimited)\n"
    "  --format FMT         Output format: names (default), table, json\n"
    "  --tags-only          List only unique matching tag names\n"
    "  --help               Show this message\n"
    "\n"
    "Examples:\n"
    "  " << prog << " '.*'\n"
    "  " << prog << " 'exp_.*' 'chunk_[0-9]+'\n"
    "  " << prog << " '/clio-core/.*' --semantic 'plasma temperature' --max 5\n"
    "  " << prog << " '.*' --since 30m\n"
    "  " << prog << " '.*' --tags-only\n";
}

static Args ParseArgs(int argc, char **argv) {
  Args a;
  if (argc < 2) {
    PrintUsage(argv[0]);
    std::exit(1);
  }

  a.tag_re  = argv[1];
  a.blob_re = ".*";

  // Consume argv[2] as blob_re only if it doesn't start with '-'
  int first_flag = 2;
  if (argc >= 3 && argv[2][0] != '-') {
    a.blob_re = argv[2];
    first_flag = 3;
  }

  int mode_count = 0;
  for (int i = first_flag; i < argc; ++i) {
    std::string flag = argv[i];

    auto need_next = [&]() -> const char * {
      if (i + 1 >= argc) {
        std::cerr << "error: " << flag << " requires an argument\n";
        std::exit(1);
      }
      return argv[++i];
    };

    if (flag == "--help" || flag == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else if (flag == "--semantic") {
      a.semantic = need_next();
      ++mode_count;
    } else if (flag == "--since") {
      a.since = need_next();
      ++mode_count;
    } else if (flag == "--time-begin") {
      a.time_begin = std::stoull(need_next());
      ++mode_count;
    } else if (flag == "--time-end") {
      a.time_end = std::stoull(need_next());
    } else if (flag == "--max") {
      a.max_results = static_cast<uint32_t>(std::stoul(need_next()));
    } else if (flag == "--format") {
      a.format = need_next();
      if (a.format != "names" && a.format != "table" && a.format != "json") {
        std::cerr << "error: --format must be names, table, or json\n";
        std::exit(1);
      }
    } else if (flag == "--tags-only") {
      a.tags_only = true;
    } else {
      std::cerr << "error: unknown option: " << flag << "\n";
      std::exit(1);
    }
  }

  if (mode_count > 1) {
    std::cerr << "error: --semantic, --since, and --time-begin are mutually exclusive\n";
    std::exit(1);
  }

  return a;
}

// ---------------------------------------------------------------------------
// Output formatters
// ---------------------------------------------------------------------------

static void PrintTagsOnly(const std::vector<std::string> &tag_names) {
  std::vector<std::string> unique = tag_names;
  std::sort(unique.begin(), unique.end());
  unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
  for (const auto &t : unique) { std::cout << t << "\n"; }
}

static void PrintNamesRegex(const std::vector<std::string> &tag_names,
                            const std::vector<std::string> &blob_names) {
  for (size_t i = 0; i < blob_names.size(); ++i) {
    std::cout << tag_names[i] << "/" << blob_names[i] << "\n";
  }
}

static void PrintTableRegex(const std::vector<std::string> &tag_names,
                            const std::vector<std::string> &blob_names) {
  std::cout << std::left << std::setw(30) << "TAG"
            << std::left << std::setw(50) << "BLOB NAME" << "\n";
  std::cout << std::string(80, '-') << "\n";
  for (size_t i = 0; i < blob_names.size(); ++i) {
    std::cout << std::left << std::setw(30) << tag_names[i]
              << std::left << std::setw(50) << blob_names[i] << "\n";
  }
  std::cout << "\n" << blob_names.size() << " result(s)\n";
}

using namespace clio::cte::core;

static std::string TagStr(const TagId &id) {
  std::ostringstream os;
  os << "(" << id.major_ << "," << id.minor_ << ")";
  return os.str();
}

static void PrintTableSemantic(const std::vector<SemanticSearchResult> &results) {
  std::cout << std::left  << std::setw(6)  << "RANK"
            << std::right << std::setw(10) << "SCORE"
            << "  "
            << std::left  << std::setw(30) << "TAG"
            << std::left  << std::setw(40) << "BLOB NAME"
            << "\n";
  std::cout << std::string(88, '-') << "\n";
  int rank = 1;
  for (const auto &r : results) {
    std::cout << std::left  << std::setw(6)  << rank
              << std::right << std::setw(10) << std::fixed
                            << std::setprecision(6) << r.score_
              << "  "
              << std::left  << std::setw(30) << TagStr(r.tag_id_)
              << std::left  << std::setw(40) << r.blob_name_
              << "\n";
    ++rank;
  }
  std::cout << "\n" << results.size() << " result(s)\n";
}

static void PrintTableTemporal(const std::vector<TemporalSearchResult> &results) {
  std::cout << std::left  << std::setw(22) << "LAST_MODIFIED (ns)"
            << std::left  << std::setw(30) << "TAG"
            << std::left  << std::setw(40) << "BLOB NAME"
            << "\n";
  std::cout << std::string(92, '-') << "\n";
  for (const auto &r : results) {
    std::cout << std::left  << std::setw(22) << r.last_modified_
              << std::left  << std::setw(30) << TagStr(r.tag_id_)
              << std::left  << std::setw(40) << r.blob_name_
              << "\n";
  }
  std::cout << "\n" << results.size() << " result(s)\n";
}

static void PrintJsonRegex(const std::vector<std::string> &tag_names,
                           const std::vector<std::string> &blob_names) {
  std::cout << "[\n";
  for (size_t i = 0; i < blob_names.size(); ++i) {
    std::cout << "  {\"tag_id\": \"" << tag_names[i]
              << "\", \"blob_name\": \"" << blob_names[i] << "\"}"
              << (i + 1 < blob_names.size() ? "," : "") << "\n";
  }
  std::cout << "]\n";
}

static void PrintJsonSemantic(const std::vector<SemanticSearchResult> &results) {
  std::cout << "[\n";
  for (size_t i = 0; i < results.size(); ++i) {
    const auto &r = results[i];
    std::cout << "  {\"tag_id\": \"" << TagStr(r.tag_id_)
              << "\", \"blob_name\": \"" << r.blob_name_ << "\", \"score\": "
              << std::fixed << std::setprecision(6) << r.score_ << "}"
              << (i + 1 < results.size() ? "," : "") << "\n";
  }
  std::cout << "]\n";
}

static void PrintJsonTemporal(const std::vector<TemporalSearchResult> &results) {
  std::cout << "[\n";
  for (size_t i = 0; i < results.size(); ++i) {
    const auto &r = results[i];
    std::cout << "  {\"tag_id\": \"" << TagStr(r.tag_id_)
              << "\", \"blob_name\": \"" << r.blob_name_
              << "\", \"last_modified\": " << r.last_modified_ << "}"
              << (i + 1 < results.size() ? "," : "") << "\n";
  }
  std::cout << "]\n";
}

// ---------------------------------------------------------------------------
// Search dispatch
// ---------------------------------------------------------------------------

enum class Mode { kRegex, kSemantic, kTemporal };

static Mode DetermineMode(const Args &a) {
  if (!a.since.empty() || a.time_begin != 0 || a.time_end != 0) {
    return Mode::kTemporal;
  }
  if (!a.semantic.empty()) { return Mode::kSemantic; }
  return Mode::kRegex;
}

static int RunSearch(clio::cte::core::Client *client, const Args &a) {
  auto pool_query = chi::PoolQuery::Broadcast();
  Mode mode = DetermineMode(a);

  if (mode == Mode::kTemporal) {
    uint64_t time_begin = a.time_begin;
    uint64_t time_end   = a.time_end;
    if (!a.since.empty()) {
      uint64_t duration_ns = ParseDurationNs(a.since);
      time_begin = MonotonicNowNs() - duration_ns;
      time_end   = 0;
    }

    auto task = client->AsyncTemporalSearch(a.tag_re, a.blob_re,
                                            time_begin, time_end,
                                            a.max_results, pool_query);
    task.Wait();
    if (task->results_.empty()) { std::cout << "(no results)\n"; return 0; }

    if (a.tags_only) {
      std::vector<std::string> tag_names;
      for (const auto &r : task->results_) { tag_names.push_back(TagStr(r.tag_id_)); }
      PrintTagsOnly(tag_names);
    } else if (a.format == "json") {
      PrintJsonTemporal(task->results_);
    } else if (a.format == "table") {
      PrintTableTemporal(task->results_);
    } else {
      for (const auto &r : task->results_) {
        std::cout << TagStr(r.tag_id_) << "/" << r.blob_name_ << "\n";
      }
    }
    return 0;
  }

  if (mode == Mode::kSemantic) {
    uint32_t k = a.max_results > 0 ? a.max_results : 10;
    auto task = client->AsyncSemanticSearch(a.tag_re, a.blob_re,
                                            a.semantic, k, pool_query);
    task.Wait();
    if (task->results_.empty()) { std::cout << "(no results)\n"; return 0; }

    if (a.tags_only) {
      std::vector<std::string> tag_names;
      for (const auto &r : task->results_) { tag_names.push_back(TagStr(r.tag_id_)); }
      PrintTagsOnly(tag_names);
    } else if (a.format == "json") {
      PrintJsonSemantic(task->results_);
    } else if (a.format == "table") {
      PrintTableSemantic(task->results_);
    } else {
      for (const auto &r : task->results_) {
        std::cout << TagStr(r.tag_id_) << "/" << r.blob_name_ << "\n";
      }
    }
    return 0;
  }

  // Regex (default)
  auto task = client->AsyncBlobQuery(a.tag_re, a.blob_re,
                                     a.max_results, pool_query);
  task.Wait();
  if (task->blob_names_.empty()) { std::cout << "(no results)\n"; return 0; }

  if (a.tags_only) {
    PrintTagsOnly(task->tag_names_);
  } else if (a.format == "json") {
    PrintJsonRegex(task->tag_names_, task->blob_names_);
  } else if (a.format == "table") {
    PrintTableRegex(task->tag_names_, task->blob_names_);
  } else {
    PrintNamesRegex(task->tag_names_, task->blob_names_);
  }
  return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  Args args = ParseArgs(argc, argv);

  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)) {
    std::cerr << "error: failed to initialize Chimaera runtime\n";
    return 1;
  }
  struct Finalizer {
    ~Finalizer() {
      auto *mgr = CLIO_RUNTIME_MANAGER;
      if (mgr != nullptr) { mgr->ClientFinalize(); }
    }
  } finalizer;

  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::cerr << "error: failed to initialize CTE client\n";
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  return RunSearch(CLIO_CTE_CLIENT, args);
}
