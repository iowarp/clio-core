/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * CAE transparent-labeling benchmark.
 *
 * Stresses the cost of LLM-driven labeling on the PutBlob fast path.
 * Each worker thread issues PutBlob calls of `--blob-size` bytes of
 * synthetic English text against a CAE pool whose label_matches rule
 * triggers labeling on every blob. The CAE container forwards the
 * blob to CTE for storage, then calls Ollama (`--model`, `--endpoint`)
 * to produce a `{blob_name}_label` blob in the same tag.
 *
 * After the run we walk every blob in every tag and ask CTE for the
 * label-blob sizes; the bench reports:
 *   - aggregate input I/O bandwidth (input bytes / wall time)
 *   - per-thread + aggregate ops/sec
 *   - average + total label blob size in bytes
 *
 * Usage:
 *   clio_cae_label_bench \
 *     [--threads N]          worker threads (default 4)
 *     [--blob-size SIZE]     bytes per input blob (default 4k)
 *     [--max-blobs N]        unique input blobs (default = threads*io-count;
 *                            keyspace is split across threads so each
 *                            thread cycles max_blobs/threads keys)
 *     [--summary-tokens N]   max output tokens per summary (Ollama
 *                            num_predict). 0 = no cap. Default 64.
 *     [--io-count N]         PutBlob ops per thread (default 8)
 *     [--context-length N]   Ollama num_ctx tokens (default 4096)
 *     [--model NAME]         Ollama model (default gemma3:1b)
 *     [--endpoint URL]       Ollama base URL (default http://127.0.0.1:11434)
 *
 * The bench writes a temporary compose YAML at /tmp so the
 * label_matches rule reflects --summary-tokens / --context-length / etc.
 * Make sure an Ollama instance with `--model` already pulled is
 * reachable at `--endpoint`.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_ctp/util/logging.h>

using namespace std::chrono;

namespace {

struct Args {
  size_t threads = 4;
  size_t blob_size = 4 * 1024;       // 4 KB default
  long max_blobs = 0;                // 0 → derive from threads*io_count
  int summary_tokens = 64;
  long io_count = 8;
  int context_length = 4096;
  std::string model = "gemma3:1b";
  std::string endpoint = "http://127.0.0.1:11434";
  bool ok = true;
};

void PrintUsage() {
  std::cerr <<
    "Usage: clio_cae_label_bench [options]\n"
    "  --threads N            worker threads (default 4)\n"
    "  --blob-size SIZE       bytes per input blob; accepts k/m suffix\n"
    "  --max-blobs N          unique input blobs across all threads\n"
    "  --summary-tokens N     Ollama num_predict (default 64; 0=no cap)\n"
    "  --io-count N           PutBlob ops per thread (default 8)\n"
    "  --context-length N     Ollama num_ctx (default 4096)\n"
    "  --model NAME           Ollama model (default gemma3:1b)\n"
    "  --endpoint URL         Ollama base URL (default http://127.0.0.1:11434)\n";
}

size_t ParseSize(const std::string &s) {
  if (s.empty()) return 0;
  size_t multi = 1;
  char suf = s.back();
  std::string num = s;
  if (suf == 'k' || suf == 'K') { multi = 1024;                 num.pop_back(); }
  else if (suf == 'm' || suf == 'M') { multi = 1024 * 1024;     num.pop_back(); }
  else if (suf == 'g' || suf == 'G') { multi = 1024 * 1024 * 1024; num.pop_back(); }
  return static_cast<size_t>(std::stoull(num)) * multi;
}

Args ParseArgs(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        a.ok = false;
        return "";
      }
      return argv[++i];
    };
    if (k == "--threads") a.threads = std::stoul(next("--threads"));
    else if (k == "--blob-size") a.blob_size = ParseSize(next("--blob-size"));
    else if (k == "--max-blobs") a.max_blobs = std::stol(next("--max-blobs"));
    else if (k == "--summary-tokens") a.summary_tokens = std::stoi(next("--summary-tokens"));
    else if (k == "--io-count") a.io_count = std::stol(next("--io-count"));
    else if (k == "--context-length") a.context_length = std::stoi(next("--context-length"));
    else if (k == "--model") a.model = next("--model");
    else if (k == "--endpoint") a.endpoint = next("--endpoint");
    else if (k == "--help" || k == "-h") { PrintUsage(); a.ok = false; return a; }
    else { std::cerr << "unknown arg: " << k << "\n"; PrintUsage(); a.ok = false; return a; }
  }
  if (a.max_blobs == 0) a.max_blobs = static_cast<long>(a.threads) * a.io_count;
  return a;
}

/**
 * Write the temporary compose YAML used by clio. CAE sits at 512.0
 * with a single labeling rule whose `model`, `context_length`, and
 * `num_predict` reflect the CLI args. CTE core is behind it at 513.0.
 */
std::string WriteTempConfig(const Args &a) {
  char tmpl[] = "/tmp/cae_label_bench_config.XXXXXX.yaml";
  int fd = mkstemps(tmpl, 5);  // 5 = len(".yaml")
  if (fd < 0) {
    HLOG(kError, "mkstemps failed for {}", tmpl);
    std::exit(1);
  }
  ::close(fd);

  std::ofstream f(tmpl);
  f << "memory:\n"
    << "  main_segment_size: \"8GB\"\n"
    << "runtime:\n"
    << "  num_threads: 8\n"
    << "  queue_depth: 1024\n"
    << "compose:\n"
    << "  - mod_name: clio_bdev\n"
    << "    pool_name: \"ram::chi_default_bdev\"\n"
    << "    pool_query: local\n"
    << "    pool_id: \"301.0\"\n"
    << "    bdev_type: ram\n"
    << "    capacity: \"4GB\"\n"
    << "  - mod_name: clio_cae_core\n"
    << "    pool_name: cae_main\n"
    << "    pool_query: local\n"
    << "    pool_id: \"512.0\"\n"
    << "    next_pool_id: \"513.0\"\n"
    << "    label_endpoint: \"" << a.endpoint << "\"\n"
    << "    label_prompts:\n"
    << "      summarize: \"Summarize the following text in one short sentence.\"\n"
    << "    label_matches:\n"
    << "      - tag_re: \".*\"\n"
    << "        blob_re: \".*\"\n"
    << "        model: \"" << a.model << "\"\n"
    << "        prompt: \"summarize\"\n"
    << "        context_length: " << a.context_length << "\n"
    << "        num_predict: " << a.summary_tokens << "\n"
    << "  - mod_name: clio_cte_core\n"
    << "    pool_name: cte_core\n"
    << "    pool_query: local\n"
    << "    pool_id: \"513.0\"\n"
    << "    targets:\n"
    << "      neighborhood: 1\n"
    << "    storage:\n"
    << "      - path: \"ram::cte_ram_tier1\"\n"
    << "        bdev_type: \"ram\"\n"
    << "        capacity_limit: \"2GB\"\n"
    << "        score: 1.0\n"
    << "    dpe:\n"
    << "      dpe_type: \"max_bw\"\n";
  f.close();
  return tmpl;
}

/**
 * Build a synthetic English-ish text payload of exactly `size` bytes
 * by cycling a fixed paragraph. Real text gives the LLM something
 * coherent to summarize, while keeping the bench reproducible.
 */
std::string MakePayload(size_t size) {
  static const std::string seed =
      "The quick brown fox jumps over the lazy dog. Apollo 11 landed on "
      "the Moon on July 20, 1969. The Wright brothers achieved the first "
      "controlled, sustained flight of a powered, heavier-than-air "
      "aircraft on December 17, 1903. The fall of the Berlin Wall on "
      "November 9, 1989 marked the symbolic end of the Cold War. ";
  std::string out;
  out.reserve(size);
  while (out.size() < size) {
    size_t remaining = size - out.size();
    out.append(seed, 0, std::min(remaining, seed.size()));
  }
  return out;
}

}  // namespace

class LabelBench {
 public:
  explicit LabelBench(Args a) : a_(std::move(a)) {}

  bool Run() {
    PrintInfo();
    auto payload = MakePayload(a_.blob_size);

    // The default CLIO_CTE_CLIENT is pointed at the CAE entrypoint
    // (pool 512.0) — that's the whole point of the bench. But CAE
    // doesn't forward every CTE method, in particular GetBlobSize.
    // For the post-run label inspection we need a second CTE client
    // that talks to the *real* CTE core at 513.0 directly. Both
    // clients share the same clio runtime / IPC.
    cte_direct_ = std::make_unique<clio::cte::core::Client>(
        clio::run::PoolId(513, 0));

    // Per-thread results.
    std::vector<long long> times_us(a_.threads);
    std::vector<long> ops(a_.threads);
    std::vector<long long> label_bytes(a_.threads);  // sum over the thread's keyspace
    std::vector<long> label_blobs_seen(a_.threads);

    long per_thread_keys = a_.max_blobs / static_cast<long>(a_.threads);
    if (per_thread_keys < 1) per_thread_keys = 1;

    auto wall_start = steady_clock::now();
    std::vector<std::thread> ts;
    for (size_t t = 0; t < a_.threads; ++t) {
      ts.emplace_back([&, t]() {
        Worker(t, per_thread_keys, payload, times_us[t], ops[t],
               label_bytes[t], label_blobs_seen[t]);
      });
    }
    for (auto &th : ts) th.join();
    auto wall_us =
        duration_cast<microseconds>(steady_clock::now() - wall_start).count();

    PrintResults(wall_us, times_us, ops, label_bytes, label_blobs_seen);
    return true;
  }

 private:
  /**
   * One thread's workload: GetOrCreateTag, then `io_count` PutBlob ops
   * cycling through `per_thread_keys` unique blob names. After PutBlobs,
   * query AsyncGetBlobSize on each `{name}_label` to accumulate the
   * label-side stats. The label-size query is *not* included in the
   * thread's I/O time — we measure PutBlob throughput separately from
   * the label inspection.
   */
  void Worker(size_t tid, long per_thread_keys, const std::string &payload,
              long long &time_us, long &ops_out, long long &label_bytes_out,
              long &label_blobs_out) {
    auto *cte = CLIO_CTE_CLIENT;
    std::string tag_name = "label_bench_t" + std::to_string(tid);
    auto tag_task = cte->AsyncGetOrCreateTag(tag_name);
    tag_task.Wait();
    if (tag_task->GetReturnCode() != 0) {
      HLOG(kError, "[t{}] tag create rc={}", tid, tag_task->GetReturnCode());
      return;
    }
    auto tag_id = tag_task->tag_id_;

    // SHM buffer reused across all PutBlobs from this thread.
    auto buf = CLIO_IPC->AllocateBuffer(a_.blob_size);
    if (buf.IsNull()) {
      HLOG(kError, "[t{}] AllocateBuffer({}) failed", tid, a_.blob_size);
      return;
    }
    std::memcpy(buf.ptr_, payload.data(), a_.blob_size);
    ctp::ipc::ShmPtr<> shm = buf.shm_.template Cast<void>();

    auto blob_name = [&](long k) {
      return "blob_t" + std::to_string(tid) + "_" + std::to_string(k);
    };

    // PutBlob loop. Synchronous Wait() per op — labeling is sync
    // inside the CAE handler, so depth>1 doesn't actually overlap
    // labeling work. Sequential is the honest measure here.
    auto t0 = steady_clock::now();
    long ok_ops = 0;
    for (long i = 0; i < a_.io_count; ++i) {
      long key = i % per_thread_keys;
      clio::cte::core::Context ctx;
      auto put = cte->AsyncPutBlob(
          tag_id, blob_name(key), 0, a_.blob_size, shm, 0.5f, ctx, 0);
      put.Wait();
      if (put->GetReturnCode() != 0) {
        HLOG(kError, "[t{}] PutBlob({}) rc={}", tid, key,
             put->GetReturnCode());
        continue;
      }
      ++ok_ops;
    }
    time_us = duration_cast<microseconds>(steady_clock::now() - t0).count();
    ops_out = ok_ops;
    CLIO_IPC->FreeBuffer(buf);

    // Label inspection. Each unique key has a `{name}_label` written
    // by the CAE handler. Some may be empty if labeling failed (e.g.
    // model unloaded mid-run); we count those separately so the
    // average is over present-and-non-empty labels only.
    //
    // CAE doesn't forward GetBlobSize today, so we query CTE directly
    // (at pool 513.0) instead of going through CAE.
    long long bytes = 0;
    long present = 0;
    for (long k = 0; k < per_thread_keys; ++k) {
      std::string lname = blob_name(k) + "_label";
      auto sz = cte_direct_->AsyncGetBlobSize(tag_id, lname);
      sz.Wait();
      if (sz->GetReturnCode() == 0 && sz->size_ > 0) {
        bytes += static_cast<long long>(sz->size_);
        ++present;
      }
    }
    label_bytes_out = bytes;
    label_blobs_out = present;
  }

  void PrintInfo() {
    HLOG(kInfo, "=== CAE Labeling Benchmark ===");
    HLOG(kInfo, "Threads: {}  blob-size: {} B  max-blobs: {}  io-count/thread: {}",
         a_.threads, a_.blob_size, a_.max_blobs, a_.io_count);
    HLOG(kInfo, "Model: {}  endpoint: {}  context_length: {}  summary_tokens: {}",
         a_.model, a_.endpoint, a_.context_length, a_.summary_tokens);
    HLOG(kInfo, "==============================");
  }

  // Project's HLOG formatter only handles `{}` (no `{:.3f}`-style
  // specifiers — see ctp::Formatter::tokenize). Pre-format any doubles
  // into strings and pass as `{}`.
  static std::string Fmt(double v, int prec) {
    std::ostringstream s;
    s.setf(std::ios::fixed);
    s.precision(prec);
    s << v;
    return s.str();
  }

  void PrintResults(long long wall_us,
                    const std::vector<long long> &times_us,
                    const std::vector<long> &ops,
                    const std::vector<long long> &label_bytes,
                    const std::vector<long> &label_blobs) {
    long total_ops = 0;
    long long total_label_bytes = 0;
    long total_label_blobs = 0;
    long long min_us = times_us.empty() ? 0 : times_us[0];
    long long max_us = 0;
    long long sum_us = 0;
    for (size_t t = 0; t < times_us.size(); ++t) {
      total_ops += ops[t];
      total_label_bytes += label_bytes[t];
      total_label_blobs += label_blobs[t];
      min_us = std::min(min_us, times_us[t]);
      max_us = std::max(max_us, times_us[t]);
      sum_us += times_us[t];
    }
    double wall_s = static_cast<double>(wall_us) / 1.0e6;
    double total_input_bytes =
        static_cast<double>(total_ops) * static_cast<double>(a_.blob_size);
    double input_mbs = (wall_s > 0)
        ? (total_input_bytes / wall_s) / (1024.0 * 1024.0)
        : 0.0;
    double ops_per_s = (wall_s > 0) ? total_ops / wall_s : 0.0;
    double avg_label_bytes = (total_label_blobs > 0)
        ? static_cast<double>(total_label_bytes) /
              static_cast<double>(total_label_blobs)
        : 0.0;
    double avg_us = times_us.empty() ? 0.0
        : static_cast<double>(sum_us) / static_cast<double>(times_us.size());

    HLOG(kInfo, "");
    HLOG(kInfo, "=== CAE Labeling Results ===");
    HLOG(kInfo, "Total PutBlobs (= labels requested): {}", total_ops);
    HLOG(kInfo, "Wall time: {} s   per-thread time min/max/avg: {}/{}/{} s",
         Fmt(wall_s, 3), Fmt(min_us / 1.0e6, 3), Fmt(max_us / 1.0e6, 3),
         Fmt(avg_us / 1.0e6, 3));
    HLOG(kInfo, "Input throughput: {} MB/s  ({} ops/s)",
         Fmt(input_mbs, 2), Fmt(ops_per_s, 2));
    HLOG(kInfo, "Label blobs observed: {} / {} unique keys",
         total_label_blobs, a_.max_blobs);
    HLOG(kInfo, "Label size - total: {} B   average: {} B",
         total_label_bytes, Fmt(avg_label_bytes, 1));
    HLOG(kInfo, "============================");
  }

  Args a_;
  std::unique_ptr<clio::cte::core::Client> cte_direct_;
};

int main(int argc, char **argv) {
  Args args = ParseArgs(argc, argv);
  if (!args.ok) return 1;

  // Write a tmp YAML that reflects --model / --endpoint / --summary-tokens /
  // --context-length, then have clio consume it via CLIO_SERVER_CONF.
  std::string config_path = WriteTempConfig(args);
  setenv("CLIO_SERVER_CONF", config_path.c_str(), 1);
  HLOG(kInfo, "Config: {}", config_path);

  HLOG(kInfo, "Initializing Clio runtime (kServer)...");
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer)) {
    HLOG(kError, "CLIO_INIT failed");
    return 1;
  }
  struct FinalizeGuard {
    ~FinalizeGuard() { clio::run::CLIO_RUNTIME_FINALIZE(); }
  } guard;
  std::this_thread::sleep_for(std::chrono::seconds(1));

  // Point the global CTE client at the CAE entrypoint (512.0). Since
  // the CAE pool already exists by ID from compose, AsyncCreate
  // returns 512.0 and the bench's CTE client lands on CAE.
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    HLOG(kError, "CLIO_CTE_CLIENT_INIT failed");
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  LabelBench bench(args);
  bool ok = bench.Run();
  // Clean up temp config (best-effort; OS will clean /tmp regardless).
  std::remove(config_path.c_str());
  return ok ? 0 : 1;
}
