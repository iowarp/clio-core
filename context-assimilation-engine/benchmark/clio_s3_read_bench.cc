/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * S3 read-path benchmark.
 *
 * Times CAE's s3:// -> CTE read path against S3. Reads `--num-objects` objects
 * through the CAE assimilator (ParseOmni -> S3FileAssimilator -> fork+exec
 * cae_s3_tool get -> CTE PutBlob) with a sliding window of `--concurrency`
 * in-flight transfers. The bench reports:
 *   - throughput in the shared clio_bench::PrintResults format
 *   - a second "Fairness" block recording the equivalence caveats needed to
 *     compare against a Zarr-on-S3 read
 *
 * Counterpart driver: jarvis_clio_core/zarr_s3_read_bench/scripts/zarr_s3_read.py,
 * which emits the same two blocks so one parser serves both.
 *
 * Usage:
 *   clio_s3_read_bench \
 *     --bucket B             S3 bucket holding the staged objects (required)
 *     --key-prefix P         key prefix; keys are <prefix>/obj_%06zu.bin
 *                            (required)
 *     --num-objects N        objects to read, before stride/offset (required)
 *     --object-size SIZE     bytes per object, e.g. 32m (required)
 *     --concurrency K        in-flight AsyncParseOmni calls (required, > 0)
 *     [--tag-prefix P]       CTE tag prefix; one tag per object (default s3rb)
 *     [--label L]            results namespace, the CSV "operation"
 *                            (default Read)
 *     [--worker-threads N]   runtime worker count to report (default 0 =
 *                            unknown)
 *     [--object-stride N]    multi-process split: take every Nth key
 *                            (default 1)
 *     [--object-offset N]    multi-process split: first key index (default 0)
 *     [--verify]             re-read tag sizes out of CTE after the run
 *     [--no-preflight]       skip the 1-byte ranged GET before timing starts
 *
 * Three properties of the assimilator shape this benchmark:
 *   1. ParseOmni processes a vector of contexts SERIALLY, so concurrency comes
 *      from issuing K separate AsyncParseOmni calls, one context each.
 *   2. S3FileAssimilator names blobs "chunk_0..chunk_N" relative to the
 *      destination tag, restarting at 0 for every context -- so every object
 *      MUST get its own tag or objects silently overwrite one another.
 *   3. The download is a fork() + blocking waitpid() on a runtime worker
 *      thread, so effective concurrency is capped by the runtime's worker
 *      count, not by K. Sweep runtime threads alongside K and compare
 *      "Requested concurrency" against measured scaling.
 *
 * The AWS SDK is deliberately NOT linked into this process: loading it into a
 * process that runs CLIO_INIT corrupts runtime startup. All S3 I/O happens
 * out-of-process via the cae_s3_tool helper.
 *
 * Environment (read by the RUNTIME process, which forks the helper -- exporting
 * these in the benchmark's own environment is not sufficient when the runtime
 * was started separately):
 *   CAE_S3_TOOL         Path to the cae_s3_tool helper (else PATH lookup).
 *   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN / AWS_PROFILE
 *   AWS_DEFAULT_REGION  Region (default us-east-1).
 *   S3_ENDPOINT / AWS_ENDPOINT_URL   Endpoint override for S3-compatible
 *                       stores; MUST stay unset for real AWS.
 *   TMPDIR              Staging dir for downloaded objects. Peak usage is
 *                       concurrency * object_size.
 */

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <clio_ctp/introspect/system_info.h>
#include <clio_ctp/util/logging.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_cae/core/core_client.h>
#include <clio_cae/core/constants.h>
#include <clio_cae/core/factory/assimilation_ctx.h>
#include <clio_cte/core/core_client.h>

#include "bench_common.h"

namespace {

using clio_bench::u64;

/** Parsed command-line configuration for one benchmark run. */
struct BenchConfig {
  std::string bucket;         /**< S3 bucket holding the staged objects. */
  std::string key_prefix;     /**< Key prefix; keys are <prefix>/obj_%06zu.bin. */
  std::string tag_prefix = "s3rb"; /**< CTE tag prefix; one tag per object. */
  std::string label = "Read"; /**< Results namespace (the CSV "operation"). */
  size_t num_objects = 0;     /**< Objects to read (before stride/offset). */
  u64 object_size = 0;        /**< Bytes per object; used for the results math. */
  size_t concurrency = 1;     /**< In-flight AsyncParseOmni calls (K). */
  size_t stride = 1;          /**< Multi-process fallback: take every Nth key. */
  size_t offset = 0;          /**< Multi-process fallback: first key index. */
  int worker_threads = 0;     /**< Reported runtime worker count (0 = unknown). */
  bool verify = false;        /**< Re-read tag sizes out of CTE after the run. */
  bool preflight = true;      /**< 1-byte ranged GET before timing starts. */
  bool ok = false;            /**< False => caller prints usage and exits. */
};

void PrintUsage(const char* argv0) {
  HLOG(kError, "Usage: {} --bucket B --key-prefix P --num-objects N", argv0);
  HLOG(kError,
       "       --object-size 32m --concurrency K [--tag-prefix s3rb]");
  HLOG(kError,
       "       [--label Read] [--worker-threads N] [--verify] [--no-preflight]");
  HLOG(kError,
       "       [--object-stride N] [--object-offset N]   (multi-process split)");
}

/**
 * Parse argv into a BenchConfig.
 *
 * @param argc Argument count from main.
 * @param argv Argument vector from main.
 * @return Config with ok=true when every required option was supplied and
 *         validated; ok=false (usage already printed) otherwise.
 */
BenchConfig ParseArgs(int argc, char** argv) {
  BenchConfig c;
  auto need = [&](int i) -> const char* {
    if (i >= argc) {
      HLOG(kError, "Missing value for option {}", argv[i - 1]);
      return nullptr;
    }
    return argv[i];
  };
  for (int i = 1; i < argc; ++i) {
    std::string f = argv[i];
    const char* v = nullptr;
    if (f == "--help" || f == "-h") {
      PrintUsage(argv[0]);
      return c;
    } else if (f == "--verify") {
      c.verify = true;
    } else if (f == "--no-preflight") {
      c.preflight = false;
    } else if (f == "--bucket") {
      v = need(++i);
      if (!v) {
        return c;
      }
      c.bucket = v;
    } else if (f == "--key-prefix") {
      v = need(++i);
      if (!v) {
        return c;
      }
      c.key_prefix = v;
    } else if (f == "--tag-prefix") {
      v = need(++i);
      if (!v) {
        return c;
      }
      c.tag_prefix = v;
    } else if (f == "--label") {
      v = need(++i);
      if (!v) {
        return c;
      }
      c.label = v;
    } else if (f == "--num-objects") {
      v = need(++i);
      if (!v) {
        return c;
      }
      c.num_objects = std::stoull(v);
    } else if (f == "--object-size") {
      v = need(++i);
      if (!v) {
        return c;
      }
      c.object_size = clio_bench::ParseSize(v);
    } else if (f == "--concurrency") {
      v = need(++i);
      if (!v) {
        return c;
      }
      c.concurrency = std::stoull(v);
    } else if (f == "--object-stride") {
      v = need(++i);
      if (!v) {
        return c;
      }
      c.stride = std::stoull(v);
    } else if (f == "--object-offset") {
      v = need(++i);
      if (!v) {
        return c;
      }
      c.offset = std::stoull(v);
    } else if (f == "--worker-threads") {
      v = need(++i);
      if (!v) {
        return c;
      }
      c.worker_threads = std::atoi(v);
    } else {
      HLOG(kError, "Unknown option: {}", f);
      PrintUsage(argv[0]);
      return c;
    }
  }
  if (c.bucket.empty() || c.key_prefix.empty() || c.num_objects == 0 ||
      c.object_size == 0 || c.concurrency == 0 || c.stride == 0) {
    HLOG(kError,
         "Missing/invalid required options (--bucket, --key-prefix, "
         "--num-objects, --object-size, --concurrency must all be set > 0)");
    PrintUsage(argv[0]);
    return c;
  }
  c.ok = true;
  return c;
}

/** Resolve the cae_s3_tool helper (CAE_S3_TOOL override, else PATH lookup). */
std::string ResolveS3Tool() {
  const char* override_path = std::getenv("CAE_S3_TOOL");
  if (override_path && *override_path) {
    return override_path;
  }
  return "cae_s3_tool";
}

/** Fork+exec a process and return its exit status (0 = success, -1 = failure). */
int RunProcess(const std::vector<std::string>& args) {
  pid_t pid = fork();
  if (pid == -1) {
    return -1;
  }
  if (pid == 0) {
    std::vector<const char*> argv;
    argv.reserve(args.size() + 1);
    for (const auto& a : args) {
      argv.push_back(a.c_str());
    }
    argv.push_back(nullptr);
    execvp(argv[0], const_cast<char* const*>(argv.data()));
    _exit(127);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

/** Build the S3 key for logical object @p idx (matches the staging script). */
std::string ObjectKey(const BenchConfig& c, size_t idx) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "obj_%06zu.bin", idx);
  return c.key_prefix + "/" + buf;
}

/** The tag this object is assimilated into. One tag PER OBJECT is mandatory:
 *  the assimilator restarts blob naming at chunk_0 for every context, so a
 *  shared tag would make later objects overwrite earlier ones. */
std::string ObjectTag(const BenchConfig& c, size_t idx) {
  return c.tag_prefix + "_" + std::to_string(idx);
}

/**
 * Fetch one byte of the first object to fail fast on bad credentials, a wrong
 * bucket/prefix, or a missing helper -- before the timed loop issues thousands
 * of doomed GETs.
 *
 * @param c Benchmark configuration (bucket, key prefix, first index).
 * @return true when the ranged GET succeeded or preflight was disabled.
 */
bool Preflight(const BenchConfig& c) {
  if (!c.preflight) {
    return true;
  }
  std::string tmp_path = "/tmp/clio_s3_read_bench_preflight_" +
                         std::to_string(static_cast<long>(::getpid())) + ".bin";
  int rc = RunProcess({ResolveS3Tool(), "get", c.bucket,
                       ObjectKey(c, c.offset), tmp_path, "0", "1"});
  ::unlink(tmp_path.c_str());
  if (rc != 0) {
    HLOG(kError,
         "Preflight GET failed (rc={}) for s3://{}/{} -- check credentials, "
         "region, bucket, key prefix, and that cae_s3_tool is on PATH",
         rc, c.bucket, ObjectKey(c, c.offset));
    return false;
  }
  HLOG(kInfo, "Preflight OK: s3://{}/{} reachable", c.bucket,
       ObjectKey(c, c.offset));
  return true;
}

/** Result of the timed read loop, fed to PrintResults and the fairness block. */
struct RunResult {
  std::vector<long long> slot_us;  /**< Cumulative busy microseconds per slot. */
  std::vector<u64> slot_ops;       /**< Objects completed per slot. */
  size_t objects_done = 0;         /**< Total objects assimilated. */
  double wall_us = 0.0;            /**< True end-to-end wall time. */
  int rc = 0;                      /**< Non-zero if any transfer failed. */
};

/**
 * Run the timed read: a sliding window of @p c.concurrency in-flight
 * AsyncParseOmni calls, one object per call, refilled as each completes.
 *
 * Note the round-robin Wait() sweep serializes OBSERVATION, not execution: a
 * slot that finishes early idles until the sweep reaches it, which inflates
 * per-slot times when object latencies are skewed. The aggregate figure to
 * trust is bytes/wall_us, reported separately as "Wire bandwidth".
 *
 * @param c          Benchmark configuration.
 * @param cae_client CAE client bound to the benchmark's CAE pool.
 * @param indices    Logical object indices this process is responsible for.
 * @return Per-slot timings, completion counts, wall time, and a return code.
 */
RunResult RunReadLoop(const BenchConfig& c, clio::cae::core::Client& cae_client,
                      const std::vector<size_t>& indices) {
  struct Slot {
    clio::run::Future<clio::cae::core::ParseOmniTask> fut;
    size_t obj_idx = 0;
    std::chrono::steady_clock::time_point t0;
    bool active = false;
  };

  const size_t total = indices.size();
  const size_t k = std::min(c.concurrency, total);
  RunResult r;
  r.slot_us.assign(k, 0);
  r.slot_ops.assign(k, 0);

  auto make_ctx = [&](size_t obj_idx) {
    clio::cae::core::AssimilationCtx ctx;
    ctx.src = "s3://" + c.bucket + "/" + ObjectKey(c, obj_idx);
    ctx.dst = "iowarp::" + ObjectTag(c, obj_idx);
    ctx.format = "binary";
    ctx.range_off = 0;
    ctx.range_size = 0;  // whole object
    // Forward THIS process's resolved AWS region/profile into the context. The
    // assimilator runs in the runtime daemon (ssh-launched, no inherited AWS_*),
    // so without this cae_s3_tool defaults to us-east-1/anonymous and every GET
    // fails (PermanentRedirect / Access Denied) even though the client's own
    // preflight -- which runs here, with these vars set -- succeeds.
    if (const char* r = std::getenv("AWS_DEFAULT_REGION")) ctx.s3_region = r;
    if (const char* p = std::getenv("AWS_PROFILE")) ctx.s3_profile = p;
    return ctx;
  };

  std::vector<Slot> slots(k);
  size_t next = 0;
  auto wall_t0 = std::chrono::steady_clock::now();

  for (size_t s = 0; s < k; ++s, ++next) {
    std::vector<clio::cae::core::AssimilationCtx> one{make_ctx(indices[next])};
    slots[s].obj_idx = indices[next];
    slots[s].t0 = std::chrono::steady_clock::now();
    slots[s].fut = cae_client.AsyncParseOmni(one);
    slots[s].active = true;
  }

  while (r.objects_done < total) {
    for (size_t s = 0; s < k && r.objects_done < total; ++s) {
      if (!slots[s].active) {
        continue;
      }
      slots[s].fut.Wait();
      auto t1 = std::chrono::steady_clock::now();
      r.slot_us[s] += std::chrono::duration_cast<std::chrono::microseconds>(
                          t1 - slots[s].t0).count();
      r.slot_ops[s] += 1;
      if (slots[s].fut->GetReturnCode() != 0 ||
          slots[s].fut->num_tasks_scheduled_ == 0) {
        HLOG(kError, "ParseOmni failed for object {} (rc={}, scheduled={})",
             slots[s].obj_idx, slots[s].fut->GetReturnCode(),
             slots[s].fut->num_tasks_scheduled_);
        r.rc = 1;
      }
      ++r.objects_done;
      slots[s].active = false;

      if (next < total) {
        std::vector<clio::cae::core::AssimilationCtx> one{
            make_ctx(indices[next])};
        slots[s].obj_idx = indices[next];
        slots[s].t0 = std::chrono::steady_clock::now();
        slots[s].fut = cae_client.AsyncParseOmni(one);
        slots[s].active = true;
        ++next;
      }
    }
  }

  r.wall_us = std::chrono::duration<double, std::micro>(
                  std::chrono::steady_clock::now() - wall_t0).count();
  return r;
}

/**
 * Confirm each object's bytes actually landed in CTE by comparing its tag size
 * against object_size plus the assimilator's "description" metadata blob.
 *
 * @param c       Benchmark configuration (object size, tag prefix).
 * @param indices Logical object indices to check.
 * @return 0 when every tag matched, 1 otherwise.
 */
int VerifyTags(const BenchConfig& c, const std::vector<size_t>& indices) {
  auto cte_client = CLIO_CTE_CLIENT;
  const std::string description =
      "binary<size=" + std::to_string(c.object_size) + ", offset=0>";
  const size_t expected = c.object_size + description.size();
  size_t bad = 0;
  for (size_t idx : indices) {
    auto tag_task = cte_client->AsyncGetOrCreateTag(ObjectTag(c, idx));
    tag_task.Wait();
    clio::cte::core::TagId tag_id = tag_task->tag_id_;
    if (tag_id.IsNull()) {
      HLOG(kError, "Verify: tag missing for object {}", idx);
      ++bad;
      continue;
    }
    auto size_task = cte_client->AsyncGetTagSize(tag_id);
    size_task.Wait();
    if (size_task->tag_size_ != expected) {
      HLOG(kError, "Verify: object {} tag size {} != expected {}", idx,
           size_task->tag_size_, expected);
      ++bad;
    }
  }
  if (bad != 0) {
    HLOG(kError, "Verify: {} of {} objects failed", bad, indices.size());
    return 1;
  }
  HLOG(kSuccess, "Verify: all {} objects present in CTE at expected size",
       indices.size());
  return 0;
}

/**
 * Emit the equivalence-caveat block that accompanies every result row.
 *
 * Every line here is scraped into a results.csv column so a reader can tell
 * what each stack actually did: how many bytes crossed the wire, how many GETs
 * it took, whether anything was compressed or decoded, and what CLIO-specific
 * overheads (per-object subprocess spawn, full temp-file staging) were paid.
 *
 * @param c Benchmark configuration.
 * @param r Timed-loop results.
 */
void PrintFairness(const BenchConfig& c, const RunResult& r) {
  const u64 bytes_moved = static_cast<u64>(r.objects_done) * c.object_size;
  HLOG(kInfo, "");
  HLOG(kInfo, "=== {} Fairness ===", c.label);
  HLOG(kInfo, "Objects read: {}", r.objects_done);
  HLOG(kInfo, "Bytes moved: {}", bytes_moved);
  HLOG(kInfo, "Logical bytes: {}", bytes_moved);
  // cae_s3_tool issues exactly one GetObject per transfer.
  HLOG(kInfo, "GET count: {}", r.objects_done);
  HLOG(kInfo, "Compression: none");
  HLOG(kInfo, "Decode step: no");
  HLOG(kInfo, "Requested concurrency: {}", c.concurrency);
  HLOG(kInfo, "Effective concurrency: {}", r.slot_ops.size());
  HLOG(kInfo, "Runtime worker threads: {}", c.worker_threads);
  HLOG(kInfo, "Wall time us: {} ({} ms)", r.wall_us, r.wall_us / 1000.0);
  HLOG(kInfo, "Wire bandwidth: {} MB/s",
       clio_bench::CalcBandwidth(bytes_moved, r.wall_us));
  // One fork+exec of cae_s3_tool per object, and each object is staged whole
  // through TMPDIR before it reaches CTE. Zarr pays neither cost.
  HLOG(kInfo, "Subprocess spawns: {}", r.objects_done);
  HLOG(kInfo, "Temp file bytes: {}", bytes_moved);
  // kMaxChunkSize in S3FileAssimilator::Schedule -- a hard-coded constexpr,
  // not tunable from here.
  HLOG(kInfo, "Transport chunk bytes: 1048576");
  HLOG(kInfo, "===================");
}

}  // namespace

int main(int argc, char** argv) {
  BenchConfig c = ParseArgs(argc, argv);
  if (!c.ok) {
    return 1;
  }

  // Objects this process owns. stride/offset let M processes partition the key
  // space disjointly when a single process cannot saturate the link.
  std::vector<size_t> indices;
  for (size_t i = c.offset; i < c.num_objects; i += c.stride) {
    indices.push_back(i);
  }
  if (indices.empty()) {
    HLOG(kError, "No objects selected (num_objects={}, stride={}, offset={})",
         c.num_objects, c.stride, c.offset);
    return 1;
  }

  HLOG(kInfo, "clio_s3_read_bench: s3://{}/{} x {} objects of {}, K={}",
       c.bucket, c.key_prefix, indices.size(),
       clio_bench::FormatSize(c.object_size), c.concurrency);

  if (!Preflight(c)) {
    return 1;
  }

  // Attach to the already-running clio_run daemon as a pure client. The
  // second arg (default_with_runtime) MUST be false here: the jarvis pipeline
  // starts the runtime via the clio_runtime package, so bootstrapping a second
  // runtime in-process collides on port 9413 ("Address already in use") and
  // aborts the row. (test_s3_assim.cc passes true because a unit test has no
  // external daemon and must start its own -- the wrong precedent to copy for
  // a pipeline client; clio_cte_bench.cc, which runs in this same pipeline
  // shape, correctly uses false.) The AWS SDK is never linked into this
  // process; all S3 I/O happens out-of-process via cae_s3_tool.
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    HLOG(kError, "Failed to initialize Clio");
    return 1;
  }
  clio::cte::core::CLIO_CTE_CLIENT_INIT();
  CLIO_CAE_CLIENT_INIT();
  clio::cae::core::Client cae_client;
  {
    clio::cae::core::CreateParams params;
    auto create_task =
        cae_client.AsyncCreate(clio::run::PoolQuery::Local(),
                               "s3_bench_cae_pool",
                               clio::cae::core::kCaePoolId, params);
    create_task.Wait();
  }

  int exit_code = 0;
  try {
    RunResult r = RunReadLoop(c, cae_client, indices);
    exit_code = r.rc;

    clio_bench::BenchArgs a;
    a.test_case = c.label;
    a.threads = r.slot_ops.size();  // one "thread" per in-flight slot
    a.depth = 1;
    a.io_size = c.object_size;
    a.io_count = static_cast<int>(indices.size());
    clio_bench::PrintResults(c.label, a, r.slot_us, r.slot_ops);
    PrintFairness(c, r);

    if (c.verify && exit_code == 0) {
      exit_code = VerifyTags(c, indices);
    }
  } catch (const std::exception& e) {
    HLOG(kError, "Exception: {}", e.what());
    exit_code = 1;
  }

  // TerminateProcessNow avoids the client-teardown hang that would otherwise
  // let the pipeline's run_timeout consume this row (see test_s3_assim.cc).
  ctp::SystemInfo::TerminateProcessNow(exit_code);
  return exit_code;
}
