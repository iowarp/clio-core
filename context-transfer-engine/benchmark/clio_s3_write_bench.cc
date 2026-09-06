/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * clio_s3_write_bench.cc - Times CLIO's CTE -> S3 bdev write path.
 *
 * Writes N blobs into CTE with a sliding window of K in-flight AsyncPutBlob
 * calls. CTE places them on an S3-backed bdev tier, whose WriteBlocks issues
 * signed PUTs from the runtime daemon. Reports throughput in the shared
 * clio_bench::PrintResults format plus a "Fairness" block recording the
 * equivalence caveats needed to compare against a Zarr-on-S3 write.
 *
 * Counterpart drivers, both emitting the same two blocks so one parser serves
 * all three:
 *   jarvis_clio_core/zarr_s3_write_bench/scripts/zarr_s3_write.py  (baseline)
 *   jarvis_clio_core/s3_raw_put_bench/scripts/s3_raw_put.py        (wire floor)
 *
 * Three properties of the write path shape this benchmark:
 *   1. The S3 PUT blocks a RUNTIME WORKER, not this client thread. So
 *      effective concurrency is capped by the runtime's worker count, not by
 *      K -- exactly as the read bench's fork+waitpid was. Sweep
 *      runtime.num_threads alongside K and compare against "Requested
 *      concurrency" before concluding anything about scaling.
 *   2. CTE must have the S3 tier as its ONLY target, or the DPE is free to
 *      place blobs on a faster local tier and the benchmark silently measures
 *      RAM. The pipeline config enforces this by configuring exactly one
 *      device; this process has no way to assert it, so read the fairness
 *      block's measured object count to confirm bytes reached the bucket.
 *   3. ONE PutBlob BECOMES ONE S3 OBJECT in the common case. The bdev does
 *      not chop a blob into fixed-size blocks: AllocateFromTarget passes the
 *      whole request to the allocator and WriteBlocks issues one PutObject
 *      per returned block (s3_bdev_transport.cc), and the allocator satisfies
 *      an unfragmented request with a single block. So "PUT count" here is
 *      blobs_done, not ceil(blob_size / some_block_size). Under heap
 *      fragmentation the allocator can return several blocks and the real
 *      count rises; that is why the jarvis package also LISTS the bucket
 *      prefix and reports objects_measured. Trust that column over this one.
 *
 * Neither the AWS SDK nor Poco is linked into this process: all S3 work
 * happens in the runtime daemon. See the sigv4.h/s3_rest.h transport.
 *
 * Environment (read by the RUNTIME process, not by this client -- exporting
 * these here is not sufficient when the runtime was started separately):
 *   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN
 *   AWS_DEFAULT_REGION  Region (default us-east-1). SigV4 is region-scoped:
 *                       a mismatch is an HTTP 301, not a 403.
 *   S3_ENDPOINT         Endpoint override (switches to path-style addressing)
 *                       for S3-compatible stores; MUST stay unset for real AWS.
 */

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
#include <clio_cte/core/core_client.h>

#include "bench_common.h"

namespace {

using clio_bench::u64;

/** Parsed command-line configuration for one benchmark run. */
struct BenchConfig {
  std::string tag_prefix = "s3wb";  /**< CTE tag prefix. */
  std::string label = "Write";      /**< Results namespace (CSV "operation"). */
  size_t num_blobs = 0;             /**< Blobs to write. */
  u64 blob_size = 0;                /**< Bytes per blob. */
  size_t concurrency = 1;           /**< In-flight AsyncPutBlob calls (K). */
  int worker_threads = 0;           /**< Reported runtime worker count. */
  bool verify = false;              /**< Read blobs back and compare bytes. */
  bool ok = false;                  /**< False => usage printed, exit. */
};

/**
 * Build stamp, greppable in the installed binary.
 *
 * Spack branch versions do not rehash when the branch moves, so `spack find`
 * happily reports a months-old build under the right name and `spack install`
 * skips the compile. The pipeline therefore greps this literal out of the
 * binary rather than trusting the spec. BUMP IT whenever a change here alters
 * what the pipeline expects to read back, and bump the matching literal in
 * pipelines/ares/clio_s3_write.yaml in the same commit.
 */
constexpr const char* kBuildMarker = "s3wb-build-marker-2026-08-26";

void PrintUsage(const char* argv0) {
  HLOG(kError, "Usage: {} --num-blobs N --blob-size 4m --concurrency K",
       argv0);
  HLOG(kError,
       "       [--tag-prefix s3wb] [--label Write] [--worker-threads N]");
  HLOG(kError, "       [--verify]");
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
    } else if (f == "--tag-prefix") {
      v = need(++i);
      if (!v) return c;
      c.tag_prefix = v;
    } else if (f == "--label") {
      v = need(++i);
      if (!v) return c;
      c.label = v;
    } else if (f == "--num-blobs") {
      v = need(++i);
      if (!v) return c;
      c.num_blobs = std::stoull(v);
    } else if (f == "--blob-size") {
      v = need(++i);
      if (!v) return c;
      c.blob_size = clio_bench::ParseSize(v);
    } else if (f == "--concurrency") {
      v = need(++i);
      if (!v) return c;
      c.concurrency = std::stoull(v);
    } else if (f == "--worker-threads") {
      v = need(++i);
      if (!v) return c;
      c.worker_threads = std::stoi(v);
    } else {
      HLOG(kError, "Unknown option: {}", f);
      PrintUsage(argv[0]);
      return c;
    }
  }
  if (c.num_blobs == 0 || c.blob_size == 0) {
    HLOG(kError, "--num-blobs and --blob-size are required and must be > 0");
    PrintUsage(argv[0]);
    return c;
  }
  if (c.concurrency == 0) {
    HLOG(kError, "--concurrency must be > 0");
    return c;
  }
  c.ok = true;
  return c;
}

/** The blob name for logical index @p idx. */
std::string BlobName(size_t idx) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "blob_%06zu", idx);
  return std::string(buf);
}

/**
 * Fill @p buf with a byte pattern derived from @p idx, so --verify can catch
 * a blob that round-tripped as the WRONG blob -- which a constant fill, or a
 * fill that ignores the index, would silently pass.
 *
 * @param buf  Destination buffer.
 * @param len  Bytes to fill.
 * @param idx  Logical blob index, mixed into the pattern.
 */
void FillPattern(char* buf, size_t len, size_t idx) {
  const u64 seed = static_cast<u64>(idx) * 2654435761ULL + 1ULL;
  for (size_t i = 0; i < len; ++i) {
    buf[i] = static_cast<char>((seed + i * 31ULL + (i >> 8)) & 0xFF);
  }
}

/** Result of the timed write loop, fed to PrintResults and the fairness block. */
struct RunResult {
  std::vector<long long> slot_us;  /**< Cumulative busy microseconds per slot. */
  std::vector<u64> slot_ops;       /**< Blobs completed per slot. */
  size_t blobs_done = 0;           /**< Total blobs written. */
  double wall_us = 0.0;            /**< True end-to-end wall time. */
  int rc = 0;                      /**< Non-zero if any put failed. */
  size_t blobs_failed = 0;         /**< Blobs whose PutBlob returned non-zero. */
};

/**
 * Run the timed write: a sliding window of @p c.concurrency in-flight
 * AsyncPutBlob calls, refilled as each completes.
 *
 * Note the round-robin Wait() sweep serializes OBSERVATION, not execution: a
 * slot that finishes early idles until the sweep reaches it, which inflates
 * per-slot times when PUT latencies are skewed. The aggregate figure to trust
 * is bytes/wall_us, reported separately as "Wire bandwidth".
 *
 * @param c      Benchmark configuration.
 * @param tag_id CTE tag every blob is written under.
 * @return Per-slot timings, completion counts, wall time, and a return code.
 */
RunResult RunWriteLoop(const BenchConfig& c,
                       const clio::cte::core::TagId& tag_id) {
  struct Slot {
    clio::run::Future<clio::cte::core::PutBlobTask> fut;
    size_t blob_idx = 0;
    std::chrono::steady_clock::time_point t0;
    bool active = false;
  };

  auto* cte = CLIO_CTE_CLIENT;
  const size_t total = c.num_blobs;
  const size_t k = std::min(c.concurrency, total);
  RunResult r;
  r.slot_us.assign(k, 0);
  r.slot_ops.assign(k, 0);

  // One SHM buffer PER SLOT. A single shared buffer would force a Wait()
  // before each new put (concurrent puts would race on the source region),
  // silently serializing writes and making --concurrency a no-op -- the same
  // trap clio_cte_bench documents on its Get path.
  std::vector<ctp::ipc::FullPtr<char>> bufs;
  std::vector<ctp::ipc::ShmPtr<>> ptrs;
  bufs.reserve(k);
  ptrs.reserve(k);
  for (size_t s = 0; s < k; ++s) {
    auto b = CLIO_IPC->AllocateBuffer(c.blob_size);
    bufs.push_back(b);
    ptrs.push_back(b.shm_.template Cast<void>());
  }

  std::vector<Slot> slots(k);
  size_t next = 0;

  auto issue = [&](size_t s, size_t idx) {
    FillPattern(bufs[s].ptr_, c.blob_size, idx);
    slots[s].blob_idx = idx;
    slots[s].t0 = std::chrono::steady_clock::now();
    // score 0.8 matches clio_cte_bench; with a single S3 target configured
    // the DPE has nowhere else to place the blob regardless.
    slots[s].fut = cte->AsyncPutBlob(tag_id, BlobName(idx), 0, c.blob_size,
                                     ptrs[s], 0.8f,
                                     clio::cte::core::Context(), 0);
    slots[s].active = true;
  };

  auto wall_t0 = std::chrono::steady_clock::now();
  for (size_t s = 0; s < k; ++s, ++next) {
    issue(s, next);
  }

  while (r.blobs_done < total) {
    for (size_t s = 0; s < k && r.blobs_done < total; ++s) {
      if (!slots[s].active) continue;
      slots[s].fut.Wait();
      auto t1 = std::chrono::steady_clock::now();
      r.slot_us[s] += std::chrono::duration_cast<std::chrono::microseconds>(
                          t1 - slots[s].t0).count();
      r.slot_ops[s] += 1;
      const int put_rc = slots[s].fut->return_code_.load();
      if (put_rc != 0) {
        // Log the FIRST failure only. When placement is broken every blob
        // fails identically, and N identical lines bury the one fact that
        // matters. rc 11..19 is CTE's `10 + alloc_result` from PlaceBlobBytes
        // (core_runtime.cc), i.e. no target could take the bytes -- almost
        // always because the S3 tier was rejected at config parse time and
        // CTE came up with zero storage devices. Say so, rather than making
        // the reader correlate this against the runtime log.
        if (r.blobs_failed == 0) {
          HLOG(kError, "PutBlob failed for blob {} (rc={})", slots[s].blob_idx,
               put_rc);
          if (put_rc >= 10 && put_rc < 20) {
            HLOG(kError,
                 "  rc {} is a CTE placement failure (alloc_result={}). Check "
                 "the runtime log for \"Invalid bdev_type 's3'\" followed by "
                 "\"No storage devices configured\": that means the runtime "
                 "predates the s3/gcs bdev_type allowlist, so the S3 tier was "
                 "dropped and there is nowhere to put the blob.",
                 put_rc, put_rc - 10);
          }
          HLOG(kError, "  suppressing further per-blob errors");
        }
        ++r.blobs_failed;
        r.rc = 1;
      }
      ++r.blobs_done;
      slots[s].active = false;
      if (next < total) {
        issue(s, next);
        ++next;
      }
    }
  }

  r.wall_us = std::chrono::duration<double, std::micro>(
                  std::chrono::steady_clock::now() - wall_t0).count();
  for (auto& b : bufs) {
    CLIO_IPC->FreeBuffer(b);
  }
  return r;
}

/**
 * Read every blob back through CTE and compare it byte-for-byte against the
 * pattern that was written.
 *
 * This is what proves bytes actually round-tripped through S3 rather than
 * being served from a CTE cache tier -- a size check alone would not.
 *
 * @param c      Benchmark configuration.
 * @param tag_id CTE tag the blobs were written under.
 * @return 0 when every blob matched, 1 otherwise.
 */
int VerifyBlobs(const BenchConfig& c, const clio::cte::core::TagId& tag_id) {
  auto* cte = CLIO_CTE_CLIENT;
  auto got = CLIO_IPC->AllocateBuffer(c.blob_size);
  std::vector<char> expect(c.blob_size);
  ctp::ipc::ShmPtr<> got_ptr = got.shm_.template Cast<void>();
  size_t bad = 0;
  for (size_t idx = 0; idx < c.num_blobs; ++idx) {
    std::memset(got.ptr_, 0, c.blob_size);
    auto t = cte->AsyncGetBlob(tag_id, BlobName(idx), 0, c.blob_size, 0,
                               got_ptr);
    t.Wait();
    if (t->return_code_.load() != 0) {
      HLOG(kError, "Verify: GetBlob rc={} for blob {}",
           t->return_code_.load(), idx);
      ++bad;
      continue;
    }
    FillPattern(expect.data(), c.blob_size, idx);
    if (std::memcmp(got.ptr_, expect.data(), c.blob_size) != 0) {
      HLOG(kError, "Verify: blob {} content mismatch", idx);
      ++bad;
    }
  }
  CLIO_IPC->FreeBuffer(got);
  if (bad != 0) {
    HLOG(kError, "Verify: {} of {} blobs failed", bad, c.num_blobs);
    return 1;
  }
  HLOG(kSuccess, "Verify: all {} blobs read back byte-identical", c.num_blobs);
  return 0;
}

/**
 * Emit the equivalence-caveat block that accompanies every result row.
 *
 * Every line here is scraped into a results.csv column so a reader can tell
 * what each stack actually did. The two zeros -- subprocess spawns and temp
 * file bytes -- are the meaningful contrast against the READ benchmark, whose
 * CAE path pays a fork+exec per object and stages each object whole through
 * TMPDIR. The bdev write path does neither.
 *
 * @param c Benchmark configuration.
 * @param r Timed-loop results.
 */
void PrintFairness(const BenchConfig& c, const RunResult& r) {
  const u64 bytes_moved = static_cast<u64>(r.blobs_done) * c.blob_size;
  // One PutBlob -> one allocator block -> one PutObject, for any request the
  // heap can satisfy contiguously. This USED to be derived as
  // ceil(blob_size / block_size) from a --block-size knob that configured
  // nothing, which overstated the count 4x at 4 MiB blobs and a 1 MiB guess.
  // The knob is gone. This is still a derivation of the common case, so the
  // jarvis package lists the bucket prefix afterwards and reports
  // objects_measured as ground truth -- if the two disagree, the allocator
  // fragmented and returned more than one block per blob.
  const u64 put_count = static_cast<u64>(r.blobs_done);
  HLOG(kInfo, "");
  HLOG(kInfo, "=== {} Fairness ===", c.label);
  HLOG(kInfo, "Objects written: {}", put_count);
  HLOG(kInfo, "Bytes moved: {}", bytes_moved);
  HLOG(kInfo, "Logical bytes: {}", bytes_moved);
  HLOG(kInfo, "PUT count: {}", put_count);
  HLOG(kInfo, "Compression: none");
  HLOG(kInfo, "Decode step: no");
  HLOG(kInfo, "Requested concurrency: {}", c.concurrency);
  HLOG(kInfo, "Effective concurrency: {}", r.slot_ops.size());
  HLOG(kInfo, "Runtime worker threads: {}", c.worker_threads);
  HLOG(kInfo, "Wall time us: {} ({} ms)", r.wall_us, r.wall_us / 1000.0);
  HLOG(kInfo, "Wire bandwidth: {} MB/s",
       clio_bench::CalcBandwidth(bytes_moved, r.wall_us));
  // The write path signs and PUTs from the runtime worker directly: no helper
  // process, no staging file. Contrast the read bench, which reports
  // one spawn and a full object staged per transfer.
  HLOG(kInfo, "Subprocess spawns: 0");
  HLOG(kInfo, "Temp file bytes: 0");
  // Bytes per PutObject, not a configured block size: WriteBlocks writes
  // min(remaining, block.size_) and the single returned block spans the whole
  // request. Equals blob_size unless the allocator fragmented.
  HLOG(kInfo, "Transport chunk bytes: {}", c.blob_size);
  HLOG(kInfo, "===================");
}

}  // namespace

int main(int argc, char** argv) {
  BenchConfig c = ParseArgs(argc, argv);
  if (!c.ok) {
    return 1;
  }

  HLOG(kInfo, "clio_s3_write_bench [{}]: {} blobs of {}, K={}", kBuildMarker,
       c.num_blobs, clio_bench::FormatSize(c.blob_size), c.concurrency);

  // Attach to the already-running clio_run daemon as a pure client. The
  // second arg (default_with_runtime) MUST be false: the jarvis pipeline
  // starts the runtime via the clio_runtime package, so bootstrapping a
  // second runtime in-process collides on port 9413 and aborts the row.
  // See the same note in clio_s3_read_bench.cc and clio_cte_bench.cc.
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    HLOG(kError, "Failed to initialize Clio");
    return 1;
  }
  clio::cte::core::CLIO_CTE_CLIENT_INIT();

  int exit_code = 0;
  try {
    auto* cte = CLIO_CTE_CLIENT;
    auto tag_task = cte->AsyncGetOrCreateTag(c.tag_prefix + "_tag");
    tag_task.Wait();
    clio::cte::core::TagId tag_id = tag_task->tag_id_;
    if (tag_id.IsNull()) {
      HLOG(kError, "Failed to create CTE tag {}_tag", c.tag_prefix);
      ctp::SystemInfo::TerminateProcessNow(1);
      return 1;
    }

    RunResult r = RunWriteLoop(c, tag_id);
    exit_code = r.rc;
    if (r.blobs_failed != 0) {
      HLOG(kError, "{}/{} blobs failed to write; the numbers below are not a "
                   "measurement of anything",
           r.blobs_failed, r.blobs_done);
    }

    clio_bench::BenchArgs a;
    a.test_case = c.label;
    a.threads = r.slot_ops.size();  // one "thread" per in-flight slot
    a.depth = 1;
    a.io_size = c.blob_size;
    a.io_count = static_cast<int>(c.num_blobs);
    clio_bench::PrintResults(c.label, a, r.slot_us, r.slot_ops);
    PrintFairness(c, r);

    if (c.verify && exit_code == 0) {
      exit_code = VerifyBlobs(c, tag_id);
    }
  } catch (const std::exception& e) {
    HLOG(kError, "Exception: {}", e.what());
    exit_code = 1;
  }

  // TerminateProcessNow avoids the client-teardown hang that would otherwise
  // let the pipeline's run_timeout consume this row.
  ctp::SystemInfo::TerminateProcessNow(exit_code);
  return exit_code;
}
