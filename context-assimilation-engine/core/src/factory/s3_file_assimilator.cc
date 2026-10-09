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

#include <clio_runtime/clio_runtime.h>
#include <clio_cae/core/factory/s3_file_assimilator.h>
#include <clio_cae/core/factory/aws_creds.h>
#include <clio_cae/core/factory/s3_conn_pool.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>

// The in-process S3 REST client (Poco::Net + SigV4). Loading the AWS SDK into
// this runtime process stack-smashes CLIO_INIT, so the read path signs and
// streams over Poco instead -- the same transport the kS3 bdev tier proved.
// cae_s3_tool (the SDK) still exists, but only as a standalone helper for the
// benchmark floors and test seeding; the assimilator no longer forks it.
#include "clio_runtime/bdev/transports/s3_rest.h"

// Include clio_cte headers after the clio_cae includes to avoid Method
// namespace collision (same ordering as BinaryFileAssimilator).
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#ifdef CAE_ENABLE_CFS_DST
#include <clio_cte/filesystem/filesystem_client.h>
#endif

namespace clio::cae::core {

namespace {

namespace s3 = clio::run::bdev::s3;

/**
 * Per-phase latency accumulators, summed across every object this process
 * assimilates and reported once by S3AssimLogPhaseTally().
 *
 * Process-wide rather than instance state on purpose: AssimilatorFactory builds
 * a fresh S3FileAssimilator inside every ParseOmni task (core_runtime.cc), so
 * instance members would not survive a single object.
 *
 * Relaxed ordering throughout -- these are counters read once at shutdown, and
 * nothing branches on them, so no happens-before relationship is needed.
 */
std::atomic<uint64_t> g_n_objects{0};   ///< objects assimilated
std::atomic<uint64_t> g_n_chunks{0};    ///< body chunks written to CTE
std::atomic<uint64_t> g_us_tag{0};      ///< AsyncGetOrCreateTag  (CTE trip 1)
std::atomic<uint64_t> g_us_creds{0};    ///< ResolveAwsCredentials (disk!)
std::atomic<uint64_t> g_us_acquire{0};  ///< pool Acquire (may connect)
std::atomic<uint64_t> g_us_begin{0};    ///< BeginGetObject (S3 round trip)
std::atomic<uint64_t> g_us_desc{0};     ///< description PutBlob (CTE trip 2)
std::atomic<uint64_t> g_us_fill{0};     ///< FillChunk -- body bytes off socket
std::atomic<uint64_t> g_us_put{0};      ///< awaiting body PutBlobs (CTE trip N)
std::atomic<uint64_t> g_us_end{0};      ///< EndGetObject (drain tail)
std::atomic<uint64_t> g_us_total{0};    ///< whole Schedule(), entry to exit

using SteadyClock = std::chrono::steady_clock;

/** Microseconds elapsed since @p t0. Safe across a CLIO_CO_AWAIT: coroutine
 *  locals are preserved over a suspend (see the ConnReturn comment below). */
inline uint64_t UsSince(const SteadyClock::time_point &t0) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          SteadyClock::now() - t0)
          .count());
}

/**
 * Build the S3 client config for one import: endpoint (and its addressing
 * style) from the environment, credentials + region from the resolver. The
 * signer reads no environment itself, so the resolved values are injected here.
 * `prefix` is left empty -- the assimilator passes whole object keys, so the
 * bdev's KeyForOffset prefixing does not apply.
 *
 * @param bucket Target bucket.
 * @param creds  Resolved access/secret/token + region.
 * @return A ready S3Config.
 */
s3::S3Config MakeS3Config(const std::string& bucket,
                          const AwsCredentials& creds) {
  // ConfigFromEnv gives us S3_ENDPOINT (+ trailing-slash stripping and the
  // path-style decision); the credential fields it reads from the environment
  // are then overridden by the resolver's result, which itself preferred the
  // environment when present, so the two agree.
  s3::S3Config cfg = s3::S3RestClient::ConfigFromEnv(bucket, /*prefix=*/"");
  cfg.region = creds.region;
  cfg.access_key = creds.access_key;
  cfg.secret_key = creds.secret_key;
  cfg.session_token = creds.session_token;
  cfg.anonymous = creds.anonymous;
  cfg.allow_bucket_create = false;  // a reader must never create a bucket
  return cfg;
}

/**
 * Fill exactly `want` bytes into `buf` from the live GET stream, resuming with a
 * ranged GET whenever the response body ends short of `want` before the whole
 * transfer is exhausted (e.g. a keep-alive socket dropped across a co_await).
 * Mirrors what the old fork+exec path got for free by staging to a whole file.
 *
 * @param client   The S3 client.
 * @param conn     The leased connection (kept across resumes).
 * @param st       In/out: the current GET stream; replaced on a resume.
 * @param key      Object key, for the resume request.
 * @param buf      Destination buffer of capacity >= want.
 * @param want     Bytes to place into buf (<= bytes remaining in the transfer).
 * @param abs_start Object offset of buf[0] (absolute, for the Range header).
 * @param max_resumes Cap on resume attempts before giving up.
 * @param resumes  In/out: running count of resumes across the whole object.
 * @param filled   Output: bytes actually placed into buf.
 * @return An S3Result: ok when want bytes were filled, else error set.
 */
s3::S3Result FillChunk(s3::S3RestClient& client, s3::S3Connection& conn,
                       s3::S3RestClient::S3GetStream& st, const std::string& key,
                       char* buf, size_t want, uint64_t abs_start,
                       int max_resumes, int* resumes, size_t* filled) {
  size_t got_total = 0;
  while (got_total < want) {
    size_t got = 0;
    s3::S3Result rr = client.ReadBody(st, buf + got_total, want - got_total,
                                      &got);
    got_total += got;
    if (got_total >= want) break;
    // The body ended (clean EOF or a mid-stream error) before this chunk was
    // full, yet the transfer is not complete. Retire the response and resume
    // from where we stopped with a ranged GET for the remaining bytes.
    if (*resumes >= max_resumes) {
      if (rr.error.empty()) {
        rr.error = "S3 GET " + key + " ended short after " +
                   std::to_string(max_resumes) + " resumes at offset " +
                   std::to_string(abs_start + got_total);
      }
      *filled = got_total;
      return rr;
    }
    ++(*resumes);
    client.EndGetObject(conn, st);
    const uint64_t resume_off = abs_start + got_total;
    const uint64_t resume_len = static_cast<uint64_t>(want - got_total);
    s3::S3Result rb = client.BeginGetObject(conn, key, resume_off, resume_len,
                                            &st);
    if (!rb.ok()) {
      *filled = got_total;
      return rb;
    }
  }
  *filled = got_total;
  return s3::S3Result{};
}

/** Size of one CTE blob of an imported object (also the clio-fs page size). */
constexpr size_t kChunkBytes = 1024 * 1024;  // 1 MiB

/**
 * Striped-fetch knobs, read per object from the environment of the RUNTIME
 * (the process that runs this assimilator):
 *
 *   CAE_S3_STREAMS  concurrent ranged GETs per object (default 8, 1 = the old
 *                   single-stream path, max 64)
 *   CAE_S3_PART_MB  bytes each GET asks for, in MiB (default 16, max 1024)
 *
 * One TCP stream to S3 is capped well below what the link can carry: through
 * the ALCF proxy one stream measured ~23 MB/s and eight concurrent ranged
 * streams ~186 MB/s. An object only stripes when it spans more than one part,
 * so small objects keep the single-stream path.
 */
struct StripeConfig {
  size_t streams = 8;
  size_t part_bytes = 16 * kChunkBytes;
};

StripeConfig StripeConfigFromEnv() {
  StripeConfig c;
  const char *s = std::getenv("CAE_S3_STREAMS");
  if (s != nullptr && *s != '\0') {
    long v = std::strtol(s, nullptr, 10);
    if (v >= 1) c.streams = static_cast<size_t>(std::min(v, 64L));
  }
  const char *p = std::getenv("CAE_S3_PART_MB");
  if (p != nullptr && *p != '\0') {
    long v = std::strtol(p, nullptr, 10);
    if (v >= 1) {
      c.part_bytes = static_cast<size_t>(std::min(v, 1024L)) * kChunkBytes;
    }
  }
  return c;
}

/**
 * One stream's share of a striped window: a contiguous run of chunk buffers,
 * fetched with ONE ranged GET on a connection of its own.
 */
struct StripeJob {
  uint64_t abs_off = 0;        ///< object offset of bufs[0]
  std::vector<char *> bufs;    ///< destination chunk buffers, in object order
  std::vector<size_t> sizes;   ///< bytes wanted in each buffer
  int resumes = 0;             ///< resumes this stream needed
  std::string error;           ///< empty on success
};

/**
 * Fill every buffer of `job` from s3://<cfg.bucket>/<key>. Runs on a plain
 * std::thread, so it touches no runtime state: the buffers were allocated by
 * the coroutine, and the connection comes from (and returns to) the
 * thread-safe pool.
 */
void FetchStripe(const s3::S3Config &cfg, S3ConnectionPool *pool,
                 const std::string &key, int max_resumes, StripeJob *job) {
  s3::S3RestClient client(cfg);
  const std::string conn_key = client.ConnectionKey(key);
  std::unique_ptr<s3::S3Connection> conn =
      pool ? pool->Acquire(conn_key) : std::make_unique<s3::S3Connection>();
  uint64_t len = 0;
  for (size_t n : job->sizes) len += n;
  s3::S3RestClient::S3GetStream st;
  s3::S3Result b = client.BeginGetObject(*conn, key, job->abs_off, len, &st);
  if (!b.ok()) {
    job->error = !b.error.empty()
                     ? b.error
                     : "S3 GET " + key + " failed: HTTP " +
                           std::to_string(b.http_status);
  } else {
    uint64_t off = job->abs_off;
    for (size_t i = 0; i < job->bufs.size(); ++i) {
      size_t filled = 0;
      s3::S3Result fr = FillChunk(client, *conn, st, key, job->bufs[i],
                                  job->sizes[i], off, max_resumes,
                                  &job->resumes, &filled);
      if (!fr.error.empty() || filled != job->sizes[i]) {
        job->error = !fr.error.empty()
                         ? fr.error
                         : "short read at offset " + std::to_string(off);
        break;
      }
      off += job->sizes[i];
    }
  }
  // A body abandoned mid-stream carries unread bytes: never pool that socket.
  if (job->error.empty()) {
    client.EndGetObject(*conn, st);
  } else if (st.body != nullptr) {
    conn->Retire();
    st.body = nullptr;
  }
  if (pool) {
    pool->Release(conn_key, std::move(conn));
  }
}

/**
 * Blob name of chunk `idx`. An iowarp:: tag keeps the historical "chunk_<n>";
 * a cfs:: file uses the clio-fs page name (the bare page index), so the FUSE
 * mount reads the blobs as the file's pages.
 */
std::string ChunkBlobName(bool to_cfs, size_t idx) {
  return to_cfs ? std::to_string(idx) : "chunk_" + std::to_string(idx);
}

}  // namespace

void S3AssimLogPhaseTally() {
  const uint64_t n = g_n_objects.load(std::memory_order_relaxed);
  if (n == 0) {
    return;  // this process assimilated nothing from S3
  }
  const uint64_t chunks = g_n_chunks.load(std::memory_order_relaxed);
  // Per-object means. Integer division is deliberate: ctp::Formatter only
  // understands bare "{}" (no format specs), and microsecond precision is far
  // finer than anything we can conclude from these anyway.
  auto per_obj = [n](const std::atomic<uint64_t> &a) {
    return a.load(std::memory_order_relaxed) / n;
  };
  HLOG(kInfo,
       "CAE S3 phase TOTAL objects={} chunks={} chunks_per_obj={} | "
       "per-object us: tag={} creds={} acquire={} get_begin={} desc={} "
       "fill={} put_wait={} end={} TOTAL={}",
       n, chunks, chunks / n, per_obj(g_us_tag), per_obj(g_us_creds),
       per_obj(g_us_acquire), per_obj(g_us_begin), per_obj(g_us_desc),
       per_obj(g_us_fill), per_obj(g_us_put), per_obj(g_us_end),
       per_obj(g_us_total));
  g_n_objects.store(0, std::memory_order_relaxed);
  g_n_chunks.store(0, std::memory_order_relaxed);
  for (std::atomic<uint64_t> *a : {&g_us_tag, &g_us_creds, &g_us_acquire,
                                   &g_us_begin, &g_us_desc, &g_us_fill,
                                   &g_us_put, &g_us_end, &g_us_total}) {
    a->store(0, std::memory_order_relaxed);
  }
}

S3FileAssimilator::S3FileAssimilator(
    std::shared_ptr<clio::cte::core::Client> cte_client,
    S3ConnectionPool* s3_pool)
    : cte_client_(cte_client), s3_pool_(s3_pool) {}

clio::run::TaskResume S3FileAssimilator::Schedule(const AssimilationCtx& ctx,
                                                  int& error_code) {
#ifdef __NVCOMPILER
  thread_local clio::run::RunContext _fb_rctx;
  clio::run::RunContext* _fp = clio::run::GetCurrentRunContextFromWorker();
  clio::run::RunContext& rctx = _fp ? *_fp : _fb_rctx;
#endif
  CLIO_TASK_BODY_BEGIN
  HLOG(kDebug,
       "S3FileAssimilator::Schedule ENTRY: src='{}', dst='{}', range_off={}, "
       "range_size={}",
       ctx.src, ctx.dst, ctx.range_off, ctx.range_size);

  // Phase timing. Only the success path accumulates: an object that bails out
  // early would otherwise drag the per-object means toward zero and make a
  // failing run look fast. See S3AssimLogPhaseTally() for how to read these.
  const SteadyClock::time_point _t_entry = SteadyClock::now();
  SteadyClock::time_point _t_phase;

  // Validate destination protocol. iowarp::<tag> stores the object as a CTE
  // tag of "chunk_<n>" blobs plus a "description" blob. cfs::<path> stores it
  // as a FILE in the clio-fs namespace (pages "0", "1", ...), so a FUSE mount
  // of that namespace reads it back with ordinary file I/O.
  std::string dst_protocol = GetUrlProtocol(ctx.dst);
  const bool to_cfs = (dst_protocol == "cfs");
#ifndef CAE_ENABLE_CFS_DST
  if (to_cfs) {
    HLOG(kError,
         "S3FileAssimilator: cfs:: destinations need the clio-fs filesystem "
         "chimod, which this build does not have");
    error_code = -1;
    CLIO_CO_RETURN;
  }
#endif
  if (dst_protocol != "iowarp" && !to_cfs) {
    HLOG(kError,
         "S3FileAssimilator: Destination protocol must be 'iowarp' or 'cfs', "
         "got '{}'",
         dst_protocol);
    error_code = -1;
    CLIO_CO_RETURN;
  }

  // Extract tag name (iowarp) or absolute file path (cfs) from the URL
  std::string tag_name = GetUrlPath(ctx.dst);
  if (tag_name.empty() || (to_cfs && tag_name[0] != '/')) {
    HLOG(kError,
         "S3FileAssimilator: Invalid destination URL '{}' (want iowarp::<tag> "
         "or cfs::/<absolute path>)",
         ctx.dst);
    error_code = -2;
    CLIO_CO_RETURN;
  }

  // Get or create the tag in CTE. A cfs:: file is opened later, once the GET
  // has succeeded, so a failed request leaves no empty file behind.
  clio::cte::core::TagId tag_id = clio::cte::core::TagId::GetNull();
  if (!to_cfs) {
    _t_phase = SteadyClock::now();
    auto tag_task = cte_client_->AsyncGetOrCreateTag(tag_name);
    CLIO_CO_AWAIT(tag_task);
    g_us_tag.fetch_add(UsSince(_t_phase), std::memory_order_relaxed);
    tag_id = tag_task->tag_id_;
    if (tag_id.IsNull()) {
      HLOG(kError, "S3FileAssimilator: Failed to get or create tag '{}'",
           tag_name);
      error_code = -3;
      CLIO_CO_RETURN;
    }
  }

  // Dependency-based scheduling is not yet supported (mirrors binary backend).
  if (!ctx.depends_on.empty()) {
    HLOG(kDebug,
         "S3FileAssimilator: Dependency handling not yet implemented "
         "(depends_on: {})",
         ctx.depends_on);
    error_code = 0;
    CLIO_CO_RETURN;
  }

  // Parse the S3 source URL into bucket + key
  std::string bucket;
  std::string key;
  if (!ParseS3Url(ctx.src, bucket, key)) {
    HLOG(kError, "S3FileAssimilator: Invalid S3 source URL '{}'", ctx.src);
    error_code = -4;
    CLIO_CO_RETURN;
  }

  // Resolve credentials + region in-process (no AWS SDK): environment keys
  // first, else the named profile from ~/.aws/credentials, else anonymous
  // (AWS_NO_SIGN_REQUEST, or nothing configured). Only the profile
  // NAME (ctx.s3_profile) ever travels in the task payload -- never a secret.
  // Timed because this can touch ~/.aws/credentials on EVERY object -- a
  // per-object filesystem cost that would be invisible in the aggregate.
  _t_phase = SteadyClock::now();
  AwsCredResult cred = ResolveAwsCredentials(ctx.s3_profile, ctx.s3_region);
  g_us_creds.fetch_add(UsSince(_t_phase), std::memory_order_relaxed);
  if (!cred.ok) {
    HLOG(kError, "S3FileAssimilator: {}", cred.error);
    error_code = -6;
    CLIO_CO_RETURN;
  }
  if (cred.creds.anonymous) {
    HLOG(kInfo,
         "S3FileAssimilator: requesting s3://{}/{} anonymously (unsigned)",
         bucket, key);
  }

  // In-process S3 client (Poco + SigV4). The connection is leased from the
  // runtime-owned pool for this object's whole lifetime, so a keep-alive socket
  // is reused across objects and is never shared with another worker even if
  // this task is migrated across a co_await (issue #785).
  s3::S3RestClient client(MakeS3Config(bucket, cred.creds));
  const std::string conn_key = client.ConnectionKey(key);
  _t_phase = SteadyClock::now();
  std::unique_ptr<s3::S3Connection> conn_owned =
      s3_pool_ ? s3_pool_->Acquire(conn_key)
               : std::make_unique<s3::S3Connection>();
  g_us_acquire.fetch_add(UsSince(_t_phase), std::memory_order_relaxed);
  s3::S3Connection& conn = *conn_owned;
  s3::S3RestClient::S3GetStream stream;
  // Return the connection on every exit path (success, error, co_return). In
  // both coroutine backends locals are destroyed at CLIO_CO_RETURN but preserved
  // across a suspend, which is exactly this lifetime. A socket whose body is
  // still open (an error bailed mid-stream) must not be pooled -- it carries
  // unconsumed bytes -- so it is retired; a clean finish nulls stream.body via
  // EndGetObject, leaving a reusable socket to pool.
  struct ConnReturn {
    S3ConnectionPool* pool;
    const std::string& key;
    std::unique_ptr<s3::S3Connection>* conn;
    s3::S3RestClient::S3GetStream* stream;
    ~ConnReturn() {
      if (!conn || !*conn) return;
      if (stream && stream->body != nullptr) (*conn)->Retire();
      if (pool) pool->Release(key, std::move(*conn));
    }
  } conn_guard{s3_pool_, conn_key, &conn_owned, &stream};

  // Whole object unless a bounded range was requested (a bare range_off with no
  // range_size means "whole object", matching the old fork+exec tool contract).
  const uint64_t req_off = (ctx.range_size > 0) ? ctx.range_off : 0;
  const uint64_t req_size = static_cast<uint64_t>(ctx.range_size);

  _t_phase = SteadyClock::now();
  s3::S3Result begin =
      client.BeginGetObject(conn, key, req_off, req_size, &stream);
  g_us_begin.fetch_add(UsSince(_t_phase), std::memory_order_relaxed);
  if (begin.not_found) {
    HLOG(kError, "S3FileAssimilator: object not found: s3://{}/{}", bucket, key);
    error_code = -7;
    CLIO_CO_RETURN;
  }
  if (!begin.ok()) {
    HLOG(kError, "S3FileAssimilator: GET failed for s3://{}/{}: {}", bucket, key,
         begin.error);
    error_code = -7;
    CLIO_CO_RETURN;
  }

  // Bytes to ingest in THIS transfer (the range length, or the whole object).
  size_t total_size = static_cast<size_t>(stream.content_length);
  size_t chunk_offset = (ctx.range_size > 0) ? ctx.range_off : 0;
  HLOG(kDebug, "S3FileAssimilator: s3://{}/{} -> {} bytes (offset {})", bucket,
       key, total_size, chunk_offset);

#ifdef CAE_ENABLE_CFS_DST
  // cfs::<path>: create the file in the clio-fs namespace and write its pages
  // straight into the file's tag. Its size is published when it is closed at
  // the end; until then readers see an empty file. Parent directories are
  // created as needed (an existing one is not an error).
  std::unique_ptr<clio::cte::filesystem::Client> cfs;
  clio::run::u64 cfs_handle = 0;
  // Close the file on every error path below, publishing no size.
  struct CfsCloseOnError {
    clio::cte::filesystem::Client *client = nullptr;
    clio::run::u64 handle = 0;
    ~CfsCloseOnError() {
      if (client != nullptr && handle != 0) {
        client->AsyncCloseDetached(handle, 0);
      }
    }
  } cfs_guard;
  if (to_cfs) {
    cfs = std::make_unique<clio::cte::filesystem::Client>(
        clio::cte::filesystem::kCfsPoolId);
    _t_phase = SteadyClock::now();
    for (size_t slash = tag_name.find('/', 1); slash != std::string::npos;
         slash = tag_name.find('/', slash + 1)) {
      auto mk = cfs->AsyncMkdir(tag_name.substr(0, slash));
      CLIO_CO_AWAIT(mk);
    }
    auto open_task = cfs->AsyncOpen(
        tag_name, static_cast<clio::run::u32>(O_CREAT | O_WRONLY | O_TRUNC),
        0644);
    CLIO_CO_AWAIT(open_task);
    g_us_tag.fetch_add(UsSince(_t_phase), std::memory_order_relaxed);
    if (open_task->GetReturnCode() != 0 || open_task->handle_ == 0) {
      HLOG(kError,
           "S3FileAssimilator: could not create clio-fs file '{}' (rc {}); is "
           "the clio_cte_filesystem pool composed?",
           tag_name, open_task->GetReturnCode());
      error_code = -3;
      CLIO_CO_RETURN;
    }
    cfs_handle = open_task->handle_;
    cfs_guard.client = cfs.get();
    cfs_guard.handle = cfs_handle;
    tag_id = clio::cte::core::TagId(
        static_cast<clio::run::u32>(open_task->tag_packed_ >> 32),
        static_cast<clio::run::u32>(open_task->tag_packed_ & 0xffffffffULL));
  }
#endif

  // Store object metadata as the "description" blob (mirrors binary backend).
  // Not for a cfs:: file: every blob of a file's tag is one of its pages.
  if (!to_cfs) {
    std::string description = "binary<size=" + std::to_string(total_size) +
                              ", offset=" + std::to_string(chunk_offset) + ">";
    size_t desc_size = description.size();
    auto desc_buffer = CLIO_IPC->AllocateBuffer(desc_size);
    std::memcpy(desc_buffer.ptr_, description.c_str(), desc_size);
    // Submitted AND awaited here, as it was before b8b84521. That commit
    // deferred the await to after the body to take this CTE round trip off the
    // critical path of every object; the targeted fine-granularity cells got
    // slower, not faster, so the deferral bought nothing measurable and cost
    // two things that are worth more than one round trip:
    //
    //   1. On the mid-body error paths (-8, -10) desc_task was abandoned
    //      while still in flight, leaving a submitted-but-never-collected
    //      future.
    //   2. A description failure was only discovered after the body had
    //      already been committed, so the object was half-written when it was
    //      reported.
    //
    // Keep the trip. The per-object cost is one CTE hop and it is honest.
    const SteadyClock::time_point _t_desc = SteadyClock::now();
    auto desc_task =
        cte_client_->AsyncPutBlob(tag_id, "description", 0, desc_size,
                                  desc_buffer.shm_.template Cast<void>(), 1.0f,
                                  clio::cte::core::Context(), 0);
    CLIO_CO_AWAIT(desc_task);
    g_us_desc.fetch_add(UsSince(_t_desc), std::memory_order_relaxed);
    if (desc_task->return_code_ != 0) {
      HLOG(kError,
           "S3FileAssimilator: Failed to store description for tag '{}' "
           "(code {})",
           tag_name, desc_task->return_code_);
      CLIO_IPC->FreeBuffer(desc_task->blob_data_.template Cast<char>());
      error_code = -9;
      CLIO_CO_RETURN;
    }
    // Pre-existing leak found while b8b84521 moved this, and kept on the
    // revert: the description buffer was allocated per object and never freed.
    // BinaryFileAssimilator:176 has the same bug -- left alone rather than
    // edited blind in an unrelated path.
    CLIO_IPC->FreeBuffer(desc_task->blob_data_.template Cast<char>());
  }

  // Stream the body into CTE in chunks, keeping up to kMaxParallelTasks PutBlob
  // tasks in flight (identical wait-and-drain shape to the binary backend). The
  // body now comes off the socket instead of a staged file; FillChunk resumes a
  // dropped keep-alive mid-object so a suspend across a PutBlob cannot truncate.
  static constexpr size_t kMaxChunkSize = kChunkBytes;
  static constexpr size_t kMaxParallelTasks = 32;
  static constexpr int kMaxResumes = 16;
  size_t chunk_idx = 0;
  size_t bytes_processed = 0;
  int resumes = 0;
  std::vector<clio::run::Future<clio::cte::core::PutBlobTask>> active_tasks;

  const StripeConfig stripe = StripeConfigFromEnv();
  if (stripe.streams > 1 && total_size > stripe.part_bytes) {
    // STRIPED: the object is fetched one window at a time. A window holds
    // kPartsPerStream parts per stream; `streams` threads pull parts from a
    // shared counter, each part one ranged GET, so a slow request delays only
    // its own thread's next pull, not the whole window. The opening GET only
    // supplied the size, so it is dropped rather than drained (draining would
    // read the whole object).
    static constexpr size_t kPartsPerStream = 4;
    conn.Retire();
    stream.body = nullptr;
    using BufPtr = decltype(CLIO_IPC->AllocateBuffer(size_t{0}));
    const size_t window = stripe.streams * kPartsPerStream * stripe.part_bytes;
    const size_t chunks_per_part = stripe.part_bytes / kChunkBytes;
    // Let a whole window's PutBlobs be in flight while the next one downloads.
    const size_t inflight_cap =
        std::max(kMaxParallelTasks, window / kChunkBytes);
    while (bytes_processed < total_size) {
      const size_t win_bytes = std::min(window, total_size - bytes_processed);
      std::vector<BufPtr> bufs;
      std::vector<size_t> sizes;
      bool alloc_ok = true;
      for (size_t off = 0; off < win_bytes; off += kChunkBytes) {
        const size_t n = std::min(kChunkBytes, win_bytes - off);
        bufs.push_back(CLIO_IPC->AllocateBuffer(n));
        sizes.push_back(n);
        if (bufs.back().ptr_ == nullptr) alloc_ok = false;
      }
      std::vector<StripeJob> jobs;
      for (size_t c = 0; c < bufs.size(); c += chunks_per_part) {
        StripeJob job;
        job.abs_off = req_off + bytes_processed + c * kChunkBytes;
        for (size_t k = c; k < std::min(c + chunks_per_part, bufs.size());
             ++k) {
          job.bufs.push_back(bufs[k].ptr_);
          job.sizes.push_back(sizes[k]);
        }
        jobs.push_back(std::move(job));
      }
      _t_phase = SteadyClock::now();
      if (alloc_ok) {
        // Threads, not coroutines: the reads block in Poco, and the runtime
        // worker blocks here anyway (as FillChunk does on the serial path).
        std::atomic<size_t> next_job{0};
        auto pull = [&]() {
          for (size_t j = next_job.fetch_add(1); j < jobs.size();
               j = next_job.fetch_add(1)) {
            FetchStripe(client.config(), s3_pool_, key, kMaxResumes, &jobs[j]);
          }
        };
        std::vector<std::thread> threads;
        const size_t nthreads = std::min(stripe.streams, jobs.size());
        for (size_t t = 1; t < nthreads; ++t) threads.emplace_back(pull);
        pull();
        for (auto& t : threads) t.join();
      }
      g_us_fill.fetch_add(UsSince(_t_phase), std::memory_order_relaxed);
      std::string err = alloc_ok ? "" : "buffer allocation failed";
      for (const StripeJob& job : jobs) {
        resumes += job.resumes;
        if (err.empty() && !job.error.empty()) err = job.error;
      }
      if (!err.empty()) {
        HLOG(kError,
             "S3FileAssimilator: striped read of s3://{}/{} failed in the "
             "window at offset {}: {}",
             bucket, key, req_off + bytes_processed, err);
        for (auto& b : bufs) {
          if (b.ptr_ != nullptr) CLIO_IPC->FreeBuffer(b);
        }
        error_code = -8;
        CLIO_CO_RETURN;
      }
      for (size_t k = 0; k < bufs.size(); ++k) {
        active_tasks.push_back(cte_client_->AsyncPutBlob(
            tag_id, ChunkBlobName(to_cfs, chunk_idx), 0, sizes[k],
            bufs[k].shm_.template Cast<void>(), 1.0f,
            clio::cte::core::Context(), 0));
        chunk_idx++;
      }
      bytes_processed += win_bytes;

      while (active_tasks.size() > inflight_cap) {
        auto& first_task = active_tasks.front();
        _t_phase = SteadyClock::now();
        CLIO_CO_AWAIT(first_task);
        g_us_put.fetch_add(UsSince(_t_phase), std::memory_order_relaxed);
        if (first_task->return_code_ != 0) {
          HLOG(kError, "S3FileAssimilator: PutBlob task failed with code {}",
               first_task->return_code_);
          CLIO_IPC->FreeBuffer(first_task->blob_data_.template Cast<char>());
          error_code = -10;
          CLIO_CO_RETURN;
        }
        CLIO_IPC->FreeBuffer(first_task->blob_data_.template Cast<char>());
        active_tasks.erase(active_tasks.begin());
      }
    }
  }

  while (bytes_processed < total_size) {
    while (active_tasks.size() < kMaxParallelTasks &&
           bytes_processed < total_size) {
      size_t current_chunk_size =
          std::min(kMaxChunkSize, total_size - bytes_processed);
      auto buffer_ptr = CLIO_IPC->AllocateBuffer(current_chunk_size);
      char* buffer = buffer_ptr.ptr_;

      size_t filled = 0;
      _t_phase = SteadyClock::now();
      s3::S3Result fr =
          FillChunk(client, conn, stream, key, buffer, current_chunk_size,
                    req_off + bytes_processed, kMaxResumes, &resumes, &filled);
      g_us_fill.fetch_add(UsSince(_t_phase), std::memory_order_relaxed);
      if (!fr.error.empty() || filled != current_chunk_size) {
        HLOG(kError,
             "S3FileAssimilator: short/failed read on chunk {} from s3://{}/{} "
             "(filled={}, want={}, err='{}')",
             chunk_idx, bucket, key, filled, current_chunk_size, fr.error);
        CLIO_IPC->FreeBuffer(buffer_ptr);
        error_code = -8;
        CLIO_CO_RETURN;
      }

      auto task = cte_client_->AsyncPutBlob(
          tag_id, ChunkBlobName(to_cfs, chunk_idx), 0, current_chunk_size,
          buffer_ptr.shm_.template Cast<void>(), 1.0f,
          clio::cte::core::Context(), 0);
      active_tasks.push_back(task);
      bytes_processed += current_chunk_size;
      chunk_idx++;
    }

    if (!active_tasks.empty()) {
      auto& first_task = active_tasks.front();
      _t_phase = SteadyClock::now();
      CLIO_CO_AWAIT(first_task);
      g_us_put.fetch_add(UsSince(_t_phase), std::memory_order_relaxed);
      if (first_task->return_code_ != 0) {
        HLOG(kError, "S3FileAssimilator: PutBlob task failed with code {}",
             first_task->return_code_);
        CLIO_IPC->FreeBuffer(first_task->blob_data_.template Cast<char>());
        error_code = -10;
        CLIO_CO_RETURN;
      }
      CLIO_IPC->FreeBuffer(first_task->blob_data_.template Cast<char>());
      active_tasks.erase(active_tasks.begin());
    }
  }

  // Drain any remaining in-flight tasks.
  for (auto& task : active_tasks) {
    _t_phase = SteadyClock::now();
    CLIO_CO_AWAIT(task);
    g_us_put.fetch_add(UsSince(_t_phase), std::memory_order_relaxed);
    if (task->return_code_ != 0) {
      HLOG(kError, "S3FileAssimilator: PutBlob task failed with code {}",
           task->return_code_);
      CLIO_IPC->FreeBuffer(task->blob_data_.template Cast<char>());
      error_code = -10;
      CLIO_CO_RETURN;
    }
    CLIO_IPC->FreeBuffer(task->blob_data_.template Cast<char>());
  }

  // Drain any unread tail so the socket is reusable, then the guard pools it.
  // (A no-op after a striped fetch: that stream was already dropped.)
  _t_phase = SteadyClock::now();
  client.EndGetObject(conn, stream);
  g_us_end.fetch_add(UsSince(_t_phase), std::memory_order_relaxed);

#ifdef CAE_ENABLE_CFS_DST
  // Every page is stored: publish the file's size by closing it.
  if (to_cfs) {
    auto close_task = cfs->AsyncClose(cfs_handle, total_size);
    CLIO_CO_AWAIT(close_task);
    cfs_guard.handle = 0;  // closed, whatever the outcome
    if (close_task->GetReturnCode() != 0) {
      HLOG(kError,
           "S3FileAssimilator: closing clio-fs file '{}' failed (rc {}); its "
           "size was not published",
           tag_name, close_task->GetReturnCode());
      error_code = -11;
      CLIO_CO_RETURN;
    }
  }
#endif

  g_n_chunks.fetch_add(chunk_idx, std::memory_order_relaxed);
  g_n_objects.fetch_add(1, std::memory_order_relaxed);
  g_us_total.fetch_add(UsSince(_t_entry), std::memory_order_relaxed);

  HLOG(kDebug,
       "S3FileAssimilator: Imported s3://{}/{} ({} chunks, {} resumes) into "
       "{} '{}'",
       bucket, key, chunk_idx, resumes, to_cfs ? "file" : "tag", tag_name);
  error_code = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

std::string S3FileAssimilator::GetUrlProtocol(const std::string& url) {
  size_t pos_standard = url.find("://");
  if (pos_standard != std::string::npos) {
    return url.substr(0, pos_standard);
  }
  size_t pos_custom = url.find("::");
  if (pos_custom != std::string::npos) {
    return url.substr(0, pos_custom);
  }
  return "";
}

std::string S3FileAssimilator::GetUrlPath(const std::string& url) {
  size_t pos_standard = url.find("://");
  if (pos_standard != std::string::npos) {
    return url.substr(pos_standard + 3);
  }
  size_t pos_custom = url.find("::");
  if (pos_custom != std::string::npos) {
    return url.substr(pos_custom + 2);
  }
  return "";
}

bool S3FileAssimilator::ParseS3Url(const std::string& url, std::string& bucket,
                                   std::string& key) {
  // Strip the scheme (`s3://` or `s3::`) -> "bucket/key".
  std::string path = GetUrlPath(url);
  if (path.empty()) {
    return false;
  }
  size_t slash = path.find('/');
  if (slash == std::string::npos || slash == 0 || slash + 1 >= path.size()) {
    return false;  // need both a bucket and a non-empty key
  }
  bucket = path.substr(0, slash);
  key = path.substr(slash + 1);
  return true;
}

}  // namespace clio::cae::core
