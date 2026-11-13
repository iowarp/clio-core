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

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <clio_cte/core/blob_transform.h>
#include <clio_cte/lz4/lz4_runtime.h>
#include <clio_ctp/compress/gpu_ans.h>
#include <clio_ctp/util/gpu_api.h>

// In-process locators (defined in the runtime core library): a blob's
// (pool, offset, stored size), and a device-backed pool's base pointer.
extern "C" int clio_cte_locate(const void *tag_id, const char *name,
                               unsigned long long *pool_u64,
                               unsigned long long *target_off,
                               unsigned long long *stored_size);
extern "C" char *clio_direct_dev_base(unsigned long long pool_id);

namespace clio::cte::lz4 {

namespace glz = ::ctp::gpu_lz4;
using clio::cte::core::kBlobTransformCompressed;
using clio::cte::core::kBlobTransformed;

namespace {

/** Staging ranges are aligned so every page starts on a device-friendly
 *  boundary. */
constexpr clio::run::u64 kArenaAlign = 512;
/** Pinned job arrays kept in the pool (one per in-flight multi-get). */
constexpr size_t kJobArrays = 64;
/** Poll period while a decode or a staging slot is outstanding. */
constexpr double kPollUs = 20.0;
/** Give up waiting for staging space after this long (then go via host). */
constexpr auto kArenaWaitMax = std::chrono::seconds(10);

/** A raw device address as a ShmPtr (null allocator: the address itself). */
ctp::ipc::ShmPtr<> RawPtr(const char *p) {
  ctp::ipc::ShmPtr<> sp;
  sp.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
  sp.off_ = reinterpret_cast<clio::run::u64>(p);
  return sp;
}

/** @return true when the context's OUT transform bits say "compressed". */
bool StoredCompressed(const Context &ctx) {
  return (ctx.transform_flags_ & kBlobTransformCompressed) != 0;
}

}  // namespace

// ------------------------------------------------------------ lifecycle

Runtime::~Runtime() {
  if (arena_ != nullptr) ctp::GpuApi::Free(arena_);
  for (void *j : jobs_all_) {
    ctp::GpuApi::FreeHost(static_cast<glz::DecodeJob *>(j));
  }
}

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  config_ = task->GetParams();
  interposer_next_pool_ = config_.next_pool_id_;  // base forwarding target
  if (!config_.next_pool_id_.IsNull()) {
    next_client_ =
        std::make_unique<clio::cte::core::Client>(config_.next_pool_id_);
  }
#if CTP_ENABLE_GPU
  // Device resources are made HERE, before any consumer kernel is resident:
  // an allocation while a kernel runs can serialize behind it.
  if (ctp::GpuApi::GetDeviceCount() > 0) {
    arena_bytes_ = static_cast<clio::run::u64>(config_.staging_mb_) << 20;
    arena_ = arena_bytes_ ? ctp::GpuApi::Malloc<char>(arena_bytes_) : nullptr;
    if (arena_ != nullptr) {
      arena_free_[0] = arena_bytes_;
    } else {
      arena_bytes_ = 0;
    }
    for (size_t i = 0; i < kJobArrays; ++i) {
      auto *j = ctp::GpuApi::MallocHost<glz::DecodeJob>(
          sizeof(glz::DecodeJob) * clio::cte::core::kPodMultiMax);
      if (j == nullptr) break;
      jobs_all_.push_back(j);
      jobs_free_.push_back(j);
    }
    HLOG(kInfo,
         "lz4: GPU decode ready ({} MB staging, {} job arrays, depth {})",
         arena_bytes_ >> 20, jobs_free_.size(), config_.depth_);
  }
#endif
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Destroy(clio::run::shared_ptr<DestroyTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Monitor(clio::run::shared_ptr<MonitorTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::cte::core::Client *Runtime::GetNextClient() {
  if (!next_client_) {
    next_client_ = std::make_unique<clio::cte::core::Client>(CorePoolId());
  }
  return next_client_.get();
}

bool Runtime::WantsLz4(const Context &ctx) {
  return (ctx.compress_lib_ == kGpuLz4Lib || ctx.compress_lib_ == kGpuAnsLib) &&
         ctx.replica_ == 0 && !ctx.emulate_;
}

// ------------------------------------------------------------ resources

clio::run::u64 Runtime::ArenaAcquire(clio::run::u64 bytes) {
  bytes = (bytes + kArenaAlign - 1) / kArenaAlign * kArenaAlign;
  std::lock_guard<std::mutex> g(arena_mu_);
  for (auto it = arena_free_.begin(); it != arena_free_.end(); ++it) {
    if (it->second < bytes) continue;
    const clio::run::u64 off = it->first;
    const clio::run::u64 rest = it->second - bytes;
    arena_free_.erase(it);
    if (rest > 0) arena_free_[off + bytes] = rest;
    return off;
  }
  return ~0ull;
}

void Runtime::ArenaRelease(clio::run::u64 off, clio::run::u64 bytes) {
  bytes = (bytes + kArenaAlign - 1) / kArenaAlign * kArenaAlign;
  std::lock_guard<std::mutex> g(arena_mu_);
  auto it = arena_free_.emplace(off, bytes).first;
  // Coalesce with the following and the preceding free range.
  auto next = std::next(it);
  if (next != arena_free_.end() && it->first + it->second == next->first) {
    it->second += next->second;
    arena_free_.erase(next);
  }
  if (it != arena_free_.begin()) {
    auto prev = std::prev(it);
    if (prev->first + prev->second == it->first) {
      prev->second += it->second;
      arena_free_.erase(it);
    }
  }
}

void *Runtime::AcquireJobs() {
  std::lock_guard<std::mutex> g(jobs_mu_);
  if (jobs_free_.empty()) return nullptr;
  void *j = jobs_free_.back();
  jobs_free_.pop_back();
  return j;
}

void Runtime::ReleaseJobs(void *jobs) {
  std::lock_guard<std::mutex> g(jobs_mu_);
  jobs_free_.push_back(jobs);
}

// ------------------------------------------------------------ puts

clio::run::TaskResume Runtime::EncodeAndPut(
    const TagId &tag_id, const std::string &name, const char *src,
    clio::run::u64 n, float score, Context ctx, clio::run::u32 flags,
    clio::run::u32 &rc, Context &out_ctx) {
  CLIO_TASK_BODY_BEGIN
  std::vector<uint8_t> enc;
  if (ctx.compress_lib_ == kGpuAnsLib) {
    glz::EncodePageAns(src, n, enc);
  } else {
    glz::EncodePage(src, n, enc, static_cast<int>(config_.depth_));
  }
  const bool shrank = enc.size() < n;
  // A page that does not shrink is stored raw, recorded as untransformed.
  const char *bytes = shrank ? reinterpret_cast<const char *>(enc.data()) : src;
  const clio::run::u64 len = shrank ? enc.size() : n;
  if (shrank) {
    ctx.transform_flags_ |= (kBlobTransformed | kBlobTransformCompressed);
    stats_.encoded.fetch_add(1, std::memory_order_relaxed);
    stats_.raw_in.fetch_add(n, std::memory_order_relaxed);
    stats_.enc_out.fetch_add(enc.size(), std::memory_order_relaxed);
  } else {
    ctx.compress_lib_ = 0;
    stats_.put_raw.fetch_add(1, std::memory_order_relaxed);
  }
  auto buf = CLIO_IPC->AllocateBuffer(len);
  if (buf.IsNull()) {
    rc = 4;
    CLIO_CO_RETURN;
  }
  std::memcpy(buf.ptr_, bytes, len);
  auto put = GetNextClient()->AsyncPutBlob(
      tag_id, name, 0, len, buf.shm_.template Cast<void>(), score, ctx, flags,
      clio::run::PoolQuery::Local());
  CLIO_CO_AWAIT(put);
  CLIO_IPC->FreeBuffer(buf);
  rc = put->GetReturnCode();
  out_ctx = put->context_;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::PutBlob(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  Context &ctx = task->context_;
  const bool whole = task->segments_.empty() && task->offset_ == 0;
  auto src = CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
  if (!WantsLz4(ctx) || !whole || task->size_ == 0 || src.ptr_ == nullptr) {
    // A partial write cannot patch an encoded page: such a put stores raw
    // bytes and must never be recorded as carrying the codec.
    if (ctx.compress_lib_ == kGpuLz4Lib || ctx.compress_lib_ == kGpuAnsLib) {
      ctx.compress_lib_ = 0;
    }
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPutBlob,
                                task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  // The encoder reads host memory. A device-resident source is staged first
  // (a pinned or registered host buffer is readable either way).
  std::vector<char> staged;
  const char *bytes = src.ptr_;
  if (ctp::IsDeviceAccessible(src.ptr_)) {
    staged.resize(static_cast<size_t>(task->size_));
    ctp::GpuApi::Memcpy(staged.data(), static_cast<const char *>(src.ptr_),
                        staged.size());
    bytes = staged.data();
  }
  clio::run::u32 rc = 0;
  Context out_ctx;
  CLIO_CO_AWAIT(EncodeAndPut(task->tag_id_, task->GetBlobName(), bytes,
                             task->size_, task->score_, ctx, task->flags_, rc,
                             out_ctx));
  task->context_ = out_ctx;
  task->return_code_ = rc;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::PodPutBlob(
    clio::run::shared_ptr<clio::cte::core::PodPutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->NormalizeBlobName();
  Context &ctx = task->context_;
  auto src = CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
  if (!WantsLz4(ctx) || task->offset_ != 0 || task->size_ == 0 ||
      src.ptr_ == nullptr) {
    if (ctx.compress_lib_ == kGpuLz4Lib || ctx.compress_lib_ == kGpuAnsLib) {
      ctx.compress_lib_ = 0;
    }
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPodPutBlob,
                                task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  // A device producer's page: stage it to the host for the encoder.
  std::vector<char> host(static_cast<size_t>(task->size_));
  ctp::GpuApi::Memcpy(host.data(), src.ptr_, host.size());
  clio::run::u32 rc = 0;
  Context out_ctx;
  CLIO_CO_AWAIT(EncodeAndPut(task->tag_id_, task->GetBlobName(), host.data(),
                             task->size_, task->score_, ctx, task->flags_, rc,
                             out_ctx));
  task->context_ = out_ctx;
  task->return_code_ = rc;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ------------------------------------------------------------ sizes

clio::run::TaskResume Runtime::GetBlobSize(
    clio::run::shared_ptr<clio::cte::core::GetBlobSizeTask> &task) {
  CLIO_TASK_BODY_BEGIN
  CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlobSize,
                              task.template Cast<clio::run::Task>()));
  // An encoded page's stored size is not its logical size: read the header
  // (its OUT context says whether the blob really is encoded).
  if (task->GetReturnCode() == 0 && task->replica_ == 0 &&
      task->size_ >= sizeof(glz::Header)) {
    auto hdr = CLIO_IPC->AllocateBuffer(sizeof(glz::Header));
    if (!hdr.IsNull()) {
      auto get = GetNextClient()->AsyncGetBlob(
          task->tag_id_, task->blob_name_.str(), 0, sizeof(glz::Header), 0,
          hdr.shm_.template Cast<void>(), clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(get);
      if (get->GetReturnCode() == 0 && StoredCompressed(get->context_)) {
        const clio::run::u64 raw = glz::RawSizeOf(
            reinterpret_cast<const uint8_t *>(hdr.ptr_), sizeof(glz::Header));
        if (raw != 0) task->size_ = raw;
      }
      CLIO_IPC->FreeBuffer(hdr);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ------------------------------------------------------------ reads

void Runtime::MaybeLogStats(clio::run::u64 every) {
  // CLIO_LZ4_STATS=N logs every N device reads (N <= 0: every `every`).
  static const long env = [] {
    const char *e = std::getenv("CLIO_LZ4_STATS");
    return e == nullptr ? -1L : std::atol(e);
  }();
  if (env < 0) return;
  if (env > 0) every = static_cast<clio::run::u64>(env);
  static std::atomic<clio::run::u64> calls{0};
  if ((calls.fetch_add(1, std::memory_order_relaxed) + 1) % every != 0) return;
  const clio::run::u64 enc = stats_.enc_out.load();
  HLOG(kInfo,
       "lz4: puts {} encoded ({:.3f}x) / {} raw; device reads: {} from a "
       "device tier, {} staged, {} via host",
       stats_.encoded.load(),
       enc ? static_cast<double>(stats_.raw_in.load()) / enc : 0.0,
       stats_.put_raw.load(), stats_.dev_tier.load(), stats_.dev_staged.load(),
       stats_.host.load());
}

clio::run::TaskResume Runtime::ReadViaHost(const TagId &tag_id,
                                           const Context &ctx,
                                           clio::run::u32 flags, DevRead &rec) {
  CLIO_TASK_BODY_BEGIN
  // The slow path: partial reads, pages stored raw, and anything the device
  // path could not serve. Stored bytes come to the host, decode there.
  auto sz = GetNextClient()->AsyncGetBlobSize(tag_id, rec.name);
  CLIO_CO_AWAIT(sz);
  if (sz->GetReturnCode() != 0) {
    rec.rc = sz->GetReturnCode();
    CLIO_CO_RETURN;
  }
  const clio::run::u64 stored = sz->size_;
  auto buf = CLIO_IPC->AllocateBuffer(stored > 0 ? stored : 1);
  if (buf.IsNull()) {
    rec.rc = 4;
    CLIO_CO_RETURN;
  }
  auto get = GetNextClient()->AsyncGetBlob(
      tag_id, rec.name, 0, stored, flags, buf.shm_.template Cast<void>(),
      clio::run::PoolQuery::Local(), ctx);
  CLIO_CO_AWAIT(get);
  rec.rc = get->GetReturnCode();
  std::vector<uint8_t> raw;
  const char *bytes = buf.ptr_;
  clio::run::u64 avail = stored;
  if (rec.rc == 0 && StoredCompressed(get->context_)) {
    const auto *p = reinterpret_cast<const uint8_t *>(buf.ptr_);
    raw.resize(static_cast<size_t>(glz::RawSizeOf(p, stored)));
    if (raw.empty() ||
        !glz::DecodePageAny(p, stored, raw.data(), raw.size())) {
      HLOG(kError, "lz4: page '{}' failed to decode", rec.name);
      rec.rc = 1;
    }
    bytes = reinterpret_cast<const char *>(raw.data());
    avail = raw.size();
  }
  if (rec.rc == 0 && rec.offset < avail) {
    const clio::run::u64 n = std::min(rec.size, avail - rec.offset);
    char *dst = CLIO_IPC->ToFullPtr<char>(rec.data.template Cast<char>()).ptr_;
    if (dst == nullptr) {
      rec.rc = 1;
    } else if (ctp::IsDeviceAccessible(dst)) {
      ctp::GpuApi::Memcpy(dst, bytes + rec.offset, n);  // fallback only
    } else {
      std::memcpy(dst, bytes + rec.offset, n);
    }
  }
  CLIO_IPC->FreeBuffer(buf);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::LaunchAndWait(void *jobs, clio::run::u32 njobs,
                                             bool &launched) {
  CLIO_TASK_BODY_BEGIN
  launched = false;
#if CTP_ENABLE_GPU
  // A pooled stream, never a new one: creating a stream while a consumer
  // kernel is resident can block until that kernel finishes.
  void *stream = nullptr;
  while ((stream = ctp::GpuApi::BorrowStream()) == nullptr) {
    CLIO_CO_AWAIT(clio::run::yield(kPollUs));
  }
  if (ctp_gpu_lz4_decode_launch(static_cast<glz::DecodeJob *>(jobs), njobs,
                                stream) == 0) {
    launched = true;
    while (!ctp::GpuApi::StreamQuery(stream)) {
      CLIO_CO_AWAIT(clio::run::yield(kPollUs));
    }
  }
  ctp::GpuApi::ReturnStream(stream);
#else
  (void)jobs;
  (void)njobs;
#endif
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::StageRecords(
    const TagId &tag_id, const Context &ctx, std::vector<DevRead> &recs,
    const std::vector<size_t> &idx, const std::vector<clio::run::u64> &stored,
    std::vector<const char *> &src, clio::run::u64 &arena_off,
    clio::run::u64 &arena_len) {
  CLIO_TASK_BODY_BEGIN
  // ONE contiguous staging range for every page that is not in a device
  // tier, then all their gets in flight at once.
  arena_len = 0;
  for (size_t i : idx) {
    arena_len += (stored[i] + kArenaAlign - 1) / kArenaAlign * kArenaAlign;
  }
  arena_off = ~0ull;
  if (arena_ != nullptr && arena_len > 0 && arena_len <= arena_bytes_) {
    const auto t0 = std::chrono::steady_clock::now();
    while ((arena_off = ArenaAcquire(arena_len)) == ~0ull &&
           std::chrono::steady_clock::now() - t0 < kArenaWaitMax) {
      CLIO_CO_AWAIT(clio::run::yield(kPollUs));
    }
  }
  if (arena_off == ~0ull) {
    arena_len = 0;
    CLIO_CO_RETURN;  // these records go via the host
  }
  using GetFut = decltype(GetNextClient()->AsyncGetBlob(
      tag_id, std::string(), clio::run::u64(0), clio::run::u64(0),
      clio::run::u32(0), ctp::ipc::ShmPtr<>(), clio::run::PoolQuery::Local(),
      ctx));
  std::vector<GetFut> futs;
  std::vector<const char *> at;
  clio::run::u64 pos = arena_off;
  for (size_t i : idx) {
    const char *slot = arena_ + pos;
    at.push_back(slot);
    futs.push_back(GetNextClient()->AsyncGetBlob(
        tag_id, recs[i].name, 0, stored[i], 0, RawPtr(slot),
        clio::run::PoolQuery::Local(), ctx));
    pos += (stored[i] + kArenaAlign - 1) / kArenaAlign * kArenaAlign;
  }
  for (auto &f : futs) CLIO_CO_AWAIT(f);
  for (size_t k = 0; k < idx.size(); ++k) {
    if (futs[k]->GetReturnCode() == 0 && StoredCompressed(futs[k]->context_)) {
      src[idx[k]] = at[k];
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ReadToDevice(const TagId &tag_id,
                                            const Context &ctx,
                                            clio::run::u32 flags,
                                            std::vector<DevRead> &recs) {
  CLIO_TASK_BODY_BEGIN
  const size_t n = recs.size();
  std::vector<const char *> src(n, nullptr);
  std::vector<clio::run::u64> stored(n, 0);
  std::vector<size_t> stage;
  void *jobs = nullptr;
#if CTP_ENABLE_GPU
  while (!jobs_all_.empty() && (jobs = AcquireJobs()) == nullptr) {
    CLIO_CO_AWAIT(clio::run::yield(kPollUs));
  }
#endif
  for (size_t i = 0; jobs != nullptr && i < n; ++i) {
    if (recs[i].offset != 0) continue;  // partial: host path
    unsigned long long pool = 0, off = 0, sz = 0;
    if (clio_cte_locate(&tag_id, recs[i].name.c_str(), &pool, &off, &sz) != 0 ||
        sz == 0) {
      continue;  // unknown or split across extents: host path
    }
    stored[i] = sz;
    // On a device tier the stored bytes are decoded where they lie.
    char *tier = clio_direct_dev_base(pool);
    if (tier != nullptr) {
      src[i] = tier + off;
    } else {
      stage.push_back(i);
    }
  }
  clio::run::u64 arena_off = ~0ull, arena_len = 0;
  if (!stage.empty()) {
    CLIO_CO_AWAIT(StageRecords(tag_id, ctx, recs, stage, stored, src,
                               arena_off, arena_len));
  }
  // ONE launch for every page whose stored bytes are now in device memory.
  std::vector<size_t> job_of;
  auto *jv = static_cast<glz::DecodeJob *>(jobs);
  for (size_t i = 0; jobs != nullptr && i < n; ++i) {
    if (src[i] == nullptr) continue;
    jv[job_of.size()] = glz::DecodeJob{
        reinterpret_cast<const uint8_t *>(src[i]), stored[i],
        reinterpret_cast<uint8_t *>(recs[i].dst), recs[i].size, 1, 0};
    job_of.push_back(i);
  }
  bool launched = false;
  if (!job_of.empty()) {
    CLIO_CO_AWAIT(LaunchAndWait(
        jobs, static_cast<clio::run::u32>(job_of.size()), launched));
  }
  std::vector<bool> done(n, false);
  std::vector<bool> staged(n, false);
  for (size_t i : stage) staged[i] = true;
  for (size_t k = 0; launched && k < job_of.size(); ++k) {
    done[job_of[k]] = jv[k].ok_ == 1u;
    if (done[job_of[k]]) {
      (staged[job_of[k]] ? stats_.dev_staged : stats_.dev_tier)
          .fetch_add(1, std::memory_order_relaxed);
    }
  }
  if (arena_len > 0) ArenaRelease(arena_off, arena_len);
  if (jobs != nullptr) ReleaseJobs(jobs);
  // Everything the device path did not serve: a page stored raw (never
  // encoded, or not worth encoding), a partial read, a page not yet placed.
  for (size_t i = 0; i < n; ++i) {
    if (done[i]) {
      recs[i].rc = 0;
    } else {
      stats_.host.fetch_add(1, std::memory_order_relaxed);
      CLIO_CO_AWAIT(ReadViaHost(tag_id, ctx, flags, recs[i]));
    }
  }
  MaybeLogStats(4096);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetBlob(
    clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Serve as-is first: an unencoded blob (and every replica-addressed or
  // vectored read) costs nothing extra; the OUT context says what it was.
  CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlob,
                              task.template Cast<clio::run::Task>()));
  if (task->GetReturnCode() != 0 || task->context_.replica_ != 0 ||
      !task->segments_.empty() || !StoredCompressed(task->context_)) {
    CLIO_CO_RETURN;
  }
  DevRead rec;
  rec.name = task->GetBlobName();
  rec.offset = task->offset_;
  rec.size = task->size_;
  rec.data = task->blob_data_;
  CLIO_CO_AWAIT(ReadViaHost(task->tag_id_, task->context_, task->flags_, rec));
  task->return_code_ = rec.rc;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::PodGetBlob(
    clio::run::shared_ptr<clio::cte::core::PodGetBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->NormalizeBlobName();
  if (!WantsLz4(task->context_)) {
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPodGetBlob,
                                task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  std::vector<DevRead> recs(1);
  recs[0].name = task->GetBlobName();
  recs[0].offset = task->offset_;
  recs[0].size = task->size_;
  recs[0].data = task->blob_data_;
  recs[0].dst =
      CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>()).ptr_;
  if (recs[0].dst != nullptr && ctp::IsDeviceAccessible(recs[0].dst)) {
    CLIO_CO_AWAIT(ReadToDevice(task->tag_id_, task->context_, task->flags_,
                               recs));
  } else {
    CLIO_CO_AWAIT(ReadViaHost(task->tag_id_, task->context_, task->flags_,
                              recs[0]));
  }
  task->return_code_ = recs[0].rc;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::PodMultiGetBlob(
    clio::run::shared_ptr<clio::cte::core::PodMultiGetBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->NormalizeBlobName();
  if (!WantsLz4(task->context_)) {
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPodMultiGetBlob,
                                task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  const clio::run::u32 n =
      std::min<clio::run::u32>(task->count_, clio::cte::core::kPodMultiMax);
  std::vector<DevRead> dev, host;
  std::vector<clio::run::u32> dev_i, host_i;
  for (clio::run::u32 i = 0; i < n; ++i) {
    auto &req = task->reqs_[i];
    DevRead r;
    r.name = std::string(req.blob_name_.c_str());
    r.offset = req.offset_;
    r.size = req.size_;
    r.data = req.data_;
    r.dst = CLIO_IPC->ToFullPtr<char>(req.data_.template Cast<char>()).ptr_;
    const bool on_dev = r.dst != nullptr && ctp::IsDeviceAccessible(r.dst);
    (on_dev ? dev : host).push_back(r);
    (on_dev ? dev_i : host_i).push_back(i);
  }
  if (!dev.empty()) {
    CLIO_CO_AWAIT(ReadToDevice(task->tag_id_, task->context_, task->flags_,
                               dev));
  }
  for (auto &r : host) {
    CLIO_CO_AWAIT(ReadViaHost(task->tag_id_, task->context_, task->flags_, r));
  }
  task->num_ok_ = 0;
  clio::run::u32 first_rc = 0;
  auto settle = [&](const std::vector<DevRead> &v,
                    const std::vector<clio::run::u32> &at) {
    for (size_t k = 0; k < v.size(); ++k) {
      task->reqs_[at[k]].rc_ = v[k].rc;
      if (v[k].rc == 0) {
        ++task->num_ok_;
      } else if (first_rc == 0) {
        first_rc = v[k].rc;
      }
    }
  };
  settle(dev, dev_i);
  settle(host, host_i);
  task->SetReturnCode(task->num_ok_ == n ? 0 : first_rc);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::lz4

// ChiMod entry points (alloc/new/name/destroy) for the module manager.
CLIO_TASK_CC(clio::cte::lz4::Runtime)
