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

#include <clio_cae/core/core_runtime.h>
#include <clio_cae/core/factory/assimilation_ctx.h>
#include <clio_cae/core/factory/assimilator_factory.h>
#ifdef CLIO_CAE_ENABLE_HDF5
#include <hdf5.h>
#include <clio_cae/core/factory/hdf5_file_assimilator.h>
// ChunkedDatasetWriter + DTypeToH5: the HDF5-writing half of ExportData, kept out
// of this file so it is testable without a runtime, a server, or a GPU.
#include <clio_cae/core/factory/hdf5_export.h>
// ChunkedDatasetReader + H5ToDType: the HDF5-reading half of ImportData, kept out
// for the same reason.
#include <clio_cae/core/factory/hdf5_import.h>
#endif

#include "clio_ctp/data_structures/serialization/global_serialize.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <climits>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <cstring>
#include "clio_ctp/util/gpu_api.h"
#include <fstream>
#include <vector>

// Include clio_cte headers before opening namespace to avoid Method namespace
// collision
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

// The kvhdf5 dataset self-description (the __meta blob), which ExportData reads
// to reconstruct a real N-D dataset. This header is a deliberate LEAF -- no CUDA,
// no other kvhdf5 headers -- precisely so this CPU-only runtime can read the
// format without acquiring a CUDA include dependency. Do not reach for any other
// kvhdf5 header here; the rest of them pull in <cuda/std/*> via defines.h.
#include <clio_cte/kvhdf5/meta_blob.h>

// Define ChiMod entry points using CLIO_TASK_CC macro
CLIO_TASK_CC(clio::cae::core::Runtime)

namespace clio::cae::core {

// LCOV_EXCL_START: admin Monitor RPC hook — a no-op status handler that the
// unit-test suite never dispatches (there is no monitoring client here).
clio::run::TaskResume Runtime::Monitor(clio::run::shared_ptr<MonitorTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}
// LCOV_EXCL_STOP

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Container is already initialized via Init() before Create is called
  // Do NOT call Init() here

  // Load create-time configuration. In compose mode this materializes the
  // PoolConfig stored on the task into a CreateParams via LoadConfig.
  CreateParams params = task->GetParams();
  next_pool_id_ = params.next_pool_id_;

  // Initialize CTE client. When CAE is wired as a transparent interceptor
  // (compose YAML sets next_pool_id), forward to that pool instead of the
  // hard-coded kCtePoolId; otherwise fall back to the legacy default so
  // existing CAE deployments keep working.
  clio::run::PoolId cte_pool = !next_pool_id_.IsNull()
                            ? next_pool_id_
                            : clio::cte::core::kCtePoolId;
  cte_client_ = std::make_shared<clio::cte::core::Client>(cte_pool);

  HLOG(kInfo,
       "Core container created and initialized for pool: {} (ID: {}), "
       "CTE next_pool_id={}",
       pool_name_, pool_id_, cte_pool);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// LCOV_EXCL_START: unreachable in practice. Create() always constructs
// cte_client_ up front, so the `if (!cte_client_)` fallbacks in the forward
// handlers never call ResolveNextPoolId(). Kept as defensive belt-and-braces.
clio::run::PoolId Runtime::ResolveNextPoolId() const {
  if (!next_pool_id_.IsNull()) return next_pool_id_;
  return clio::cte::core::kCtePoolId;
}
// LCOV_EXCL_STOP

clio::run::PoolQuery Runtime::ScheduleTask(
    const clio::run::shared_ptr<clio::run::Task> &task) {
  // Interceptor methods always run locally — they immediately forward
  // synchronously to the configured CTE core pool, so there is no value
  // in bouncing the task across nodes. Mirrors compressor_runtime.cc
  // which routes non-Compress Dynamic methods to Local.
  switch (task->method_) {
    case Method::kPutBlob:
    case Method::kGetBlob:
    case Method::kGetOrCreateTag:
    case Method::kSemanticSearch:
      return clio::run::PoolQuery::Local();
    default:
      return clio::run::PoolQuery::Local();
  }
}

clio::run::TaskResume Runtime::PutBlob(clio::run::shared_ptr<PutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (!cte_client_) {
    cte_client_ = std::make_shared<clio::cte::core::Client>(ResolveNextPoolId());
  }
  // Pure forward. CO_AWAIT yields the worker while the downstream CTE task
  // runs — Wait() would block this worker and deadlock when only a single
  // worker is available (or when this and the downstream task happen to land
  // on the same worker).
  auto fwd = cte_client_->AsyncPutBlob(
      task->tag_id_, task->blob_name_.str(), task->offset_, task->size_,
      task->blob_data_, task->score_, task->context_, task->flags_,
      clio::run::PoolQuery::Local());
  CLIO_CO_AWAIT(fwd);
  task->context_ = fwd->context_;
  task->SetReturnCode(fwd->GetReturnCode());
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

namespace {
/**
 * "page:<id>" -> id. The prefix is what makes pagify OPT-IN per request:
 * a deployment that does not use it never sees a behaviour change, and a
 * blob legitimately named like a page is not something the assimilators
 * produce (their names are "<tag>_b<fam>_pi<page>" or "w").
 */
bool ParsePageName(const std::string &name, clio::run::u64 *page_id) {
  static constexpr char kPrefix[] = "page:";
  static constexpr size_t kPrefixLen = sizeof(kPrefix) - 1;
  if (name.size() <= kPrefixLen || name.compare(0, kPrefixLen, kPrefix) != 0) {
    return false;
  }
  clio::run::u64 v = 0;
  for (size_t i = kPrefixLen; i < name.size(); ++i) {
    const char c = name[i];
    if (c < '0' || c > '9') return false;
    v = v * 10 + (clio::run::u64) (c - '0');
  }
  *page_id = v;
  return true;
}
}  // namespace

namespace {
/** Strip "<proto>::" or "<proto>://" from a URL. */
std::string StripProto(const std::string &url) {
  size_t p = url.find("://");
  if (p != std::string::npos) return url.substr(p + 3);
  p = url.find("::");
  if (p != std::string::npos) return url.substr(p + 2);
  return url;
}
}  // namespace

// ---------------------------------------------------------------------------
// Ingest on fault: stage counters (CLIO_CAE_FAULT_STATS=1 logs them every 5 s)
// ---------------------------------------------------------------------------
namespace {
struct FaultStats {
  std::atomic<long long> entered{0}, read_done{0}, stored{0}, store_fail{0},
      done{0}, alloc_fail{0}, read_fail{0};
};
FaultStats &FStats() {
  static FaultStats *s = new FaultStats();
  return *s;
}
void MaybeLogFaultStats() {
  static const bool on = [] {
    const char *e = std::getenv("CLIO_CAE_FAULT_STATS");
    return e != nullptr && *e == '1';
  }();
  if (!on) return;
  static std::atomic<long long> last{0};
  const long long now = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
  long long prev = last.load();
  if (now - prev < 5 || !last.compare_exchange_strong(prev, now)) return;
  FaultStats &f = FStats();
  HLOG(kInfo,
       "CAE_FAULT_STATS entered={} read_done={} stored={} store_fail={} "
       "done={} inflight={} alloc_fail={} read_fail={}",
       f.entered.load(), f.read_done.load(), f.stored.load(),
       f.store_fail.load(), f.done.load(), f.entered.load() - f.done.load(),
       f.alloc_fail.load(), f.read_fail.load());
}
}  // namespace

// ---------------------------------------------------------------------------
// Ingest on fault: context registry
// ---------------------------------------------------------------------------
namespace {
void PutField(std::string *o, const std::string &v) {
  *o += std::to_string(v.size());
  *o += ':';
  *o += v;
}
bool GetField(const std::string &in, size_t *pos, std::string *v) {
  const size_t c = in.find(':', *pos);
  if (c == std::string::npos) return false;
  const size_t n = std::strtoull(in.c_str() + *pos, nullptr, 10);
  if (c + 1 + n > in.size()) return false;
  *v = in.substr(c + 1, n);
  *pos = c + 1 + n;
  return true;
}
}  // namespace

std::string Runtime::EncodeCtx(const AssimilationCtx &c) {
  // Length-prefixed fields: everything an assimilator needs to ingest one
  // blob later, on any node.
  std::string o;
  for (const std::string *f :
       {&c.src, &c.dst, &c.format, &c.src_token, &c.dst_token, &c.s3_region,
        &c.s3_profile, &c.src_data, &c.blob_prefix}) {
    PutField(&o, *f);
  }
  PutField(&o, std::to_string(c.range_off));
  PutField(&o, std::to_string(c.range_size));
  PutField(&o, std::to_string(c.chunk_size));
  return o;
}

bool Runtime::DecodeCtx(const char *buf, size_t n, AssimilationCtx *c) {
  const std::string in(buf, strnlen(buf, n));
  size_t pos = 0;
  for (std::string *f :
       {&c->src, &c->dst, &c->format, &c->src_token, &c->dst_token,
        &c->s3_region, &c->s3_profile, &c->src_data, &c->blob_prefix}) {
    if (!GetField(in, &pos, f)) return false;
  }
  std::string ro, rs, cs;
  if (!GetField(in, &pos, &ro) || !GetField(in, &pos, &rs) ||
      !GetField(in, &pos, &cs)) {
    return false;
  }
  c->range_off = std::strtoull(ro.c_str(), nullptr, 10);
  c->range_size = std::strtoull(rs.c_str(), nullptr, 10);
  c->chunk_size = std::strtoull(cs.c_str(), nullptr, 10);
  c->on_fault = true;
  return true;
}

std::string Runtime::OnFaultId(const AssimilationCtx &ctx) {
  const std::string k = EncodeCtx(ctx);
  clio::run::u64 h = 1469598103934665603ull;  // FNV-1a 64
  for (unsigned char ch : k) {
    h ^= ch;
    h *= 1099511628211ull;
  }
  char id[32];
  std::snprintf(id, sizeof(id), "%s%016llx", kOnFaultPrefix,
                (unsigned long long)h);
  return id;
}

std::shared_ptr<Runtime::OnFault> Runtime::MakeOnFault(
    const AssimilationCtx &ctx) {
#ifdef CLIO_ENABLE_S3_REST
  AssimilatorFactory factory(cte_client_, &s3_conn_pool_);
#else
  AssimilatorFactory factory(cte_client_);
#endif
  std::unique_ptr<BaseAssimilator> a = factory.Get(ctx.src);
  if (!a) return nullptr;
  auto of = std::make_shared<OnFault>();
  of->ctx = ctx;
  of->blob_bytes = a->BlobBytes(ctx);
  of->assimilator = std::move(a);
  if (of->blob_bytes == 0) {
    HLOG(kError, "CAE on_fault: assimilator for '{}' cannot ingest per blob",
         ctx.src);
    return nullptr;
  }
  return of;
}

namespace {
/**
 * Blocking source I/O for by-reference faults runs HERE, never on a runtime
 * worker: a pread on a parallel filesystem can take milliseconds to seconds,
 * and a worker parked in one stalls every task queued behind it (measured:
 * all compute workers stalled on 1-3 s CAE GetBlobs). The fault task hands
 * the read to this pool and yields until it completes.
 */
class RefIoPool {
 public:
  static RefIoPool &Get() {
    // Leaked on purpose: its threads are detached and park on cv_, so
    // destroying the mutex/cv at static teardown hung process exit.
    static RefIoPool *pool = new RefIoPool();
    return *pool;
  }
  void Submit(std::function<void()> fn) {
    {
      std::lock_guard<std::mutex> g(mu_);
      q_.push_back(std::move(fn));
    }
    cv_.notify_one();
  }

 private:
  RefIoPool() {
    const char *e = std::getenv("CLIO_CAE_REF_IO_THREADS");
    unsigned n = (e != nullptr && *e) ? (unsigned)std::strtoul(e, nullptr, 10)
                                      : 32u;
    if (n == 0) n = 1;
    for (unsigned i = 0; i < n; ++i) {
      std::thread([this]() {
        for (;;) {
          std::function<void()> fn;
          {
            std::unique_lock<std::mutex> l(mu_);
            cv_.wait(l, [this]() { return !q_.empty(); });
            fn = std::move(q_.front());
            q_.pop_front();
          }
          fn();
        }
      }).detach();
    }
  }
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> q_;
};
}  // namespace

clio::run::TaskResume Runtime::GetBlob(clio::run::shared_ptr<GetBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (!cte_client_) {
    cte_client_ = std::make_shared<clio::cte::core::Client>(ResolveNextPoolId());
  }
  // INGEST ON FAULT: CTE dispatched a miss on a tag assimilated with
  // on_fault. Have the source's assimilator ingest exactly this blob, store
  // it in the tag, and serve the read from it.
  if (std::strncmp(task->context_.fault_params_, kOnFaultPrefix,
                   std::strlen(kOnFaultPrefix)) == 0) {
    const std::string key(task->context_.fault_params_);
    std::shared_ptr<OnFault> of;
    {
      std::lock_guard<std::mutex> g(on_fault_mu_);
      auto it = on_fault_.find(key);
      if (it != on_fault_.end()) of = it->second;
    }
    if (!of) {
      // Registered on another node: load the context from the registry.
      auto tf = cte_client_->AsyncGetOrCreateTag(kOnFaultRegistryTag);
      CLIO_CO_AWAIT(tf);
      auto dbuf = CLIO_IPC->AllocateBuffer(kOnFaultDescBytes);
      if (tf->GetReturnCode() != 0 || dbuf.IsNull()) {
        task->SetReturnCode(1);
        CLIO_CO_RETURN;
      }
      std::memset(dbuf.ptr_, 0, kOnFaultDescBytes);
      clio::cte::core::Context gctx;
      gctx.op_flags_ |= clio::cte::core::Context::kNoFault;
      auto gf = cte_client_->AsyncGetBlob(
          tf->tag_id_, key.c_str(), 0, kOnFaultDescBytes, 0,
          dbuf.shm_.template Cast<void>(), clio::run::PoolQuery::Dynamic(),
          gctx);
      CLIO_CO_AWAIT(gf);
      AssimilationCtx ctx;
      const bool ok = gf->GetReturnCode() == 0 &&
                      DecodeCtx(dbuf.ptr_, kOnFaultDescBytes, &ctx);
      CLIO_IPC->FreeBuffer(dbuf);
      if (ok) of = MakeOnFault(ctx);
      if (!of) {
        HLOG(kError, "CAE on_fault: no context for {}", key);
        task->SetReturnCode(1);
        CLIO_CO_RETURN;
      }
      std::lock_guard<std::mutex> g(on_fault_mu_);
      auto ins = on_fault_.emplace(key, of);
      of = ins.first->second;  // a concurrent fault may have won
    }
    const clio::run::u64 cap = of->blob_bytes;
    FStats().entered++;
    MaybeLogFaultStats();
    auto stage = CLIO_IPC->AllocateBuffer(cap);
    if (stage.IsNull()) {
      FStats().alloc_fail++;
      FStats().done++;
      task->SetReturnCode(4);
      CLIO_CO_RETURN;
    }
    // The assimilator's read is blocking source I/O: run it off the workers.
    auto st = std::make_shared<std::atomic<long long>>(LLONG_MIN);
    {
      const std::string bname = task->GetBlobName();
      char *buf = stage.ptr_;
      RefIoPool::Get().Submit([of, bname, buf, cap, st]() {
        st->store(of->assimilator->ReadBlob(of->ctx, bname, buf, cap),
                  std::memory_order_release);
      });
      clio::run::u64 polls = 0;
      while (st->load(std::memory_order_acquire) == LLONG_MIN) {
        CLIO_CO_AWAIT(clio::run::yield(polls++ < 64 ? 20.0 : 200.0));
      }
    }
    const long long n = st->load();
    FStats().read_done++;
    if (n < 0) {
      FStats().read_fail++;
      FStats().done++;
      HLOG(kError, "CAE on_fault: ingest of {} from '{}' failed ({})",
           task->GetBlobName(), of->ctx.src, n);
      CLIO_IPC->FreeBuffer(stage);
      task->SetReturnCode(static_cast<clio::run::u32>(-n));
      CLIO_CO_RETURN;
    }
    // A short last blob reads as zeros past the source, like a page.
    if (static_cast<clio::run::u64>(n) < cap) {
      std::memset(stage.ptr_ + n, 0, cap - static_cast<clio::run::u64>(n));
    }
    // CLIO_CAE_FAULT_STORE=0: never store (pure read-through on fault).
    static const bool kStore = [] {
      const char *e = std::getenv("CLIO_CAE_FAULT_STORE");
      return !(e != nullptr && *e == '0');
    }();
    // Never take the last of the tier (see tier_free_): keep this much free
    // for everything that is NOT re-ingestable. CLIO_CAE_FAULT_RESERVE_MB.
    static const long long kReserve = [] {
      const char *e = std::getenv("CLIO_CAE_FAULT_RESERVE_MB");
      return ((e != nullptr && *e) ? std::atoll(e) : 512LL) << 20;
    }();
    // A size-0 "materialise" fault precedes a PARTIAL WRITE: storing the blob
    // IS the write path there, so it always stores (no headroom, no skip).
    const bool materialise = task->size_ == 0 || task->blob_data_.IsNull();
    bool store = n > 0 && (kStore || materialise);
    if (store && !materialise) {
      // Refresh the free-space reading at most every 250 ms, one refresher
      // at a time; everyone else uses (and debits) the cached value.
      const long long now_ns = (long long)std::chrono::duration_cast<
          std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count();
      if (now_ns - tier_free_at_ns_.load() > 250LL * 1000 * 1000 &&
          !tier_refreshing_.exchange(true)) {
        auto lt = cte_client_->AsyncListTargets(clio::run::PoolQuery::Local());
        CLIO_CO_AWAIT(lt);
        long long free_b = 0;
        if (lt->GetReturnCode() == 0) {
          for (size_t ti = 0; ti < lt->target_names_.size(); ++ti) {
            auto gi = cte_client_->AsyncGetTargetInfo(
                lt->target_names_[ti], clio::run::PoolQuery::Local());
            CLIO_CO_AWAIT(gi);
            if (gi->GetReturnCode() == 0) {
              free_b += (long long)gi->remaining_space_;
            }
          }
          tier_free_.store(free_b);
        }
        tier_free_at_ns_.store(now_ns);
        tier_refreshing_.store(false);
      }
      const long long free_now = tier_free_.load();
      if (free_now >= 0 && free_now - (long long)cap < kReserve) {
        store = false;  // keep the headroom: serve from the source only
      } else if (free_now >= 0) {
        tier_free_.fetch_sub((long long)cap);
      }
    }
    if (store && !materialise &&
        of->skip_stores.load(std::memory_order_relaxed) > 0) {
      of->skip_stores.fetch_sub(1, std::memory_order_relaxed);
      store = false;  // tier recently full: serve from the source only
    }
    if (store) {
      // Ingest: the blob now lives in the tag, exactly as eager ingestion
      // would have stored it, so the next read does not fault (kNoFault: the
      // put must not fault back into us).
      clio::cte::core::Context pctx;
      pctx.op_flags_ |= clio::cte::core::Context::kNoFault;
      auto put = cte_client_->AsyncPutBlob(task->tag_id_, task->GetBlobName(),
                                           0, cap,
                                           stage.shm_.template Cast<void>(),
                                           -1.0f, pctx);
      CLIO_CO_AWAIT(put);
      const clio::run::u32 prc = put->GetReturnCode();
      if (prc == 0) FStats().stored++;
      if (prc != 0) {
        FStats().store_fail++;
        // A failed put can leave a partial blob behind; a later read would
        // find it and never fault. Drop it so the next read ingests again.
        auto del = cte_client_->AsyncDelBlob(task->tag_id_,
                                             task->GetBlobName(),
                                             clio::run::PoolQuery::Dynamic());
        CLIO_CO_AWAIT(del);
        of->skip_stores.store(256, std::memory_order_relaxed);
        if (!of->warned_full.exchange(true)) {
          HLOG(kWarning,
               "CAE on_fault: tag of '{}' cannot store more (rc={}); serving "
               "faults from the source, retrying every 256 faults",
               of->ctx.src, prc);
        }
      }
    }
    // Serve the read that faulted (none for a size-0 "materialise" request
    // CTE sends before a partial write: the put above was the point).
    if (task->size_ != 0 && !task->blob_data_.IsNull()) {
      auto dst = CLIO_IPC->ToFullPtr<char>(task->blob_data_.Cast<char>());
      const clio::run::u64 off = task->offset_;
      const clio::run::u64 len =
          off < cap ? std::min<clio::run::u64>(task->size_, cap - off) : 0;
      if (len > 0) {
        if (ctp::IsDeviceAccessible(dst.ptr_)) {
          ctp::DeviceAwareMemcpy(dst.ptr_, stage.ptr_ + off, (size_t)len);
        } else {
          std::memcpy(dst.ptr_, stage.ptr_ + off, (size_t)len);
        }
      }
    }
    CLIO_IPC->FreeBuffer(stage);
    FStats().done++;
    MaybeLogFaultStats();
    task->SetReturnCode(0);
    CLIO_CO_RETURN;
  }
  // pagify: a request named "page:<id>" asks for a UNIFIED PAGE NUMBER, not
  // a blob. The caller (e.g. the gpu_vector) knows only page 0, page 1, ...
  // and it is this layer's job to turn that into the specific blobs that
  // compose the page and to construct it -- including pages made of several
  // blobs, pages holding only part of one, and gaps that must read as zero.
  //
  // Anything not named that way forwards unchanged, so pagify is opt-in per
  // request and costs an unpagified deployment one string comparison.
  {
    clio::run::u64 page_id = 0;
    if (ParsePageName(task->blob_name_.str(), &page_id)) {
      // Load the map once per tag. It is published as a plain blob by the
      // assimilator that wrote the data, so the mapping travels WITH the
      // data instead of being configured separately and drifting from it.
      if (page_maps_.find(task->tag_id_) == page_maps_.end()) {
        auto sz = cte_client_->AsyncGetBlobSize(task->tag_id_, "pagemap");
        CLIO_CO_AWAIT(sz);
        const clio::run::u64 msz = sz->size_;
        if (sz->GetReturnCode() != 0 || msz == 0) {
          HLOG(kWarning, "pagify: tag has no 'pagemap' blob; page read fails");
          task->SetReturnCode(1);
          CLIO_CO_RETURN;
        }
        auto mbuf = CLIO_IPC->AllocateBuffer(msz);
        if (mbuf.IsNull()) {
          task->SetReturnCode(4);
          CLIO_CO_RETURN;
        }
        ctp::ipc::ShmPtr<> mshm(mbuf.shm_);
        auto mget = cte_client_->AsyncGetBlob(task->tag_id_, "pagemap", 0, msz,
                                              0, mshm);
        CLIO_CO_AWAIT(mget);
        if (mget->GetReturnCode() != 0) {
          task->SetReturnCode(mget->GetReturnCode());
          CLIO_CO_RETURN;
        }
        page_maps_[task->tag_id_] = clio::cae::core::Pagify(
            clio::cae::core::PageMap::Parse(
                std::string(mbuf.ptr_, (size_t) msz)));
      }
      const clio::cae::core::Pagify &pg = page_maps_[task->tag_id_];

      auto dst = CLIO_IPC->ToFullPtr<char>(task->blob_data_.Cast<char>());
      const clio::run::u64 psz = pg.PageSize();
      const clio::run::u64 want = task->size_ ? task->size_ : psz;
      if (dst.ptr_ == nullptr || psz == 0 || want > psz) {
        task->SetReturnCode(1);
        CLIO_CO_RETURN;
      }
      // Zero first: a region no extent maps to must read as zero, not as
      // whatever the caller's buffer happened to hold.
      std::memset(dst.ptr_, 0, (size_t) want);

      int rc = 0;
      for (const clio::cae::core::PageSlice &s : pg.Map().ToPage(page_id)) {
        if (s.dst_off >= want) continue;
        const clio::run::u64 n = std::min(s.size, want - s.dst_off);
        // Point the slice read directly at its place in the caller's
        // buffer, so the page is assembled in place with no staging copy.
        ctp::ipc::ShmPtr<> sub(task->blob_data_.alloc_id_,
                               task->blob_data_.off_.load() + s.dst_off);
        auto g = cte_client_->AsyncGetBlob(task->tag_id_, s.blob_name,
                                           s.blob_off, n, 0, sub);
        CLIO_CO_AWAIT(g);
        if (g->GetReturnCode() != 0) {
          rc = g->GetReturnCode();
          break;
        }
      }
      task->SetReturnCode(rc);
      CLIO_CO_RETURN;
    }
  }
  auto fwd = cte_client_->AsyncGetBlob(
      task->tag_id_, task->blob_name_.str(), task->offset_, task->size_,
      task->flags_, task->blob_data_, clio::run::PoolQuery::Local());
  CLIO_CO_AWAIT(fwd);
  task->SetReturnCode(fwd->GetReturnCode());
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetOrCreateTag(
    clio::run::shared_ptr<GetOrCreateTagTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (!cte_client_) {
    cte_client_ = std::make_shared<clio::cte::core::Client>(ResolveNextPoolId());
  }
  std::string tag_name = task->tag_name_.str();
  auto fwd = cte_client_->AsyncGetOrCreateTag(
      tag_name, task->tag_id_, clio::run::PoolQuery::Local());
  CLIO_CO_AWAIT(fwd);
  task->tag_id_ = fwd->tag_id_;
  task->SetReturnCode(fwd->GetReturnCode());
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::SemanticSearch(
    clio::run::shared_ptr<SemanticSearchTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (!cte_client_) {
    cte_client_ = std::make_shared<clio::cte::core::Client>(ResolveNextPoolId());
  }
  // Forward verbatim. CAE doesn't add any semantic-search logic on
  // top of CTE today — the only reason this handler exists is so a
  // CTE client pointed at the CAE entrypoint pool keeps working
  // without any client-side rewiring.
  auto fwd = cte_client_->AsyncSemanticSearch(
      task->tag_regex_.str(), task->blob_regex_.str(),
      task->query_text_.str(), task->k_, clio::run::PoolQuery::Local());
  CLIO_CO_AWAIT(fwd);
  task->results_ = fwd->results_;
  task->SetReturnCode(fwd->GetReturnCode());
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::u64 Runtime::GetWorkRemaining() const {
  // CAE doesn't currently track work remaining
  // Return 0 to indicate no pending work
  return 0;
}

clio::run::TaskResume Runtime::ParseOmni(clio::run::shared_ptr<ParseOmniTask> &task) {
  CLIO_TASK_BODY_BEGIN
  HLOG(kInfo, "ParseOmni called with {} bytes of serialized data",
       task->serialized_ctx_.size());

  // Deserialize the vector of AssimilationCtx
  std::vector<AssimilationCtx> assimilation_contexts;
  try {
    std::string data = task->serialized_ctx_.str();
    std::vector<char> buf(data.begin(), data.end());
    ctp::ipc::GlobalDeserialize<std::vector<char>> ar(buf);
    ar(assimilation_contexts);
  } catch (const std::exception& e) {
    HLOG(kError, "ParseOmni: Failed to deserialize AssimilationCtx vector: {}",
         e.what());
    task->result_code_ = -1;
    task->error_message_ = e.what();
    task->num_tasks_scheduled_ = 0;
    CLIO_CO_RETURN;
  }

  HLOG(kInfo, "ParseOmni: Processing {} assimilation contexts",
       assimilation_contexts.size());

  // Reject unsatisfiable dependencies before anything is scheduled, so a bad
  // OMNI file fails whole instead of half-transferred. Satisfiable ones need no
  // further handling: the loop below runs the contexts in order.
  const std::string dep_error = CheckDependencies(assimilation_contexts);
  if (!dep_error.empty()) {
    HLOG(kError, "ParseOmni: {}", dep_error);
    task->result_code_ = -3;
    task->error_message_ = dep_error;
    task->num_tasks_scheduled_ = 0;
    task->SetReturnCode(static_cast<clio::run::u32>(-3));
    CLIO_CO_RETURN;
  }

  // Process each assimilation context
  clio::run::u32 tasks_scheduled = 0;
#ifdef CLIO_ENABLE_S3_REST
  AssimilatorFactory factory(cte_client_, &s3_conn_pool_);
#else
  AssimilatorFactory factory(cte_client_);
#endif

  for (size_t i = 0; i < assimilation_contexts.size(); ++i) {
    const auto& assimilation_ctx = assimilation_contexts[i];

    HLOG(kInfo,
         "ParseOmni: Processing context {}/{} - src: {}, dst: {}, format: {}",
         i + 1, assimilation_contexts.size(), assimilation_ctx.src,
         assimilation_ctx.dst, assimilation_ctx.format);

    // on_fault: ingest nothing now; register as the tag's fault handler.
    if (assimilation_ctx.on_fault) {
      std::shared_ptr<OnFault> of = MakeOnFault(assimilation_ctx);
      int rc = of ? 0 : -20;
      const std::string id = OnFaultId(assimilation_ctx);
      const std::string tag_name = StripProto(assimilation_ctx.dst);
      if (rc == 0) {
        {
          std::lock_guard<std::mutex> g(on_fault_mu_);
          on_fault_.emplace(id, of);
        }
        // Publish the context so the CAE on any node can serve faults.
        const std::string desc = EncodeCtx(assimilation_ctx);
        auto rt = cte_client_->AsyncGetOrCreateTag(kOnFaultRegistryTag);
        CLIO_CO_AWAIT(rt);
        rc = (int)rt->GetReturnCode();
        auto dbuf = CLIO_IPC->AllocateBuffer(kOnFaultDescBytes);
        if (rc == 0 && (dbuf.IsNull() || desc.size() >= kOnFaultDescBytes)) {
          rc = -13;
        }
        if (rc == 0) {
          std::memset(dbuf.ptr_, 0, kOnFaultDescBytes);
          std::memcpy(dbuf.ptr_, desc.data(), desc.size());
          clio::cte::core::Context pctx;
          pctx.op_flags_ |= clio::cte::core::Context::kNoFault;
          auto pf = cte_client_->AsyncPutBlob(
              rt->tag_id_, id.c_str(), 0, kOnFaultDescBytes,
              dbuf.shm_.template Cast<void>(), -1.0f, pctx);
          CLIO_CO_AWAIT(pf);
          rc = (int)pf->GetReturnCode();
        }
        if (!dbuf.IsNull()) CLIO_IPC->FreeBuffer(dbuf);
      }
      if (rc == 0) {
        // Register the handler for the canonical tag on EVERY CTE container:
        // a read is resolved by the container it lands on.
        auto tf = cte_client_->AsyncGetOrCreateTag(tag_name);
        CLIO_CO_AWAIT(tf);
        rc = (int)tf->GetReturnCode();
        const clio::cte::core::TagId tid = tf->tag_id_;
        const clio::run::PoolInfo *cinfo =
            CLIO_POOL_MANAGER->GetPoolInfo(ResolveNextPoolId());
        const clio::run::u32 ncont =
            cinfo != nullptr ? cinfo->num_containers_ : 1;
        for (clio::run::u32 c = 0; rc == 0 && c < ncont; ++c) {
          auto cf = cte_client_->AsyncGetOrCreateTag(
              tag_name, pool_id_, pool_name_, id, tid,
              clio::run::PoolQuery::DirectId(
                  static_cast<clio::run::ContainerId>(c)));
          CLIO_CO_AWAIT(cf);
          rc = (int)cf->GetReturnCode();
        }
      }
      if (rc != 0) {
        task->result_code_ = rc;
        task->error_message_ = "on_fault assimilation setup failed";
        task->num_tasks_scheduled_ = tasks_scheduled;
        task->SetReturnCode(static_cast<clio::run::u32>(rc));
        CLIO_CO_RETURN;
      }
      HLOG(kInfo, "CAE on_fault: '{}' -> tag '{}' ({}), {} B per blob",
           assimilation_ctx.src, tag_name, id, of->blob_bytes);
      tasks_scheduled++;
      continue;
    }

    // Get appropriate assimilator for this context
    auto assimilator = factory.Get(assimilation_ctx.src);

    if (!assimilator) {
      HLOG(kError, "ParseOmni: No assimilator found for source: {}",
           assimilation_ctx.src);
      task->result_code_ = -2;
      task->error_message_ =
          "No assimilator found for source: " + assimilation_ctx.src;
      task->num_tasks_scheduled_ = tasks_scheduled;
      // Mirror onto the task-framework code: clients that check GetReturnCode()
      // would otherwise see 0 and report success.
      task->SetReturnCode(static_cast<clio::run::u32>(-2));
      CLIO_CO_RETURN;
    }

    // Schedule the assimilation using co_await
    int result = 0;
    CLIO_CO_AWAIT(assimilator->Schedule(assimilation_ctx, result));
    if (result != 0) {
      HLOG(
          kError,
          "ParseOmni: Assimilator failed for context {}/{} with error code: {}",
          i + 1, assimilation_contexts.size(), result);
      task->result_code_ = result;
      task->error_message_ = std::string("Assimilator failed");
      task->num_tasks_scheduled_ = tasks_scheduled;
      task->SetReturnCode(static_cast<clio::run::u32>(result));
      CLIO_CO_RETURN;
    }

    tasks_scheduled++;
  }

  // Success
  task->result_code_ = 0;
  task->error_message_ = "";
  task->num_tasks_scheduled_ = tasks_scheduled;

  HLOG(kInfo, "ParseOmni: Successfully scheduled {} assimilations",
       tasks_scheduled);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ProcessHdf5Dataset(
    clio::run::shared_ptr<ProcessHdf5DatasetTask> &task) {
  CLIO_TASK_BODY_BEGIN
#ifdef CLIO_CAE_ENABLE_HDF5
  HLOG(kInfo, "ProcessHdf5Dataset: file='{}', dataset='{}', tag_prefix='{}'",
       task->file_path_.str(), task->dataset_path_.str(),
       task->tag_prefix_.str());

  // Open the HDF5 file
  hid_t file_id =
      H5Fopen(task->file_path_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    HLOG(kError, "ProcessHdf5Dataset: Failed to open HDF5 file: {}",
         task->file_path_.str());
    task->result_code_ = -1;
    task->error_message_ =
        clio::run::priv::string("Failed to open HDF5 file", CTP_MALLOC);
    CLIO_CO_RETURN;
  }

  // Create assimilator and process the dataset
  clio::cae::core::Hdf5FileAssimilator assimilator(cte_client_);
  int result = 0;
  CLIO_CO_AWAIT(assimilator.ProcessDataset(file_id, task->dataset_path_.str(),
                                      task->tag_prefix_.str(), result));

  // Close the HDF5 file
  H5Fclose(file_id);

  if (result != 0) {
    HLOG(kError,
         "ProcessHdf5Dataset: Failed to process dataset '{}' (error: {})",
         task->dataset_path_.str(), result);
    task->result_code_ = result;
    task->error_message_ =
        clio::run::priv::string("Dataset processing failed", CTP_MALLOC);
  } else {
    HLOG(kInfo, "ProcessHdf5Dataset: Successfully processed dataset '{}'",
         task->dataset_path_.str());
    task->result_code_ = 0;
  }
#else
  task->result_code_ = -1;
  task->error_message_ =
      clio::run::priv::string("HDF5 support not compiled in", CTP_MALLOC);
#endif
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ExportData(clio::run::shared_ptr<ExportDataTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->result_code_ = 0;
  task->bytes_exported_ = 0;

  const std::string tag_name = task->tag_name_.str();
  const std::string output_path = task->output_path_.str();
  const std::string format = task->format_.str();

  HLOG(kInfo, "ExportData: tag='{}', output='{}', format='{}'",
       tag_name, output_path, format);

  // Step 1: resolve the tag ID
  auto tag_future = cte_client_->AsyncGetOrCreateTag(tag_name);
  CLIO_CO_AWAIT(tag_future);
  const auto &tag_id = tag_future->tag_id_;
  if (tag_id.IsNull()) {
    HLOG(kError, "ExportData: tag '{}' not found", tag_name);
    task->result_code_ = -1;
    task->error_message_ = clio::run::priv::string("Tag not found", CTP_MALLOC);
    CLIO_CO_RETURN;
  }

  // Step 2: list all blobs in the tag
  auto blobs_future = cte_client_->AsyncGetContainedBlobs(tag_id);
  CLIO_CO_AWAIT(blobs_future);
  const auto &blob_names = blobs_future->blob_names_;

  if (blob_names.empty()) {
    HLOG(kInfo, "ExportData: tag '{}' has no blobs", tag_name);
    CLIO_CO_RETURN;
  }

  if (format == "hdf5") {
#ifdef CLIO_CAE_ENABLE_HDF5
    // ---- Is this a self-describing kvhdf5 dataset? -----------------------
    // A kvhdf5 dataset publishes its Layout into its own tag as the reserved
    // __meta blob (kvhdf5/meta_blob.h). If it is there, we can reconstruct the
    // real N-D dataset -- dims, chunk geometry, element type -- and each chunk
    // blob maps 1:1 onto an HDF5 chunk. If it is NOT there, the tag is just an
    // opaque bag of blobs (a VOL-written tag, a raw producer, anything), and we
    // fall back to the legacy behavior below: one flat 1-D uint8 dataset per
    // blob. That fallback is the only thing that was ever possible without
    // dims/dtype, and it stays byte-for-byte what it was.
    // A __meta blob is PRESENT vs. absent is a different question from whether it
    // could be READ. Absent -> this is an opaque tag, fall back to the flat dump
    // (legitimate, no error). Present but unreadable (transient OOM on the read,
    // a get failure, a corrupt/newer-version blob) -> the tag CLAIMS to be a
    // self-describing dataset and we cannot honor that, so it is a hard error: a
    // silent flat dump reported as success would be a structurally wrong .h5 that
    // looks fine. Only fall back when __meta genuinely is not there.
    kvhdf5::MetaBlob meta{};
    bool have_meta = false;
    const bool meta_present =
        std::find(blob_names.begin(), blob_names.end(),
                  std::string(kvhdf5::kMetaBlobName)) != blob_names.end();
    if (meta_present) {
      auto meta_size_future =
          cte_client_->AsyncGetBlobSize(tag_id, kvhdf5::kMetaBlobName);
      CLIO_CO_AWAIT(meta_size_future);
      const clio::run::u64 meta_size = meta_size_future->size_;

      auto *ipc_manager = CLIO_IPC;
      ctp::ipc::FullPtr<char> meta_buf = ipc_manager->AllocateBuffer(meta_size);
      if (!meta_buf.IsNull()) {
        ctp::ipc::ShmPtr<> meta_shm(meta_buf.shm_);
        auto meta_get = cte_client_->AsyncGetBlob(tag_id, kvhdf5::kMetaBlobName,
                                                  0, meta_size, 0, meta_shm);
        CLIO_CO_AWAIT(meta_get);
        if (meta_get->GetReturnCode() == 0) {
          have_meta = kvhdf5::DecodeMeta(meta_buf.ptr_, meta_size, &meta);
        }
        ipc_manager->FreeBuffer(meta_buf);
      }
      if (!have_meta) {
        HLOG(kError,
             "ExportData: tag '{}' has a __meta blob but it could not be read or "
             "decoded (transient OOM, get failure, or a corrupt/newer-version "
             "blob). Refusing to export it as a flat dump that would look like a "
             "successful N-D export. Treating as a failure.",
             tag_name);
        task->result_code_ = -6;
        task->error_message_ = clio::run::priv::string(
            "Unreadable __meta blob", CTP_MALLOC);
        CLIO_CO_RETURN;
      }
    }

    if (have_meta) {
      // ---- Self-describing path: rebuild the real N-D chunked dataset -----
      // The HDF5 mechanics live in ChunkedDatasetWriter (hdf5_export.h) rather
      // than inline here, so they are unit-testable without a runtime or a GPU.
      // All this loop does is feed it chunk bytes.
      ChunkedDatasetWriter writer;
      if (!writer.Create(output_path, tag_name, meta)) {
        HLOG(kError, "ExportData: failed to create dataset '{}' in '{}'",
             tag_name, output_path);
        task->result_code_ = -4;
        task->error_message_ =
            clio::run::priv::string("Failed to create HDF5 dataset", CTP_MALLOC);
        CLIO_CO_RETURN;
      }

      const clio::run::u64 chunk_bytes = meta.ChunkBytes();
      auto *ipc_manager = CLIO_IPC;
      // ONE buffer, reused for every chunk -- they are all exactly chunk_bytes.
      // This is what keeps peak memory at a single chunk, so a dataset far
      // larger than RAM still exports.
      ctp::ipc::FullPtr<char> buf = ipc_manager->AllocateBuffer(chunk_bytes);
      if (buf.IsNull()) {
        HLOG(kError, "ExportData: allocation failed ({} bytes)", chunk_bytes);
        task->result_code_ = -2;
        task->error_message_ =
            clio::run::priv::string("Chunk buffer allocation failed", CTP_MALLOC);
        CLIO_CO_RETURN;
      }
      ctp::ipc::ShmPtr<> shm_ptr(buf.shm_);

      for (const auto &blob_name : blob_names) {
        // Skips __meta (and any future reserved key): a chunk name is only ever
        // '_'-separated decimal digits.
        if (!kvhdf5::IsChunkBlobName(blob_name)) continue;

        auto size_future = cte_client_->AsyncGetBlobSize(tag_id, blob_name);
        CLIO_CO_AWAIT(size_future);
        const clio::run::u64 blob_size = size_future->size_;
        if (blob_size != chunk_bytes) {
          // Not the chunk __meta says it is. Writing it would silently corrupt
          // the dataset; skip it and let the completeness check below turn this
          // into a hard failure.
          HLOG(kError, "ExportData: chunk '{}' is {} bytes, expected {}; skipped",
               blob_name, blob_size, chunk_bytes);
          continue;
        }

        auto get_future = cte_client_->AsyncGetBlob(tag_id, blob_name, 0,
                                                     chunk_bytes, 0, shm_ptr);
        CLIO_CO_AWAIT(get_future);
        if (get_future->GetReturnCode() != 0) {
          HLOG(kError, "ExportData: GetBlob failed for chunk '{}'", blob_name);
          continue;
        }

        if (!writer.WriteChunk(blob_name, buf.ptr_,
                               static_cast<size_t>(chunk_bytes))) {
          HLOG(kError, "ExportData: failed to write chunk '{}' of tag '{}'",
               blob_name, tag_name);
          continue;
        }
        task->bytes_exported_ += chunk_bytes;
      }

      ipc_manager->FreeBuffer(buf);

      const clio::run::u64 written = writer.ChunksWritten();
      const clio::run::u64 expected = writer.ChunksExpected();
      // A dataset with holes is NOT a successful export. We asked for
      // FILL_TIME_NEVER, so a chunk nobody wrote is uninitialized bytes that read
      // back as plausible-looking data. Fail loudly rather than hand back a .h5
      // that looks fine and is not.
      if (!writer.Complete()) {
        HLOG(kError,
             "ExportData: tag '{}' exported {}/{} chunks -- the dataset in '{}' "
             "HAS HOLES and its unwritten chunks are uninitialized. Treating as "
             "a failure.",
             tag_name, written, expected, output_path);
        task->result_code_ = -5;
        task->error_message_ = clio::run::priv::string(
            "Incomplete dataset: missing chunks", CTP_MALLOC);
        CLIO_CO_RETURN;
      }

      HLOG(kInfo,
           "ExportData: wrote N-D dataset '{}' ({} chunks, {} bytes) to '{}'",
           tag_name, written, task->bytes_exported_, output_path);
      CLIO_CO_RETURN;
    }

    hid_t file_id = H5Fcreate(output_path.c_str(), H5F_ACC_TRUNC,
                               H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
      HLOG(kError, "ExportData: failed to create HDF5 file '{}'", output_path);
      task->result_code_ = -2;
      task->error_message_ =
          clio::run::priv::string("Failed to create HDF5 file", CTP_MALLOC);
      CLIO_CO_RETURN;
    }

    // ---- Fallback: opaque tag, one flat 1-D uint8 dataset per blob ---------
    // Reached only when no __meta is present (see above), but a partially-written
    // kvhdf5 tag can still carry chunk-coordinate blobs; regardless, the internal
    // __meta blob is never user data, so never emit it as a dataset.
    for (const auto &blob_name : blob_names) {
      if (blob_name == kvhdf5::kMetaBlobName) continue;
      // Get blob size
      auto size_future = cte_client_->AsyncGetBlobSize(tag_id, blob_name);
      CLIO_CO_AWAIT(size_future);
      clio::run::u64 blob_size = size_future->size_;
      if (blob_size == 0) continue;

      // Allocate buffer and read blob
      auto *ipc_manager = CLIO_IPC;
      ctp::ipc::FullPtr<char> buf = ipc_manager->AllocateBuffer(blob_size);
      if (buf.IsNull()) {
        HLOG(kError, "ExportData: allocation failed for blob '{}'", blob_name);
        continue;
      }
      ctp::ipc::ShmPtr<> shm_ptr(buf.shm_);
      auto get_future = cte_client_->AsyncGetBlob(tag_id, blob_name, 0,
                                                   blob_size, 0, shm_ptr);
      CLIO_CO_AWAIT(get_future);

      if (get_future->GetReturnCode() == 0) {
        hsize_t dims[1] = {static_cast<hsize_t>(blob_size)};
        hid_t space = H5Screate_simple(1, dims, nullptr);
        hid_t ds = H5Dcreate2(file_id, blob_name.c_str(), H5T_NATIVE_UINT8,
                               space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (ds >= 0) {
          H5Dwrite(ds, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   buf.ptr_);
          H5Dclose(ds);
          task->bytes_exported_ += blob_size;
        }
        H5Sclose(space);
      }
      ipc_manager->FreeBuffer(buf);
    }

    H5Fclose(file_id);
    HLOG(kInfo, "ExportData: wrote {} bytes to HDF5 '{}'",
         task->bytes_exported_, output_path);
#else
    task->result_code_ = -3;
    task->error_message_ =
        clio::run::priv::string("HDF5 support not compiled in", CTP_MALLOC);
#endif
  } else {
    // Binary format: sequential blob data with a simple header per blob
    std::ofstream ofs(output_path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
      HLOG(kError, "ExportData: failed to open '{}' for writing", output_path);
      task->result_code_ = -2;
      task->error_message_ =
          clio::run::priv::string("Failed to open output file", CTP_MALLOC);
      CLIO_CO_RETURN;
    }

    for (const auto &blob_name : blob_names) {
      // The internal __meta blob is not user data; a downstream reader of this
      // stream would parse its layout POD as a record. Never emit it.
      if (blob_name == kvhdf5::kMetaBlobName) continue;
      auto size_future = cte_client_->AsyncGetBlobSize(tag_id, blob_name);
      CLIO_CO_AWAIT(size_future);
      clio::run::u64 blob_size = size_future->size_;
      if (blob_size == 0) continue;

      auto *ipc_manager = CLIO_IPC;
      ctp::ipc::FullPtr<char> buf = ipc_manager->AllocateBuffer(blob_size);
      if (buf.IsNull()) {
        HLOG(kError, "ExportData: allocation failed for blob '{}'", blob_name);
        continue;
      }
      ctp::ipc::ShmPtr<> shm_ptr(buf.shm_);
      auto get_future = cte_client_->AsyncGetBlob(tag_id, blob_name, 0,
                                                   blob_size, 0, shm_ptr);
      CLIO_CO_AWAIT(get_future);

      if (get_future->GetReturnCode() == 0) {
        // Header: name length (u32) + name + data length (u64) + data
        uint32_t name_len = static_cast<uint32_t>(blob_name.size());
        ofs.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
        ofs.write(blob_name.data(), name_len);
        ofs.write(reinterpret_cast<const char *>(&blob_size), sizeof(blob_size));
        ofs.write(buf.ptr_, static_cast<std::streamsize>(blob_size));
        task->bytes_exported_ += blob_size;
      }
      ipc_manager->FreeBuffer(buf);
    }

    HLOG(kInfo, "ExportData: wrote {} bytes to binary '{}'",
         task->bytes_exported_, output_path);
  }

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ImportData(clio::run::shared_ptr<ImportDataTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->result_code_ = 0;
  task->bytes_imported_ = 0;

  const std::string tag_name = task->tag_name_.str();
  const std::string input_path = task->input_path_.str();
  const std::string format = task->format_.str();

  HLOG(kInfo, "ImportData: tag='{}', input='{}', format='{}'", tag_name,
       input_path, format);

  if (format != "hdf5") {
    HLOG(kError, "ImportData: unsupported format '{}' (only 'hdf5')", format);
    task->result_code_ = -1;
    task->error_message_ =
        clio::run::priv::string("Unsupported import format", CTP_MALLOC);
    CLIO_CO_RETURN;
  }

#ifdef CLIO_CAE_ENABLE_HDF5
  // Open the dataset and derive its kvhdf5 layout. All the HDF5 mechanics live
  // in ChunkedDatasetReader (hdf5_import.h) so they are unit-testable without a
  // runtime or a GPU; this coroutine only moves bytes into the store. The
  // in-file dataset path IS the tag name -- the exact inverse of ExportData.
  ChunkedDatasetReader reader;
  if (!reader.Open(input_path, tag_name)) {
    HLOG(kError,
         "ImportData: cannot import dataset '{}' from '{}' (missing/unreadable, "
         "or an unsupported element type or shape)",
         tag_name, input_path);
    task->result_code_ = -2;
    task->error_message_ = clio::run::priv::string(
        "Failed to open importable HDF5 dataset", CTP_MALLOC);
    CLIO_CO_RETURN;
  }
  const kvhdf5::MetaBlob meta = reader.Meta();

  // Resolve (or create) the destination tag.
  auto tag_future = cte_client_->AsyncGetOrCreateTag(tag_name);
  CLIO_CO_AWAIT(tag_future);
  const auto &tag_id = tag_future->tag_id_;
  if (tag_id.IsNull()) {
    HLOG(kError, "ImportData: could not get/create tag '{}'", tag_name);
    task->result_code_ = -3;
    task->error_message_ =
        clio::run::priv::string("Tag create failed", CTP_MALLOC);
    CLIO_CO_RETURN;
  }

  auto *ipc_manager = CLIO_IPC;

  // Publish __meta first, so the tag is self-describing even if the chunk loop is
  // interrupted -- a consumer that sees any chunk can always recover the layout.
  {
    ctp::ipc::FullPtr<char> meta_buf =
        ipc_manager->AllocateBuffer(sizeof(kvhdf5::MetaBlob));
    if (meta_buf.IsNull()) {
      HLOG(kError, "ImportData: __meta allocation failed");
      task->result_code_ = -4;
      task->error_message_ = clio::run::priv::string(
          "Meta buffer allocation failed", CTP_MALLOC);
      CLIO_CO_RETURN;
    }
    kvhdf5::EncodeMeta(meta, meta_buf.ptr_);
    auto put_meta = cte_client_->AsyncPutBlob(
        tag_id, kvhdf5::kMetaBlobName, 0, sizeof(kvhdf5::MetaBlob),
        meta_buf.shm_.template Cast<void>(), 1.0f, clio::cte::core::Context(), 0);
    CLIO_CO_AWAIT(put_meta);
    const int rc = put_meta->return_code_;
    ipc_manager->FreeBuffer(meta_buf);
    if (rc != 0) {
      HLOG(kError, "ImportData: failed to write __meta for tag '{}' (rc {})",
           tag_name, rc);
      task->result_code_ = -5;
      task->error_message_ =
          clio::run::priv::string("Meta PutBlob failed", CTP_MALLOC);
      CLIO_CO_RETURN;
    }
  }

  // Stream the chunks through ONE reused buffer -- each PutBlob is awaited before
  // the next read overwrites it, so peak memory is a single chunk (unless the
  // dataset had to collapse to one chunk; see the reader's chunking policy).
  const clio::run::u64 chunk_bytes = meta.ChunkBytes();
  const clio::run::u64 chunk_count = meta.ChunkCount();
  ctp::ipc::FullPtr<char> buf = ipc_manager->AllocateBuffer(chunk_bytes);
  if (buf.IsNull()) {
    HLOG(kError, "ImportData: chunk allocation failed ({} bytes)", chunk_bytes);
    task->result_code_ = -6;
    task->error_message_ =
        clio::run::priv::string("Chunk buffer allocation failed", CTP_MALLOC);
    CLIO_CO_RETURN;
  }

  for (clio::run::u64 k = 0; k < chunk_count; ++k) {
    std::string chunk_name;
    if (!reader.ReadChunk(k, buf.ptr_, static_cast<size_t>(chunk_bytes),
                          &chunk_name)) {
      HLOG(kError, "ImportData: failed to read chunk {} of '{}'", k, tag_name);
      ipc_manager->FreeBuffer(buf);
      task->result_code_ = -7;
      task->error_message_ =
          clio::run::priv::string("Chunk read failed", CTP_MALLOC);
      CLIO_CO_RETURN;
    }
    auto put = cte_client_->AsyncPutBlob(
        tag_id, chunk_name, 0, chunk_bytes, buf.shm_.template Cast<void>(), 1.0f,
        clio::cte::core::Context(), 0);
    CLIO_CO_AWAIT(put);
    if (put->return_code_ != 0) {
      HLOG(kError, "ImportData: PutBlob failed for chunk '{}' (rc {})",
           chunk_name, put->return_code_);
      ipc_manager->FreeBuffer(buf);
      task->result_code_ = -8;
      task->error_message_ =
          clio::run::priv::string("Chunk PutBlob failed", CTP_MALLOC);
      CLIO_CO_RETURN;
    }
    task->bytes_imported_ += chunk_bytes;
  }

  ipc_manager->FreeBuffer(buf);
  HLOG(kInfo, "ImportData: imported '{}' from '{}' ({} chunks, {} bytes)",
       tag_name, input_path, chunk_count, task->bytes_imported_);
#else
  task->result_code_ = -9;
  task->error_message_ =
      clio::run::priv::string("HDF5 support not compiled in", CTP_MALLOC);
#endif
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cae::core
