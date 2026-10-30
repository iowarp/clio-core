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
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
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
/** "k=v;k=v" option lookup in AssimilationCtx::src_data. */
clio::run::u64 OptU64(const std::string &opts, const std::string &key,
                      clio::run::u64 dflt) {
  size_t p = 0;
  while (p < opts.size()) {
    size_t e = opts.find(';', p);
    if (e == std::string::npos) e = opts.size();
    const std::string kv = opts.substr(p, e - p);
    const size_t eq = kv.find('=');
    if (eq != std::string::npos && kv.substr(0, eq) == key) {
      return std::strtoull(kv.c_str() + eq + 1, nullptr, 10);
    }
    p = e + 1;
  }
  return dflt;
}
}  // namespace

int Runtime::AssimilateByReference(const AssimilationCtx &ctx,
                                   std::string *ref_id,
                                   std::string *tag_name) {
  RefExtent ref;
  ref.path = StripProto(ctx.src);
  ref.fd = ::open(ref.path.c_str(), O_RDONLY | O_CLOEXEC);
  if (ref.fd < 0) {
    HLOG(kError, "CAE reference: cannot open '{}': {}", ref.path,
         std::strerror(errno));
    return -10;
  }
  struct stat st;
  if (::fstat(ref.fd, &st) != 0) {
    ::close(ref.fd);
    return -11;
  }
  const clio::run::u64 fsz = (clio::run::u64)st.st_size;
  ref.base = ctx.range_off;
  if (ref.base > fsz) {
    ::close(ref.fd);
    return -12;
  }
  ref.size = ctx.range_size ? std::min<clio::run::u64>(ctx.range_size,
                                                       fsz - ref.base)
                            : fsz - ref.base;
  ref.page_size = OptU64(ctx.src_data, "page_size", 4ull << 20);
  if (ref.page_size == 0) ref.page_size = 4ull << 20;
#ifdef POSIX_FADV_SEQUENTIAL
  ::posix_fadvise(ref.fd, (off_t)ref.base, (off_t)ref.size,
                  POSIX_FADV_SEQUENTIAL);
#endif
  {
    // Content-derived id: identical on every node, distinct per extent.
    const std::string k = ref.path + "|" + std::to_string(ref.base) + "|" +
                          std::to_string(ref.size) + "|" +
                          std::to_string(ref.page_size);
    clio::run::u64 h = 1469598103934665603ull;  // FNV-1a 64
    for (unsigned char c : k) { h ^= c; h *= 1099511628211ull; }
    char hex[24];
    std::snprintf(hex, sizeof(hex), "ref:%016llx", (unsigned long long)h);
    *ref_id = hex;
  }
  *tag_name = StripProto(ctx.dst);
  HLOG(kInfo,
       "CAE reference: tag '{}' -> '{}' [{}, +{}) page_size={} ({})",
       *tag_name, ref.path, ref.base, ref.size, ref.page_size, *ref_id);
  {
    std::lock_guard<std::mutex> g(refs_mu_);
    if (refs_.find(*ref_id) != refs_.end()) {
      ::close(ref.fd);  // same extent assimilated again: keep the first
    } else {
      refs_[*ref_id] = std::move(ref);
    }
  }
  return 0;
}

std::string Runtime::EncodeRefDesc(const RefExtent &r) {
  return std::to_string(r.base) + " " + std::to_string(r.size) + " " +
         std::to_string(r.page_size) + " " + r.path;
}

bool Runtime::DecodeRefDesc(const char *buf, size_t n, RefExtent *r) {
  std::string d(buf, strnlen(buf, n));
  unsigned long long b = 0, sz = 0, ps = 0;
  int used = 0;
  if (std::sscanf(d.c_str(), "%llu %llu %llu %n", &b, &sz, &ps, &used) != 3 ||
      used <= 0 || (size_t)used >= d.size() || ps == 0) {
    return false;
  }
  r->base = b; r->size = sz; r->page_size = ps;
  r->path = d.substr((size_t)used);
  r->fd = ::open(r->path.c_str(), O_RDONLY | O_CLOEXEC);
  if (r->fd < 0) {
    HLOG(kError, "CAE reference: cannot open '{}' on this node: {}", r->path,
         std::strerror(errno));
    return false;
  }
  return true;
}

int Runtime::ServeRefFault(const RefExtent &ref, const std::string &blob_name,
                           clio::run::u64 offset, clio::run::u64 size,
                           void *dst) {
  // The pager names pages by their decimal index; anything else (a _pi
  // suffix, a metadata blob) is not part of a referenced extent.
  if (blob_name.empty()) return 1;
  clio::run::u64 page = 0;
  for (char c : blob_name) {
    if (c < '0' || c > '9') return 1;
    page = page * 10 + (clio::run::u64)(c - '0');
  }
  if (dst == nullptr) return 1;
  const clio::run::u64 psz = ref.page_size;
  if (size == 0) size = psz > offset ? psz - offset : 0;
  const clio::run::u64 lo = page * psz + offset;  // within the extent
  clio::run::u64 avail = lo < ref.size ? ref.size - lo : 0;
  const clio::run::u64 n = std::min(size, avail);
#ifdef POSIX_FADV_WILLNEED
  // Passes stream the extent in page order: when a fault opens a new window
  // of R pages, ask the filesystem to start fetching the NEXT window so the
  // faults that follow find it in flight. A hint only -- no buffer of ours.
  {
    static const clio::run::u64 kRa = [] {
      const char *e = std::getenv("CLIO_CAE_REF_READAHEAD");
      return (e != nullptr && *e) ? std::strtoull(e, nullptr, 10) : 16ull;
    }();
    if (kRa > 0 && offset == 0 && page % kRa == 0) {
      const clio::run::u64 nlo = (page + kRa) * psz;
      if (nlo < ref.size) {
        ::posix_fadvise(ref.fd, (off_t)(ref.base + nlo),
                        (off_t)std::min(kRa * psz, ref.size - nlo),
                        POSIX_FADV_WILLNEED);
      }
    }
  }
#endif
  const bool dev = ctp::IsDeviceAccessible(dst);
  // Device destinations go through a per-thread host bounce; host
  // destinations are read in place with no staging copy at all.
  thread_local std::vector<char> bounce;
  char *io = static_cast<char *>(dst);
  if (dev) {
    if (bounce.size() < size) bounce.resize(size);
    io = bounce.data();
  }
  clio::run::u64 done = 0;
  while (done < n) {
    const ssize_t r = ::pread(ref.fd, io + done, (size_t)(n - done),
                              (off_t)(ref.base + lo + done));
    if (r < 0) {
      if (errno == EINTR) continue;
      HLOG(kError, "CAE reference: pread '{}' page {} failed: {}", ref.path,
           page, std::strerror(errno));
      return 5;
    }
    if (r == 0) break;  // short file: tail reads as zero below
    done += (clio::run::u64)r;
  }
  if (done < size) std::memset(io + done, 0, (size_t)(size - done));
  {
    // CLIO_CAE_REF_TRACE=N: log the first N faults served by this process.
    static const long kTrace = [] {
      const char *e = std::getenv("CLIO_CAE_REF_TRACE");
      return (e != nullptr && *e) ? std::strtol(e, nullptr, 10) : 0L;
    }();
    static std::atomic<long> traced{0};
    if (kTrace > 0 && traced.fetch_add(1) < kTrace) {
      char host[64] = {0};
      ::gethostname(host, sizeof(host) - 1);
      float v[4] = {0, 0, 0, 0};
      std::memcpy(v, io, std::min<size_t>(sizeof(v), (size_t)size));
      HLOG(kInfo,
           "CAE_REF_TRACE host={} path={} page={} off={} size={} read={} "
           "dev={} v0={} v1={} v2={} v3={}",
           host, ref.path, page, offset, size, done, dev ? 1 : 0, v[0], v[1],
           v[2], v[3]);
    }
  }
  if (dev) ctp::DeviceAwareMemcpy(dst, io, (size_t)size);
  return 0;
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
  // By-reference fault: CTE dispatched a miss on a tag CAE assimilated by
  // reference. Read the page straight from the source object.
  if (std::strncmp(task->context_.fault_params_, "ref:", 4) == 0) {
    const std::string ref_key(task->context_.fault_params_);
    std::map<std::string, RefExtent>::iterator it;
    bool have = false;
    {
      std::lock_guard<std::mutex> g(refs_mu_);
      it = refs_.find(ref_key);
      have = it != refs_.end();
    }
    if (!have) {
      // Assimilated on another node: fetch its descriptor from the registry.
      auto tf = cte_client_->AsyncGetOrCreateTag(kRefRegistryTag);
      CLIO_CO_AWAIT(tf);
      auto dbuf = CLIO_IPC->AllocateBuffer(kRefDescSize);
      if (tf->GetReturnCode() != 0 || dbuf.IsNull()) {
        task->SetReturnCode(1);
        CLIO_CO_RETURN;
      }
      std::memset(dbuf.ptr_, 0, kRefDescSize);
      clio::cte::core::Context gctx;
      gctx.op_flags_ |= clio::cte::core::Context::kNoFault;
      auto gf = cte_client_->AsyncGetBlob(
          tf->tag_id_, ref_key.c_str(), 0, kRefDescSize, 0,
          dbuf.shm_.template Cast<void>(), clio::run::PoolQuery::Dynamic(),
          gctx);
      CLIO_CO_AWAIT(gf);
      RefExtent loaded;
      const bool ok = gf->GetReturnCode() == 0 &&
                      DecodeRefDesc(dbuf.ptr_, kRefDescSize, &loaded);
      CLIO_IPC->FreeBuffer(dbuf);
      if (!ok) {
        HLOG(kError, "CAE reference: no descriptor for {}", ref_key);
        task->SetReturnCode(1);
        CLIO_CO_RETURN;
      }
      std::lock_guard<std::mutex> g(refs_mu_);
      it = refs_.find(ref_key);
      if (it == refs_.end()) {
        it = refs_.emplace(ref_key, std::move(loaded)).first;
      } else {
        ::close(loaded.fd);  // a concurrent fault loaded it first
      }
    }
    // MATERIALISE request: CTE sends a size-0, null-buffer fault before a
    // PutBlob into a page this tag does not hold, so a PARTIAL write lands
    // on top of the source bytes instead of zeros. Read the whole page into
    // a staging buffer and store it in the tag (kNoFault: the put must not
    // fault back into us). Only written pages ever become resident.
    ctp::ipc::FullPtr<char> stage;
    const bool materialise =
        task->size_ == 0 || task->blob_data_.IsNull();
    if (materialise) {
      stage = CLIO_IPC->AllocateBuffer(it->second.page_size);
      if (stage.IsNull()) {
        task->SetReturnCode(4);
        CLIO_CO_RETURN;
      }
    }
    auto dst = materialise
                   ? stage
                   : CLIO_IPC->ToFullPtr<char>(task->blob_data_.Cast<char>());
    auto st = std::make_shared<std::atomic<int>>(-1);
    {
      const RefExtent *ref = &it->second;  // refs_ entries are never erased
      std::string bname = task->GetBlobName();
      const clio::run::u64 off = materialise ? 0 : task->offset_;
      const clio::run::u64 sz = materialise ? it->second.page_size : task->size_;
      void *d = dst.ptr_;
      RefIoPool::Get().Submit([this, ref, bname, off, sz, d, st]() {
        st->store(ServeRefFault(*ref, bname, off, sz, d),
                  std::memory_order_release);
      });
    }
    {
      // Poll fast first (a cached read lands in microseconds), then back off.
      clio::run::u64 polls = 0;
      while (st->load(std::memory_order_acquire) < 0) {
        CLIO_CO_AWAIT(clio::run::yield(polls++ < 64 ? 20.0 : 200.0));
      }
    }
    if (materialise) {
      int rc = st->load();
      if (rc == 0) {
        clio::cte::core::Context pctx;
        pctx.op_flags_ |= clio::cte::core::Context::kNoFault;
        auto put = cte_client_->AsyncPutBlob(
            task->tag_id_, task->GetBlobName(), 0, it->second.page_size,
            stage.shm_.template Cast<void>(), -1.0f, pctx);
        CLIO_CO_AWAIT(put);
        rc = (int)put->GetReturnCode();
      }
      CLIO_IPC->FreeBuffer(stage);
      task->SetReturnCode((clio::run::u32)rc);
      CLIO_CO_RETURN;
    }
    task->SetReturnCode((clio::run::u32)st->load());
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

    // format "reference": map the source as an object, do not copy it.
    if (assimilation_ctx.format == "reference" ||
        assimilation_ctx.format == "ref") {
      std::string ref_id, tag_name;
      int rc = AssimilateByReference(assimilation_ctx, &ref_id, &tag_name);
      if (rc == 0) {
        // Publish the descriptor so the CAE on any node can serve faults.
        auto rt = cte_client_->AsyncGetOrCreateTag(kRefRegistryTag);
        CLIO_CO_AWAIT(rt);
        rc = (int)rt->GetReturnCode();
        if (rc == 0) {
          std::string desc;
          {
            std::lock_guard<std::mutex> g(refs_mu_);
            desc = EncodeRefDesc(refs_[ref_id]);
          }
          auto dbuf = CLIO_IPC->AllocateBuffer(kRefDescSize);
          if (dbuf.IsNull() || desc.size() >= kRefDescSize) {
            rc = -13;
          } else {
            std::memset(dbuf.ptr_, 0, kRefDescSize);
            std::memcpy(dbuf.ptr_, desc.data(), desc.size());
            clio::cte::core::Context pctx;
            pctx.op_flags_ |= clio::cte::core::Context::kNoFault;
            auto pf = cte_client_->AsyncPutBlob(
                rt->tag_id_, ref_id.c_str(), 0, kRefDescSize,
                dbuf.shm_.template Cast<void>(), -1.0f, pctx);
            CLIO_CO_AWAIT(pf);
            rc = (int)pf->GetReturnCode();
            CLIO_IPC->FreeBuffer(dbuf);
          }
        }
      }
      if (rc == 0) {
        // Register CAE as the tag's fault handler on EVERY CTE container:
        // a GetBlob is routed to the page's hashed owner container, and that
        // container consults ITS OWN TagInfo for the handler. Registering
        // only locally left remote-owned pages unhandled (served as zeros).
        // First resolve the canonical id, then pin it on each container.
        // Resolve the CANONICAL tag id with the same routing every other
        // client uses (the name's hash owner): a Local create here minted a
        // second, node-local tag of the same name that the dataset never
        // used, so the handler landed on the wrong tag.
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
              tag_name, pool_id_, pool_name_, ref_id, tid,
              clio::run::PoolQuery::DirectId(
                  static_cast<clio::run::ContainerId>(c)));
          CLIO_CO_AWAIT(cf);
          rc = (int)cf->GetReturnCode();
          if (rc == 0 && !(cf->tag_id_ == tid)) {
            HLOG(kError,
                 "CAE reference: tag '{}' id differs on container {}",
                 tag_name, c);
            rc = -14;
          }
        }
      }
      if (rc != 0) {
        task->result_code_ = rc;
        task->error_message_ = "reference assimilation failed";
        task->num_tasks_scheduled_ = tasks_scheduled;
        task->SetReturnCode(static_cast<clio::run::u32>(rc));
        CLIO_CO_RETURN;
      }
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
