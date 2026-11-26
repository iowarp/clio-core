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

/** Implementation of fs_ns_shm.h (included at its end). */
#ifndef CLIO_CTE_FILESYSTEM_FS_NS_SHM_IMPL_H_
#define CLIO_CTE_FILESYSTEM_FS_NS_SHM_IMPL_H_

#include <algorithm>
#include <chrono>
#include <csignal>
#include <mutex>
#if !defined(_WIN32)
#include <unistd.h>  // getuid, getgid: the root directory's owner
#endif

namespace clio::cte::filesystem::ns {

namespace detail {
/** Round up to a power of two (at least 64). */
inline u32 Pow2(u32 n) {
  u32 p = 64;
  while (p < n) p <<= 1;
  return p;
}
/** Round up to a multiple of 64. */
inline u64 Align64(u64 n) { return (n + 63) & ~u64(63); }
/** Mix a 64-bit key. */
inline u64 Mix(u64 x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdull;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ull;
  x ^= x >> 33;
  return x;
}
constexpr u64 kFreedId = ~0ull;
}  // namespace detail

// ---------------------------------------------------------------------------
// NsLock
// ---------------------------------------------------------------------------

namespace detail {

/** Fault injection for crash-repair tests: SIGKILL this process the n-th
 *  time it reaches the point (set from a forked test child). */
enum NsCrashPoint : int {
  kNsCpNone = 0,
  kNsCpCreateLinked,  ///< Create: entries inserted and linked, op unlogged
};
struct NsCrashCfg {
  int point_ = kNsCpNone;
  u64 left_ = 0;
};
inline NsCrashCfg &NsCrash() {
  static NsCrashCfg cfg;
  return cfg;
}
inline void NsCrashHit(int point) {
  NsCrashCfg &c = NsCrash();
  if (c.point_ == point && c.left_ != 0 && --c.left_ == 0) ::raise(SIGKILL);
}

/** Back off while waiting on the namespace lock (spin, yield, then sleep). */
inline void NsBackoff(u32 spin) {
  if (spin < 64) {
    ctp::CpuRelax();
  } else if (spin < 1024) {
    std::this_thread::yield();
  } else {
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

/** Locks mapped in this process (by address): a thread unregisters at exit
 *  only from locks still mapped. */
inline std::mutex &NsMappedMu() {
  static std::mutex mu;
  return mu;
}
inline std::set<const NsLock *> &NsMapped() {
  static auto *s = new std::set<const NsLock *>();
  return *s;  // leaked: thread exits may follow static destruction
}

/** This thread's registration in each namespace lock it used. */
struct NsThreadCache {
  struct Entry {
    const NsLock *lock_ = nullptr;
    u64 format_id_ = 0;
    NsThread *t_ = nullptr;
    unsigned gen_ = 0;
  };
  Entry e_[4];
  ~NsThreadCache() {
    std::lock_guard<std::mutex> g(NsMappedMu());
    const unsigned gen = ctp::detail::ProcIdentityGen();
    for (Entry &x : e_) {
      if (x.t_ != nullptr && x.gen_ == gen && NsMapped().count(x.lock_) &&
          x.lock_->format_id_ == x.format_id_ && x.t_->nest_ == 0) {
        x.t_->app_n_[0].store(0);
        x.t_->app_n_[1].store(0);
        x.t_->app_id_.store(0);
        x.t_->pid_.store(0, std::memory_order_release);  // free
      }
    }
  }
};

inline NsThreadCache &MyNsThreads() {
  static thread_local NsThreadCache cache;
  return cache;
}

}  // namespace detail

inline void NsLock::Init(u64 format_id) {
  writer_.store(0);
  wcount_.store(0);
  format_id_ = format_id;
  hw_.store(0);
  for (auto &w : readers_) w.store(0);
  std::memset(static_cast<void *>(threads_), 0, sizeof(threads_));
}

inline NsThread *NsLock::Me() {
  detail::NsThreadCache &c = detail::MyNsThreads();
  const unsigned gen = ctp::detail::ProcIdentityGen();
  detail::NsThreadCache::Entry *spare = nullptr;
  for (auto &x : c.e_) {
    if (x.lock_ == this && x.format_id_ == format_id_ && x.gen_ == gen &&
        x.t_ != nullptr) {
      return x.t_;
    }
    if (spare == nullptr && (x.t_ == nullptr || x.gen_ != gen)) spare = &x;
  }
  NsThread *t = Claim();
  if (t == nullptr) return nullptr;
  if (spare == nullptr) spare = &c.e_[0];
  *spare = {this, format_id_, t, gen};
  return t;
}

inline NsThread *NsLock::Claim() {
  const u32 pid = static_cast<u32>(ctp::SystemInfo::GetPid());
  const u64 start = ctp::TimedMutex::GetProcessStartTime(pid);
  for (u32 i = 0; i < kThreads; ++i) {
    u32 expect = 0;
    if (!threads_[i].pid_.compare_exchange_strong(expect, pid)) continue;
    threads_[i].start_ = start;
    threads_[i].nest_ = 0;
    threads_[i].app_id_.store(0);
    threads_[i].app_n_[0].store(0);
    threads_[i].app_n_[1].store(0);
    u32 hw = hw_.load();
    while (hw < i + 1 && !hw_.compare_exchange_weak(hw, i + 1)) {
    }
    return &threads_[i];
  }
  return nullptr;
}

inline void NsLock::ReadLock() {
  NsThread *t = nullptr;
  for (u32 spin = 0; (t = Me()) == nullptr; ++spin) {
    detail::NsBackoff(spin);  // every thread slot taken: wait for one
  }
  if (t->nest_++ > 0) return;
  const u32 i = static_cast<u32>(t - threads_);
  const u64 bit = 1ull << (i % 64);
  std::atomic<u64> &w = readers_[i / 64];
  for (u32 spin = 0;; ++spin) {
    w.fetch_or(bit, std::memory_order_seq_cst);
    if (writer_.load(std::memory_order_seq_cst) == 0) return;
    w.fetch_and(~bit, std::memory_order_seq_cst);  // a writer: step aside
    while (writer_.load(std::memory_order_acquire) != 0) {
      detail::NsBackoff(spin++);
    }
  }
}

inline void NsLock::ReadUnlock() {
  NsThread *t = Me();
  if (t == nullptr || t->nest_ == 0 || --t->nest_ > 0) return;
  const u32 i = static_cast<u32>(t - threads_);
  readers_[i / 64].fetch_and(~(1ull << (i % 64)), std::memory_order_release);
}

inline void NsLock::WaitReaders(const std::function<void()> &stalled) {
  for (u32 spin = 0;; ++spin) {
    bool any = false;
    for (u32 w = 0; w < kWords && !any; ++w) {
      any = readers_[w].load(std::memory_order_seq_cst) != 0;
    }
    if (!any) return;
    detail::NsBackoff(spin);
    if (stalled && spin % kStallSpins == kStallSpins - 1) stalled();
  }
}

inline u64 NsLock::NewWriterId() {
  return (static_cast<u64>(ctp::TimedMutex::MyPid()) << 32) |
         ((wcount_.fetch_add(1) + 1) & 0x7FFFFFFFull);
}

inline void NsLock::WriteLock() {
  const u64 me = NewWriterId();
  for (u32 spin = 0;; ++spin) {
    u64 expect = 0;
    if (writer_.compare_exchange_weak(expect, me, std::memory_order_seq_cst)) {
      break;
    }
    detail::NsBackoff(spin);
  }
  WaitReaders();  // new readers step aside once the writer word is set
}

inline void NsLock::WriteUnlock() {
  writer_.store(0, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// Segment
// ---------------------------------------------------------------------------

inline bool View::Format(const std::string &name, const Limits &lim,
                         bool case_insensitive, u64 root_id,
                         bool root_complete, u32 node_id,
                         bool namespace_on) {
  const u32 ni = detail::Pow2(lim.max_inodes_);
  const u32 nd = detail::Pow2(lim.max_dentries_);
  const u32 nt = std::max<u32>(lim.max_targets_, 16);
  const u64 lb = std::max<u64>((lim.log_bytes_ + 7) & ~u64(7), 1u << 20);
  const u64 root_b = detail::Align64(sizeof(Root));
  const u64 ino_off = root_b;
  const u64 den_off = detail::Align64(ino_off + u64(ni) * sizeof(Inode));
  const u64 tgt_off = detail::Align64(den_off + u64(nd) * sizeof(Dentry));
  const u32 na = detail::Pow2(lim.max_append_files_);
  const u64 app_off = detail::Align64(tgt_off + u64(nt) * sizeof(Target));
  const u32 nx = detail::Pow2(lim.max_xattrs_);
  const u64 xat_off = detail::Align64(app_off + u64(na) * sizeof(AppendSlot));
  const u64 log_off = detail::Align64(xat_off + u64(nx) * sizeof(XattrSlot));
  const u64 total = log_off + lb + 4096;
  const ctp::ipc::MemoryBackendId bid(0x4653u,
                                      static_cast<u32>(detail::Mix(
                                          std::hash<std::string>{}(name))));
  if (!backend_.shm_init(bid, total + ctp::ipc::kBackendHeaderSize, name)) {
    return false;
  }
  mapped_ = true;
  owner_ = true;
  base_ = backend_.data_;
  std::memset(base_, 0, total);  // tables start empty (id 0, kDentEmpty)
  root_ = reinterpret_cast<Root *>(base_);
  root_->magic_ = Root::kMagic;
  root_->version_ = Root::kVersion;
  root_->case_insensitive_ = case_insensitive ? 1u : 0u;
  root_->node_id_ = node_id;
  root_->namespace_on_ = namespace_on ? 1u : 0u;
  root_->lock_.Init(static_cast<u64>(
      std::chrono::steady_clock::now().time_since_epoch().count()));
  root_->root_id_ = root_id;
  root_->n_inodes_ = ni;
  root_->n_dentries_ = nd;
  root_->n_targets_ = nt;
  root_->inodes_off_ = ino_off;
  root_->dentries_off_ = den_off;
  root_->targets_off_ = tgt_off;
  root_->appends_off_ = app_off;
  root_->xattrs_off_ = xat_off;
  root_->n_xattrs_ = nx;
  root_->n_appends_ = na;
  root_->log_off_ = log_off;
  root_->log_bytes_ = lb;
  InitRootInodeLocked(root_complete);
  NoteMapped();
  root_->ready_.store(1, std::memory_order_release);
  return true;
}

inline void View::InitRootInodeLocked(bool complete) {
  Inode *r = InsertInodeLocked(root_->root_id_);
  r->type_ = kFsTypeDir;
  r->mode_ = 0755;
  // Owned by the user running the runtime (the mounting user in every
  // deployment), as the task path reports it. Left at 0 (root), a fresh
  // volume refused every mkdir/create at its top to a non-root user (EACCES).
#if !defined(_WIN32)
  r->uid_ = static_cast<u32>(getuid());
  r->gid_ = static_cast<u32>(getgid());
#endif
  r->nlink_ = 2;
  r->first_child_ = kNoSlot;
  r->target_ = kNoSlot;
  r->flags_ = complete ? kInodeComplete : 0u;
  r->flags_ |= complete ? kInodeXattrsKnown : 0u;
}

inline void View::NoteMapped() {
  std::lock_guard<std::mutex> g(detail::NsMappedMu());
  detail::NsMapped().insert(&root_->lock_);
}

inline bool View::Attach(const std::string &name) {
  if (mapped_) return Ready();
  if (!backend_.shm_attach(name)) return false;
  mapped_ = true;
  base_ = backend_.data_;
  root_ = reinterpret_cast<Root *>(base_);
  if (root_->magic_ != Root::kMagic || root_->version_ != Root::kVersion ||
      root_->ready_.load(std::memory_order_acquire) != 1) {
    Detach();
    return false;
  }
  NoteMapped();
  return true;
}

inline void View::Detach() {
  if (!mapped_) return;
  if (root_ != nullptr) {
    std::lock_guard<std::mutex> g(detail::NsMappedMu());
    detail::NsMapped().erase(&root_->lock_);
  }
  if (owner_) {
    backend_.shm_destroy();
  } else {
    backend_.shm_detach();
  }
  mapped_ = false;
  owner_ = false;
  base_ = nullptr;
  root_ = nullptr;
}

// ---------------------------------------------------------------------------
// Tables (lock held)
// ---------------------------------------------------------------------------

inline u64 View::NameHash(u64 parent, const std::string &canon) {
  u64 h = 1469598103934665603ull ^ detail::Mix(parent);
  for (unsigned char c : canon) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h == 0 ? 1 : h;
}

inline Inode *View::FindInodeLocked(u64 id) {
  const u32 mask = root_->n_inodes_ - 1;
  Inode *t = Inodes();
  u32 i = static_cast<u32>(detail::Mix(id)) & mask;
  for (u32 probes = 0; probes <= mask; ++probes, i = (i + 1) & mask) {
    if (t[i].id_ == id) return &t[i];
    if (t[i].id_ == 0) return nullptr;
  }
  return nullptr;
}

inline Inode *View::InsertInodeLocked(u64 id) {
  if (Inode *hit = FindInodeLocked(id)) return hit;
  const u32 mask = root_->n_inodes_ - 1;
  Inode *t = Inodes();
  u32 i = static_cast<u32>(detail::Mix(id)) & mask;
  for (u32 probes = 0; probes <= mask; ++probes, i = (i + 1) & mask) {
    if (t[i].id_ == 0 || t[i].id_ == detail::kFreedId) {
      std::memset(&t[i], 0, sizeof(Inode));
      t[i].id_ = id;
      t[i].first_child_ = kNoSlot;
      t[i].target_ = kNoSlot;
      root_->used_inodes_.fetch_add(1, std::memory_order_relaxed);
      return &t[i];
    }
  }
  return nullptr;
}

inline void View::FreeInodeLocked(Inode *ino) {
  if (ino->nxattr_ != 0) FreeXattrsLocked(ino);
  if (ino->target_ != kNoSlot) {
    Targets()[ino->target_].id_ = 0;
  }
  ino->id_ = detail::kFreedId;
  root_->used_inodes_.fetch_sub(1, std::memory_order_relaxed);
}

inline Dentry *View::FindDentLocked(u64 parent, const std::string &name,
                                    u64 *hash) {
  const std::string canon = Canon(name);
  const u64 h = NameHash(parent, canon);
  if (hash != nullptr) *hash = h;
  const u32 mask = root_->n_dentries_ - 1;
  Dentry *t = Dentries();
  u32 i = static_cast<u32>(h) & mask;
  for (u32 probes = 0; probes <= mask; ++probes, i = (i + 1) & mask) {
    Dentry &d = t[i];
    if (d.state_ == kDentEmpty) return nullptr;
    if (d.state_ == kDentTomb || d.parent_ != parent || d.hash_ != h) continue;
    if (Canon(std::string(d.name_, d.len_)) == canon) return &d;
  }
  return nullptr;
}

inline Dentry *View::InsertDentLocked(u64 parent, u64 hash,
                                      const std::string &name) {
  if (name.size() > kMaxName) return nullptr;
  const u32 mask = root_->n_dentries_ - 1;
  // Keep a free eighth: open addressing degrades as it fills.
  const u64 high = u64(root_->n_dentries_) * 7 / 8;
  if (root_->used_dentries_.load(std::memory_order_relaxed) >= high) {
    ReapNegLocked();
    if (root_->used_dentries_.load(std::memory_order_relaxed) >= high) {
      return nullptr;
    }
  }
  Dentry *t = Dentries();
  u32 i = static_cast<u32>(hash) & mask;
  for (u32 probes = 0; probes <= mask; ++probes, i = (i + 1) & mask) {
    if (t[i].state_ == kDentEmpty || t[i].state_ == kDentTomb) {
      Dentry &d = t[i];
      d.parent_ = parent;
      d.hash_ = hash;
      d.child_ = 0;
      d.seq_ = 0;
      d.type_ = 0;
      d.state_ = kDentNeg;
      d.next_ = kNoSlot;
      d.prev_ = kNoSlot;
      d.len_ = static_cast<u16>(name.size());
      std::memcpy(d.name_, name.data(), name.size());
      d.name_[name.size()] = '\0';
      root_->used_dentries_.fetch_add(1, std::memory_order_relaxed);
      return &d;
    }
  }
  return nullptr;
}

inline void View::LinkDentLocked(Inode *dir, Dentry *d) {
  const u32 slot = SlotOf(d);
  d->prev_ = kNoSlot;
  d->next_ = dir->first_child_;
  if (dir->first_child_ != kNoSlot) {
    Dentries()[dir->first_child_].prev_ = slot;
  }
  dir->first_child_ = slot;
}

inline void View::UnlinkDentLocked(Inode *dir, Dentry *d) {
  Dentry *t = Dentries();
  if (d->prev_ != kNoSlot) {
    t[d->prev_].next_ = d->next_;
  } else if (dir != nullptr) {
    dir->first_child_ = d->next_;
  }
  if (d->next_ != kNoSlot) t[d->next_].prev_ = d->prev_;
  d->next_ = d->prev_ = kNoSlot;
}

inline void View::ReapNegLocked() {
  // Negative entries whose removal the module has applied carry nothing the
  // module lacks: free them.
  const u64 applied = root_->applied_seq_.load(std::memory_order_acquire);
  Dentry *t = Dentries();
  for (u32 i = 0; i < root_->n_dentries_; ++i) {
    Dentry &d = t[i];
    if (d.state_ != kDentNeg || d.seq_ > applied) continue;
    UnlinkDentLocked(FindInodeLocked(d.parent_), &d);
    d.state_ = kDentTomb;
    root_->used_dentries_.fetch_sub(1, std::memory_order_relaxed);
  }
}

// ---------------------------------------------------------------------------
// Resolution (lock held)
// ---------------------------------------------------------------------------

inline int View::ResolveDirLocked(const std::vector<std::string> &comps,
                                  size_t n, Inode **dir) {
  Inode *cur = FindInodeLocked(root_->root_id_);
  if (cur == nullptr) return kMiss;
  for (size_t i = 0; i < n; ++i) {
    Dentry *d = FindDentLocked(cur->id_, comps[i], nullptr);
    if (d == nullptr) {
      return (cur->flags_ & kInodeComplete) ? ENOENT : kMiss;
    }
    if (d->state_ != kDentLive) return ENOENT;
    if (d->type_ != kFsTypeDir) return ENOTDIR;
    cur = FindInodeLocked(d->child_);
    if (cur == nullptr) return kMiss;
  }
  *dir = cur;
  return 0;
}

inline int View::ResolveLeafLocked(const std::string &path, Inode **parent,
                                   Dentry **d, u64 *hash, std::string *leaf) {
  const std::vector<std::string> comps = SplitPath(path);
  if (comps.empty()) return EINVAL;  // the root has no parent entry
  int rc = ResolveDirLocked(comps, comps.size() - 1, parent);
  if (rc != 0) return rc;
  *leaf = comps.back();
  *d = FindDentLocked((*parent)->id_, *leaf, hash);
  if (*d == nullptr && ((*parent)->flags_ & kInodeComplete) == 0) {
    return kMiss;
  }
  return 0;
}

inline void View::FillAttr(const Inode &ino, Attr *out) {
  out->id = ino.id_;
  out->type = ino.type_;
  out->mode = ino.mode_;
  out->uid = ino.uid_;
  out->gid = ino.gid_;
  out->nlink = ino.nlink_;
  out->size = ino.size_;
  out->atime = ino.atime_;
  out->mtime = ino.mtime_;
  out->ctime = ino.ctime_;
}

// ---------------------------------------------------------------------------
// Op log
// ---------------------------------------------------------------------------

inline bool View::HasLogRoomLocked(size_t plen, size_t p2len) {
  const u64 cap = root_->log_bytes_;
  const u64 rec = OpRecBytes(plen, p2len);
  const u64 head = root_->log_head_.load(std::memory_order_acquire);
  const u64 tail = root_->log_tail_.load(std::memory_order_relaxed);
  const u64 pos = tail % cap;
  const u64 need = (pos + rec > cap) ? (cap - pos) + rec : rec;
  return cap - (tail - head) >= need;
}

inline u64 View::PushOpLocked(u32 type, const std::string &path,
                              const std::string &path2, u64 id, u64 size,
                              u64 t1, u64 t2, u32 mode, u32 uid, u32 gid,
                              u32 flags) {
  const u64 cap = root_->log_bytes_;
  const u32 bytes = OpRecBytes(path.size(), path2.size());
  u64 tail = root_->log_tail_.load(std::memory_order_relaxed);
  u64 pos = tail % cap;
  if (pos + bytes > cap) {
    // Not enough room before the end: a filler record, then wrap.
    OpRec *wrap = reinterpret_cast<OpRec *>(Log() + pos);
    wrap->len_ = static_cast<u32>(cap - pos);
    wrap->type_ = kOpWrap;
    tail += cap - pos;
    pos = 0;
  }
  const u64 seq = root_->pushed_seq_.load(std::memory_order_relaxed) + 1;
  OpRec *r = reinterpret_cast<OpRec *>(Log() + pos);
  r->len_ = bytes;
  r->type_ = type;
  r->seq_ = seq;
  r->id_ = id;
  r->size_ = size;
  r->t1_ = t1;
  r->t2_ = t2;
  r->mode_ = mode;
  r->uid_ = uid;
  r->gid_ = gid;
  r->flags_ = flags;
  r->plen_ = static_cast<u16>(path.size());
  r->p2len_ = static_cast<u16>(path2.size());
  r->pad_ = 0;
  char *p = reinterpret_cast<char *>(r + 1);
  std::memcpy(p, path.data(), path.size());
  std::memcpy(p + path.size(), path2.data(), path2.size());
  root_->pushed_seq_.store(seq, std::memory_order_relaxed);
  root_->log_tail_.store(tail + bytes, std::memory_order_release);
  return seq;
}

inline bool View::PeekOp(OpRec *rec, std::string *path, std::string *path2) {
  const u64 cap = root_->log_bytes_;
  for (;;) {
    const u64 head = root_->log_head_.load(std::memory_order_relaxed);
    const u64 tail = root_->log_tail_.load(std::memory_order_acquire);
    if (head == tail) return false;
    const OpRec *r = reinterpret_cast<const OpRec *>(Log() + head % cap);
    if (r->type_ == kOpWrap) {
      root_->log_head_.store(head + r->len_, std::memory_order_release);
      continue;
    }
    *rec = *r;
    const char *p = reinterpret_cast<const char *>(r + 1);
    path->assign(p, r->plen_);
    path2->assign(p + r->plen_, r->p2len_);
    return true;
  }
}

inline void View::PopOp() {
  const u64 cap = root_->log_bytes_;
  const u64 head = root_->log_head_.load(std::memory_order_relaxed);
  const OpRec *r = reinterpret_cast<const OpRec *>(Log() + head % cap);
  root_->log_head_.store(head + r->len_, std::memory_order_release);
}

inline void View::MarkApplied(u64 seq) {
  root_->applied_seq_.store(seq, std::memory_order_release);
}

inline int View::Mutate(size_t plen, size_t p2len,
                        const std::function<int()> &fn) {
  if (plen > kMaxPath || p2len > kMaxPath) return ENAMETOOLONG;
  for (;;) {
    root_->lock_.WriteLock();
    if (HasLogRoomLocked(plen, p2len)) {
      const int rc = fn();
      root_->lock_.WriteUnlock();
      return rc;
    }
    root_->lock_.WriteUnlock();
    // The module drains the log continuously; room follows.
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
}

// ---------------------------------------------------------------------------
// Ids
// ---------------------------------------------------------------------------

inline bool View::MintIdLocked(u64 *id) {
  const u64 n = root_->id_next_.load(std::memory_order_relaxed);
  if (n >= root_->id_limit_.load(std::memory_order_relaxed)) return false;
  root_->id_next_.store(n + 1, std::memory_order_relaxed);
  *id = (u64(root_->id_major_) << 32) | (n & 0xFFFFFFFFull);
  return true;
}

inline void View::SetIdBlock(u32 major, u64 first, u64 count) {
  ModuleGuard g(this);
  root_->id_major_ = major;
  root_->id_next_.store(first, std::memory_order_relaxed);
  root_->id_limit_.store(first + count, std::memory_order_relaxed);
}

inline u64 View::IdsLeft() const {
  const u64 n = root_->id_next_.load(std::memory_order_relaxed);
  const u64 l = root_->id_limit_.load(std::memory_order_relaxed);
  return l > n ? l - n : 0;
}

// ---------------------------------------------------------------------------
// Client operations
// ---------------------------------------------------------------------------

inline int View::Lookup(const std::string &path, Attr *out) {
  root_->lock_.ReadLock();
  int rc = 0;
  if (SplitPath(path).empty()) {
    Inode *r = FindInodeLocked(root_->root_id_);
    if (r == nullptr) {
      rc = kMiss;
    } else {
      FillAttr(*r, out);
    }
  } else {
    Inode *parent = nullptr;
    Dentry *d = nullptr;
    std::string leaf;
    rc = ResolveLeafLocked(path, &parent, &d, nullptr, &leaf);
    if (rc == 0) {
      if (d == nullptr || d->state_ != kDentLive) {
        rc = ENOENT;
      } else if (Inode *ino = FindInodeLocked(d->child_)) {
        FillAttr(*ino, out);
      } else {
        rc = kMiss;
      }
    }
  }
  root_->lock_.ReadUnlock();
  return rc;
}

inline int View::Create(const std::string &path, u32 mode, u32 uid, u32 gid,
                        bool excl, u64 now, Attr *out, bool *created) {
  *created = false;
  return Mutate(path.size(), 0, [&]() -> int {
    Inode *parent = nullptr;
    Dentry *d = nullptr;
    u64 h = 0;
    std::string leaf;
    int rc = ResolveLeafLocked(path, &parent, &d, &h, &leaf);
    if (rc != 0) return rc;
    if (d != nullptr && d->state_ == kDentLive) {
      if (excl) return EEXIST;
      Inode *ino = FindInodeLocked(d->child_);
      if (ino == nullptr) return kMiss;
      if (ino->type_ == kFsTypeDir) return EISDIR;
      FillAttr(*ino, out);
      return 0;
    }
    if (leaf.size() > kMaxName) return ENAMETOOLONG;
    u64 id = 0;
    if (!MintIdLocked(&id)) return EAGAIN;
    Inode *ino = InsertInodeLocked(id);
    if (ino == nullptr) return ENOSPC;
    if (d == nullptr) {
      d = InsertDentLocked(parent->id_, h, leaf);
      if (d == nullptr) {
        FreeInodeLocked(ino);
        return ENOSPC;
      }
      LinkDentLocked(parent, d);
    } else {
      d->len_ = static_cast<u16>(leaf.size());  // this spelling now
      std::memcpy(d->name_, leaf.data(), leaf.size());
      d->name_[leaf.size()] = '\0';
    }
    ino->type_ = kFsTypeFile;
    ino->flags_ = kInodeXattrsKnown;  // a new file has none
    ino->mode_ = mode & 07777u;
    ino->uid_ = uid;
    ino->gid_ = gid;
    ino->nlink_ = 1;
    ino->atime_ = ino->mtime_ = ino->ctime_ = now;
    d->child_ = id;
    d->type_ = kFsTypeFile;
    d->state_ = kDentLive;
    parent->mtime_ = parent->ctime_ = now;
    detail::NsCrashHit(detail::kNsCpCreateLinked);
    d->seq_ = PushOpLocked(kOpCreate, path, std::string(), id, 0, now, 0,
                           mode & 07777u, uid, gid, 0);
    FillAttr(*ino, out);
    *created = true;
    return 0;
  });
}

inline int View::Mkdir(const std::string &path, u32 mode, u32 uid, u32 gid,
                       u64 now) {
  return Mutate(path.size(), 0, [&]() -> int {
    Inode *parent = nullptr;
    Dentry *d = nullptr;
    u64 h = 0;
    std::string leaf;
    int rc = ResolveLeafLocked(path, &parent, &d, &h, &leaf);
    if (rc != 0) return rc;
    if (d != nullptr && d->state_ == kDentLive) return EEXIST;
    if (leaf.size() > kMaxName) return ENAMETOOLONG;
    u64 id = 0;
    if (!MintIdLocked(&id)) return EAGAIN;
    Inode *ino = InsertInodeLocked(id);
    if (ino == nullptr) return ENOSPC;
    if (d == nullptr) {
      d = InsertDentLocked(parent->id_, h, leaf);
      if (d == nullptr) {
        FreeInodeLocked(ino);
        return ENOSPC;
      }
      LinkDentLocked(parent, d);
    } else {
      d->len_ = static_cast<u16>(leaf.size());
      std::memcpy(d->name_, leaf.data(), leaf.size());
      d->name_[leaf.size()] = '\0';
    }
    ino->type_ = kFsTypeDir;
    ino->mode_ = mode & 07777u;
    ino->uid_ = uid;
    ino->gid_ = gid;
    ino->nlink_ = 2;
    // A new directory has no other entries and no xattrs.
    ino->flags_ = kInodeComplete | kInodeXattrsKnown;
    ino->atime_ = ino->mtime_ = ino->ctime_ = now;
    d->child_ = id;
    d->type_ = kFsTypeDir;
    d->state_ = kDentLive;
    parent->mtime_ = parent->ctime_ = now;
    d->seq_ = PushOpLocked(kOpMkdir, path, std::string(), id, 0, now, 0,
                           mode & 07777u, uid, gid, 0);
    return 0;
  });
}

inline int View::Symlink(const std::string &target, const std::string &path,
                         u32 uid, u32 gid, u64 now) {
  if (target.size() > kMaxTarget) return ENAMETOOLONG;
  return Mutate(path.size(), target.size(), [&]() -> int {
    Inode *parent = nullptr;
    Dentry *d = nullptr;
    u64 h = 0;
    std::string leaf;
    int rc = ResolveLeafLocked(path, &parent, &d, &h, &leaf);
    if (rc != 0) return rc;
    if (d != nullptr && d->state_ == kDentLive) return EEXIST;
    if (leaf.size() > kMaxName) return ENAMETOOLONG;
    Target *tg = Targets();
    u32 slot = kNoSlot;
    for (u32 i = 0; i < root_->n_targets_; ++i) {
      if (tg[i].id_ == 0) {
        slot = i;
        break;
      }
    }
    if (slot == kNoSlot) return ENOSPC;
    u64 id = 0;
    if (!MintIdLocked(&id)) return EAGAIN;
    Inode *ino = InsertInodeLocked(id);
    if (ino == nullptr) return ENOSPC;
    if (d == nullptr) {
      d = InsertDentLocked(parent->id_, h, leaf);
      if (d == nullptr) {
        FreeInodeLocked(ino);
        return ENOSPC;
      }
      LinkDentLocked(parent, d);
    }
    tg[slot].id_ = id;
    tg[slot].len_ = static_cast<u16>(target.size());
    std::memcpy(tg[slot].s_, target.data(), target.size());
    tg[slot].s_[target.size()] = '\0';
    ino->type_ = kFsTypeSymlink;
    ino->flags_ = kInodeXattrsKnown;
    ino->mode_ = 0777;
    ino->uid_ = uid;
    ino->gid_ = gid;
    ino->nlink_ = 1;
    ino->size_ = target.size();
    ino->target_ = slot;
    ino->atime_ = ino->mtime_ = ino->ctime_ = now;
    d->child_ = id;
    d->type_ = kFsTypeSymlink;
    d->state_ = kDentLive;
    parent->mtime_ = parent->ctime_ = now;
    d->seq_ = PushOpLocked(kOpSymlink, path, target, id, 0, now, 0, 0777, uid,
                           gid, 0);
    return 0;
  });
}

inline int View::Readlink(const std::string &path, std::string *target) {
  root_->lock_.ReadLock();
  Inode *parent = nullptr;
  Dentry *d = nullptr;
  std::string leaf;
  int rc = ResolveLeafLocked(path, &parent, &d, nullptr, &leaf);
  if (rc == 0) {
    Inode *ino = (d != nullptr && d->state_ == kDentLive)
                     ? FindInodeLocked(d->child_)
                     : nullptr;
    if (d == nullptr || d->state_ != kDentLive) {
      rc = ENOENT;
    } else if (ino == nullptr || (ino->type_ == kFsTypeSymlink &&
                                  ino->target_ == kNoSlot)) {
      rc = kMiss;
    } else if (ino->type_ != kFsTypeSymlink) {
      rc = EINVAL;
    } else {
      const Target &t = Targets()[ino->target_];
      target->assign(t.s_, t.len_);
    }
  }
  root_->lock_.ReadUnlock();
  return rc;
}

inline int View::RemoveLocked(const std::string &path, bool dir, u64 now) {
  Inode *parent = nullptr;
  Dentry *d = nullptr;
  std::string leaf;
  int rc = ResolveLeafLocked(path, &parent, &d, nullptr, &leaf);
  if (rc != 0) return rc;
  if (d == nullptr || d->state_ != kDentLive) return ENOENT;
  Inode *ino = FindInodeLocked(d->child_);
  if (ino == nullptr) return kMiss;
  if (dir) {
    if (ino->type_ != kFsTypeDir) return ENOTDIR;
    if ((ino->flags_ & kInodeComplete) == 0) return kMiss;  // emptiness
    for (u32 s = ino->first_child_; s != kNoSlot;
         s = Dentries()[s].next_) {
      if (Dentries()[s].state_ == kDentLive) return ENOTEMPTY;
    }
    // Its negative entries can never be consulted again: free them.
    for (u32 s = ino->first_child_; s != kNoSlot;) {
      Dentry &c = Dentries()[s];
      s = c.next_;
      c.state_ = kDentTomb;
      root_->used_dentries_.fetch_sub(1, std::memory_order_relaxed);
    }
    FreeInodeLocked(ino);
  } else {
    if (ino->type_ == kFsTypeDir) return EISDIR;
    if (ino->nlink_ > 1) {
      ino->nlink_--;
      ino->ctime_ = now;
    } else {
      FreeInodeLocked(ino);
    }
  }
  d->state_ = kDentNeg;
  d->child_ = 0;
  parent->mtime_ = parent->ctime_ = now;
  d->seq_ = PushOpLocked(dir ? kOpRmdir : kOpUnlink, path, std::string(), 0,
                         0, now, 0, 0, 0, 0, 0);
  return 0;
}

inline int View::Unlink(const std::string &path, u64 now) {
  return Mutate(path.size(), 0,
                [&]() -> int { return RemoveLocked(path, false, now); });
}

inline int View::Rmdir(const std::string &path, u64 now) {
  return Mutate(path.size(), 0,
                [&]() -> int { return RemoveLocked(path, true, now); });
}

inline int View::Rename(const std::string &src, const std::string &dst,
                        bool noreplace, u64 now) {
  return Mutate(src.size(), dst.size(), [&]() -> int {
    Inode *sp = nullptr;
    Inode *dp = nullptr;
    Dentry *sd = nullptr;
    Dentry *dd = nullptr;
    u64 dh = 0;
    std::string sleaf;
    std::string dleaf;
    int rc = ResolveLeafLocked(src, &sp, &sd, nullptr, &sleaf);
    if (rc != 0) return rc;
    if (sd == nullptr || sd->state_ != kDentLive) return ENOENT;
    Inode *sino = FindInodeLocked(sd->child_);
    if (sino == nullptr) return kMiss;
    // The destination's parent chain must not pass through the source
    // directory (moving a directory under itself).
    const std::vector<std::string> dcomps = SplitPath(dst);
    if (dcomps.empty()) return EBUSY;
    {
      Inode *cur = FindInodeLocked(root_->root_id_);
      for (size_t i = 0; cur != nullptr && i + 1 < dcomps.size(); ++i) {
        if (cur->id_ == sino->id_) return EINVAL;
        Dentry *c = FindDentLocked(cur->id_, dcomps[i], nullptr);
        if (c == nullptr || c->state_ != kDentLive) break;
        cur = FindInodeLocked(c->child_);
      }
      if (cur != nullptr && cur->id_ == sino->id_) return EINVAL;
    }
    rc = ResolveLeafLocked(dst, &dp, &dd, &dh, &dleaf);
    if (rc != 0) return rc;
    if (dleaf.size() > kMaxName) return ENAMETOOLONG;
    if (dd == sd) {
      // Same entry (a case-only respelling on a case-insensitive volume).
      dd->len_ = static_cast<u16>(dleaf.size());
      std::memcpy(dd->name_, dleaf.data(), dleaf.size());
      dd->name_[dleaf.size()] = '\0';
      dd->seq_ = PushOpLocked(kOpRename, src, dst, sino->id_, 0, now, 0, 0, 0,
                              0, noreplace ? 1u : 0u);
      return 0;
    }
    if (dd != nullptr && dd->state_ == kDentLive) {
      if (noreplace) return EEXIST;
      if (dd->child_ == sd->child_) return 0;  // two names of one file
      Inode *dino = FindInodeLocked(dd->child_);
      if (dino == nullptr) return kMiss;
      if (sino->type_ == kFsTypeDir && dino->type_ != kFsTypeDir) {
        return ENOTDIR;
      }
      if (sino->type_ != kFsTypeDir && dino->type_ == kFsTypeDir) {
        return EISDIR;
      }
      if (dino->type_ == kFsTypeDir) {
        if ((dino->flags_ & kInodeComplete) == 0) return kMiss;
        for (u32 s = dino->first_child_; s != kNoSlot;
             s = Dentries()[s].next_) {
          if (Dentries()[s].state_ == kDentLive) return ENOTEMPTY;
        }
        FreeInodeLocked(dino);
      } else if (dino->nlink_ > 1) {
        dino->nlink_--;
      } else {
        FreeInodeLocked(dino);
      }
    }
    if (dd == nullptr) {
      dd = InsertDentLocked(dp->id_, dh, dleaf);
      if (dd == nullptr) return ENOSPC;
      LinkDentLocked(dp, dd);
    } else {
      dd->len_ = static_cast<u16>(dleaf.size());
      std::memcpy(dd->name_, dleaf.data(), dleaf.size());
      dd->name_[dleaf.size()] = '\0';
    }
    dd->child_ = sd->child_;
    dd->type_ = sd->type_;
    dd->state_ = kDentLive;
    sd->state_ = kDentNeg;
    sd->child_ = 0;
    sino->ctime_ = now;
    sp->mtime_ = sp->ctime_ = now;
    dp->mtime_ = dp->ctime_ = now;
    const u64 seq = PushOpLocked(kOpRename, src, dst, sino->id_, 0, now, 0, 0,
                                 0, 0, noreplace ? 1u : 0u);
    sd->seq_ = seq;
    dd->seq_ = seq;
    return 0;
  });
}

inline int View::Link(const std::string &src, const std::string &dst,
                      u64 now) {
  return Mutate(src.size(), dst.size(), [&]() -> int {
    Inode *sp = nullptr;
    Inode *dp = nullptr;
    Dentry *sd = nullptr;
    Dentry *dd = nullptr;
    u64 dh = 0;
    std::string sleaf;
    std::string dleaf;
    int rc = ResolveLeafLocked(src, &sp, &sd, nullptr, &sleaf);
    if (rc != 0) return rc;
    if (sd == nullptr || sd->state_ != kDentLive) return ENOENT;
    Inode *sino = FindInodeLocked(sd->child_);
    if (sino == nullptr) return kMiss;
    if (sino->type_ == kFsTypeDir) return EPERM;
    rc = ResolveLeafLocked(dst, &dp, &dd, &dh, &dleaf);
    if (rc != 0) return rc;
    if (dd != nullptr && dd->state_ == kDentLive) return EEXIST;
    if (dleaf.size() > kMaxName) return ENAMETOOLONG;
    if (dd == nullptr) {
      dd = InsertDentLocked(dp->id_, dh, dleaf);
      if (dd == nullptr) return ENOSPC;
      LinkDentLocked(dp, dd);
    }
    dd->child_ = sino->id_;
    dd->type_ = sino->type_;
    dd->state_ = kDentLive;
    sino->nlink_++;
    sino->ctime_ = now;
    dp->mtime_ = dp->ctime_ = now;
    dd->seq_ = PushOpLocked(kOpLink, src, dst, sino->id_, 0, now, 0, 0, 0, 0,
                            0);
    return 0;
  });
}

/**
 * Apply an attribute change (body_, which sees `ino`) to the inode a path
 * names and log it as one op of type_ with the given fields.
 */
#define CLIO_NS_SETATTR(path_, body_, type_, size_, t1_, t2_, mode_, uid_, \
                        gid_)                                              \
  Mutate((path_).size(), 0, [&]() -> int {                                 \
    Inode *ino = nullptr;                                                  \
    int rc = ResolveAnyLocked(path_, &ino);                                \
    if (rc != 0) return rc;                                                \
    body_;                                                                 \
    PushOpLocked(type_, path_, std::string(), ino->id_, size_, t1_, t2_,   \
                 mode_, uid_, gid_, 0);                                    \
    return 0;                                                              \
  })

inline int View::Chmod(const std::string &path, u32 mode, u64 now) {
  return CLIO_NS_SETATTR(path,
                         (ino->mode_ = mode & 07777u, ino->ctime_ = now),
                         kOpChmod, 0, 0, 0, mode & 07777u, 0, 0);
}

inline int View::Chown(const std::string &path, u32 uid, u32 gid, u64 now) {
  return CLIO_NS_SETATTR(
      path,
      (ino->uid_ = (uid == 0xFFFFFFFFu ? ino->uid_ : uid),
       ino->gid_ = (gid == 0xFFFFFFFFu ? ino->gid_ : gid), ino->ctime_ = now),
      kOpChown, 0, 0, 0, 0, uid, gid);
}

inline int View::Utimens(const std::string &path, u64 atime, u64 mtime,
                         u64 now) {
  return CLIO_NS_SETATTR(
      path,
      (ino->atime_ = (atime != 0 ? atime : ino->atime_),
       ino->mtime_ = (mtime != 0 ? mtime : ino->mtime_), ino->ctime_ = now),
      kOpUtimens, 0, atime, mtime, 0, 0, 0);
}

inline bool View::WaitApplied(u64 seq, double timeout_ms) {
  if (root_->applied_seq_.load(std::memory_order_acquire) >= seq) return true;
  Urge();
  const auto t0 = std::chrono::steady_clock::now();
  bool ok = true;
  while (root_->applied_seq_.load(std::memory_order_acquire) < seq) {
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    if (ms >= timeout_ms) {
      ok = false;
      break;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  Unurge();
  return ok;
}

inline int View::Truncate(const std::string &path, u64 size, u64 now,
                          u64 *id, u64 *old_size, u64 *seq) {
  return Mutate(path.size(), 0, [&]() -> int {
    Inode *ino = nullptr;
    int rc = ResolveAnyLocked(path, &ino);
    if (rc != 0) return rc;
    if (ino->type_ == kFsTypeDir) return EISDIR;
    if (ino->type_ != kFsTypeFile) return EINVAL;
    *old_size = ino->size_;
    ino->size_ = size;
    ino->mtime_ = ino->ctime_ = now;
    *id = ino->id_;
    *seq = PushOpLocked(kOpTruncate, path, std::string(), ino->id_, size, now,
                        0, 0, 0, 0, 0);
    return 0;
  });
}

inline int View::TruncateId(u64 id, u64 size, u64 now, u64 *old_size,
                            u64 *seq) {
  return Mutate(0, 0, [&]() -> int {
    Inode *ino = FindInodeLocked(id);
    if (ino == nullptr) return kMiss;
    if (ino->type_ == kFsTypeDir) return EISDIR;
    if (ino->type_ != kFsTypeFile) return EINVAL;
    *old_size = ino->size_;
    ino->size_ = size;
    ino->mtime_ = ino->ctime_ = now;
    // No path: the module applies a truncate with an id by inode.
    *seq = PushOpLocked(kOpTruncate, std::string(), std::string(), id, size,
                        now, 0, 0, 0, 0, 0);
    return 0;
  });
}

#undef CLIO_NS_SETATTR

inline int View::Readdir(const std::string &path, std::vector<DirEnt> *out) {
  root_->lock_.ReadLock();
  Inode *dir = nullptr;
  int rc = ResolveAnyLocked(path, &dir);
  if (rc == 0) {
    if (dir->type_ != kFsTypeDir) {
      rc = ENOTDIR;
    } else if ((dir->flags_ & kInodeComplete) == 0) {
      rc = kMiss;
    } else {
      for (u32 s = dir->first_child_; s != kNoSlot;
           s = Dentries()[s].next_) {
        const Dentry &d = Dentries()[s];
        if (d.state_ != kDentLive) continue;
        DirEnt e;
        e.name.assign(d.name_, d.len_);
        e.id = d.child_;
        e.type = d.type_;
        out->push_back(std::move(e));
      }
    }
  }
  root_->lock_.ReadUnlock();
  return rc;
}

inline void View::RaiseSize(u64 id, u64 end, u64 mtime) {
  ModuleGuard g(this);
  // Frozen: the tables are about to be rebuilt from the module, which
  // holds this size already.
  if (!g.Tables()) return;
  if (Inode *ino = FindInodeLocked(id)) {
    if (end > ino->size_) ino->size_ = end;
    if (mtime > ino->mtime_) ino->mtime_ = mtime;
    if (mtime > ino->ctime_) ino->ctime_ = mtime;
  }
}

inline bool View::StatId(u64 id, Attr *out) {
  root_->lock_.ReadLock();
  Inode *ino = FindInodeLocked(id);
  if (ino != nullptr) FillAttr(*ino, out);
  root_->lock_.ReadUnlock();
  return ino != nullptr;
}

inline bool View::HasInode(u64 id) {
  ModuleGuard g(this);
  if (!g.Tables()) return true;  // frozen: assume so (keeps the size)
  return FindInodeLocked(id) != nullptr;
}

inline int View::FillEntry(const std::string &parent_path,
                           const std::string &name, const Attr &a,
                           const std::string &target) {
  if (name.size() > kMaxName || a.id == 0) return EINVAL;
  root_->lock_.WriteLock();
  Inode *parent = nullptr;
  int rc = ResolveAnyLocked(parent_path, &parent);
  if (rc == 0 && parent->type_ != kFsTypeDir) rc = ENOTDIR;
  if (rc == 0) {
    u64 h = 0;
    Dentry *d = FindDentLocked(parent->id_, name, &h);
    // Never replace what the table holds: a live or negative entry here is
    // newer than the module's answer (its op may not be applied yet). A
    // live name whose inode is not held yet (a listing fill) gets it.
    if (d != nullptr && d->state_ == kDentLive && d->child_ == a.id &&
        FindInodeLocked(a.id) == nullptr) {
      Inode *ino = InsertInodeLocked(a.id);
      if (ino == nullptr) {
        rc = ENOSPC;
      } else {
        ino->type_ = a.type;
        ino->mode_ = a.mode;
        ino->uid_ = a.uid;
        ino->gid_ = a.gid;
        ino->nlink_ = a.nlink;
        ino->size_ = a.size;
        ino->atime_ = a.atime;
        ino->mtime_ = a.mtime;
        ino->ctime_ = a.ctime;
      }
    }
    if (d == nullptr) {
      Inode *ino = InsertInodeLocked(a.id);
      d = ino != nullptr ? InsertDentLocked(parent->id_, h, name) : nullptr;
      if (d == nullptr) {
        rc = ENOSPC;
      } else {
        if (ino->type_ == 0) {
          ino->type_ = a.type;
          ino->mode_ = a.mode;
          ino->uid_ = a.uid;
          ino->gid_ = a.gid;
          ino->nlink_ = a.nlink;
          ino->size_ = a.size;
          ino->atime_ = a.atime;
          ino->mtime_ = a.mtime;
          ino->ctime_ = a.ctime;
        }
        if (a.type == kFsTypeSymlink && ino->target_ == kNoSlot &&
            !target.empty() && target.size() <= kMaxTarget) {
          Target *tg = Targets();
          for (u32 i = 0; i < root_->n_targets_; ++i) {
            if (tg[i].id_ != 0) continue;
            tg[i].id_ = a.id;
            tg[i].len_ = static_cast<u16>(target.size());
            std::memcpy(tg[i].s_, target.data(), target.size());
            tg[i].s_[target.size()] = '\0';
            ino->target_ = i;
            break;
          }
        }
        d->child_ = a.id;
        d->type_ = a.type;
        d->state_ = kDentLive;
        LinkDentLocked(parent, d);
      }
    }
  }
  root_->lock_.WriteUnlock();
  return rc;
}

inline int View::FillAbsent(const std::string &parent_path,
                            const std::string &name) {
  if (name.size() > kMaxName) return EINVAL;
  root_->lock_.WriteLock();
  Inode *parent = nullptr;
  int rc = ResolveAnyLocked(parent_path, &parent);
  if (rc == 0 && parent->type_ != kFsTypeDir) rc = ENOTDIR;
  if (rc == 0) {
    u64 h = 0;
    if (FindDentLocked(parent->id_, name, &h) == nullptr) {
      // seq 0: already "applied", so it carries no pending removal.
      Dentry *d = InsertDentLocked(parent->id_, h, name);
      if (d == nullptr) {
        rc = ENOSPC;
      } else {
        LinkDentLocked(parent, d);
      }
    }
  }
  root_->lock_.WriteUnlock();
  return rc;
}

inline int View::FillName(const std::string &parent_path,
                          const std::string &name, u64 id, u32 type) {
  if (name.size() > kMaxName || id == 0) return EINVAL;
  root_->lock_.WriteLock();
  Inode *parent = nullptr;
  int rc = ResolveAnyLocked(parent_path, &parent);
  if (rc == 0 && parent->type_ != kFsTypeDir) rc = ENOTDIR;
  if (rc == 0) {
    u64 h = 0;
    if (FindDentLocked(parent->id_, name, &h) == nullptr) {
      Dentry *d = InsertDentLocked(parent->id_, h, name);
      if (d == nullptr) {
        rc = ENOSPC;
      } else {
        d->child_ = id;
        d->type_ = type;
        d->state_ = kDentLive;
        LinkDentLocked(parent, d);
      }
    }
  }
  root_->lock_.WriteUnlock();
  return rc;
}

inline int View::MarkComplete(const std::string &dir_path) {
  root_->lock_.WriteLock();
  Inode *dir = nullptr;
  int rc = ResolveAnyLocked(dir_path, &dir);
  if (rc == 0 && dir->type_ != kFsTypeDir) rc = ENOTDIR;
  if (rc == 0) dir->flags_ |= kInodeComplete;
  root_->lock_.WriteUnlock();
  return rc;
}

inline int View::ResolveAnyLocked(const std::string &path, Inode **ino) {
  const std::vector<std::string> comps = SplitPath(path);
  if (comps.empty()) {
    *ino = FindInodeLocked(root_->root_id_);
    return *ino == nullptr ? kMiss : 0;
  }
  Inode *parent = nullptr;
  int rc = ResolveDirLocked(comps, comps.size() - 1, &parent);
  if (rc != 0) return rc;
  Dentry *d = FindDentLocked(parent->id_, comps.back(), nullptr);
  if (d == nullptr) {
    return (parent->flags_ & kInodeComplete) ? ENOENT : kMiss;
  }
  if (d->state_ != kDentLive) return ENOENT;
  *ino = FindInodeLocked(d->child_);
  return *ino == nullptr ? kMiss : 0;
}

// ---------------------------------------------------------------------------
// Appends
// ---------------------------------------------------------------------------

inline AppendSlot *View::AppendSlotLocked(u64 id, bool insert) {
  const u32 mask = root_->n_appends_ - 1;
  AppendSlot *t = Appends();
  u32 i = static_cast<u32>(detail::Mix(id)) & mask;
  AppendSlot *free_slot = nullptr;
  for (u32 probes = 0; probes <= mask; ++probes, i = (i + 1) & mask) {
    if (t[i].id_ == id) return &t[i];
    if (t[i].id_ == detail::kFreedId) {
      if (free_slot == nullptr) free_slot = &t[i];
      continue;
    }
    if (t[i].id_ == 0) {
      if (free_slot == nullptr) free_slot = &t[i];
      break;
    }
  }
  if (!insert || free_slot == nullptr) return nullptr;
  std::memset(free_slot, 0, sizeof(AppendSlot));
  free_slot->id_ = id;
  return free_slot;
}

inline bool View::Appending(u64 id, int parity) const {
  const NsLock &l = root_->lock_;
  const u32 hw = l.hw_.load(std::memory_order_acquire);
  for (u32 i = 0; i < hw; ++i) {
    const NsThread &t = l.threads_[i];
    if (t.pid_.load(std::memory_order_acquire) == 0 ||
        t.app_id_.load(std::memory_order_acquire) != id) {
      continue;
    }
    const u32 n0 = t.app_n_[0].load(std::memory_order_acquire);
    const u32 n1 = t.app_n_[1].load(std::memory_order_acquire);
    if (parity < 0 ? n0 + n1 != 0 : (parity == 0 ? n0 : n1) != 0) return true;
  }
  return false;
}

inline void View::MaybeFreeAppendLocked(AppendSlot *a) {
  if (a->locked_ == 0 && a->bytes_[0] == 0 && a->bytes_[1] == 0 &&
      !Appending(a->id_, -1)) {
    a->id_ = detail::kFreedId;
  }
}

inline int View::AppendLock(u64 id) {
  for (;;) {
    root_->lock_.WriteLock();
    AppendSlot *a = AppendSlotLocked(id, true);
    if (a == nullptr) {
      root_->lock_.WriteUnlock();
      return ENOSPC;
    }
    if (a->locked_ == 0) {
      // The holder's pid: the module frees it if the holder dies.
      a->locked_ = static_cast<u32>(ctp::TimedMutex::MyPid());
      root_->lock_.WriteUnlock();
      return 0;
    }
    root_->lock_.WriteUnlock();
    std::this_thread::yield();
  }
}

inline void View::AppendUnlock(u64 id) {
  root_->lock_.WriteLock();
  if (AppendSlot *a = AppendSlotLocked(id, false)) {
    a->locked_ = 0;
    MaybeFreeAppendLocked(a);
  }
  root_->lock_.WriteUnlock();
}

inline int View::AppendStage(u64 id, u64 len, u32 *gen, u64 *off) {
  root_->lock_.WriteLock();
  AppendSlot *a = AppendSlotLocked(id, true);
  if (a == nullptr) {
    root_->lock_.WriteUnlock();
    return ENOSPC;
  }
  const u32 p = a->gen_ & 1u;
  NsThread *me = root_->lock_.Me();
  if (me == nullptr ||
      (me->app_id_.load() != id && me->app_id_.load() != 0 &&
       me->app_n_[0].load() + me->app_n_[1].load() != 0)) {
    root_->lock_.WriteUnlock();
    return EBUSY;  // unrecordable here: the caller takes the task path
  }
  *gen = a->gen_;
  *off = a->bytes_[p];
  a->bytes_[p] += len;
  // In flight while this thread's journal counts it (an exact, per-thread
  // record: the module clears a dead thread's, where a shared count could
  // not be attributed).
  me->app_n_[p].fetch_add(1, std::memory_order_relaxed);
  me->app_id_.store(id, std::memory_order_release);
  root_->lock_.WriteUnlock();
  return 0;
}

inline int View::AppendStaged(const std::string &path, u64 id, u32 gen) {
  return Mutate(path.size(), 0, [&]() -> int {
    PushOpLocked(kOpAppend, path, std::string(), id, 0, gen, 0, 0, 0, 0, 0);
    // After the push: the module may merge it now.
    NsThread *me = root_->lock_.Me();
    if (me != nullptr && me->app_id_.load() == id &&
        me->app_n_[gen & 1u].load() != 0) {
      me->app_n_[gen & 1u].fetch_sub(1, std::memory_order_release);
      if (me->app_n_[0].load() + me->app_n_[1].load() == 0) {
        me->app_id_.store(0, std::memory_order_release);
      }
    }
    return 0;
  });
}

inline bool View::AppendIdle(u64 id) {
  root_->lock_.ReadLock();
  const AppendSlot *a = AppendSlotLocked(id, false);
  const bool idle = a == nullptr ||
                    (a->bytes_[0] == 0 && a->bytes_[1] == 0 &&
                     !Appending(id, -1));
  root_->lock_.ReadUnlock();
  return idle;
}

inline bool View::AppendSeal(u64 id, u32 *gen) {
  ModuleGuard g(this);  // appends survive a repair: usable while frozen
  AppendSlot *a = AppendSlotLocked(id, false);
  bool sealed = false;
  if (a != nullptr && a->bytes_[a->gen_ & 1u] != 0) {
    *gen = a->gen_;
    a->gen_++;  // new appends stage into the next generation
    sealed = true;
  }
  return sealed;
}

inline bool View::AppendInflight(u64 id, u32 gen, u64 *len) {
  ModuleGuard g(this);
  const AppendSlot *a = AppendSlotLocked(id, false);
  bool busy = false;
  *len = 0;
  if (a != nullptr) {
    busy = Appending(id, static_cast<int>(gen & 1u));
    *len = a->bytes_[gen & 1u];
  }
  return busy;
}

inline void View::AppendMerged(u64 id, u32 gen) {
  ModuleGuard g(this);
  if (AppendSlot *a = AppendSlotLocked(id, false)) {
    a->bytes_[gen & 1u] = 0;
    MaybeFreeAppendLocked(a);
  }
}

// ---------------------------------------------------------------------------
// Extended attributes
// ---------------------------------------------------------------------------

inline XattrSlot *View::XattrSlotLocked(u64 id, const std::string &name,
                                        bool insert) {
  const u32 mask = root_->n_xattrs_ - 1;
  XattrSlot *t = Xattrs();
  u64 h = detail::Mix(id);
  for (unsigned char c : name) {
    h ^= c;
    h *= 1099511628211ull;
  }
  u32 i = static_cast<u32>(h) & mask;
  XattrSlot *free_slot = nullptr;
  for (u32 probes = 0; probes <= mask; ++probes, i = (i + 1) & mask) {
    XattrSlot &x = t[i];
    if (x.id_ == 0) {
      if (free_slot == nullptr) free_slot = &x;
      break;
    }
    if (x.id_ == detail::kFreedId) {
      if (free_slot == nullptr) free_slot = &x;
      continue;
    }
    if (x.id_ == id && x.hash_ == h && x.name_len_ == name.size() &&
        std::memcmp(x.name_, name.data(), name.size()) == 0) {
      return &x;
    }
  }
  if (!insert || free_slot == nullptr) return nullptr;
  free_slot->id_ = id;
  free_slot->hash_ = h;
  free_slot->name_len_ = static_cast<u16>(name.size());
  std::memcpy(free_slot->name_, name.data(), name.size());
  free_slot->name_[name.size()] = '\0';
  free_slot->val_len_ = 0;
  return free_slot;
}

inline void View::FreeXattrsLocked(Inode *ino) {
  XattrSlot *t = Xattrs();
  for (u32 i = 0; i < root_->n_xattrs_ && ino->nxattr_ != 0; ++i) {
    if (t[i].id_ == ino->id_) {
      t[i].id_ = detail::kFreedId;
      ino->nxattr_--;
    }
  }
  ino->nxattr_ = 0;
}

inline int View::Setxattr(const std::string &path, const std::string &name,
                          const std::string &value, u32 flags, u64 now) {
  if (name.empty() || name.size() > kMaxName) return ERANGE;
  if (value.size() > kMaxXattrValue) return kTooBig;
  std::string rec2 = name;
  rec2.push_back('\0');
  rec2.append(value);
  return Mutate(path.size(), rec2.size(), [&]() -> int {
    Inode *ino = nullptr;
    int rc = ResolveAnyLocked(path, &ino);
    if (rc != 0) return rc;
    if ((ino->flags_ & kInodeXattrsKnown) == 0) return kMiss;
    XattrSlot *x = XattrSlotLocked(ino->id_, name, false);
    if (x != nullptr && (flags & 1u)) return EEXIST;   // XATTR_CREATE
    if (x == nullptr && (flags & 2u)) return ENODATA;  // XATTR_REPLACE
    if (x == nullptr) {
      x = XattrSlotLocked(ino->id_, name, true);
      if (x == nullptr) return ENOSPC;
      ino->nxattr_++;
    }
    x->val_len_ = static_cast<u32>(value.size());
    std::memcpy(x->val_, value.data(), value.size());
    ino->ctime_ = now;
    PushOpLocked(kOpSetxattr, path, rec2, ino->id_, 0, now, 0, 0, 0, 0,
                 flags);
    return 0;
  });
}

inline int View::Getxattr(const std::string &path, const std::string &name,
                          std::string *value) {
  root_->lock_.ReadLock();
  Inode *ino = nullptr;
  int rc = ResolveAnyLocked(path, &ino);
  if (rc == 0) {
    if ((ino->flags_ & kInodeXattrsKnown) == 0) {
      rc = kMiss;
    } else if (XattrSlot *x = XattrSlotLocked(ino->id_, name, false)) {
      value->assign(x->val_, x->val_len_);
    } else {
      rc = ENODATA;
    }
  }
  root_->lock_.ReadUnlock();
  return rc;
}

inline int View::Listxattr(const std::string &path, std::string *names) {
  root_->lock_.ReadLock();
  Inode *ino = nullptr;
  int rc = ResolveAnyLocked(path, &ino);
  if (rc == 0) {
    if ((ino->flags_ & kInodeXattrsKnown) == 0) {
      rc = kMiss;
    } else {
      names->clear();
      const XattrSlot *t = Xattrs();
      u32 left = ino->nxattr_;
      for (u32 i = 0; i < root_->n_xattrs_ && left != 0; ++i) {
        if (t[i].id_ != ino->id_) continue;
        names->append(t[i].name_, t[i].name_len_);
        names->push_back('\0');
        --left;
      }
    }
  }
  root_->lock_.ReadUnlock();
  return rc;
}

inline int View::Removexattr(const std::string &path, const std::string &name,
                             u64 now) {
  return Mutate(path.size(), name.size(), [&]() -> int {
    Inode *ino = nullptr;
    int rc = ResolveAnyLocked(path, &ino);
    if (rc != 0) return rc;
    if ((ino->flags_ & kInodeXattrsKnown) == 0) return kMiss;
    XattrSlot *x = XattrSlotLocked(ino->id_, name, false);
    if (x == nullptr) return ENODATA;
    x->id_ = detail::kFreedId;
    ino->nxattr_--;
    ino->ctime_ = now;
    PushOpLocked(kOpRemovexattr, path, name, ino->id_, 0, now, 0, 0, 0, 0,
                 0);
    return 0;
  });
}

inline int View::FillXattrs(
    const std::string &path,
    const std::vector<std::pair<std::string, std::string>> &attrs) {
  for (const auto &a : attrs) {
    if (a.second.size() > kMaxXattrValue || a.first.size() > kMaxName) {
      return kTooBig;
    }
  }
  root_->lock_.WriteLock();
  Inode *ino = nullptr;
  int rc = ResolveAnyLocked(path, &ino);
  if (rc == 0 && (ino->flags_ & kInodeXattrsKnown) == 0) {
    for (const auto &a : attrs) {
      XattrSlot *x = XattrSlotLocked(ino->id_, a.first, true);
      if (x == nullptr) {
        rc = ENOSPC;
        break;
      }
      ino->nxattr_++;  // unknown until now, so none was held
      x->val_len_ = static_cast<u32>(a.second.size());
      std::memcpy(x->val_, a.second.data(), a.second.size());
    }
    if (rc == 0) ino->flags_ |= kInodeXattrsKnown;
  }
  root_->lock_.WriteUnlock();
  return rc;
}

inline int View::ForgetXattrs(const std::string &path) {
  root_->lock_.WriteLock();
  Inode *ino = nullptr;
  const int rc = ResolveAnyLocked(path, &ino);
  if (rc == 0) {
    if (ino->nxattr_ != 0) FreeXattrsLocked(ino);
    ino->flags_ &= ~kInodeXattrsKnown;
  }
  root_->lock_.WriteUnlock();
  return rc;
}


// ---------------------------------------------------------------------------
// Crash repair (the module; SHM_CACHE_DESIGN.md "Crash recovery")
// ---------------------------------------------------------------------------

inline View::ModuleGuard::ModuleGuard(View *v) : v_(v) {
  NsLock &l = v_->root_->lock_;
  const u64 me = l.NewWriterId();
  for (u32 spin = 0;; ++spin) {
    u64 expect = 0;
    if (l.writer_.compare_exchange_weak(expect, me,
                                        std::memory_order_seq_cst)) {
      // A dead reader's bit would hold this module thread (maybe the one
      // the periodic repair needs): clear dead holds inline.
      l.WaitReaders([this] { v_->RepairDeadInline(); });
      write_ = true;
      return;
    }
    if (expect & NsLock::kFrozen) {
      frozen_ = std::unique_lock<std::mutex>(v_->frozen_mu_);
      return;
    }
    detail::NsBackoff(spin);
    if (spin % NsLock::kStallSpins == NsLock::kStallSpins - 1) {
      v_->RepairDeadInline();  // a dead writer is frozen out
    }
  }
}

inline View::ModuleGuard::~ModuleGuard() {
  if (write_) v_->root_->lock_.WriteUnlock();
}

inline void View::RepairDeadInline() {
  std::unique_lock<std::mutex> g(repair_mu_, std::try_to_lock);
  if (!g.owns_lock()) return;
  std::set<u32> dead;
  ReapDeadThreads(&dead);
  FreezeDeadWriter(dead);
}

inline bool View::FreezeDeadWriter(const std::set<u32> &dead) {
  NsLock &l = root_->lock_;
  const u64 w = l.writer_.load(std::memory_order_acquire);
  const u32 wpid = static_cast<u32>(w >> 32);
  const u32 self = static_cast<u32>(ctp::SystemInfo::GetPid());
  if (w == 0 || (w & NsLock::kFrozen) != 0 || wpid == self ||
      (!dead.count(wpid) && ctp::TimedMutex::IsProcessAlive(wpid, 0))) {
    return false;
  }
  // A writer died mid-change: its table edits may be half-done and were
  // never logged. Lock clients out (frozen) until the log drains, then
  // rebuild the tables from the module (ResetTables).
  const u64 frozen = (static_cast<u64>(self) << 32) | NsLock::kFrozen;
  u64 expect = w;
  return l.writer_.compare_exchange_strong(expect, frozen);
}

inline u32 View::RepairDead() {
  std::lock_guard<std::mutex> g(repair_mu_);
  std::set<u32> dead;
  const u32 freed = ReapDeadThreads(&dead);
  NsLock &l = root_->lock_;
  const u32 self = static_cast<u32>(ctp::SystemInfo::GetPid());
  const u32 wpid = static_cast<u32>(l.writer_.load() >> 32);
  if (FreezeDeadWriter(dead)) dead.insert(wpid);
  FreeDeadAppendLocks(dead);
  if (l.Frozen() && static_cast<u32>(l.writer_.load() >> 32) == self &&
      root_->log_head_.load(std::memory_order_acquire) ==
          root_->log_tail_.load(std::memory_order_acquire)) {
    ResetTables();
  }
  return freed;
}

inline u32 View::ReapDeadThreads(std::set<u32> *dead) {
  NsLock &l = root_->lock_;
  const u32 self = static_cast<u32>(ctp::SystemInfo::GetPid());
  const u32 hw = l.hw_.load(std::memory_order_acquire);
  std::set<u32> alive;
  u32 freed = 0;
  for (u32 i = 0; i < hw; ++i) {
    NsThread &t = l.threads_[i];
    const u32 pid = t.pid_.load(std::memory_order_acquire);
    if (pid == 0 || pid == self || alive.count(pid)) continue;
    if (!dead->count(pid)) {
      if (ctp::TimedMutex::IsProcessAlive(pid, t.start_)) {
        alive.insert(pid);
        continue;
      }
      dead->insert(pid);
    }
    // Its read hold (exactly its bit) and its append record go; then the
    // slot is free for another thread.
    l.readers_[i / 64].fetch_and(~(1ull << (i % 64)), std::memory_order_acq_rel);
    t.app_n_[0].store(0);
    t.app_n_[1].store(0);
    t.app_id_.store(0, std::memory_order_release);
    t.nest_ = 0;
    t.pid_.store(0, std::memory_order_release);
    ++freed;
  }
  return freed;
}

inline void View::FreeDeadAppendLocks(const std::set<u32> &dead) {
  // The holder of an append lock need not be a registered thread (writers
  // are not): check each holder's liveness, once per pid.
  AppendSlot *a = Appends();
  std::set<u32> gone(dead);
  std::set<u32> alive;
  bool any = false;
  for (u32 i = 0; i < root_->n_appends_; ++i) {
    const u32 h = a[i].locked_;
    if (h == 0 || a[i].id_ == 0 || a[i].id_ == detail::kFreedId ||
        alive.count(h) || gone.count(h)) {
      any = any || gone.count(h) != 0;
      continue;
    }
    if (ctp::TimedMutex::IsProcessAlive(h, 0)) {
      alive.insert(h);
    } else {
      gone.insert(h);
      any = true;
    }
  }
  if (!any) return;
  ModuleGuard g(this);
  for (u32 i = 0; i < root_->n_appends_; ++i) {
    if (a[i].id_ != 0 && a[i].id_ != detail::kFreedId && a[i].locked_ != 0 &&
        gone.count(a[i].locked_)) {
      a[i].locked_ = 0;
      MaybeFreeAppendLocked(&a[i]);
    }
  }
}

inline void View::ResetTables() {
  std::lock_guard<std::mutex> g(frozen_mu_);  // module threads stay out
  std::memset(static_cast<void *>(Inodes()), 0,
              sizeof(Inode) * root_->n_inodes_);
  std::memset(static_cast<void *>(Dentries()), 0,
              sizeof(Dentry) * root_->n_dentries_);
  std::memset(static_cast<void *>(Targets()), 0,
              sizeof(Target) * root_->n_targets_);
  std::memset(static_cast<void *>(Xattrs()), 0,
              sizeof(XattrSlot) * root_->n_xattrs_);
  root_->used_inodes_.store(0);
  root_->used_dentries_.store(0);
  // Nothing remains to apply (the log is empty): a seq the dead writer
  // took without publishing its record must not keep waiters waiting.
  root_->applied_seq_.store(root_->pushed_seq_.load(),
                            std::memory_order_release);
  // Everything below the root is filled from the module again.
  InitRootInodeLocked(false);
  root_->lock_.writer_.store(0, std::memory_order_release);
}

}  // namespace clio::cte::filesystem::ns

#endif  // CLIO_CTE_FILESYSTEM_FS_NS_SHM_IMPL_H_
