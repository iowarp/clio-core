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
// Segment
// ---------------------------------------------------------------------------

inline bool View::Format(const std::string &name, const Limits &lim,
                         bool case_insensitive, u64 root_id,
                         bool root_complete) {
  const u32 ni = detail::Pow2(lim.max_inodes_);
  const u32 nd = detail::Pow2(lim.max_dentries_);
  const u32 nt = std::max<u32>(lim.max_targets_, 16);
  const u64 lb = std::max<u64>((lim.log_bytes_ + 7) & ~u64(7), 1u << 20);
  const u64 root_b = detail::Align64(sizeof(Root));
  const u64 ino_off = root_b;
  const u64 den_off = detail::Align64(ino_off + u64(ni) * sizeof(Inode));
  const u64 tgt_off = detail::Align64(den_off + u64(nd) * sizeof(Dentry));
  const u64 log_off = detail::Align64(tgt_off + u64(nt) * sizeof(Target));
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
  root_->lock_.Init();
  root_->root_id_ = root_id;
  root_->n_inodes_ = ni;
  root_->n_dentries_ = nd;
  root_->n_targets_ = nt;
  root_->inodes_off_ = ino_off;
  root_->dentries_off_ = den_off;
  root_->targets_off_ = tgt_off;
  root_->log_off_ = log_off;
  root_->log_bytes_ = lb;
  Inode *r = InsertInodeLocked(root_id);
  r->type_ = kFsTypeDir;
  r->mode_ = 0755;
  r->nlink_ = 2;
  r->first_child_ = kNoSlot;
  r->target_ = kNoSlot;
  r->flags_ = root_complete ? kInodeComplete : 0u;
  root_->ready_.store(1, std::memory_order_release);
  return true;
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
  return true;
}

inline void View::Detach() {
  if (!mapped_) return;
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

inline void View::PopOp(u64 seq) {
  const u64 cap = root_->log_bytes_;
  const u64 head = root_->log_head_.load(std::memory_order_relaxed);
  const OpRec *r = reinterpret_cast<const OpRec *>(Log() + head % cap);
  root_->log_head_.store(head + r->len_, std::memory_order_release);
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
  root_->lock_.WriteLock();
  root_->id_major_ = major;
  root_->id_next_.store(first, std::memory_order_relaxed);
  root_->id_limit_.store(first + count, std::memory_order_relaxed);
  root_->lock_.WriteUnlock();
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
    ino->mode_ = mode & 07777u;
    ino->uid_ = uid;
    ino->gid_ = gid;
    ino->nlink_ = 1;
    ino->atime_ = ino->mtime_ = ino->ctime_ = now;
    d->child_ = id;
    d->type_ = kFsTypeFile;
    d->state_ = kDentLive;
    parent->mtime_ = parent->ctime_ = now;
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
    ino->flags_ = kInodeComplete;  // a new directory has no other entries
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

inline int View::Truncate(const std::string &path, u64 size, u64 now,
                          u64 *id) {
  return Mutate(path.size(), 0, [&]() -> int {
    Inode *ino = nullptr;
    int rc = ResolveAnyLocked(path, &ino);
    if (rc != 0) return rc;
    if (ino->type_ == kFsTypeDir) return EISDIR;
    if (ino->type_ != kFsTypeFile) return EINVAL;
    ino->size_ = size;
    ino->mtime_ = ino->ctime_ = now;
    *id = ino->id_;
    PushOpLocked(kOpTruncate, path, std::string(), ino->id_, size, now, 0, 0,
                 0, 0, 0);
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
  root_->lock_.WriteLock();
  if (Inode *ino = FindInodeLocked(id)) {
    if (end > ino->size_) ino->size_ = end;
    if (mtime > ino->mtime_) ino->mtime_ = mtime;
    if (mtime > ino->ctime_) ino->ctime_ = mtime;
  }
  root_->lock_.WriteUnlock();
}

inline bool View::StatId(u64 id, Attr *out) {
  root_->lock_.ReadLock();
  Inode *ino = FindInodeLocked(id);
  if (ino != nullptr) FillAttr(*ino, out);
  root_->lock_.ReadUnlock();
  return ino != nullptr;
}

inline bool View::HasInode(u64 id) {
  root_->lock_.ReadLock();
  const bool has = FindInodeLocked(id) != nullptr;
  root_->lock_.ReadUnlock();
  return has;
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
    // newer than the module's answer (its op may not be applied yet).
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

}  // namespace clio::cte::filesystem::ns

#endif  // CLIO_CTE_FILESYSTEM_FS_NS_SHM_IMPL_H_
