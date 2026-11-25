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
 * The clio-fs shared-memory namespace (see filesystem/SHM_CACHE_DESIGN.md).
 *
 * One segment per node, formatted by the filesystem module and mapped by
 * every client process on the node. It holds the namespace this node uses:
 * inodes, directory entries and symlink targets. Clients perform every
 * namespace operation against it directly -- lookup, stat, readdir, create,
 * mkdir, unlink, rmdir, rename, link, symlink, readlink, chmod, chown,
 * utimens, truncate -- and append the change to the segment's op log. No
 * operation submits a task; the module drains the log in the background and
 * applies it, in order, at each name's home (NsDrain).
 *
 * Rules that keep it correct:
 *  - Every mutation and its log record happen under the segment's writer
 *    lock, so the log order is the mutation order.
 *  - A removed name stays as a NEGATIVE entry until the module has applied
 *    the removal (Root::applied_seq_). A lookup can therefore never fetch a
 *    stale copy of it from the module.
 *  - A name absent from the table is authoritative ENOENT only under a
 *    directory whose listing is COMPLETE (every entry present). Elsewhere
 *    the caller fills the missing entry from the module (FillEntry) and
 *    retries; a fill never replaces an entry the table already holds.
 *  - Inode ids come from blocks of the module's durable id reservation
 *    (Root::id_next_/id_limit_), so they are unique across restarts.
 */
#ifndef CLIO_CTE_FILESYSTEM_FS_NS_SHM_H_
#define CLIO_CTE_FILESYSTEM_FS_NS_SHM_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <clio_ctp/data_structures/ipc/ring_buffer.h>
#include <clio_ctp/memory/backend/posix_shm_mmap.h>
#include <clio_cte/filesystem/fs_canonical.h>
#include <clio_cte/filesystem/fs_shard.h>

namespace clio::cte::filesystem::ns {

using u16 = std::uint16_t;
using clio::run::u32;
using clio::run::u64;

/** Longest name component held in the table (POSIX NAME_MAX). */
static constexpr u32 kMaxName = 255;
/** Longest symlink target held in the table. */
static constexpr u32 kMaxTarget = 1023;
/** Longest path an op log record carries. */
static constexpr u32 kMaxPath = 4095;
/** "No slot" in an index field. */
static constexpr u32 kNoSlot = 0xFFFFFFFFu;
/** Returned by lookups that need a fill from the module. */
static constexpr int kMiss = -1;
/** An xattr value too large to hold inline: the module keeps it. */
static constexpr int kTooBig = -3;
/** Largest xattr value held in the segment. */
static constexpr u32 kMaxXattrValue = 2048;

/** Inode flags. */
enum : u32 {
  kInodeComplete = 1u << 0,  ///< directory: every entry is in the table
  kInodeXattrsKnown = 1u << 1,  ///< every xattr of the inode is held here
};

/** Directory entry states. */
enum : u32 {
  kDentEmpty = 0,
  kDentLive = 1,
  kDentNeg = 2,   ///< removed here; kept until the removal is applied
  kDentTomb = 3,  ///< freed slot (keeps probe sequences intact)
};

/** Op log record types. */
enum : u32 {
  kOpCreate = 1,    ///< path, id, mode, uid, gid, t1 = ctime
  kOpMkdir = 2,     ///< path, id, mode, uid, gid
  kOpUnlink = 3,    ///< path
  kOpRmdir = 4,     ///< path
  kOpRename = 5,    ///< path -> path2, flags
  kOpLink = 6,      ///< existing path -> new path2
  kOpSymlink = 7,   ///< new path, path2 = target
  kOpChmod = 8,     ///< path, mode
  kOpChown = 9,     ///< path, uid, gid
  kOpUtimens = 10,  ///< path, t1 = atime, t2 = mtime
  kOpTruncate = 11, ///< path, id, size
  kOpAppend = 12,   ///< path, id: the file has staged appends to merge
  kOpSetxattr = 13, ///< path, path2 = name NUL value, flags
  kOpRemovexattr = 14,  ///< path, path2 = name
  kOpWrap = 0xFFFFFFFFu,  ///< filler at the end of the ring
};

/** Writer-preferring reader/writer spin lock in shared memory. */
struct RwSpin {
  std::atomic<u32> state_;    ///< readers, or kWriter
  std::atomic<u32> waiting_;  ///< writers waiting for the lock
  static constexpr u32 kWriter = 0x80000000u;

  void Init() {
    state_.store(0);
    waiting_.store(0);
  }
  void ReadLock() {
    for (;;) {
      u32 s = state_.load(std::memory_order_relaxed);
      if ((s & kWriter) == 0 &&
          waiting_.load(std::memory_order_relaxed) == 0 &&
          state_.compare_exchange_weak(s, s + 1, std::memory_order_acquire)) {
        return;
      }
      std::this_thread::yield();
    }
  }
  void ReadUnlock() { state_.fetch_sub(1, std::memory_order_release); }
  void WriteLock() {
    waiting_.fetch_add(1, std::memory_order_relaxed);
    for (;;) {
      u32 expect = 0;
      if (state_.compare_exchange_weak(expect, kWriter,
                                       std::memory_order_acquire)) {
        break;
      }
      std::this_thread::yield();
    }
    waiting_.fetch_sub(1, std::memory_order_relaxed);
  }
  void WriteUnlock() { state_.store(0, std::memory_order_release); }
};

/** One inode. id 0 = empty slot, ~0 = freed slot. */
struct Inode {
  u64 id_;
  u32 type_;  ///< kFsType*
  u32 flags_;
  u32 mode_;  ///< permission bits
  u32 uid_;
  u32 gid_;
  u32 nlink_;
  u64 size_;  ///< logical size (files), target length (symlinks)
  u64 atime_;
  u64 mtime_;
  u64 ctime_;
  u32 first_child_;  ///< directory: head of its entry list (dentry slot)
  u32 target_;       ///< symlink: target slot
  u32 nxattr_;       ///< xattr slots it holds
  u32 pad_;
};

/** One directory entry, keyed by (parent id, canonical name). */
struct Dentry {
  u64 parent_;
  u64 hash_;
  u64 child_;
  u64 seq_;  ///< op seq of its last change (NEG: removable once applied)
  u32 type_;
  u32 state_;
  u32 next_;  ///< sibling list
  u32 prev_;
  u16 len_;
  char name_[kMaxName + 1];  ///< as spelled
};

/** One symlink target. */
struct Target {
  u64 id_;  ///< owning inode, 0 = free
  u16 len_;
  char s_[kMaxTarget + 1];
};

/** One extended attribute. id 0 = empty, ~0 = freed. */
struct XattrSlot {
  u64 id_;
  u64 hash_;
  u16 name_len_;
  u16 pad_;
  u32 val_len_;
  char name_[kMaxName + 1];
  char val_[kMaxXattrValue];
};

/**
 * Per-file append state (see SHM_CACHE_DESIGN.md "Appends"). id 0 = empty,
 * ~0 = freed. Generation g stages into blob app.<node>.<g> of the file's
 * tag; bytes_/inflight_ are indexed by g & 1 (one generation is staged into
 * while the previous one merges).
 */
struct AppendSlot {
  u64 id_;
  u32 gen_;       ///< generation new appends stage into
  u32 locked_;    ///< one-node append path: the file's append lock
  u64 bytes_[2];  ///< bytes staged per generation (by parity)
  u32 inflight_[2];  ///< appends copying into a generation right now
};

/** Header of one op log record; path bytes follow. */
struct OpRec {
  u32 len_;   ///< whole record, 8-byte aligned
  u32 type_;
  u64 seq_;
  u64 id_;
  u64 size_;
  u64 t1_;
  u64 t2_;
  u32 mode_;
  u32 uid_;
  u32 gid_;
  u32 flags_;
  u16 plen_;
  u16 p2len_;
  u32 pad_;
};

/** Attributes of one name, as lookups report them. */
struct Attr {
  u64 id = 0;
  u32 type = 0;
  u32 mode = 0;
  u32 uid = 0;
  u32 gid = 0;
  u32 nlink = 0;
  u64 size = 0;
  u64 atime = 0;
  u64 mtime = 0;
  u64 ctime = 0;
};

/** One readdir entry. */
struct DirEnt {
  std::string name;
  u64 id = 0;
  u32 type = 0;
};

/** Segment sizing (from the filesystem module's config). */
struct Limits {
  u32 max_inodes_ = 256 * 1024;
  u32 max_dentries_ = 256 * 1024;
  u32 max_targets_ = 4096;
  u32 max_append_files_ = 4096;
  u32 max_xattrs_ = 4096;
  u64 log_bytes_ = 16ull * 1024 * 1024;
};

/** Segment root. */
struct Root {
  static constexpr u64 kMagic = 0x3153534E53464C43ull;  // "CLFSNSS1"
  static constexpr u32 kVersion = 4;
  u64 magic_;
  u32 version_;
  std::atomic<u32> ready_;
  u32 case_insensitive_;
  u32 node_id_;  ///< the node this segment belongs to
  RwSpin lock_;
  u64 root_id_;
  // Id block handed out by the module (durably reserved).
  u32 id_major_;
  u32 pad1_;
  std::atomic<u64> id_next_;
  std::atomic<u64> id_limit_;
  // Tables.
  u32 n_inodes_;     ///< power of two
  u32 n_dentries_;   ///< power of two
  u32 n_targets_;
  u32 n_appends_;  ///< power of two
  /** Nonzero when clients run the namespace here (one node owns it);
   *  zero: the segment carries appends only (multi-node volume). */
  u32 namespace_on_;
  u32 pad2_;
  u64 inodes_off_;
  u64 dentries_off_;
  u64 targets_off_;
  u64 appends_off_;
  u64 xattrs_off_;
  u32 n_xattrs_;  ///< power of two
  u32 pad5_;
  u64 log_off_;
  u64 log_bytes_;
  // Op log: byte offsets, monotonically increasing.
  std::atomic<u64> log_head_;  ///< consumer (module)
  std::atomic<u64> log_tail_;  ///< producers (under lock_)
  std::atomic<u64> pushed_seq_;
  std::atomic<u64> applied_seq_;
  std::atomic<u64> used_inodes_;
  std::atomic<u64> used_dentries_;
  /** Clients waiting on the drain (WaitApplied, a pending size): the
   *  module also drains sizes on its fast period while nonzero. */
  std::atomic<u32> urgent_;
};

/**
 * Published in the runtime's metadata segment and registered in its
 * MetadataDirectory under LocatorKey(pool): how a client finds the segment.
 */
struct Locator {
  static constexpr u64 kMagic = 0x31434F4C534E4643ull;  // "CFNSLOC1"
  u64 magic_;
  std::atomic<u32> ready_;
  u32 pid_;
};

/** MetadataDirectory key of a filesystem pool's Locator. */
inline u64 LocatorKey(u64 pool_u64) { return pool_u64 ^ (1ull << 63); }

/** Segment name for a filesystem pool of a given runtime. */
inline std::string SegmentName(u32 runtime_pid, u64 pool_u64) {
  return "clio_fsns_" + std::to_string(runtime_pid) + "_" +
         std::to_string(pool_u64);
}

/** Bytes one op record takes (8-aligned). */
inline u32 OpRecBytes(size_t plen, size_t p2len) {
  const size_t raw = sizeof(OpRec) + plen + p2len;
  return static_cast<u32>((raw + 7) & ~size_t(7));
}

/** Split a normalized absolute path into its components. */
inline std::vector<std::string> SplitPath(const std::string &path) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < path.size()) {
    while (i < path.size() && path[i] == '/') ++i;
    size_t j = i;
    while (j < path.size() && path[j] != '/') ++j;
    if (j > i) out.emplace_back(path.substr(i, j - i));
    i = j;
  }
  return out;
}

/**
 * A process's view of the namespace segment. The module formats it; clients
 * attach. Every operation below is complete in this process.
 */
class View {
 public:
  View() = default;
  ~View() { Detach(); }
  View(const View &) = delete;
  View &operator=(const View &) = delete;

  /**
   * Create and format the segment (the module, once).
   * @param name segment name
   * @param lim table sizes
   * @param case_insensitive whether names fold case
   * @param root_id the root directory's packed id
   * @param root_complete whether the root starts with a complete listing
   *        (a fresh filesystem: nothing exists that the table lacks)
   * @param node_id this node
   * @param namespace_on clients run the namespace here (one node)
   * @return true when formatted
   */
  bool Format(const std::string &name, const Limits &lim,
              bool case_insensitive, u64 root_id, bool root_complete,
              u32 node_id = 0, bool namespace_on = true);

  /** @return the node this segment belongs to */
  u32 NodeId() const { return root_->node_id_; }

  /** @return true when clients run the namespace in this segment */
  bool NamespaceOn() const { return root_->namespace_on_ != 0; }

  /**
   * Map an existing segment (a client).
   * @param name segment name
   * @return true when mapped and ready
   */
  bool Attach(const std::string &name);

  /** Unmap (the module also destroys it). */
  void Detach();

  /** @return true when mapped and ready */
  bool Ready() const {
    return root_ != nullptr &&
           root_->ready_.load(std::memory_order_acquire) == 1;
  }

  /** @return the segment root (null when not mapped) */
  Root *GetRoot() { return root_; }

  // ---- client operations: 0, an errno, or kMiss (fill, then retry) ----

  /**
   * Look a path up.
   * @param path normalized absolute path
   * @param out receives its attributes
   * @return 0, ENOENT, ENOTDIR, or kMiss
   */
  int Lookup(const std::string &path, Attr *out);

  /**
   * Open-or-create a regular file.
   * @param path file path
   * @param mode permission bits for a new file
   * @param uid owner of a new file
   * @param gid group of a new file
   * @param excl O_EXCL
   * @param now wall ns
   * @param out the file's attributes
   * @param created set true when this call created it
   * @return 0, EEXIST, EISDIR, ENOENT (parent), ENOTDIR, EAGAIN (no ids
   *         yet), ENOSPC (table full), or kMiss
   */
  int Create(const std::string &path, u32 mode, u32 uid, u32 gid, bool excl,
             u64 now, Attr *out, bool *created);

  /** mkdir(2); same returns as Create. */
  int Mkdir(const std::string &path, u32 mode, u32 uid, u32 gid, u64 now);

  /** symlink(2): `path` names a new link to `target`. */
  int Symlink(const std::string &target, const std::string &path, u32 uid,
              u32 gid, u64 now);

  /** readlink(2). */
  int Readlink(const std::string &path, std::string *target);

  /** unlink(2). */
  int Unlink(const std::string &path, u64 now);

  /** rmdir(2). */
  int Rmdir(const std::string &path, u64 now);

  /**
   * rename(2).
   * @param noreplace RENAME_NOREPLACE
   */
  int Rename(const std::string &src, const std::string &dst, bool noreplace,
             u64 now);

  /** link(2): `dst` becomes another name of `src`. */
  int Link(const std::string &src, const std::string &dst, u64 now);

  /** chmod(2). */
  int Chmod(const std::string &path, u32 mode, u64 now);

  /** chown(2); kNoChange (~0) leaves a field alone. */
  int Chown(const std::string &path, u32 uid, u32 gid, u64 now);

  /** utimens(2); 0 leaves a time alone. */
  int Utimens(const std::string &path, u64 atime, u64 mtime, u64 now);

  /**
   * truncate(2).
   * @param id receives the file's id
   * @param old_size receives its size before the call
   * @param seq receives the op's log seq (a shrink waits for it: WaitApplied)
   */
  int Truncate(const std::string &path, u64 size, u64 now, u64 *id,
               u64 *old_size, u64 *seq);

  /**
   * ftruncate(2) through an open descriptor: by inode, never by name. The
   * name may be gone (unlinked while open) and must not be resolved.
   * @param id the open file's id
   * @param size new size
   * @param now change time (ns)
   * @param old_size receives its size before the call
   * @param seq receives the op's log seq (WaitApplied)
   * @return 0, EISDIR, EINVAL (not a file), or kMiss (inode not in the table)
   */
  int TruncateId(u64 id, u64 size, u64 now, u64 *old_size, u64 *seq);

  /**
   * Wait until the module has applied the log through `seq` (a wait on the
   * background drain, never a task).
   * @param seq a seq a mutation returned, or PushedSeq()
   * @param timeout_ms give up after this long
   * @return true when applied
   */
  bool WaitApplied(u64 seq, double timeout_ms);

  /** A client starts / stops waiting on the drain (see Root::urgent_). */
  void Urge() { root_->urgent_.fetch_add(1, std::memory_order_acq_rel); }
  void Unurge() { root_->urgent_.fetch_sub(1, std::memory_order_acq_rel); }

  /** @return true while any client waits on the drain */
  bool Urgent() const {
    return root_->urgent_.load(std::memory_order_acquire) != 0;
  }

  /** @return true when every pushed record has been consumed */
  bool LogEmpty() const {
    return root_->log_head_.load(std::memory_order_acquire) ==
           root_->log_tail_.load(std::memory_order_acquire);
  }

  /** @return the seq of the last op pushed */
  u64 PushedSeq() const {
    return root_->pushed_seq_.load(std::memory_order_acquire);
  }

  /**
   * List a directory.
   * @param path directory path
   * @param out receives its entries
   * @return 0, ENOENT, ENOTDIR, or kMiss (listing incomplete)
   */
  int Readdir(const std::string &path, std::vector<DirEnt> *out);

  /**
   * Raise a file's size and mtime after a page write (no log record: the
   * size travels through the size table).
   */
  void RaiseSize(u64 id, u64 end, u64 mtime);

  /** Attributes of an inode by id; false when not held. */
  bool StatId(u64 id, Attr *out);

  // ---- fills (a client that missed) ----

  /**
   * Insert one name the module reported, unless the table already holds it.
   * @param parent_path its directory (must resolve)
   * @param name its name
   * @param a its attributes (a.id, a.type required)
   * @param target symlink target, if any
   * @return 0, or an errno
   */
  int FillEntry(const std::string &parent_path, const std::string &name,
                const Attr &a, const std::string &target = std::string());

  /**
   * Record that a name does not exist at the module (a negative fill), so
   * a create under an incomplete directory can proceed.
   * @param parent_path its directory (must resolve)
   * @param name the absent name
   * @return 0, or an errno
   */
  int FillAbsent(const std::string &parent_path, const std::string &name);

  /**
   * Insert one name of a listing the module reported (id and type only; its
   * inode's attributes are filled on first stat), unless the table holds it.
   * @param parent_path its directory (must resolve)
   * @param name the name
   * @param id the inode it names
   * @param type its kFsType*
   * @return 0, or an errno
   */
  int FillName(const std::string &parent_path, const std::string &name,
               u64 id, u32 type);

  /** Record a directory's listing as complete (after filling it). */
  int MarkComplete(const std::string &dir_path);

  // ---- extended attributes ----

  /**
   * setxattr(2).
   * @param path the file
   * @param name attribute name
   * @param value attribute value
   * @param flags XATTR_CREATE (1) / XATTR_REPLACE (2)
   * @param now wall ns (the file's new ctime)
   * @return 0, EEXIST, ENODATA, ENOENT, kMiss (fill the file's xattrs), or
   *         kTooBig (the value is held by the module: use its task)
   */
  int Setxattr(const std::string &path, const std::string &name,
               const std::string &value, u32 flags, u64 now);

  /**
   * getxattr(2).
   * @param value receives the value
   * @return 0, ENODATA, ENOENT, or kMiss
   */
  int Getxattr(const std::string &path, const std::string &name,
               std::string *value);

  /**
   * listxattr(2).
   * @param names receives every name, each NUL-terminated
   * @return 0, ENOENT, or kMiss
   */
  int Listxattr(const std::string &path, std::string *names);

  /**
   * removexattr(2).
   * @return 0, ENODATA, ENOENT, or kMiss
   */
  int Removexattr(const std::string &path, const std::string &name, u64 now);

  /**
   * Hold a file's xattrs as the module reported them (a fill), unless they
   * are held already.
   * @param path the file
   * @param attrs its (name, value) pairs
   * @return 0, kTooBig (a value too large: they stay with the module), or
   *         an errno
   */
  int FillXattrs(const std::string &path,
                 const std::vector<std::pair<std::string, std::string>> &attrs);

  /**
   * Forget a file's xattrs (one was set through the module): the next
   * access fills them again.
   */
  int ForgetXattrs(const std::string &path);

  // ---- appends ----

  /**
   * One-node append: take the file's append lock (spins while another
   * append to the file holds it). The caller writes at the inode's size and
   * calls AppendUnlock.
   * @param id the file's id
   * @return 0, or ENOSPC (append table full)
   */
  int AppendLock(u64 id);

  /** Release AppendLock. */
  void AppendUnlock(u64 id);

  /**
   * Merge path, step 1a: reserve `len` bytes in the file's current append
   * generation.
   * @param id the file's id
   * @param len bytes to stage
   * @param gen receives the generation (the blob is app.<node>.<gen>)
   * @param off receives the offset inside that blob
   * @return 0, or ENOSPC (append table full)
   */
  int AppendStage(u64 id, u64 len, u32 *gen, u64 *off);

  /**
   * Merge path, step 1b: the bytes are in the blob. Logs kOpAppend.
   * @param path the file's path
   * @param id the file's id
   * @param gen the generation AppendStage returned
   * @return 0
   */
  int AppendStaged(const std::string &path, u64 id, u32 gen);

  /**
   * @param id a file's id
   * @return true when it has no staged bytes and no append in progress
   */
  bool AppendIdle(u64 id);

  /**
   * Module: seal the file's current generation (new appends go to the
   * next one).
   * @param id the file's id
   * @param gen receives the sealed generation
   * @return false when nothing is staged
   */
  bool AppendSeal(u64 id, u32 *gen);

  /**
   * Module: whether appends are still copying into a sealed generation.
   * @param id the file's id
   * @param gen the sealed generation
   * @param len receives its staged bytes
   * @return true while any are
   */
  bool AppendInflight(u64 id, u32 gen, u64 *len);

  /** Module: the sealed generation is merged; reset it. */
  void AppendMerged(u64 id, u32 gen);

  // ---- module side ----

  /**
   * Hand the clients a new block of ids.
   * @param major the ids' major
   * @param first first minor
   * @param count minors in the block
   */
  void SetIdBlock(u32 major, u64 first, u64 count);

  /** @return ids left in the current block */
  u64 IdsLeft() const;

  /**
   * Next op record to apply, without consuming it.
   * @param rec receives the header
   * @param path receives the path
   * @param path2 receives the second path (rename/link/symlink target)
   * @return false when the log is empty
   */
  bool PeekOp(OpRec *rec, std::string *path, std::string *path2);

  /**
   * Consume the record PeekOp returned (frees its log space). It is not
   * applied yet: MarkApplied once its effect has landed.
   */
  void PopOp();

  /**
   * Every record through `seq` has landed at its home: waiters
   * (WaitApplied: fsync, truncate, xattrs) may proceed, and negative
   * entries they covered may be reclaimed.
   * @param seq the last landed record's seq
   */
  void MarkApplied(u64 seq);

  /** @return true when an inode with this id is in the table */
  bool HasInode(u64 id);

 private:
  // All *Locked helpers require lock_ (writer for mutations).
  char *Ptr(u64 off) { return base_ + off; }
  Inode *Inodes() { return reinterpret_cast<Inode *>(Ptr(root_->inodes_off_)); }
  Dentry *Dentries() {
    return reinterpret_cast<Dentry *>(Ptr(root_->dentries_off_));
  }
  Target *Targets() {
    return reinterpret_cast<Target *>(Ptr(root_->targets_off_));
  }
  char *Log() { return Ptr(root_->log_off_); }

  std::string Canon(const std::string &name) const {
    return FsCanonicalName(name, root_->case_insensitive_ != 0);
  }
  static u64 NameHash(u64 parent, const std::string &canon);

  Inode *FindInodeLocked(u64 id);
  Inode *InsertInodeLocked(u64 id);
  void FreeInodeLocked(Inode *ino);
  Dentry *FindDentLocked(u64 parent, const std::string &name, u64 *hash);
  Dentry *InsertDentLocked(u64 parent, u64 hash, const std::string &name);
  void LinkDentLocked(Inode *dir, Dentry *d);
  void UnlinkDentLocked(Inode *dir, Dentry *d);
  /** Free negative entries whose removal has been applied. */
  void ReapNegLocked();
  u32 SlotOf(const Dentry *d) { return static_cast<u32>(d - Dentries()); }
  XattrSlot *Xattrs() {
    return reinterpret_cast<XattrSlot *>(Ptr(root_->xattrs_off_));
  }
  /** Find (or insert) an xattr slot; lock held. */
  XattrSlot *XattrSlotLocked(u64 id, const std::string &name, bool insert);
  /** Free every xattr slot of an inode; lock held. */
  void FreeXattrsLocked(Inode *ino);
  AppendSlot *Appends() {
    return reinterpret_cast<AppendSlot *>(Ptr(root_->appends_off_));
  }
  /** Find (or insert) a file's append slot; lock held. */
  AppendSlot *AppendSlotLocked(u64 id, bool insert);
  /** Free an idle append slot; lock held. */
  void MaybeFreeAppendLocked(AppendSlot *a);

  /**
   * Resolve a directory path to its inode.
   * @return 0, ENOENT, ENOTDIR, or kMiss
   */
  int ResolveDirLocked(const std::vector<std::string> &comps, size_t n,
                       Inode **dir);

  /**
   * Resolve the last component under its parent.
   * @param parent receives the parent directory
   * @param d receives the entry (null when absent)
   * @return 0 (d may be null or NEG), ENOENT/ENOTDIR (parent), or kMiss
   *         (parent incomplete and name absent)
   */
  int ResolveLeafLocked(const std::string &path, Inode **parent, Dentry **d,
                        u64 *hash, std::string *leaf);

  /**
   * Resolve any path (the root included) to its inode.
   * @return 0, ENOENT, ENOTDIR, or kMiss
   */
  int ResolveAnyLocked(const std::string &path, Inode **ino);

  bool HasLogRoomLocked(size_t plen, size_t p2len);
  u64 PushOpLocked(u32 type, const std::string &path,
                   const std::string &path2, u64 id, u64 size, u64 t1,
                   u64 t2, u32 mode, u32 uid, u32 gid, u32 flags);
  bool MintIdLocked(u64 *id);
  void FillAttr(const Inode &ino, Attr *out);
  int RemoveLocked(const std::string &path, bool dir, u64 now);

  /**
   * Run a mutation under the writer lock, retrying while the log has no
   * room (the module drains it).
   */
  int Mutate(size_t plen, size_t p2len, const std::function<int()> &fn);

  ctp::ipc::PosixShmMmap backend_;
  bool mapped_ = false;
  bool owner_ = false;
  char *base_ = nullptr;
  Root *root_ = nullptr;
};

}  // namespace clio::cte::filesystem::ns

#include <clio_cte/filesystem/fs_ns_shm_impl.h>

#endif  // CLIO_CTE_FILESYSTEM_FS_NS_SHM_H_
