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

// LOW-LEVEL FUSE adapter (issue #1007 follow-up, experimental target
// clio_cte_fuse_ll): the high-level adapter's remaining checkout cost is the
// FUSE transition COUNT itself (measured: 110 us/op wall, 9.6 us daemon CPU
// — 91% kernel transit; a kernel clone issues ~900k ops). The low-level API
// cannot make one transition cheaper, but it ELIMINATES transitions:
//
//   - create/mkdir/symlink replies carry fuse_entry_param — the kernel gets
//     the attrs WITH the entry, killing the follow-up GETATTR the high-level
//     libfuse issues per create (~95k/clone) and the LOOKUP re-probe.
//   - readdirplus returns attrs with names (one op per directory page
//     instead of one LOOKUP+GETATTR per entry for `git status`/`ls -l`).
//
// Every operation resolves ino -> path and DELEGATES to the exported
// high-level implementations in fuse_cte.cc (mirror-first getattr,
// complete-dir negatives, minted creates, sieve-direct I/O, the closer
// thread) — this file adds no filesystem semantics of its own.
//
// Deliberate simplifications (experimental perf target, not the default):
//   - Inos are minted per PATH (sequential); hard-link aliases therefore get
//     distinct inos (the high-level adapter reports the shared tag-derived
//     ino). st_nlink is still correct; only ino-equality across aliases is
//     not. The default adapter remains clio_cte_fuse.
//   - xattr ops are not implemented (kernel latches them off via ENOSYS).

#ifndef FUSE_USE_VERSION
#define FUSE_USE_VERSION 35
#endif

#include <fuse3/fuse.h>            // high-level types the delegates use
#include <fuse3/fuse_lowlevel.h>   // the session API this file drives

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "fuse_cte.h"

namespace {

// ============================================================================
// Ino table: ino <-> current path, with kernel lookup-count refcounting.
// ============================================================================
struct InoEntry {
  std::string path;
  uint64_t nlookup = 0;
};

std::mutex g_ino_mtx;
std::unordered_map<fuse_ino_t, InoEntry> g_ino_to_path;
std::unordered_map<std::string, fuse_ino_t> g_path_to_ino;
fuse_ino_t g_next_ino = 2;  // 1 is the root

double g_ttl = 1.0;  // entry+attr TTL, CLIO_FUSE_ATTR_CACHE_S

// Return the ino for `path`, minting one if new; bumps nlookup by `add`.
fuse_ino_t InternPath(const std::string &path, uint64_t add) {
  if (path == "/") return FUSE_ROOT_ID;
  std::lock_guard<std::mutex> lk(g_ino_mtx);
  auto it = g_path_to_ino.find(path);
  if (it != g_path_to_ino.end()) {
    g_ino_to_path[it->second].nlookup += add;
    return it->second;
  }
  fuse_ino_t ino = g_next_ino++;
  g_path_to_ino.emplace(path, ino);
  g_ino_to_path.emplace(ino, InoEntry{path, add});
  return ino;
}

// Current path of `ino`; empty when unknown (kernel bug or post-forget).
std::string PathOf(fuse_ino_t ino) {
  if (ino == FUSE_ROOT_ID) return "/";
  std::lock_guard<std::mutex> lk(g_ino_mtx);
  auto it = g_ino_to_path.find(ino);
  return it != g_ino_to_path.end() ? it->second.path : std::string();
}

std::string ChildPath(fuse_ino_t parent, const char *name) {
  std::string p = PathOf(parent);
  if (p.empty()) return p;
  if (p == "/") return "/" + std::string(name);
  return p + "/" + name;
}

void Forget(fuse_ino_t ino, uint64_t n) {
  if (ino == FUSE_ROOT_ID) return;
  std::lock_guard<std::mutex> lk(g_ino_mtx);
  auto it = g_ino_to_path.find(ino);
  if (it == g_ino_to_path.end()) return;
  if (it->second.nlookup <= n) {
    g_path_to_ino.erase(it->second.path);
    g_ino_to_path.erase(it);
  } else {
    it->second.nlookup -= n;
  }
}

// Rename support: the ino keeps identity, its PATH string changes — for a
// renamed DIRECTORY every descendant path shifts too. Rename is rare in the
// workloads this targets, so a full-table prefix rewrite is fine.
void RenamePaths(const std::string &from, const std::string &to) {
  std::lock_guard<std::mutex> lk(g_ino_mtx);
  std::vector<std::pair<fuse_ino_t, std::string>> moves;
  for (auto &kv : g_ino_to_path) {
    const std::string &p = kv.second.path;
    if (p == from) {
      moves.emplace_back(kv.first, to);
    } else if (p.size() > from.size() && p.compare(0, from.size(), from) == 0 &&
               p[from.size()] == '/') {
      moves.emplace_back(kv.first, to + p.substr(from.size()));
    }
  }
  for (auto &m : moves) {
    g_path_to_ino.erase(g_ino_to_path[m.first].path);
    g_ino_to_path[m.first].path = m.second;
    g_path_to_ino[m.second] = m.first;
  }
  // The destination may have had its own (replaced) ino: leave it — its
  // getattr will report ENOENT and the kernel forgets it naturally.
}

// Build a fuse_entry_param for `path` from the delegated stat. `add` is the
// nlookup bump this reply hands the kernel (1 for lookup/create/direntplus).
int FillEntry(const std::string &path, struct fuse_file_info *fi,
              uint64_t add, struct fuse_entry_param *e) {
  std::memset(e, 0, sizeof(*e));
  struct stat st;
  int rc = cte_fuse_getattr_stat(path.c_str(), &st, fi);
  if (rc != 0) return rc;
  fuse_ino_t ino = InternPath(path, add);
  st.st_ino = static_cast<ino_t>(ino);
  e->ino = ino;
  e->attr = st;
  e->attr_timeout = g_ttl;
  e->entry_timeout = g_ttl;
  return 0;
}

// ============================================================================
// Operations
// ============================================================================

void ll_init(void *userdata, struct fuse_conn_info *conn) {
  (void)userdata;
  if (const char *ttl_env = getenv("CLIO_FUSE_ATTR_CACHE_S")) {
    if (*ttl_env != '\0') g_ttl = atof(ttl_env);
  }
  if (conn->capable & FUSE_CAP_ASYNC_READ) conn->want |= FUSE_CAP_ASYNC_READ;
#ifdef FUSE_CAP_HANDLE_KILLPRIV_V2
  if (conn->capable & FUSE_CAP_HANDLE_KILLPRIV_V2) {
    conn->want |= FUSE_CAP_HANDLE_KILLPRIV_V2;
  }
#endif
#ifdef FUSE_CAP_PARALLEL_DIROPS
  if (conn->capable & FUSE_CAP_PARALLEL_DIROPS) {
    conn->want |= FUSE_CAP_PARALLEL_DIROPS;
  }
#endif
  if (conn->max_write < (1u << 20)) conn->max_write = 1u << 20;
  if (cte_fuse_bootstrap_clients() != 0) {
    fprintf(stderr, "ERROR: client bootstrap failed; aborting mount\n");
    abort();
  }
}

void ll_lookup(fuse_req_t req, fuse_ino_t parent, const char *name) {
  std::string path = ChildPath(parent, name);
  if (path.empty()) return (void)fuse_reply_err(req, ENOENT);
  struct fuse_entry_param e;
  int rc = FillEntry(path, nullptr, 1, &e);
  if (rc == -ENOENT) {
    // Negative entry WITH a TTL: e.ino == 0 caches the absence in the
    // kernel (the reply_err path would re-issue the lookup every time).
    std::memset(&e, 0, sizeof(e));
    e.entry_timeout = g_ttl;
    fuse_reply_entry(req, &e);
    return;
  }
  if (rc != 0) return (void)fuse_reply_err(req, -rc);
  fuse_reply_entry(req, &e);
}

void ll_forget(fuse_req_t req, fuse_ino_t ino, uint64_t nlookup) {
  Forget(ino, nlookup);
  fuse_reply_none(req);
}

void ll_forget_multi(fuse_req_t req, size_t count,
                     struct fuse_forget_data *forgets) {
  for (size_t i = 0; i < count; ++i) Forget(forgets[i].ino, forgets[i].nlookup);
  fuse_reply_none(req);
}

void ll_getattr(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi) {
  std::string path = PathOf(ino);
  if (path.empty()) return (void)fuse_reply_err(req, ENOENT);
  struct stat st;
  int rc = cte_fuse_getattr_stat(path.c_str(), &st, fi);
  if (rc != 0) return (void)fuse_reply_err(req, -rc);
  st.st_ino = static_cast<ino_t>(ino);
  fuse_reply_attr(req, &st, g_ttl);
}

void ll_setattr(fuse_req_t req, fuse_ino_t ino, struct stat *attr, int to_set,
                struct fuse_file_info *fi) {
  std::string path = PathOf(ino);
  if (path.empty()) return (void)fuse_reply_err(req, ENOENT);
  int rc = 0;
  if (rc == 0 && (to_set & FUSE_SET_ATTR_MODE)) {
    rc = cte_fuse_chmod(path.c_str(), attr->st_mode, fi);
  }
  if (rc == 0 && (to_set & (FUSE_SET_ATTR_UID | FUSE_SET_ATTR_GID))) {
    uid_t u = (to_set & FUSE_SET_ATTR_UID) ? attr->st_uid : (uid_t)-1;
    gid_t g = (to_set & FUSE_SET_ATTR_GID) ? attr->st_gid : (gid_t)-1;
    rc = cte_fuse_chown(path.c_str(), u, g, fi);
  }
  if (rc == 0 && (to_set & FUSE_SET_ATTR_SIZE)) {
    rc = cte_fuse_truncate(path.c_str(), attr->st_size, fi);
  }
  if (rc == 0 &&
      (to_set & (FUSE_SET_ATTR_ATIME | FUSE_SET_ATTR_MTIME |
                 FUSE_SET_ATTR_ATIME_NOW | FUSE_SET_ATTR_MTIME_NOW))) {
    struct timespec tv[2];
    tv[0].tv_nsec = UTIME_OMIT;
    tv[1].tv_nsec = UTIME_OMIT;
    if (to_set & FUSE_SET_ATTR_ATIME_NOW) {
      tv[0].tv_nsec = UTIME_NOW;
    } else if (to_set & FUSE_SET_ATTR_ATIME) {
      tv[0] = attr->st_atim;
    }
    if (to_set & FUSE_SET_ATTR_MTIME_NOW) {
      tv[1].tv_nsec = UTIME_NOW;
    } else if (to_set & FUSE_SET_ATTR_MTIME) {
      tv[1] = attr->st_mtim;
    }
    rc = cte_fuse_utimens(path.c_str(), tv, fi);
  }
  if (rc != 0) return (void)fuse_reply_err(req, -rc);
  struct stat st;
  rc = cte_fuse_getattr_stat(path.c_str(), &st, fi);
  if (rc != 0) return (void)fuse_reply_err(req, -rc);
  st.st_ino = static_cast<ino_t>(ino);
  fuse_reply_attr(req, &st, g_ttl);
}

void ll_mkdir(fuse_req_t req, fuse_ino_t parent, const char *name,
              mode_t mode) {
  std::string path = ChildPath(parent, name);
  if (path.empty()) return (void)fuse_reply_err(req, ENOENT);
  int rc = cte_fuse_mkdir(path.c_str(), mode);
  if (rc != 0) return (void)fuse_reply_err(req, -rc);
  struct fuse_entry_param e;
  rc = FillEntry(path, nullptr, 1, &e);
  if (rc != 0) return (void)fuse_reply_err(req, -rc);
  fuse_reply_entry(req, &e);
}

void ll_create(fuse_req_t req, fuse_ino_t parent, const char *name,
               mode_t mode, struct fuse_file_info *fi) {
  std::string path = ChildPath(parent, name);
  if (path.empty()) return (void)fuse_reply_err(req, ENOENT);
  int rc = cte_fuse_create(path.c_str(), mode, fi);
  if (rc != 0) return (void)fuse_reply_err(req, -rc);
  struct fuse_entry_param e;
  rc = FillEntry(path, fi, 1, &e);
  if (rc != 0) {
    cte_fuse_release(path.c_str(), fi);
    return (void)fuse_reply_err(req, -rc);
  }
  fuse_reply_create(req, &e, fi);
}

void ll_open(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi) {
  std::string path = PathOf(ino);
  if (path.empty()) return (void)fuse_reply_err(req, ENOENT);
  int rc = cte_fuse_open(path.c_str(), fi);
  if (rc != 0) return (void)fuse_reply_err(req, -rc);
  fuse_reply_open(req, fi);
}

void ll_read(fuse_req_t req, fuse_ino_t ino, size_t size, off_t off,
             struct fuse_file_info *fi) {
  (void)ino;
  std::unique_ptr<char[]> buf(new char[size]);
  int got = cte_fuse_read(nullptr, buf.get(), size, off, fi);
  if (got < 0) return (void)fuse_reply_err(req, -got);
  fuse_reply_buf(req, buf.get(), static_cast<size_t>(got));
}

void ll_write(fuse_req_t req, fuse_ino_t ino, const char *buf, size_t size,
              off_t off, struct fuse_file_info *fi) {
  (void)ino;
  int n = cte_fuse_write(nullptr, buf, size, off, fi);
  if (n < 0) return (void)fuse_reply_err(req, -n);
  fuse_reply_write(req, static_cast<size_t>(n));
}

void ll_flush(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi) {
  (void)ino;
  int rc = cte_fuse_flush(nullptr, fi);
  fuse_reply_err(req, rc == 0 ? 0 : -rc);
}

void ll_release(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi) {
  std::string path = PathOf(ino);
  int rc = cte_fuse_release(path.empty() ? nullptr : path.c_str(), fi);
  fuse_reply_err(req, rc == 0 ? 0 : -rc);
}

void ll_fsync(fuse_req_t req, fuse_ino_t ino, int datasync,
              struct fuse_file_info *fi) {
  std::string path = PathOf(ino);
  if (path.empty()) return (void)fuse_reply_err(req, ENOENT);
  int rc = cte_fuse_fsync(path.c_str(), datasync, fi);
  fuse_reply_err(req, rc == 0 ? 0 : -rc);
}

void ll_unlink(fuse_req_t req, fuse_ino_t parent, const char *name) {
  std::string path = ChildPath(parent, name);
  if (path.empty()) return (void)fuse_reply_err(req, ENOENT);
  int rc = cte_fuse_unlink(path.c_str());
  fuse_reply_err(req, rc == 0 ? 0 : -rc);
}

void ll_rmdir(fuse_req_t req, fuse_ino_t parent, const char *name) {
  std::string path = ChildPath(parent, name);
  if (path.empty()) return (void)fuse_reply_err(req, ENOENT);
  int rc = cte_fuse_rmdir(path.c_str());
  fuse_reply_err(req, rc == 0 ? 0 : -rc);
}

void ll_rename(fuse_req_t req, fuse_ino_t parent, const char *name,
               fuse_ino_t newparent, const char *newname, unsigned int flags) {
  std::string from = ChildPath(parent, name);
  std::string to = ChildPath(newparent, newname);
  if (from.empty() || to.empty()) return (void)fuse_reply_err(req, ENOENT);
  int rc = cte_fuse_rename(from.c_str(), to.c_str(), flags);
  if (rc == 0) RenamePaths(from, to);
  fuse_reply_err(req, rc == 0 ? 0 : -rc);
}

void ll_symlink(fuse_req_t req, const char *target, fuse_ino_t parent,
                const char *name) {
  std::string path = ChildPath(parent, name);
  if (path.empty()) return (void)fuse_reply_err(req, ENOENT);
  int rc = cte_fuse_symlink(target, path.c_str());
  if (rc != 0) return (void)fuse_reply_err(req, -rc);
  struct fuse_entry_param e;
  rc = FillEntry(path, nullptr, 1, &e);
  if (rc != 0) return (void)fuse_reply_err(req, -rc);
  fuse_reply_entry(req, &e);
}

void ll_readlink(fuse_req_t req, fuse_ino_t ino) {
  std::string path = PathOf(ino);
  if (path.empty()) return (void)fuse_reply_err(req, ENOENT);
  char buf[4096];
  int rc = cte_fuse_readlink(path.c_str(), buf, sizeof(buf));
  if (rc != 0) return (void)fuse_reply_err(req, -rc);
  fuse_reply_readlink(req, buf);
}

void ll_link(fuse_req_t req, fuse_ino_t ino, fuse_ino_t newparent,
             const char *newname) {
  std::string from = PathOf(ino);
  std::string to = ChildPath(newparent, newname);
  if (from.empty() || to.empty()) return (void)fuse_reply_err(req, ENOENT);
  int rc = cte_fuse_link(from.c_str(), to.c_str());
  if (rc != 0) return (void)fuse_reply_err(req, -rc);
  struct fuse_entry_param e;
  rc = FillEntry(to, nullptr, 1, &e);
  if (rc != 0) return (void)fuse_reply_err(req, -rc);
  fuse_reply_entry(req, &e);
}

void ll_statfs(fuse_req_t req, fuse_ino_t ino) {
  (void)ino;
  struct statvfs st;
  int rc = cte_fuse_statfs("/", &st);
  if (rc != 0) return (void)fuse_reply_err(req, -rc);
  fuse_reply_statfs(req, &st);
}

// ---------------------------------------------------------------------------
// Directories: opendir snapshots the listing ONCE via the delegated readdir
// (collector filler), then readdir(plus) pages out of the snapshot.
// ---------------------------------------------------------------------------
struct DirSnapshot {
  std::vector<std::pair<std::string, struct stat>> entries;
  std::string path;
};

int CollectFiller(void *buf, const char *name, const struct stat *stbuf,
                  off_t off, enum fuse_fill_dir_flags flags) {
  (void)off;
  (void)flags;
  auto *snap = static_cast<DirSnapshot *>(buf);
  struct stat st;
  std::memset(&st, 0, sizeof(st));
  if (stbuf != nullptr) st = *stbuf;
  snap->entries.emplace_back(name, st);
  return 0;
}

void ll_opendir(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi) {
  std::string path = PathOf(ino);
  if (path.empty()) return (void)fuse_reply_err(req, ENOENT);
  auto *snap = new DirSnapshot();
  snap->path = path;
  int rc = cte_fuse_readdir(path.c_str(), snap, CollectFiller, 0, nullptr,
                            static_cast<enum fuse_readdir_flags>(0));
  if (rc != 0) {
    delete snap;
    return (void)fuse_reply_err(req, -rc);
  }
  fi->fh = reinterpret_cast<uint64_t>(snap);
  fuse_reply_open(req, fi);
}

void ll_releasedir(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi) {
  (void)ino;
  delete reinterpret_cast<DirSnapshot *>(fi->fh);
  fuse_reply_err(req, 0);
}

void EmitDir(fuse_req_t req, fuse_ino_t ino, size_t size, off_t off,
             struct fuse_file_info *fi, bool plus) {
  auto *snap = reinterpret_cast<DirSnapshot *>(fi->fh);
  if (snap == nullptr) return (void)fuse_reply_err(req, EBADF);
  std::unique_ptr<char[]> buf(new char[size]);
  size_t used = 0;
  for (size_t i = static_cast<size_t>(off); i < snap->entries.size(); ++i) {
    const std::string &name = snap->entries[i].first;
    size_t room = size - used;
    size_t need;
    if (plus) {
      std::string child = snap->path == "/" ? "/" + name
                                            : snap->path + "/" + name;
      struct fuse_entry_param e;
      std::memset(&e, 0, sizeof(e));
      if (name != "." && name != ".." &&
          FillEntry(child, nullptr, 1, &e) != 0) {
        // Entry vanished between snapshot and page-out: report it attrless
        // (ino 0 = kernel does its own lookup if it cares).
        std::memset(&e, 0, sizeof(e));
      }
      need = fuse_add_direntry_plus(req, buf.get() + used, room, name.c_str(),
                                    &e, static_cast<off_t>(i + 1));
      if (need > room && e.ino != 0) Forget(e.ino, 1);  // didn't fit: undo
    } else {
      struct stat st = snap->entries[i].second;
      need = fuse_add_direntry(req, buf.get() + used, room, name.c_str(), &st,
                               static_cast<off_t>(i + 1));
    }
    if (need > room) break;
    used += need;
  }
  fuse_reply_buf(req, buf.get(), used);
}

void ll_readdir(fuse_req_t req, fuse_ino_t ino, size_t size, off_t off,
                struct fuse_file_info *fi) {
  EmitDir(req, ino, size, off, fi, /*plus=*/false);
}

void ll_readdirplus(fuse_req_t req, fuse_ino_t ino, size_t size, off_t off,
                    struct fuse_file_info *fi) {
  EmitDir(req, ino, size, off, fi, /*plus=*/true);
}

const struct fuse_lowlevel_ops kLLOps = [] {
  struct fuse_lowlevel_ops ops = {};
  ops.init = ll_init;
  ops.lookup = ll_lookup;
  ops.forget = ll_forget;
  ops.forget_multi = ll_forget_multi;
  ops.getattr = ll_getattr;
  ops.setattr = ll_setattr;
  ops.mkdir = ll_mkdir;
  ops.create = ll_create;
  ops.open = ll_open;
  ops.read = ll_read;
  ops.write = ll_write;
  ops.flush = ll_flush;
  ops.release = ll_release;
  ops.fsync = ll_fsync;
  ops.unlink = ll_unlink;
  ops.rmdir = ll_rmdir;
  ops.rename = ll_rename;
  ops.symlink = ll_symlink;
  ops.readlink = ll_readlink;
  ops.link = ll_link;
  ops.statfs = ll_statfs;
  ops.opendir = ll_opendir;
  ops.readdir = ll_readdir;
  ops.readdirplus = ll_readdirplus;
  ops.releasedir = ll_releasedir;
  return ops;
}();

}  // namespace

int main(int argc, char *argv[]) {
  struct fuse_args args = FUSE_ARGS_INIT(argc, argv);
  struct fuse_cmdline_opts opts;
  if (fuse_parse_cmdline(&args, &opts) != 0) return 1;
  if (opts.show_help || opts.mountpoint == nullptr) {
    printf("usage: %s [options] <mountpoint>\n", argv[0]);
    fuse_cmdline_help();
    fuse_lowlevel_help();
    free(opts.mountpoint);
    fuse_opt_free_args(&args);
    return opts.show_help ? 0 : 1;
  }
  struct fuse_session *se =
      fuse_session_new(&args, &kLLOps, sizeof(kLLOps), nullptr);
  if (se == nullptr) return 1;
  if (fuse_set_signal_handlers(se) != 0) return 1;
  if (fuse_session_mount(se, opts.mountpoint) != 0) return 1;
  fuse_daemonize(opts.foreground);
  int rc;
  if (opts.singlethread) {
    rc = fuse_session_loop(se);
  } else {
    struct fuse_loop_config config;
    std::memset(&config, 0, sizeof(config));
    config.clone_fd = opts.clone_fd;
    config.max_idle_threads = 10;
    rc = fuse_session_loop_mt(se, &config);
  }
  fuse_session_unmount(se);
  fuse_remove_signal_handlers(se);
  fuse_session_destroy(se);
  free(opts.mountpoint);
  fuse_opt_free_args(&args);
  return rc != 0 ? 1 : 0;
}
