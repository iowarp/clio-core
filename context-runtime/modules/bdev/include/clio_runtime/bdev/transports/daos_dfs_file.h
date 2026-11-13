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

#ifndef CLIO_BDEV_DAOS_DFS_FILE_H_
#define CLIO_BDEV_DAOS_DFS_FILE_H_

/**
 * @file daos_dfs_file.h
 * DAOS client primitives of the kDaos bdev transport: the process-wide
 * library reference and per-worker event queues (DaosProcess), one open DFS
 * file (DfsFile) and asynchronous reads/writes on it (IoRequest, SubmitWrite,
 * SubmitRead, PollEq).
 *
 * Deliberately free of any runtime (clio) dependency: errors come back as
 * codes and strings instead of log lines, so the exact code the transport
 * runs can also be exercised by a small standalone program inside a job.
 * Only daos_bdev_transport.cc includes this header in the build.
 *
 * Error-code conventions of the DAOS client library, as used here:
 *  - libdaos calls (daos_init, pool, container, event queue) return 0 or a
 *    NEGATIVE DER_* code (d_errstr names it);
 *  - libdfs calls (dfs_mount, dfs_open, dfs_read, ...) return 0 or a
 *    POSITIVE errno;
 *  - a completed event carries the operation's result in ev_error, which for
 *    DFS I/O is either of the two depending on where the failure came from.
 */

#include <daos.h>
#include <daos_fs.h>

#include <fcntl.h>
#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

namespace clio::run::bdev::daosio {

/**
 * Describe a libdaos return code.
 * @param rc negative DER_* code
 * @return "DER_NONEXIST(-1005): The specified entity does not exist"
 */
inline std::string DerText(int rc) {
  return std::string(d_errstr(rc)) + "(" + std::to_string(rc) +
         "): " + d_errdesc(rc);
}

/**
 * Describe a libdfs return code.
 * @param err positive errno
 * @return "No such file or directory (errno 2)"
 */
inline std::string ErrnoText(int err) {
  return std::string(strerror(err)) + " (errno " + std::to_string(err) + ")";
}

/**
 * Describe the result code of a completed event (see the file comment).
 * @param code ev_error: negative DER_* code or positive errno
 * @return the matching DerText or ErrnoText
 */
inline std::string CompletionText(int code) {
  return code < 0 ? DerText(code) : ErrnoText(code);
}

/**
 * Convert the result code of a completed event to an errno.
 * @param code ev_error: 0, a negative DER_* code or a positive errno
 * @return 0 on success, else the closest errno (EIO when there is none)
 */
inline int CompletionErrno(int code) {
  if (code >= 0) {
    return code;
  }
  switch (-code) {
    case DER_NOSPACE:  return ENOSPC;
    case DER_NO_PERM:  return EACCES;
    case DER_NONEXIST: return ENOENT;
    case DER_INVAL:    return EINVAL;
    case DER_NOMEM:    return ENOMEM;
    case DER_TIMEDOUT: return ETIMEDOUT;
    case DER_BUSY:     return EBUSY;
    case DER_AGAIN:    return EAGAIN;
    default:           return EIO;
  }
}

/**
 * The process's use of the DAOS client library: one reference-counted
 * daos_init() and one lazily created event queue per runtime worker, shared
 * by every DAOS bdev in the process. Event queues are shared rather than
 * created per bdev because each one owns a network context, which is
 * expensive and limited in number on some fabrics (see daos_eq_create).
 */
class DaosProcess {
 public:
  /**
   * The process-wide instance.
   * @return pointer to the singleton (never null)
   */
  static DaosProcess *GetInstance() {
    static DaosProcess instance;
    return &instance;
  }

  /**
   * Take a reference; the first one initialises the library.
   * @param num_slots number of worker ids that may ask for an event queue
   *        (fixed by the first reference of a generation)
   * @return 0, or the negative DER_* code of daos_init()
   */
  int Acquire(size_t num_slots) {
    std::lock_guard<std::mutex> lock(mu_);
    if (refs_ == 0) {
      const int rc = daos_init();
      if (rc != 0) {
        return rc;
      }
      slots_.reset(new EqSlot[num_slots]);
      num_slots_ = num_slots;
    }
    ++refs_;
    return 0;
  }

  /**
   * Drop a reference; the last one destroys the event queues and finalises
   * the library. No I/O may be in flight on a queue at that point.
   * @param warnings out (appended): what could not be torn down cleanly
   */
  void Release(std::string *warnings) {
    std::lock_guard<std::mutex> lock(mu_);
    if (refs_ == 0 || --refs_ > 0) {
      return;
    }
    for (size_t i = 0; i < num_slots_; ++i) {
      if (!slots_[i].ready_.load(std::memory_order_acquire)) {
        continue;
      }
      const int rc = daos_eq_destroy(slots_[i].eqh_, DAOS_EQ_DESTROY_FORCE);
      if (rc != 0) {
        *warnings += " daos_eq_destroy(worker " + std::to_string(i) +
                     "): " + DerText(rc) + ";";
      }
    }
    slots_.reset();
    num_slots_ = 0;
    const int rc = daos_fini();
    if (rc != 0) {
      *warnings += " daos_fini: " + DerText(rc) + ";";
    }
  }

  /**
   * The event queue of a worker, created on its first use.
   * @param worker_id id of the calling worker
   * @param eqh out: the event-queue handle
   * @return 0, -DER_INVAL for an id outside the reserved range (or when the
   *         library is not initialised), or the code of daos_eq_create()
   */
  int GetWorkerEq(size_t worker_id, daos_handle_t *eqh) {
    if (worker_id >= num_slots_) {
      return -DER_INVAL;
    }
    // Workers share a small number of event queues. Each queue owns a
    // network context, and one per worker (16-32 per runtime) left most of
    // them unable to reach the DAOS servers on Aurora (NA_HOSTUNREACH on
    // ofi+cxi) and the tier stalled. CLIO_DAOS_EQ_COUNT sets how many queues
    // the workers are spread over (default 1). A queue may be polled by any
    // of its workers: a poll that reaps another worker's event marks that
    // request done.
    static const size_t eq_count = [] {
      const char *e = std::getenv("CLIO_DAOS_EQ_COUNT");
      const long v = (e != nullptr && *e != '\0') ? std::atol(e) : 1;
      return static_cast<size_t>(v > 0 ? v : 1);
    }();
    EqSlot &slot = slots_[worker_id % std::min(eq_count, num_slots_)];
    if (!slot.ready_.load(std::memory_order_acquire)) {
      std::lock_guard<std::mutex> lock(mu_);
      if (!slot.ready_.load(std::memory_order_relaxed)) {
        const int rc = daos_eq_create(&slot.eqh_);
        if (rc != 0) {
          return rc;
        }
        slot.ready_.store(true, std::memory_order_release);
      }
    }
    *eqh = slot.eqh_;
    return 0;
  }

 private:
  /** One worker's event queue. */
  struct EqSlot {
    std::atomic<bool> ready_{false}; /**< eqh_ is a created queue. */
    daos_handle_t eqh_{};            /**< The queue handle. */
  };

  DaosProcess() = default;

  std::mutex mu_;                  /**< Guards refs_ and slot creation. */
  size_t refs_ = 0;                /**< References taken by Acquire. */
  std::unique_ptr<EqSlot[]> slots_; /**< Indexed by worker id. */
  size_t num_slots_ = 0;           /**< Length of slots_. */
};

/** What to open: a file of a POSIX container, by labels and path. */
struct DfsFileSpec {
  std::string pool_;               /**< Pool label or UUID string. */
  std::string cont_;               /**< Container label or UUID string. */
  std::vector<std::string> dirs_;  /**< Parent directories, outermost first. */
  std::string file_;               /**< File name. */
  uint64_t chunk_size_ = 0;        /**< DFS chunk size (0 = container default). */
  std::string oclass_;             /**< Object class name (empty = default). */
};

/**
 * One open DFS file together with everything opened to reach it: the library
 * reference, the pool connection, the container handle and the DFS mount.
 * Open either succeeds completely or leaves nothing open.
 */
class DfsFile {
 public:
  DfsFile() = default;
  ~DfsFile() {
    std::string ignored;
    Close(&ignored);
  }
  DfsFile(const DfsFile &) = delete;
  DfsFile &operator=(const DfsFile &) = delete;

  /**
   * Initialise the library, connect to the pool, open the container, mount
   * it and create-or-open the file (creating its parent directories).
   * Blocking: every step is a round trip to the DAOS servers.
   * @param spec what to open
   * @param num_eq_slots worker ids to reserve event queues for
   * @param error out: the failing step with its DAOS code, pool and container
   * @return true when the file is open
   */
  bool Open(const DfsFileSpec &spec, size_t num_eq_slots, std::string *error) {
    where_ = "pool '" + spec.pool_ + "' container '" + spec.cont_ + "'";
    const bool ok = AcquireLibrary(num_eq_slots, error) &&
                    ConnectPool(spec, error) && OpenContainer(spec, error) &&
                    Mount(error) && OpenFile(spec, error);
    if (!ok) {
      std::string ignored;
      Close(&ignored);
    }
    return ok;
  }

  /**
   * Release everything Open acquired, in reverse order (idempotent).
   * @param warnings out (appended): steps that reported an error
   */
  void Close(std::string *warnings) {
    if (obj_ != nullptr) {
      Note(dfs_release(obj_), false, "dfs_release", warnings);
      obj_ = nullptr;
    }
    if (dfs_ != nullptr) {
      Note(dfs_umount(dfs_), false, "dfs_umount", warnings);
      dfs_ = nullptr;
    }
    if (cont_open_) {
      Note(daos_cont_close(coh_, nullptr), true, "daos_cont_close", warnings);
      cont_open_ = false;
    }
    if (pool_open_) {
      Note(daos_pool_disconnect(poh_, nullptr), true, "daos_pool_disconnect",
           warnings);
      pool_open_ = false;
    }
    if (library_ref_) {
      DaosProcess *process = DaosProcess::GetInstance();
      process->Release(warnings);
      library_ref_ = false;
    }
  }

  /**
   * Current size of the file (blocking round trip).
   * @param size out: size in bytes
   * @param error out: failure text
   * @return true on success
   */
  bool GetSize(uint64_t *size, std::string *error) {
    daos_size_t bytes = 0;
    const int rc = dfs_get_size(dfs_, obj_, &bytes);
    if (rc != 0) {
      *error = "dfs_get_size failed in " + where_ + ": " + ErrnoText(rc);
      return false;
    }
    *size = bytes;
    return true;
  }

  /** @return the DFS mount (null unless open). */
  dfs_t *dfs() const { return dfs_; }
  /** @return the open file object (null unless open). */
  dfs_obj_t *obj() const { return obj_; }
  /** @return whether Open succeeded and Close has not run. */
  bool IsOpen() const { return obj_ != nullptr; }

 private:
  /**
   * Append "<what>: <code text>;" to `warnings` when a close step failed.
   * @param rc the step's return code
   * @param is_der true for a libdaos (DER) code, false for a libdfs errno
   * @param what name of the call
   * @param warnings out (appended)
   */
  static void Note(int rc, bool is_der, const char *what,
                   std::string *warnings) {
    if (rc != 0) {
      *warnings += std::string(" ") + what + ": " +
                   (is_der ? DerText(rc) : ErrnoText(rc)) + ";";
    }
  }

  /**
   * Take the process's library reference (daos_init on the first one).
   * @param num_eq_slots worker ids to reserve event queues for
   * @param error out: failure text
   * @return true on success
   */
  bool AcquireLibrary(size_t num_eq_slots, std::string *error) {
    DaosProcess *process = DaosProcess::GetInstance();
    const int rc = process->Acquire(num_eq_slots);
    if (rc != 0) {
      *error = "daos_init failed (" + where_ + "): " + DerText(rc) +
               ". Is the DAOS agent reachable (DAOS_AGENT_DRPC_DIR) and was "
               "the process started where the fabric allows DAOS traffic?";
      return false;
    }
    library_ref_ = true;
    return true;
  }

  /**
   * Connect to the pool read-write, by label.
   * @param spec names the pool
   * @param error out: failure text
   * @return true on success
   */
  bool ConnectPool(const DfsFileSpec &spec, std::string *error) {
    const int rc = daos_pool_connect(spec.pool_.c_str(), nullptr, DAOS_PC_RW,
                                     &poh_, nullptr, nullptr);
    if (rc != 0) {
      *error = "daos_pool_connect failed for " + where_ + ": " + DerText(rc);
      return false;
    }
    pool_open_ = true;
    return true;
  }

  /**
   * Open the container read-write, by label.
   * @param spec names the container
   * @param error out: failure text
   * @return true on success
   */
  bool OpenContainer(const DfsFileSpec &spec, std::string *error) {
    const int rc = daos_cont_open(poh_, spec.cont_.c_str(), DAOS_COO_RW,
                                  &coh_, nullptr, nullptr);
    if (rc != 0) {
      *error = "daos_cont_open failed for " + where_ + ": " + DerText(rc);
      return false;
    }
    cont_open_ = true;
    return true;
  }

  /**
   * Mount the container's DFS namespace read-write. Fails for a container
   * that is not of type POSIX.
   * @param error out: failure text
   * @return true on success
   */
  bool Mount(std::string *error) {
    const int rc = dfs_mount(poh_, coh_, O_RDWR, &dfs_);
    if (rc != 0) {
      dfs_ = nullptr;
      *error = "dfs_mount failed for " + where_ + ": " + ErrnoText(rc) +
               " (the container must be of type POSIX)";
      return false;
    }
    return true;
  }

  /**
   * Create (if missing) and open the directory chain of the file.
   * @param dirs directory names, outermost first
   * @param parent out: the innermost directory, to be released by the
   *        caller; null when `dirs` is empty (the container root)
   * @param error out: failure text
   * @return true on success; on failure nothing stays open
   */
  bool OpenParentDir(const std::vector<std::string> &dirs, dfs_obj_t **parent,
                     std::string *error) {
    const mode_t dir_mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
    dfs_obj_t *cur = nullptr;
    for (const std::string &name : dirs) {
      int rc = dfs_mkdir(dfs_, cur, name.c_str(), dir_mode, 0);
      const char *step = "dfs_mkdir";
      dfs_obj_t *next = nullptr;
      if (rc == 0 || rc == EEXIST) {
        mode_t mode = 0;
        step = "dfs_lookup_rel";
        rc = dfs_lookup_rel(dfs_, cur, name.c_str(), O_RDWR, &next, &mode,
                            nullptr);
        if (rc == 0 && !S_ISDIR(mode)) {
          dfs_release(next);
          rc = ENOTDIR;
        }
      }
      if (cur != nullptr) {
        dfs_release(cur);
      }
      if (rc != 0) {
        *error = std::string(step) + " failed for directory '" + name +
                 "' in " + where_ + ": " + ErrnoText(rc);
        return false;
      }
      cur = next;
    }
    *parent = cur;
    return true;
  }

  /**
   * Create-or-open the file inside its (created) parent directory.
   * @param spec names the directories, the file and its creation attributes
   * @param error out: failure text
   * @return true on success
   */
  bool OpenFile(const DfsFileSpec &spec, std::string *error) {
    daos_oclass_id_t cid = 0;  // 0: the container's default object class
    if (!spec.oclass_.empty()) {
      cid = static_cast<daos_oclass_id_t>(
          daos_oclass_name2id(spec.oclass_.c_str()));
      if (cid == 0) {
        *error = "unknown DAOS object class '" + spec.oclass_ + "' (" +
                 where_ + ")";
        return false;
      }
    }
    dfs_obj_t *parent = nullptr;
    if (!OpenParentDir(spec.dirs_, &parent, error)) {
      return false;
    }
    const int rc = dfs_open(dfs_, parent, spec.file_.c_str(),
                            S_IFREG | S_IRUSR | S_IWUSR | S_IRGRP,
                            O_RDWR | O_CREAT, cid, spec.chunk_size_, nullptr,
                            &obj_);
    if (parent != nullptr) {
      dfs_release(parent);
    }
    if (rc != 0) {
      obj_ = nullptr;
      *error = "dfs_open failed for file '" + spec.file_ + "' in " + where_ +
               ": " + ErrnoText(rc);
      return false;
    }
    return true;
  }

  std::string where_;        /**< "pool 'P' container 'C'" for messages. */
  bool library_ref_ = false; /**< Holds a DaosProcess reference. */
  bool pool_open_ = false;   /**< poh_ is a live connection. */
  bool cont_open_ = false;   /**< coh_ is an open container. */
  daos_handle_t poh_{};      /**< Pool connection. */
  daos_handle_t coh_{};      /**< Container handle. */
  dfs_t *dfs_ = nullptr;     /**< DFS mount. */
  dfs_obj_t *obj_ = nullptr; /**< The open file. */
};

/**
 * One asynchronous read or write. Heap-allocated and owned through
 * IoRequestPtr: while the operation is in flight DAOS holds pointers into
 * this object (the event, the scatter/gather list, read_size_), so it must
 * not be freed before its event has been pulled out of the event queue. The
 * event is the first member so PollEq can map a completed event back to its
 * request.
 */
struct IoRequest {
  daos_event_t ev_;            /**< DAOS event; must stay the first member. */
  d_iov_t iov_;                /**< The single data buffer. */
  d_sg_list_t sgl_;            /**< Scatter/gather list over iov_. */
  daos_size_t read_size_ = 0;  /**< Bytes a read returned (set by DAOS). */
  int submit_rc_ = 0;          /**< errno the submit call returned, if any. */
  int raw_rc_ = 0;             /**< Result as DAOS reported it (for logs). */
  int rc_ = 0;                 /**< Result as an errno; valid once done_. */
  bool inited_ = false;        /**< ev_ went through daos_event_init. */
  bool lost_ = false;          /**< State unknown: never finalise or free. */
  std::atomic<bool> done_{false};      /**< Finished; rc_ is valid. */
  std::atomic<bool> abandoned_{false}; /**< Owner left; PollEq frees it. */

  /** @return whether the operation has finished. */
  bool IsDone() const { return done_.load(std::memory_order_acquire); }

  /**
   * The request a completed event belongs to.
   * @param ev event returned by daos_eq_poll
   * @return its request
   */
  static IoRequest *FromEvent(daos_event_t *ev) {
    return reinterpret_cast<IoRequest *>(ev);
  }
};
static_assert(std::is_standard_layout<IoRequest>::value,
              "IoRequest::FromEvent needs the event at offset 0");
static_assert(offsetof(IoRequest, ev_) == 0,
              "IoRequest::FromEvent needs the event at offset 0");

/**
 * Give up ownership of a request. A finished request is finalised and freed;
 * one still in flight is marked abandoned and freed by PollEq when its event
 * completes (so a coroutine destroyed while waiting never frees memory DAOS
 * still writes to); one whose state is unknown is leaked on purpose.
 * @param req request to release (may be null)
 */
inline void ReleaseRequest(IoRequest *req) {
  if (req == nullptr || req->lost_) {
    return;
  }
  if (!req->inited_) {
    delete req;
    return;
  }
  if (req->IsDone()) {
    daos_event_fini(&req->ev_);
    delete req;
    return;
  }
  req->abandoned_.store(true, std::memory_order_release);
}

/** Deleter that routes through ReleaseRequest. */
struct IoRequestDeleter {
  /** @param req request to release */
  void operator()(IoRequest *req) const { ReleaseRequest(req); }
};
/** Owning pointer to an IoRequest. */
using IoRequestPtr = std::unique_ptr<IoRequest, IoRequestDeleter>;

/**
 * Allocate a request.
 * @return an empty request
 */
inline IoRequestPtr NewRequest() { return IoRequestPtr(new IoRequest()); }

/**
 * Mark a request finished.
 * @param req the request
 * @param raw_rc result as DAOS reported it (DER code or errno; 0 = success)
 */
inline void CompleteRequest(IoRequest *req, int raw_rc) {
  req->raw_rc_ = raw_rc;
  req->rc_ = CompletionErrno(raw_rc);
  req->done_.store(true, std::memory_order_release);
}

/**
 * Reap completed events of an event queue without blocking, and drive the
 * queue's network progress (nothing else does: an event queue makes progress
 * only while it is being polled). Marks the request of every reaped event
 * done; frees the ones their owner abandoned.
 * @param eqh the worker's event queue
 * @return number of events reaped (>= 0), or a negative DER_* code
 */
inline int PollEq(daos_handle_t eqh) {
  constexpr unsigned int kBatch = 64;
  daos_event_t *events[kBatch];
  const int n = daos_eq_poll(eqh, 0, DAOS_EQ_NOWAIT, kBatch, events);
  for (int i = 0; i < n; ++i) {
    IoRequest *req = IoRequest::FromEvent(events[i]);
    if (req->abandoned_.load(std::memory_order_acquire)) {
      daos_event_fini(&req->ev_);
      delete req;
      continue;
    }
    const int ev_rc = events[i]->ev_error;
    CompleteRequest(req, ev_rc != 0 ? ev_rc : req->submit_rc_);
  }
  return n;
}

/**
 * Fail a request whose event queue can no longer be polled. The operation
 * may still be in flight, so the request is never finalised or freed.
 * @param req the request
 * @param poll_rc the negative DER_* code PollEq returned
 */
inline void FailLostRequest(IoRequest *req, int poll_rc) {
  req->lost_ = true;
  req->raw_rc_ = poll_rc;
  req->rc_ = EIO;
  req->done_.store(true, std::memory_order_release);
}

/**
 * Prepare a request's event and buffer description for one operation.
 * @param eqh event queue the completion is delivered to
 * @param req the request
 * @param buf data buffer; must stay valid until the request is done
 * @param len buffer length in bytes
 * @return true when the request can be submitted; false when it has been
 *         completed with the daos_event_init error
 */
inline bool PrepareRequest(daos_handle_t eqh, IoRequest *req, void *buf,
                           size_t len) {
  const int rc = daos_event_init(&req->ev_, eqh, nullptr);
  if (rc != 0) {
    CompleteRequest(req, rc);
    return false;
  }
  req->inited_ = true;
  d_iov_set(&req->iov_, buf, len);
  req->sgl_.sg_nr = 1;
  req->sgl_.sg_nr_out = 0;
  req->sgl_.sg_iovs = &req->iov_;
  return true;
}

/**
 * Settle a request after its submit call returned. A zero return means the
 * operation was launched and completes through the event queue. A non-zero
 * return normally means it was never launched; daos_event_test tells the
 * cases apart (it reports "finished" for an event that is not in flight and,
 * for one that already completed, pulls it out of the queue).
 * @param req the request
 * @param rc errno the dfs_read/dfs_write call returned
 */
inline void SettleSubmit(IoRequest *req, int rc) {
  if (rc == 0) {
    return;
  }
  bool finished = false;
  const int test_rc = daos_event_test(&req->ev_, DAOS_EQ_NOWAIT, &finished);
  if (test_rc != 0) {
    // The event's state cannot be determined: report the failure, keep the
    // memory (the operation may still complete into it).
    req->lost_ = true;
    CompleteRequest(req, rc);
  } else if (finished) {
    CompleteRequest(req, rc);
  } else {
    // Launched after all: wait for the event, but keep the error.
    req->submit_rc_ = rc;
  }
}

/**
 * Start an asynchronous write. Never blocks. Afterwards the caller polls the
 * event queue (PollEq) until req->IsDone(), then reads req->rc_.
 * @param file the open file
 * @param eqh event queue of the calling worker
 * @param req the request (from NewRequest)
 * @param buf data to write; must stay valid until the request is done
 * @param len number of bytes (> 0)
 * @param off file offset
 */
inline void SubmitWrite(const DfsFile &file, daos_handle_t eqh,
                        IoRequest *req, const void *buf, size_t len,
                        uint64_t off) {
  if (!PrepareRequest(eqh, req, const_cast<void *>(buf), len)) {
    return;
  }
  SettleSubmit(req,
               dfs_write(file.dfs(), file.obj(), &req->sgl_, off, &req->ev_));
}

/**
 * Start an asynchronous read. Never blocks. Afterwards the caller polls the
 * event queue (PollEq) until req->IsDone(), then reads req->rc_ and
 * req->read_size_ (which is smaller than `len` when the read ran past the
 * end of the file).
 * @param file the open file
 * @param eqh event queue of the calling worker
 * @param req the request (from NewRequest)
 * @param buf destination; must stay valid until the request is done
 * @param len number of bytes (> 0)
 * @param off file offset
 */
inline void SubmitRead(const DfsFile &file, daos_handle_t eqh, IoRequest *req,
                       void *buf, size_t len, uint64_t off) {
  if (!PrepareRequest(eqh, req, buf, len)) {
    return;
  }
  req->read_size_ = 0;
  SettleSubmit(req, dfs_read(file.dfs(), file.obj(), &req->sgl_, off,
                             &req->read_size_, &req->ev_));
}

}  // namespace clio::run::bdev::daosio

#endif  // CLIO_BDEV_DAOS_DFS_FILE_H_
