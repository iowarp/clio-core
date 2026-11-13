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
 * library reference and its pool of dedicated I/O threads (DaosProcess,
 * IoThread), one open DFS file (DfsFile) and asynchronous reads/writes on it
 * (IoRequest, SubmitWrite, SubmitRead).
 *
 * Threading: all DAOS I/O is issued and reaped by the I/O threads. Each one
 * creates its own event queue when it starts and is the only thread that
 * ever uses it. Any other thread (a runtime worker) only describes a request,
 * hands it over (SubmitWrite / SubmitRead: a mutex-protected queue plus a
 * condition-variable wake-up) and watches the request's atomic done flag.
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
#include <pthread.h>
#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
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

/** Clock the layer stamps request stages with (cycles, ns, ...; 0 = off). */
using StampFn = unsigned long long (*)();

/**
 * The slot holding the stage clock.
 * @return the process-wide slot (null until SetStampFn)
 */
inline std::atomic<StampFn> &StampFnSlot() {
  static std::atomic<StampFn> slot{nullptr};
  return slot;
}

/**
 * Install the clock request stages are stamped with (IoRequest::t_enq_,
 * t_issue_, t_reap_). Without one every stamp is 0.
 * @param fn clock function, or null to switch stamping off
 */
inline void SetStampFn(StampFn fn) {
  StampFnSlot().store(fn, std::memory_order_release);
}

/**
 * Read the stage clock.
 * @return the clock value, or 0 when no clock is installed
 */
inline unsigned long long Stamp() {
  const StampFn fn = StampFnSlot().load(std::memory_order_acquire);
  return fn != nullptr ? fn() : 0;
}

/**
 * Counts the requests of one open file that DAOS may still be working on
 * (queued on or issued by an I/O thread). DfsFile::Close waits for it to
 * reach zero before it releases the file's handles. Heap-allocated so a file
 * that could not be drained can leave it behind for the I/O threads.
 */
struct IoGate {
  std::atomic<size_t> pending_{0}; /**< Requests not yet finished. */
};

/**
 * One asynchronous read or write. Reference-counted: the submitter holds one
 * reference (IoRequestPtr) and, from the moment the request is queued, the
 * I/O thread holds another; whichever drops last frees the request. While
 * the operation is in flight DAOS holds pointers into this object (the event,
 * the scatter/gather list, read_size_), which the I/O thread's reference
 * keeps alive even when the submitter leaves early. The event is the first
 * member so a completed event maps back to its request.
 *
 * After a successful submit only the I/O thread touches the request until
 * done_ is set; the submitter may only call IsDone().
 */
struct IoRequest {
  daos_event_t ev_;            /**< DAOS event; must stay the first member. */
  d_iov_t iov_;                /**< The single data buffer. */
  d_sg_list_t sgl_;            /**< Scatter/gather list over iov_. */
  daos_size_t read_size_ = 0;  /**< Bytes a read returned (set by DAOS). */
  dfs_t *dfs_ = nullptr;       /**< Mount the operation runs on. */
  dfs_obj_t *obj_ = nullptr;   /**< File the operation runs on. */
  IoGate *gate_ = nullptr;     /**< The file's pending-request counter. */
  void *buf_ = nullptr;        /**< Data buffer. */
  size_t len_ = 0;             /**< Buffer length in bytes. */
  uint64_t off_ = 0;           /**< File offset. */
  bool is_write_ = false;      /**< dfs_write (true) or dfs_read. */
  int submit_rc_ = 0;          /**< errno the dfs call returned, if any. */
  int raw_rc_ = 0;             /**< Result as DAOS reported it (for logs). */
  int rc_ = 0;                 /**< Result as an errno; valid once done_. */
  bool inited_ = false;        /**< ev_ went through daos_event_init. */
  bool lost_ = false;          /**< State unknown: never finalise or free. */
  size_t slot_ = 0;            /**< Index in the I/O thread's in-flight list. */
  unsigned long long t_enq_ = 0;   /**< Stamp: handed to the layer. */
  unsigned long long t_issue_ = 0; /**< Stamp: dfs call about to be issued. */
  unsigned long long t_reap_ = 0;  /**< Stamp: completion reaped. */
  /** Keeps a buffer the submitter owns alive until the request is freed
   *  (for a submitter that may leave before the operation completes). */
  std::shared_ptr<void> keepalive_;
  std::atomic<int> refs_{1};           /**< Submitter + I/O thread. */
  std::atomic<bool> done_{false};      /**< Finished; rc_ is valid. */

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
 * Drop one reference to a request; the last one frees it. The event has
 * already been finalised by the I/O thread by then (or was never
 * initialised), so this is safe on any thread. A request whose state is
 * unknown (lost_) keeps the I/O thread's reference forever and is therefore
 * never freed.
 * @param req request to release (may be null)
 */
inline void UnrefRequest(IoRequest *req) {
  if (req != nullptr &&
      req->refs_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    delete req;
  }
}

/** Deleter that routes through UnrefRequest. */
struct IoRequestDeleter {
  /** @param req request to release */
  void operator()(IoRequest *req) const { UnrefRequest(req); }
};
/** The submitter's reference to an IoRequest. */
using IoRequestPtr = std::unique_ptr<IoRequest, IoRequestDeleter>;

/**
 * Allocate a request.
 * @return an empty request holding the submitter's reference
 */
inline IoRequestPtr NewRequest() { return IoRequestPtr(new IoRequest()); }

/**
 * Mark a request finished: store the result, then set the done flag.
 * @param req the request
 * @param raw_rc result as DAOS reported it (DER code or errno; 0 = success)
 */
inline void CompleteRequest(IoRequest *req, int raw_rc) {
  req->raw_rc_ = raw_rc;
  req->rc_ = CompletionErrno(raw_rc);
  req->done_.store(true, std::memory_order_release);
}

/**
 * One dedicated DAOS I/O thread. It creates its own event queue when it
 * starts and is the only thread that ever initialises events on, issues I/O
 * on, polls or destroys that queue. Other threads hand it requests through a
 * mutex-protected queue (Enqueue) and learn the outcome from the request's
 * done flag.
 *
 * Loop: take queued requests while fewer than kMaxInFlight are in flight and
 * issue them; poll the event queue (for at most poll_us microseconds) while
 * anything is in flight; sleep on a condition variable when nothing is in
 * flight and nothing is queued. An Enqueue wakes a sleeping thread at once.
 */
class IoThread {
 public:
  /** Most operations one thread keeps in flight; the rest stay queued. */
  static constexpr size_t kMaxInFlight = 64;
  /** Most requests (queued + in flight) one thread accepts; beyond it
   *  Enqueue reports kFull and the submitter retries later. */
  static constexpr size_t kMaxOutstanding = 256;

  /** Outcome of Enqueue. */
  enum class Enq {
    kQueued,  /**< Accepted; the thread now owns a reference. */
    kFull,    /**< At kMaxOutstanding; try again later. */
    kStopped  /**< The thread is shutting down. */
  };

  /**
   * @param index position of the thread in the pool (for names/messages)
   * @param poll_us event-queue poll timeout while I/O is in flight, in
   *        microseconds (0 = non-blocking polls, i.e. spin while in flight)
   */
  IoThread(size_t index, int poll_us) : index_(index), poll_us_(poll_us) {}

  /**
   * Start the thread and wait until it has created its event queue.
   * @param self the thread object (the running thread shares ownership, so
   *        it stays valid even if the thread has to be detached later)
   * @return 0, or the negative DER_* code of daos_eq_create (the thread has
   *         then already exited and been joined)
   */
  static int Start(const std::shared_ptr<IoThread> &self) {
    self->thread_ = std::thread([self] { self->Run(); });
    std::unique_lock<std::mutex> lock(self->mu_);
    self->state_cv_.wait(lock, [&self] { return self->started_; });
    const int rc = self->start_rc_;
    lock.unlock();
    if (rc != 0) {
      self->thread_.join();
    }
    return rc;
  }

  /**
   * Hand a request to the thread. Never blocks beyond a short mutex hold.
   * @param req the request, fully described (see SubmitWrite / SubmitRead)
   * @return kQueued when the thread took it (and a reference to it)
   */
  Enq Enqueue(IoRequest *req) {
    if (outstanding_.fetch_add(1, std::memory_order_acq_rel) >=
        kMaxOutstanding) {
      outstanding_.fetch_sub(1, std::memory_order_acq_rel);
      return Enq::kFull;
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (stop_) {
        outstanding_.fetch_sub(1, std::memory_order_acq_rel);
        return Enq::kStopped;
      }
      req->refs_.fetch_add(1, std::memory_order_relaxed);
      queue_.push_back(req);
      queued_.fetch_add(1, std::memory_order_release);
    }
    work_cv_.notify_one();
    return Enq::kQueued;
  }

  /**
   * Ask the thread to finish: requests still queued are completed with
   * -DER_CANCELED, operations in flight are reaped, then the thread destroys
   * its event queue and exits.
   */
  void RequestStop() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
      stop_flag_.store(true, std::memory_order_release);
    }
    work_cv_.notify_one();
  }

  /**
   * Wait for the thread to exit after RequestStop, then join it; a thread
   * still busy at the deadline is detached instead (it keeps this object
   * alive and finishes on its own).
   * @param deadline latest time to wait until
   * @param warnings out (appended): a detach or an event-queue destroy error
   * @return true when the thread exited and was joined
   */
  bool JoinUntil(std::chrono::steady_clock::time_point deadline,
                 std::string *warnings) {
    if (!thread_.joinable()) {
      return true;
    }
    std::unique_lock<std::mutex> lock(mu_);
    const bool exited =
        state_cv_.wait_until(lock, deadline, [this] { return exited_; });
    lock.unlock();
    if (!exited) {
      thread_.detach();
      *warnings += " DAOS I/O thread " + std::to_string(index_) +
                   " still has " + std::to_string(InFlight()) +
                   " operation(s) in flight; detached;";
      return false;
    }
    thread_.join();
    if (destroy_rc_ != 0) {
      *warnings += " daos_eq_destroy(I/O thread " + std::to_string(index_) +
                   "): " + DerText(destroy_rc_) + ";";
    }
    return true;
  }

  /** @return requests queued or in flight on this thread. */
  size_t Outstanding() const {
    return outstanding_.load(std::memory_order_relaxed);
  }
  /** @return operations issued and not yet reaped. */
  size_t InFlight() const {
    return inflight_n_.load(std::memory_order_relaxed);
  }
  /** @return requests accepted and not yet issued. */
  size_t Queued() const { return queued_.load(std::memory_order_relaxed); }

 private:
  /** Thread body: create the queue, serve requests, destroy the queue. */
  void Run() {
#if defined(__linux__)
    const std::string name = "clio-daos-io" + std::to_string(index_);
    pthread_setname_np(pthread_self(), name.substr(0, 15).c_str());
#endif
    const int rc = daos_eq_create(&eqh_);
    {
      std::lock_guard<std::mutex> lock(mu_);
      start_rc_ = rc;
      started_ = true;
    }
    state_cv_.notify_all();
    if (rc == 0) {
      Loop();
      destroy_rc_ = daos_eq_destroy(eqh_, DAOS_EQ_DESTROY_FORCE);
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      exited_ = true;
    }
    state_cv_.notify_all();
  }

  /** Serve requests until asked to stop and nothing is in flight. */
  void Loop() {
    std::vector<IoRequest *> batch;
    batch.reserve(kMaxInFlight);
    for (;;) {
      bool stopping = stop_flag_.load(std::memory_order_acquire);
      // The lock is taken only when there may be something to take (or to
      // sleep); a busy thread with a full window just keeps polling.
      if (inflight_.empty() || stopping ||
          (queued_.load(std::memory_order_acquire) > 0 &&
           inflight_.size() < kMaxInFlight)) {
        stopping = TakeWork(&batch);
      }
      for (IoRequest *req : batch) {
        if (stopping) {
          Finish(req, -DER_CANCELED);
        } else {
          Issue(req);
        }
      }
      batch.clear();
      if (!inflight_.empty()) {
        Reap();
      } else if (stopping) {
        return;
      }
    }
  }

  /**
   * Move queued requests into `batch`: as many as the in-flight window has
   * room for, or all of them when stopping (to be cancelled). Sleeps while
   * nothing is in flight and nothing is queued.
   * @param batch out: the requests taken
   * @return true when the thread has been asked to stop
   */
  bool TakeWork(std::vector<IoRequest *> *batch) {
    std::unique_lock<std::mutex> lock(mu_);
    if (inflight_.empty()) {
      work_cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
    }
    size_t room = stop_ ? queue_.size() : kMaxInFlight - inflight_.size();
    while (room > 0 && !queue_.empty()) {
      batch->push_back(queue_.front());
      queue_.pop_front();
      --room;
    }
    queued_.fetch_sub(batch->size(), std::memory_order_release);
    return stop_;
  }

  /**
   * Issue one request's dfs_write / dfs_read with an event on this thread's
   * queue. A request that could not be launched is finished here.
   * @param req the request
   */
  void Issue(IoRequest *req) {
    req->t_issue_ = Stamp();
    int rc = daos_event_init(&req->ev_, eqh_, nullptr);
    if (rc != 0) {
      Finish(req, rc);
      return;
    }
    req->inited_ = true;
    d_iov_set(&req->iov_, req->buf_, req->len_);
    req->sgl_.sg_nr = 1;
    req->sgl_.sg_nr_out = 0;
    req->sgl_.sg_iovs = &req->iov_;
    if (req->is_write_) {
      rc = dfs_write(req->dfs_, req->obj_, &req->sgl_, req->off_, &req->ev_);
    } else {
      req->read_size_ = 0;
      rc = dfs_read(req->dfs_, req->obj_, &req->sgl_, req->off_,
                    &req->read_size_, &req->ev_);
    }
    if (rc != 0 && !LaunchedDespiteError(req, rc)) {
      Finish(req, rc);
      return;
    }
    req->slot_ = inflight_.size();
    inflight_.push_back(req);
    inflight_n_.store(inflight_.size(), std::memory_order_relaxed);
  }

  /**
   * Settle a request whose dfs call returned an error. Normally the
   * operation was never launched; daos_event_test tells the cases apart (it
   * reports "finished" for an event that is not in flight and, for one that
   * already completed, pulls it out of the queue). When the event's state
   * cannot be determined the request is marked lost: the failure is
   * reported, the memory is kept (the operation may still complete into it).
   * @param req the request
   * @param rc errno the dfs call returned
   * @return true when the operation is in flight after all (the error is
   *         kept in submit_rc_ and reported when the event completes)
   */
  bool LaunchedDespiteError(IoRequest *req, int rc) {
    bool finished = false;
    const int test_rc = daos_event_test(&req->ev_, DAOS_EQ_NOWAIT, &finished);
    if (test_rc != 0) {
      req->lost_ = true;
      return false;
    }
    if (finished) {
      return false;
    }
    req->submit_rc_ = rc;
    return true;
  }

  /**
   * Reap completed events (waiting at most poll_us_) and finish their
   * requests. Polling is also what drives the queue's network progress.
   */
  void Reap() {
    constexpr unsigned int kBatch = 64;
    daos_event_t *events[kBatch];
    const int n = daos_eq_poll(eqh_, 0, poll_us_, kBatch, events);
    if (n < 0) {
      FailInFlight(n);
      return;
    }
    for (int i = 0; i < n; ++i) {
      IoRequest *req = IoRequest::FromEvent(events[i]);
      // Swap-remove from the in-flight list.
      IoRequest *last = inflight_.back();
      inflight_[req->slot_] = last;
      last->slot_ = req->slot_;
      inflight_.pop_back();
      const int ev_rc = events[i]->ev_error;
      Finish(req, ev_rc != 0 ? ev_rc : req->submit_rc_);
    }
    inflight_n_.store(inflight_.size(), std::memory_order_relaxed);
  }

  /**
   * Finish a request that is not (or no longer) in flight: finalise its
   * event on this thread, publish the result, release the thread's hold on
   * the file and drop the thread's reference. A lost request is only
   * reported: its event is not finalised, the file stays pinned and the
   * reference is kept, so nothing DAOS may still use is ever freed.
   * @param req the request (already out of the in-flight list)
   * @param raw_rc result as DAOS reported it
   */
  void Finish(IoRequest *req, int raw_rc) {
    req->t_reap_ = Stamp();
    const bool lost = req->lost_;
    IoGate *gate = req->gate_;
    if (req->inited_ && !lost) {
      daos_event_fini(&req->ev_);
    }
    CompleteRequest(req, raw_rc);
    outstanding_.fetch_sub(1, std::memory_order_acq_rel);
    if (!lost) {
      gate->pending_.fetch_sub(1, std::memory_order_acq_rel);
      UnrefRequest(req);
    }
  }

  /**
   * Fail every in-flight request after the queue could not be polled. The
   * operations may still be running, so the requests are lost: reported as
   * EIO, never finalised or freed, and their files stay pinned.
   * @param poll_rc the negative DER_* code daos_eq_poll returned
   */
  void FailInFlight(int poll_rc) {
    for (IoRequest *req : inflight_) {
      req->t_reap_ = Stamp();
      req->lost_ = true;
      req->raw_rc_ = poll_rc;
      req->rc_ = EIO;
      req->done_.store(true, std::memory_order_release);
      outstanding_.fetch_sub(1, std::memory_order_acq_rel);
    }
    inflight_.clear();
    inflight_n_.store(0, std::memory_order_relaxed);
  }

  const size_t index_;  /**< Position in the pool. */
  const int poll_us_;   /**< Poll timeout while in flight (us; 0 = spin). */
  std::thread thread_;  /**< The thread. */
  daos_handle_t eqh_{}; /**< Its event queue (thread-private). */
  std::vector<IoRequest *> inflight_; /**< Issued, not reaped (thread-private). */

  std::mutex mu_;       /**< Guards queue_, stop_, started_, exited_. */
  std::condition_variable work_cv_;  /**< Wakes the thread: work or stop. */
  std::condition_variable state_cv_; /**< Announces started_ / exited_. */
  std::deque<IoRequest *> queue_;    /**< Accepted, not yet issued. */
  bool stop_ = false;      /**< Asked to finish. */
  bool started_ = false;   /**< start_rc_ is valid. */
  bool exited_ = false;    /**< The thread body has returned. */
  int start_rc_ = 0;       /**< Result of daos_eq_create. */
  int destroy_rc_ = 0;     /**< Result of daos_eq_destroy (after exit). */

  std::atomic<bool> stop_flag_{false};   /**< stop_, readable lock-free. */
  std::atomic<size_t> queued_{0};        /**< queue_.size(). */
  std::atomic<size_t> inflight_n_{0};    /**< inflight_.size(). */
  std::atomic<size_t> outstanding_{0};   /**< Queued + in flight. */
};

/**
 * Number of I/O threads to run.
 * @param requested explicit count (the io_threads URI option), or 0
 * @return `requested`, else CLIO_DAOS_IO_THREADS, else 4; within 1..64
 */
inline size_t ResolveIoThreads(size_t requested) {
  constexpr size_t kDefault = 4;
  constexpr size_t kMax = 64;
  size_t n = requested;
  if (n == 0) {
    const char *e = std::getenv("CLIO_DAOS_IO_THREADS");
    const long v = (e != nullptr && *e != '\0') ? std::atol(e) : 0;
    n = v > 0 ? static_cast<size_t>(v) : kDefault;
  }
  return std::min(n, kMax);
}

/**
 * Event-queue poll timeout of an I/O thread while it has I/O in flight.
 * @return CLIO_DAOS_POLL_US, else 200; within 0..100000 microseconds
 *         (0 = non-blocking polls: the thread spins while I/O is in flight)
 */
inline int ResolvePollUs() {
  constexpr long kDefault = 200;
  constexpr long kMax = 100000;
  const char *e = std::getenv("CLIO_DAOS_POLL_US");
  const long v = (e != nullptr && *e != '\0') ? std::atol(e) : kDefault;
  return static_cast<int>(std::min(std::max(v, 0L), kMax));
}

/**
 * The process's use of the DAOS client library: one reference-counted
 * daos_init() and one pool of I/O threads (IoThread), shared by every DAOS
 * bdev in the process. The threads are shared rather than created per bdev
 * because each one owns an event queue and with it a network context, which
 * is expensive and limited in number on some fabrics (see daos_eq_create).
 */
class DaosProcess {
 public:
  /** How long Release waits for the I/O threads to exit, in milliseconds. */
  static constexpr int kStopWaitMs = 10000;

  /**
   * The process-wide instance.
   * @return pointer to the singleton (never null)
   */
  static DaosProcess *GetInstance() {
    static DaosProcess instance;
    return &instance;
  }

  /**
   * Take a reference; the first one initialises the library and starts the
   * I/O threads, each of which creates its event queue before this returns.
   * @param io_threads explicit thread count or 0 (see ResolveIoThreads);
   *        only the first reference of a generation decides it
   * @param step out: the call that failed ("daos_init", "daos_eq_create ...")
   * @return 0, or the negative DER_* code of the failing step
   */
  int Acquire(size_t io_threads, std::string *step) {
    std::lock_guard<std::mutex> lock(mu_);
    if (refs_ == 0) {
      int rc = daos_init();
      if (rc != 0) {
        *step = "daos_init";
        return rc;
      }
      rc = StartThreads(ResolveIoThreads(io_threads), step);
      if (rc != 0) {
        std::string ignored;
        if (StopThreads(&ignored)) {
          daos_fini();
        }
        return rc;
      }
    }
    ++refs_;
    return 0;
  }

  /**
   * Drop a reference; the last one stops the I/O threads (each finishes what
   * it has in flight and destroys its own event queue) and finalises the
   * library. A thread that does not finish within kStopWaitMs is detached
   * and the library is then left initialised, since a detached thread still
   * uses it.
   * @param warnings out (appended): what could not be torn down cleanly
   */
  void Release(std::string *warnings) {
    std::lock_guard<std::mutex> lock(mu_);
    if (refs_ == 0 || --refs_ > 0) {
      return;
    }
    if (!StopThreads(warnings)) {
      *warnings += " daos_fini skipped (I/O threads still running);";
      return;
    }
    const int rc = daos_fini();
    if (rc != 0) {
      *warnings += " daos_fini: " + DerText(rc) + ";";
    }
  }

  /**
   * Hand a request to the least-loaded I/O thread (ties go round-robin).
   * Only for requests of a file that holds a library reference, so the
   * threads exist for the duration of the call.
   * @param req the request, fully described
   * @return kQueued; kFull when every thread is at its bound (retry later);
   *         kStopped when there is no running thread to take it
   */
  IoThread::Enq Submit(IoRequest *req) {
    const size_t n = threads_.size();
    if (n == 0) {
      return IoThread::Enq::kStopped;
    }
    const size_t start = next_.fetch_add(1, std::memory_order_relaxed) % n;
    size_t best = start;
    size_t best_load = threads_[start]->Outstanding();
    for (size_t k = 1; k < n && best_load > 0; ++k) {
      const size_t i = (start + k) % n;
      const size_t load = threads_[i]->Outstanding();
      if (load < best_load) {
        best = i;
        best_load = load;
      }
    }
    return threads_[best]->Enqueue(req);
  }

  /** @return number of running I/O threads (0 when not initialised). */
  size_t NumIoThreads() {
    std::lock_guard<std::mutex> lock(mu_);
    return threads_.size();
  }

  /**
   * Load of one I/O thread.
   * @param i thread index (< NumIoThreads())
   * @param queued out: requests accepted and not yet issued
   * @param in_flight out: operations issued and not yet reaped
   * @return false for an index out of range
   */
  bool ThreadLoad(size_t i, size_t *queued, size_t *in_flight) {
    std::lock_guard<std::mutex> lock(mu_);
    if (i >= threads_.size()) {
      return false;
    }
    *queued = threads_[i]->Queued();
    *in_flight = threads_[i]->InFlight();
    return true;
  }

 private:
  DaosProcess() = default;

  /**
   * Start `n` I/O threads (mu_ held). Stops at the first that cannot create
   * its event queue; the ones started so far stay in threads_.
   * @param n number of threads
   * @param step out: the failing call
   * @return 0 or the negative DER_* code of daos_eq_create
   */
  int StartThreads(size_t n, std::string *step) {
    const int poll_us = ResolvePollUs();
    for (size_t i = 0; i < n; ++i) {
      auto thread = std::make_shared<IoThread>(i, poll_us);
      const int rc = IoThread::Start(thread);
      if (rc != 0) {
        *step = "daos_eq_create on I/O thread " + std::to_string(i) + " of " +
                std::to_string(n);
        return rc;
      }
      threads_.push_back(std::move(thread));
    }
    return 0;
  }

  /**
   * Stop and join every I/O thread (mu_ held), waiting kStopWaitMs in total.
   * @param warnings out (appended): detached threads, queue-destroy errors
   * @return true when every thread exited (none was detached)
   */
  bool StopThreads(std::string *warnings) {
    for (const auto &thread : threads_) {
      thread->RequestStop();
    }
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(kStopWaitMs);
    bool all_exited = true;
    for (const auto &thread : threads_) {
      all_exited = thread->JoinUntil(deadline, warnings) && all_exited;
    }
    threads_.clear();
    return all_exited;
  }

  std::mutex mu_;      /**< Guards refs_ and thread start/stop. */
  size_t refs_ = 0;    /**< References taken by Acquire. */
  /** The I/O threads; fixed between the first Acquire and the last Release. */
  std::vector<std::shared_ptr<IoThread>> threads_;
  std::atomic<size_t> next_{0}; /**< Round-robin cursor of Submit. */
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
 *
 * Close waits for the file's requests to leave the I/O threads before it
 * releases anything. Submitting must not race with Close.
 */
class DfsFile {
 public:
  /** How long Close waits for the file's requests, in milliseconds. */
  static constexpr int kCloseDrainMs = 30000;

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
   * @param io_threads I/O threads to start if this is the process's first
   *        DAOS file (0 = CLIO_DAOS_IO_THREADS, else the default)
   * @param error out: the failing step with its DAOS code, pool and container
   * @return true when the file is open
   */
  bool Open(const DfsFileSpec &spec, size_t io_threads, std::string *error) {
    where_ = "pool '" + spec.pool_ + "' container '" + spec.cont_ + "'";
    const bool ok = AcquireLibrary(io_threads, error) &&
                    ConnectPool(spec, error) && OpenContainer(spec, error) &&
                    Mount(error) && OpenFile(spec, error);
    if (ok) {
      gate_ = new IoGate();
    } else {
      std::string ignored;
      Close(&ignored);
    }
    return ok;
  }

  /**
   * Release everything Open acquired, in reverse order (idempotent). First
   * waits (at most kCloseDrainMs) until no request of this file is queued or
   * in flight on an I/O thread. If some remain -- DAOS is not answering, or a
   * request's state is unknown -- NOTHING is released: the handles, the
   * library reference and with it the I/O threads are left in place for the
   * operations that may still complete, and the file just stops being open.
   * @param warnings out (appended): steps that reported an error
   */
  void Close(std::string *warnings) {
    if (gate_ != nullptr && !Drain()) {
      *warnings += " " +
                   std::to_string(gate_->pending_.load(
                       std::memory_order_acquire)) +
                   " request(s) still pending after " +
                   std::to_string(kCloseDrainMs) +
                   " ms; DAOS handles and I/O threads left open;";
      gate_ = nullptr;  // left behind for the pending requests
      obj_ = nullptr;
      dfs_ = nullptr;
      cont_open_ = false;
      pool_open_ = false;
      library_ref_ = false;
      return;
    }
    delete gate_;
    gate_ = nullptr;
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
  bool IsOpen() const { return gate_ != nullptr; }
  /** @return the pending-request counter (null unless open). */
  IoGate *gate() const { return gate_; }

 private:
  /**
   * Wait until no request of this file is queued or in flight.
   * @return false if some were still pending after kCloseDrainMs
   */
  bool Drain() {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(kCloseDrainMs);
    while (gate_->pending_.load(std::memory_order_acquire) != 0) {
      if (std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    return true;
  }

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
   * Take the process's library reference (the first one runs daos_init and
   * starts the I/O threads with their event queues).
   * @param io_threads see Open
   * @param error out: failure text
   * @return true on success
   */
  bool AcquireLibrary(size_t io_threads, std::string *error) {
    DaosProcess *process = DaosProcess::GetInstance();
    std::string step;
    const int rc = process->Acquire(io_threads, &step);
    if (rc != 0) {
      *error = step + " failed (" + where_ + "): " + DerText(rc) +
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
  IoGate *gate_ = nullptr;   /**< Pending-request counter; set while open. */
};

/**
 * Describe one operation in a request and hand it to an I/O thread. Never
 * blocks and never calls DAOS. Must not race with file.Close().
 * @param file the open file
 * @param req the request (from NewRequest)
 * @param is_write dfs_write (true) or dfs_read
 * @param buf data buffer; must stay valid until the request is done
 * @param len number of bytes (> 0)
 * @param off file offset
 * @return true when the request was taken (wait for req->IsDone(); a request
 *         that could not be queued at all is already done with an error);
 *         false when every I/O thread is at its bound -- nothing was queued,
 *         call again later with the same arguments
 */
inline bool SubmitIo(const DfsFile &file, IoRequest *req, bool is_write,
                     void *buf, size_t len, uint64_t off) {
  if (req->t_enq_ == 0) {
    req->t_enq_ = Stamp();
  }
  IoGate *gate = file.gate();
  if (gate == nullptr) {
    CompleteRequest(req, -DER_NO_HDL);
    return true;
  }
  req->dfs_ = file.dfs();
  req->obj_ = file.obj();
  req->gate_ = gate;
  req->buf_ = buf;
  req->len_ = len;
  req->off_ = off;
  req->is_write_ = is_write;
  gate->pending_.fetch_add(1, std::memory_order_acq_rel);
  DaosProcess *process = DaosProcess::GetInstance();
  const IoThread::Enq result = process->Submit(req);
  if (result == IoThread::Enq::kQueued) {
    return true;
  }
  gate->pending_.fetch_sub(1, std::memory_order_acq_rel);
  if (result == IoThread::Enq::kFull) {
    return false;
  }
  CompleteRequest(req, -DER_SHUTDOWN);
  return true;
}

/**
 * Start an asynchronous write on an I/O thread. Afterwards the caller waits
 * for req->IsDone() (without calling DAOS), then reads req->rc_.
 * @param file the open file
 * @param req the request (from NewRequest)
 * @param buf data to write; must stay valid until the request is done
 * @param len number of bytes (> 0)
 * @param off file offset
 * @return see SubmitIo (false = no room, retry later)
 */
inline bool SubmitWrite(const DfsFile &file, IoRequest *req, const void *buf,
                        size_t len, uint64_t off) {
  return SubmitIo(file, req, /*is_write=*/true, const_cast<void *>(buf), len,
                  off);
}

/**
 * Start an asynchronous read on an I/O thread. Afterwards the caller waits
 * for req->IsDone() (without calling DAOS), then reads req->rc_ and
 * req->read_size_ (which is smaller than `len` when the read ran past the
 * end of the file).
 * @param file the open file
 * @param req the request (from NewRequest)
 * @param buf destination; must stay valid until the request is done
 * @param len number of bytes (> 0)
 * @param off file offset
 * @return see SubmitIo (false = no room, retry later)
 */
inline bool SubmitRead(const DfsFile &file, IoRequest *req, void *buf,
                       size_t len, uint64_t off) {
  return SubmitIo(file, req, /*is_write=*/false, buf, len, off);
}

}  // namespace clio::run::bdev::daosio

#endif  // CLIO_BDEV_DAOS_DFS_FILE_H_
