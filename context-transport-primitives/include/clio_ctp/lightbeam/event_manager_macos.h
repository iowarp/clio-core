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

#pragma once

#include <fcntl.h>
#include <pthread.h>
#include <sys/event.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <csignal>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "clio_ctp/util/logging.h"

namespace ctp::lbm {

namespace detail {

/** Per-EventManager signal socketpair registry.
 *
 *  Signal(pid, tid) finds the write-end of the socketpair registered by
 *  AddSignalEvent() for that (pid, tid) pair and writes a wakeup byte.
 *  This mirrors the Windows named-event scheme and avoids the Linux-only
 *  tgkill + signalfd mechanism.
 */
struct MacSignalRegistry {
  std::mutex mtx_;
  std::unordered_map<uint64_t, int> map_;  // key=(pid<<32|lo32(tid)), val=write_fd

  static MacSignalRegistry& Get() {
    static MacSignalRegistry r;
    return r;
  }

  static uint64_t Key(int pid, int tid) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(pid)) << 32) |
           static_cast<uint32_t>(tid);
  }

  void Register(int pid, int tid, int wfd) {
    std::lock_guard<std::mutex> lk(mtx_);
    map_[Key(pid, tid)] = wfd;
  }

  void Unregister(int pid, int tid) {
    std::lock_guard<std::mutex> lk(mtx_);
    map_.erase(Key(pid, tid));
  }

  int Find(int pid, int tid) {
    std::lock_guard<std::mutex> lk(mtx_);
    auto it = map_.find(Key(pid, tid));
    return it != map_.end() ? it->second : -1;
  }
};

}  // namespace detail

class EventManager {
 public:
  EventManager()
      : kqueue_fd_(::kqueue()),
        signal_rfd_(-1),
        signal_wfd_(-1),
        next_event_id_(0),
        reg_pid_(-1),
        reg_tid_(-1) {}

  ~EventManager() {
    if (signal_rfd_ >= 0) {
      detail::MacSignalRegistry::Get().Unregister(reg_pid_, reg_tid_);
      ::close(signal_rfd_);
      signal_rfd_ = -1;
    }
    if (signal_wfd_ >= 0) {
      ::close(signal_wfd_);
      signal_wfd_ = -1;
    }
    if (kqueue_fd_ >= 0) {
      ::close(kqueue_fd_);
      kqueue_fd_ = -1;
    }
  }

  EventManager(const EventManager&) = delete;
  EventManager& operator=(const EventManager&) = delete;

  int AddEvent(int fd, uint32_t /*events*/ = 0, EventAction* action = nullptr) {
    struct kevent ev;
    EV_SET(&ev, static_cast<uintptr_t>(fd), EVFILT_READ, EV_ADD | EV_ENABLE,
           0, 0, nullptr);
    if (::kevent(kqueue_fd_, &ev, 1, nullptr, 0, nullptr) == -1) {
      if (errno != EEXIST) {
        HLOG(kError, "EventManager::AddEvent: kevent ADD failed for fd={}: {}",
             fd, strerror(errno));
        return -1;
      }
    }
    auto it = fd_to_reg_.find(fd);
    if (it != fd_to_reg_.end()) {
      it->second.action_ = action;
      return it->second.event_id_;
    }
    int event_id = next_event_id_++;
    fd_to_reg_[fd] = {fd, event_id, action};
    return event_id;
  }

  void RemoveEvent(int fd) {
    auto it = fd_to_reg_.find(fd);
    if (it == fd_to_reg_.end()) return;
    struct kevent ev;
    EV_SET(&ev, static_cast<uintptr_t>(fd), EVFILT_READ, EV_DELETE, 0, 0,
           nullptr);
    if (::kevent(kqueue_fd_, &ev, 1, nullptr, 0, nullptr) == -1 &&
        errno != ENOENT && errno != EBADF) {
      HLOG(kError, "EventManager::RemoveEvent: kevent DEL fd={}: {}", fd,
           strerror(errno));
    }
    fd_to_reg_.erase(it);
  }

  int AddSignalEvent(EventAction* action = nullptr) {
    int sv[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == -1) {
      HLOG(kError, "EventManager::AddSignalEvent: socketpair failed: {}",
           strerror(errno));
      return -1;
    }
    for (int i = 0; i < 2; ++i) {
      int fl = ::fcntl(sv[i], F_GETFL, 0);
      ::fcntl(sv[i], F_SETFL, fl | O_NONBLOCK);
      ::fcntl(sv[i], F_SETFD, FD_CLOEXEC);
    }
    signal_rfd_ = sv[0];
    signal_wfd_ = sv[1];

    // Use the same (pid, tid) derivation as callers of Signal().
    // pthread_threadid_np matches SystemInfo::GetTid() on macOS.
    reg_pid_ = static_cast<int>(::getpid());
    uint64_t mac_tid = 0;
    ::pthread_threadid_np(nullptr, &mac_tid);
    reg_tid_ = static_cast<int>(mac_tid);

    detail::MacSignalRegistry::Get().Register(reg_pid_, reg_tid_, signal_wfd_);

    // Cross-process wakeup (#619 fallback-runtime punting): the socketpair and
    // its registry are per-process, so a worker living in another runtime
    // process cannot be reached via Find()/write(). Also watch
    // EVFILT_SIGNAL(SIGUSR1) so a kill(pid, SIGUSR1) from another runtime wakes
    // this worker's kqueue. macOS only DELIVERS (and thus EVFILT_SIGNAL only
    // MONITORS) signals that are not ignored, so install a no-op handler — NOT
    // SIG_IGN (which would suppress delivery) and NOT SIG_DFL (which would
    // terminate). The actual wake is the kqueue event; the handler does nothing.
    struct sigaction sa;
    sa.sa_handler = [](int) {};
    ::sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    ::sigaction(SIGUSR1, &sa, nullptr);
    struct kevent sigev;
    EV_SET(&sigev, SIGUSR1, EVFILT_SIGNAL, EV_ADD | EV_ENABLE | EV_CLEAR, 0, 0,
           nullptr);
    ::kevent(kqueue_fd_, &sigev, 1, nullptr, 0, nullptr);

    return AddEvent(signal_rfd_, 0, action);
  }

  /** Wake the EventManager registered for (runtime_pid, tid).
   *
   *  On macOS, tid comes from SystemInfo::GetTid() which calls
   *  pthread_threadid_np — consistent with AddSignalEvent(). */
  static int Signal(pid_t runtime_pid, pid_t tid) {
    // Same-process fast path: write the wakeup byte to the worker's socketpair,
    // which wakes exactly that worker's kqueue.
    int wfd = detail::MacSignalRegistry::Get().Find(runtime_pid, tid);
    if (wfd != -1) {
      char b = 1;
      if (::write(wfd, &b, 1) == 1) return 0;
      // fall through to the cross-process path on a transient write failure
    }
    // Cross-process path (#619 fallback-runtime punting): the target worker is
    // in another runtime process and not in our per-process registry. SIGUSR1
    // to that process trips EVFILT_SIGNAL on every worker's kqueue there (see
    // AddSignalEvent); they wake and re-check their queues. A broadcast wake,
    // but correct.
    int rc = ::kill(runtime_pid, SIGUSR1);
    HLOG(kInfo, "EventManager::Signal: cross-process SIGUSR1 -> pid={} rc={}",
         static_cast<int>(runtime_pid), rc);
    return (rc == 0) ? 0 : -1;
  }

  int Wait(int timeout_us = -1) {
    struct timespec ts, *pts = nullptr;
    if (timeout_us >= 0) {
      ts.tv_sec = timeout_us / 1000000;
      ts.tv_nsec = static_cast<long>(timeout_us % 1000000) * 1000L;
      pts = &ts;
    }
    struct kevent kev[kMaxEvents];
    int nfds = ::kevent(kqueue_fd_, nullptr, 0, kev, kMaxEvents, pts);
    if (nfds < 0) {
      if (errno == EINTR) return 0;
      return nfds;
    }
    for (int i = 0; i < nfds; ++i) {
      // Cross-process SIGUSR1 wake (#619): EVFILT_SIGNAL reports the signal
      // number in `ident`, not a fd. The wake itself is the effect — the worker
      // re-checks its queues after Wait() returns — so there is no per-fd action.
      if (kev[i].filter == EVFILT_SIGNAL) {
        HLOG(kInfo, "EventManager::Wait: woke on cross-process SIGUSR1");
        continue;
      }
      int fd = static_cast<int>(kev[i].ident);
      auto it = fd_to_reg_.find(fd);
      if (it == fd_to_reg_.end()) continue;
      const EventRegistration& reg = it->second;
      if (fd == signal_rfd_) {
        // Drain the wakeup byte(s) written by Signal().
        char buf[64];
        while (::read(signal_rfd_, buf, sizeof(buf)) > 0) {}
      }
      if (reg.action_) {
        EventInfo info;
        info.trigger_ = {reg.fd_, reg.event_id_};
        info.events_ = kDefaultReadEvent;
        info.action_ = reg.action_;
        reg.action_->Run(info);
      }
    }
    return nfds;
  }

  /** Returns the kqueue fd (analogous to the Linux epoll fd). */
  int GetEpollFd() const { return kqueue_fd_; }

  /** Returns the read-end of the signal socketpair, or -1 if not registered. */
  int GetSignalFd() const { return signal_rfd_; }

 private:
  static constexpr int kMaxEvents = 256;

  int kqueue_fd_;
  int signal_rfd_;
  int signal_wfd_;
  int next_event_id_;
  int reg_pid_;
  int reg_tid_;

  struct EventRegistration {
    int fd_;
    int event_id_;
    EventAction* action_;
  };
  std::unordered_map<int, EventRegistration> fd_to_reg_;
};

}  // namespace ctp::lbm
