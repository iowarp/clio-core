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

#include <signal.h>
#include <sys/epoll.h>
#include <sys/signalfd.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "clio_ctp/util/msan.h"

namespace ctp::lbm {

class EventManager {
 public:
  EventManager()
      : epoll_fd_(epoll_create1(0)),
        signal_fd_(-1),
        next_event_id_(0) {
  }

  ~EventManager() {
    if (signal_fd_ >= 0) {
      close(signal_fd_);
    }
    if (epoll_fd_ >= 0) {
      close(epoll_fd_);
    }
  }

  EventManager(const EventManager &) = delete;
  EventManager &operator=(const EventManager &) = delete;

  int AddEvent(int fd, uint32_t events = EPOLLIN,
               EventAction *action = nullptr) {
    struct epoll_event ev;
    ev.events = events;
    ev.data.fd = fd;
    auto it = fd_to_reg_.find(fd);
    if (it != fd_to_reg_.end()) {
      // Stale-entry-safe MOD: callers (e.g. socket_transport on accept of a
      // recycled fd number) can leave fd_to_reg_ holding a dangling entry
      // while the kernel auto-dropped the fd from epoll on close. ENOENT
      // here means "fd not in epoll" — recover by promoting the call to
      // EPOLL_CTL_ADD instead of erroring out.
      if (epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, fd, &ev) == -1) {
        if (errno == ENOENT) {
          if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &ev) == -1) {
            HLOG(kError,
                 "EventManager::AddEvent: MOD->ADD fallback failed for fd={}: {}",
                 fd, strerror(errno));
            return -1;
          }
        } else {
          HLOG(kError,
               "EventManager::AddEvent: epoll_ctl MOD failed for fd={}: {}", fd,
               strerror(errno));
          return -1;
        }
      }
      it->second.action_ = action;
      return it->second.event_id_;
    }
    int event_id = next_event_id_++;
    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &ev) == -1) {
      // Symmetric recovery for the rare case where the kernel still has the
      // fd registered (e.g. another caller added it) but our local map is
      // empty — promote to MOD.
      if (errno == EEXIST &&
          epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, fd, &ev) != -1) {
        // ok
      } else {
        HLOG(kError,
             "EventManager::AddEvent: epoll_ctl ADD failed for fd={}: {}",
             fd, strerror(errno));
        return -1;
      }
    }
    fd_to_reg_[fd] = {fd, event_id, action};
    return event_id;
  }

  /** Remove an fd from epoll and the bookkeeping map.
   *  Must be called *before* sock::Close(fd); otherwise the kernel may
   *  recycle the fd number for a different file and a later AddEvent call
   *  trips MOD on a dangling entry. ENOENT here is benign — the fd was
   *  already absent from epoll. */
  void RemoveEvent(int fd) {
    auto it = fd_to_reg_.find(fd);
    if (it == fd_to_reg_.end()) {
      return;
    }
    if (epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr) == -1 &&
        errno != ENOENT && errno != EBADF) {
      HLOG(kError, "EventManager::RemoveEvent: epoll_ctl DEL fd={}: {}", fd,
           strerror(errno));
    }
    fd_to_reg_.erase(it);
  }

  int AddSignalEvent(EventAction *action = nullptr) {
    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGUSR1);
    if (pthread_sigmask(SIG_BLOCK, &mask, nullptr) != 0) {
      HLOG(kError, "EventManager::AddSignalEvent: pthread_sigmask failed");
      return -1;
    }
    signal_fd_ = signalfd(-1, &mask, SFD_NONBLOCK | SFD_CLOEXEC);
    if (signal_fd_ == -1) {
      HLOG(kError, "EventManager::AddSignalEvent: signalfd failed: {}",
           strerror(errno));
      return -1;
    }
    return AddEvent(signal_fd_, EPOLLIN, action);
  }

  static int Signal(pid_t runtime_pid, pid_t tid) {
    return syscall(SYS_tgkill, runtime_pid, tid, SIGUSR1);
  }

  int Wait(int timeout_us = -1) {
    int nfds;
    if (timeout_us < 0) {
      // Indefinite wait — use epoll_wait for simplicity
      nfds = epoll_wait(epoll_fd_, epoll_events_, kMaxEvents, -1);
    } else {
      // Try epoll_pwait2 for microsecond precision (kernel 5.11+, syscall 441)
      struct timespec ts;
      ts.tv_sec = timeout_us / 1000000;
      ts.tv_nsec = (timeout_us % 1000000) * 1000L;
      nfds = static_cast<int>(syscall(441, epoll_fd_, epoll_events_,
                                       kMaxEvents, &ts, nullptr, 0));
      // The kernel filled epoll_events_, but MSan only intercepts the libc
      // epoll_wait/epoll_pwait wrappers -- a raw syscall() goes straight past
      // it, so the shadow for these entries keeps whatever poison the memory
      // carried before (this buffer is recycled every call). Without this the
      // fd read out below is reported as uninitialized on every wakeup that
      // takes the epoll_pwait2 path.
      if (nfds > 0) {
        CTP_MSAN_UNPOISON(epoll_events_,
                          static_cast<size_t>(nfds) * sizeof(epoll_events_[0]));
      }
      if (nfds == -1 && errno == ENOSYS) {
        // Fallback to epoll_wait if kernel doesn't support epoll_pwait2
        int timeout_ms = static_cast<int>((timeout_us + 999) / 1000);
        if (timeout_ms < 1 && timeout_us > 0) timeout_ms = 1;
        nfds = epoll_wait(epoll_fd_, epoll_events_, kMaxEvents, timeout_ms);
      }
    }
    for (int i = 0; i < nfds; ++i) {
      int fd = epoll_events_[i].data.fd;
      auto it = fd_to_reg_.find(fd);
      if (it == fd_to_reg_.end()) continue;
      const EventRegistration &reg = it->second;
      if (fd == signal_fd_) {
        struct signalfd_siginfo si;
        ssize_t bytes = read(signal_fd_, &si, sizeof(si));
        (void)bytes;
      }
      if (reg.action_) {
        EventInfo info;
        info.trigger_ = {reg.fd_, reg.event_id_};
        info.events_ = epoll_events_[i].events;
        info.action_ = reg.action_;
        reg.action_->Run(info);
      }
    }
    return nfds;
  }

  int GetEpollFd() const { return epoll_fd_; }

  int GetSignalFd() const { return signal_fd_; }

 private:
  int epoll_fd_;
  int signal_fd_;
  int next_event_id_;
  static constexpr int kMaxEvents = 256;
  struct epoll_event epoll_events_[kMaxEvents];

  struct EventRegistration {
    int fd_;
    int event_id_;
    EventAction *action_;
  };
  std::unordered_map<int, EventRegistration> fd_to_reg_;
};

}  // namespace ctp::lbm
