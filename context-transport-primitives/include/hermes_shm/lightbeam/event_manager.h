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

#include <unordered_map>

#include "hermes_shm/util/logging.h"

namespace hshm::lbm {

struct EventTrigger {
  int fd_;
  int event_id_;
};

struct EventInfo;

class EventAction {
 public:
  virtual ~EventAction() = default;
  virtual void Run(const EventInfo &event) = 0;
};

struct EventInfo {
  EventTrigger trigger_;
  uint32_t epoll_events_;
  EventAction *action_;
};

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
    int event_id = next_event_id_++;
    struct epoll_event ev;
    ev.events = events;
    ev.data.fd = fd;
    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &ev) == -1) {
      HLOG(kError, "EventManager::AddEvent: epoll_ctl ADD failed for fd={}: {}",
           fd, strerror(errno));
      return -1;
    }
    fd_to_reg_[fd] = {fd, event_id, action};
    return event_id;
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
    int timeout_ms;
    if (timeout_us < 0) {
      timeout_ms = -1;
    } else {
      timeout_ms = static_cast<int>((timeout_us + 999) / 1000);
      if (timeout_ms < 1 && timeout_us > 0) {
        timeout_ms = 1;
      }
    }
    int nfds = epoll_wait(epoll_fd_, epoll_events_, kMaxEvents, timeout_ms);
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
        info.epoll_events_ = epoll_events_[i].events;
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

}  // namespace hshm::lbm
