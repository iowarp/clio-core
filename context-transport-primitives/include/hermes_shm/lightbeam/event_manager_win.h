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

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <windows.h>
#ifdef Yield
#undef Yield
#endif
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#include <vector>

namespace hshm::lbm {

class EventManager {
 public:
  EventManager()
      : signal_event_(nullptr),
        next_event_id_(0) {
  }

  ~EventManager() {
    if (signal_event_) {
      CloseHandle(signal_event_);
    }
  }

  EventManager(const EventManager &) = delete;
  EventManager &operator=(const EventManager &) = delete;

  int AddEvent(int fd, uint32_t events = POLLRDNORM,
               EventAction *action = nullptr) {
    int event_id = next_event_id_++;
    WSAPOLLFD pfd;
    pfd.fd = static_cast<SOCKET>(fd);
    pfd.events = static_cast<SHORT>(events);
    pfd.revents = 0;
    poll_fds_.push_back(pfd);
    registrations_.push_back({fd, event_id, action});
    return event_id;
  }

  int AddSignalEvent(EventAction *action = nullptr) {
    signal_event_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!signal_event_) {
      HLOG(kError, "EventManager::AddSignalEvent: CreateEvent failed");
      return -1;
    }
    signal_action_ = action;
    return next_event_id_++;
  }

  static int Signal(HANDLE event_handle) {
    return SetEvent(event_handle) ? 0 : -1;
  }

  static int Signal(int runtime_pid, int tid) {
    (void)runtime_pid;
    (void)tid;
    return 0;
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

    // Check signal event first (non-blocking)
    if (signal_event_) {
      ::DWORD wait_result = WaitForSingleObject(signal_event_, 0);
      if (wait_result == WAIT_OBJECT_0 && signal_action_) {
        EventInfo info;
        info.trigger_ = {-1, 0};
        info.events_ = 0;
        info.action_ = signal_action_;
        signal_action_->Run(info);
      }
    }

    if (poll_fds_.empty()) return 0;

    int nfds = WSAPoll(poll_fds_.data(),
                       static_cast<ULONG>(poll_fds_.size()),
                       timeout_ms);
    if (nfds <= 0) return nfds;

    int fired = 0;
    for (size_t i = 0; i < poll_fds_.size(); ++i) {
      if (poll_fds_[i].revents != 0) {
        const EventRegistration &reg = registrations_[i];
        if (reg.action_) {
          EventInfo info;
          info.trigger_ = {reg.fd_, reg.event_id_};
          info.events_ = poll_fds_[i].revents;
          info.action_ = reg.action_;
          reg.action_->Run(info);
        }
        poll_fds_[i].revents = 0;
        fired++;
      }
    }
    return fired;
  }

  int GetEpollFd() const { return -1; }

  int GetSignalFd() const { return -1; }

 private:
  HANDLE signal_event_;
  EventAction *signal_action_ = nullptr;
  int next_event_id_;

  struct EventRegistration {
    int fd_;
    int event_id_;
    EventAction *action_;
  };

  std::vector<WSAPOLLFD> poll_fds_;
  std::vector<EventRegistration> registrations_;
};

}  // namespace hshm::lbm
