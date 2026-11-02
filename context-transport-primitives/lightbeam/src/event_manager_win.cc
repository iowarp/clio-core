/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * Windows EventManager implementation. All Win32 / Winsock API usage
 * is contained in this translation unit — the header
 * (event_manager_win.h) is Win32-free.
 */

#ifdef _WIN32

#include "clio_ctp/lightbeam/event_manager.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <windows.h>
#ifdef Yield
#undef Yield
#endif
#ifdef SendMessage
#undef SendMessage
#endif
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#include <cstdio>
#include <string>
#include <vector>

#include "clio_ctp/util/logging.h"

namespace ctp::lbm {

struct EventManager::Impl {
  HANDLE signal_event = nullptr;
  EventAction *signal_action = nullptr;
  int next_event_id = 0;

  struct EventRegistration {
    int fd;
    int event_id;
    WSAEVENT wsa_event;
    long wsa_mask;
    EventAction *action;
  };
  std::vector<EventRegistration> registrations;
};

namespace {

std::string SignalEventName(int pid, int tid) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "Local\\chi_em_%d_%d", pid, tid);
  return std::string(buf);
}

long TranslatePollMaskToWsa(uint32_t poll_mask) {
  // POLLRDNORM=0x0100 POLLWRNORM=0x0010 POLLIN=0x0001 POLLOUT=0x0004
  // POLLHUP=0x0002 POLLERR=0x0008
  long m = 0;
  if (poll_mask & (0x0100 | 0x0001)) m |= FD_READ | FD_ACCEPT;
  if (poll_mask & (0x0010 | 0x0004)) m |= FD_WRITE | FD_CONNECT;
  if (poll_mask & (0x0002 | 0x0008)) m |= FD_CLOSE;
  if (m == 0) m = FD_READ | FD_ACCEPT | FD_CLOSE;
  return m;
}

uint32_t TranslateWsaMaskToPoll(long wsa_mask) {
  uint32_t out = 0;
  if (wsa_mask & (FD_READ | FD_ACCEPT)) out |= 0x0100;     // POLLRDNORM
  if (wsa_mask & (FD_WRITE | FD_CONNECT)) out |= 0x0010;   // POLLWRNORM
  if (wsa_mask & FD_CLOSE) out |= 0x0002;                  // POLLHUP
  return out;
}

}  // namespace

EventManager::EventManager() : impl_(std::make_unique<Impl>()) {}

EventManager::~EventManager() {
  for (auto &reg : impl_->registrations) {
    if (reg.fd != static_cast<int>(INVALID_SOCKET)) {
      ::WSAEventSelect(static_cast<SOCKET>(reg.fd), nullptr, 0);
    }
    if (reg.wsa_event != WSA_INVALID_EVENT) {
      ::WSACloseEvent(reg.wsa_event);
    }
  }
  if (impl_->signal_event) {
    ::CloseHandle(impl_->signal_event);
  }
}

int EventManager::AddEvent(int fd, uint32_t events, EventAction *action) {
  long wsa_mask = TranslatePollMaskToWsa(events);

  for (auto &reg : impl_->registrations) {
    if (reg.fd == fd) {
      if (::WSAEventSelect(static_cast<SOCKET>(fd), reg.wsa_event, wsa_mask) ==
          SOCKET_ERROR) {
        HLOG(kError, "EventManager::AddEvent: WSAEventSelect MOD failed: {}",
             ::WSAGetLastError());
        return -1;
      }
      reg.action = action;
      reg.wsa_mask = wsa_mask;
      return reg.event_id;
    }
  }

  WSAEVENT ev = ::WSACreateEvent();
  if (ev == WSA_INVALID_EVENT) {
    HLOG(kError, "EventManager::AddEvent: WSACreateEvent failed: {}",
         ::WSAGetLastError());
    return -1;
  }
  if (::WSAEventSelect(static_cast<SOCKET>(fd), ev, wsa_mask) == SOCKET_ERROR) {
    HLOG(kError, "EventManager::AddEvent: WSAEventSelect ADD failed: {}",
         ::WSAGetLastError());
    ::WSACloseEvent(ev);
    return -1;
  }
  int event_id = impl_->next_event_id++;
  impl_->registrations.push_back({fd, event_id, ev, wsa_mask, action});
  return event_id;
}

void EventManager::RemoveEvent(int fd) {
  for (size_t i = 0; i < impl_->registrations.size();) {
    if (impl_->registrations[i].fd == fd) {
      ::WSAEventSelect(static_cast<SOCKET>(fd), nullptr, 0);
      if (impl_->registrations[i].wsa_event != WSA_INVALID_EVENT) {
        ::WSACloseEvent(impl_->registrations[i].wsa_event);
      }
      impl_->registrations.erase(impl_->registrations.begin() + i);
    } else {
      ++i;
    }
  }
}

int EventManager::AddSignalEvent(EventAction *action) {
  std::string name = SignalEventName(
      static_cast<int>(::GetCurrentProcessId()),
      static_cast<int>(::GetCurrentThreadId()));
  impl_->signal_event = ::CreateEventA(nullptr,
                                        /*manual_reset=*/FALSE,
                                        /*initial_state=*/FALSE,
                                        name.c_str());
  if (!impl_->signal_event) {
    HLOG(kError, "EventManager::AddSignalEvent: CreateEventA('{}') failed: {}",
         name, ::GetLastError());
    return -1;
  }
  impl_->signal_action = action;
  return impl_->next_event_id++;
}

int EventManager::Signal(int runtime_pid, int tid) {
  std::string name = SignalEventName(runtime_pid, tid);
  HANDLE h = ::OpenEventA(EVENT_MODIFY_STATE, FALSE, name.c_str());
  if (!h) return -1;
  bool ok = ::SetEvent(h) != 0;
  ::CloseHandle(h);
  return ok ? 0 : -1;
}

int EventManager::Wait(int timeout_us) {
  DWORD timeout_ms;
  if (timeout_us < 0) {
    timeout_ms = INFINITE;
  } else {
    timeout_ms = static_cast<DWORD>((timeout_us + 999) / 1000);
    if (timeout_ms < 1 && timeout_us > 0) timeout_ms = 1;
  }

  std::vector<HANDLE> wait_handles;
  wait_handles.reserve(1 + impl_->registrations.size());
  if (impl_->signal_event) wait_handles.push_back(impl_->signal_event);
  for (auto &reg : impl_->registrations) wait_handles.push_back(reg.wsa_event);

  if (wait_handles.empty()) {
    if (timeout_ms != 0 && timeout_ms != INFINITE) ::Sleep(timeout_ms);
    return 0;
  }
  if (wait_handles.size() > MAXIMUM_WAIT_OBJECTS) {
    HLOG(kError,
         "EventManager::Wait: handle count {} exceeds MAXIMUM_WAIT_OBJECTS ({})",
         wait_handles.size(), MAXIMUM_WAIT_OBJECTS);
    return -1;
  }

  DWORD r = ::WaitForMultipleObjectsEx(
      static_cast<DWORD>(wait_handles.size()), wait_handles.data(),
      /*wait_all=*/FALSE, timeout_ms, /*alertable=*/FALSE);
  if (r == WAIT_TIMEOUT) return 0;
  if (r == WAIT_FAILED) {
    HLOG(kError, "EventManager::Wait: WFMO failed: {}", ::GetLastError());
    return -1;
  }

  int fired = 0;

  bool signal_ready =
      impl_->signal_event &&
      ::WaitForSingleObject(impl_->signal_event, 0) == WAIT_OBJECT_0;
  if (signal_ready) {
    if (impl_->signal_action) {
      EventInfo info;
      info.trigger_ = {-1, 0};
      info.events_ = 0;
      info.action_ = impl_->signal_action;
      impl_->signal_action->Run(info);
    }
    ++fired;
  }

  for (auto &reg : impl_->registrations) {
    WSANETWORKEVENTS net_events;
    if (::WSAEnumNetworkEvents(static_cast<SOCKET>(reg.fd), reg.wsa_event,
                                &net_events) == SOCKET_ERROR) {
      continue;
    }
    if (net_events.lNetworkEvents == 0) continue;
    if (reg.action) {
      EventInfo info;
      info.trigger_ = {reg.fd, reg.event_id};
      info.events_ = TranslateWsaMaskToPoll(net_events.lNetworkEvents);
      info.action_ = reg.action;
      reg.action->Run(info);
    }
    ++fired;
  }
  return fired;
}

int EventManager::GetEpollFd() const { return -1; }

int EventManager::GetSignalFd() const {
  return impl_->signal_event
             ? static_cast<int>(
                   reinterpret_cast<intptr_t>(impl_->signal_event) & 0x7FFFFFFF)
             : -1;
}

}  // namespace ctp::lbm

#endif  // _WIN32
