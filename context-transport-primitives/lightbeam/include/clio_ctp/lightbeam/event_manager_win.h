/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

#pragma once

#include <cstdint>
#include <memory>

#include "clio_ctp/constants/macros.h"

namespace ctp::lbm {

class EventAction;

/**
 * Windows EventManager — Linux signalfd + epoll equivalent.
 *
 * The implementation lives entirely in event_manager_win.cc — this
 * header is Win32-free (no <windows.h>, no <winsock2.h>) so consumers
 * don't drag platform macros (Yield, SendMessage, min, max, ...) into
 * their translation units.
 *
 * Demultiplexes two kinds of wakeups onto a single Wait() call:
 *  1. Socket I/O readiness (AddEvent / RemoveEvent). Each socket is
 *     associated with a WSAEVENT via WSAEventSelect internally so its
 *     readiness can be observed alongside the signal event in a single
 *     WaitForMultipleObjects.
 *  2. Cross-thread / cross-process wakeup (AddSignalEvent / Signal).
 *     The recipient creates a named auto-reset Win32 event keyed by
 *     (pid, tid); the sender opens the same name and SetEvent. This is
 *     the Win32 analogue of tgkill+signalfd+SIGUSR1.
 */
class EventManager {
 public:
  CTP_DLL EventManager();
  CTP_DLL ~EventManager();

  EventManager(const EventManager &) = delete;
  EventManager &operator=(const EventManager &) = delete;

  /** Register a socket for readiness events. events is a POLLRDNORM/
   *  POLLWRNORM/POLLERR-style mask. */
  CTP_DLL int AddEvent(int fd, uint32_t events = 0x0100 /*POLLRDNORM*/,
                       EventAction *action = nullptr);

  /** Detach an fd before the socket is closed. Must run *before*
   *  closesocket() so a recycled socket-number doesn't pick up a
   *  dangling handler. */
  CTP_DLL void RemoveEvent(int fd);

  /** Create the per-thread named signal event. Signal(pid, tid) on any
   *  thread opens the same name and calls SetEvent to wake us. */
  CTP_DLL int AddSignalEvent(EventAction *action = nullptr);

  /** Cross-thread / cross-process wakeup of (pid, tid). Returns 0 on
   *  success, -1 if the target hasn't registered yet. */
  CTP_DLL static int Signal(int runtime_pid, int tid);

  /** Block until at least one socket becomes ready or the signal event
   *  fires. timeout_us < 0 waits indefinitely. */
  CTP_DLL int Wait(int timeout_us = -1);

  CTP_DLL int GetEpollFd() const;
  CTP_DLL int GetSignalFd() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ctp::lbm
