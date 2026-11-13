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
// Pipelined socket I/O for SocketTransport (TCP).
//
// A process-wide pool of I/O threads (CLIO_SOCK_IO_THREADS, default 4) owns
// the sockets. Each connection is pinned to one I/O thread, which is the only
// thread that reads it, so connections are read in parallel and frame
// reassembly needs no lock. Complete frames go to the owning transport's
// Inbox; Recv just pops one. Reading the next frame therefore overlaps the
// caller dispatching the last one.
//
// Send never blocks on the network: the caller writes what the kernel takes
// right away (unless CLIO_SOCK_DIRECT_SEND=0), and the rest of the frame is
// copied to the connection's out-queue, which its I/O thread flushes when the
// socket is writable. A frame is queued whole, so it still arrives whole or
// (if the connection breaks) not at all.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "clio_ctp/lightbeam/posix_socket.h"
#include "clio_ctp/lightbeam/socket_frame.h"

namespace ctp::lbm::sock {

struct Inbox;

/** One connection owned by the engine. */
struct EngineConn {
  socket_t fd = kInvalidSocket;
  int id = 0;                       ///< id within its Inbox
  int worker = -1;                  ///< owning I/O thread
  bool tcp = true;                  ///< TCP (set TCP_NODELAY) vs unix socket
  std::shared_ptr<Inbox> inbox;     ///< where its frames go

  // Send side (under mu).
  std::mutex mu;
  std::condition_variable cv;       ///< queue drained below the cap
  // Output not yet taken by the kernel. A frame that is partly written is
  // `cur`; it always finishes first. Whole frames wait in small_q (frames
  // under kSmallFrameBytes: replies, metadata, probes) or big_q, and small_q
  // goes first, so small frames never wait behind queued bulk.
  std::vector<char> cur;
  size_t cur_off = 0;               ///< bytes of cur already sent
  std::deque<std::vector<char>> small_q;
  std::deque<std::vector<char>> big_q;
  size_t queued = 0;                ///< bytes waiting in cur + queues
  bool HasOutput() const {
    return cur_off < cur.size() || !small_q.empty() || !big_q.empty();
  }
  bool broken = false;
  std::atomic<bool> want_out{false};  ///< outq non-empty (poll for writable)

  // Receive side (owning I/O thread only).
  FrameReader reader;

  CTP_DLL ~EngineConn();
};

/** A transport's receive queue and connection table. */
struct Inbox {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<std::pair<int, Frame>> frames;  ///< (connection id, frame)

  std::mutex conns_mu;
  std::unordered_map<int, std::shared_ptr<EngineConn>> conns;
  int next_id = 1;

  /** Add a connection and give it an id. */
  CTP_DLL int Register(const std::shared_ptr<EngineConn>& c);
  /** Forget a connection. */
  CTP_DLL void Unregister(int id);
  /** @return the live connection with this id, or null */
  CTP_DLL std::shared_ptr<EngineConn> Find(int id);
  /** Forget every connection (breaks the conn <-> inbox reference cycle). */
  CTP_DLL std::vector<std::shared_ptr<EngineConn>> TakeAll();

  CTP_DLL void Push(int id, Frame&& f);
  /** Pop one frame without waiting. @return false if empty */
  CTP_DLL bool Pop(int* id, Frame* f);
  /** Wait up to timeout_ms for a frame. @return true if one is queued */
  CTP_DLL bool Wait(int timeout_ms);
};

/** A listening socket owned by the engine; accepted connections join its
 *  Inbox. Closes the socket when the last reference goes. */
struct EngineListener {
  socket_t fd = kInvalidSocket;
  int worker = -1;
  bool tcp = true;
  std::shared_ptr<Inbox> inbox;
  CTP_DLL ~EngineListener();
};

/** The process-wide I/O thread pool. */
class IoEngine {
 public:
  CTP_DLL static IoEngine& Instance();
  /** True if SocketTransport should use the engine (CLIO_SOCK_ENGINE,
   *  default on). */
  CTP_DLL static bool Enabled();

  /** Apply socket options, register c with its inbox, and hand it to an I/O
   *  thread. c->fd and c->inbox must be set. */
  CTP_DLL void AddConn(const std::shared_ptr<EngineConn>& c);
  /** Stop servicing c (its socket closes once no thread holds it). */
  CTP_DLL void RemoveConn(const std::shared_ptr<EngineConn>& c);
  CTP_DLL std::shared_ptr<EngineListener> AddListener(
      socket_t fd, const std::shared_ptr<Inbox>& inbox, bool tcp);
  CTP_DLL void RemoveListener(const std::shared_ptr<EngineListener>& l);

  /** Send one frame (iov: header, meta, bulks) on c: write what the socket
   *  takes now and queue the rest. Blocks only while c's out-queue is over
   *  its cap. @return 0, or EPIPE if c is broken */
  CTP_DLL int Send(const std::shared_ptr<EngineConn>& c, const IoBuffer* iov,
                   size_t count);

  /** Frames under this size go ahead of queued bulk frames. */
  static constexpr size_t kSmallFrameBytes = 64 * 1024;

  ~IoEngine();

 private:
  struct Worker {
    std::thread th;
    std::mutex mu;
    std::vector<std::shared_ptr<EngineConn>> conns;
    std::vector<std::shared_ptr<EngineListener>> listeners;
    std::atomic<uint64_t> gen{0};  ///< bumped when conns/listeners change
  };

  IoEngine();
  void Loop(Worker* w);
  void ReadConn(const std::shared_ptr<EngineConn>& c);
  void Flush(const std::shared_ptr<EngineConn>& c);
  void Accept(const std::shared_ptr<EngineListener>& l);
  void Drop(const std::shared_ptr<EngineConn>& c);
  static void Fail(EngineConn* c);  // under c->mu

  std::vector<std::unique_ptr<Worker>> workers_;
  std::atomic<unsigned> rr_{0};
  std::atomic<bool> stop_{false};
  bool direct_send_ = true;
  size_t max_queued_ = 0;
};

}  // namespace ctp::lbm::sock
