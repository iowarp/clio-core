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
// SocketTransport: tasks over plain TCP / Unix-domain sockets.
//
// The contract is deliberately small: a task is delivered whole or not at all.
//   Send  writes one complete frame (socket_frame.h) under a per-connection
//         lock, so concurrent senders never interleave. If the write fails
//         part-way the connection is shut down, so the receiver can never
//         take half a frame for a whole one, and the error is returned.
//   Recv  reassembles frames from whatever bytes have arrived, on any
//         connection, and returns only complete ones. A large bulk is read
//         straight into its buffer and does not hold up other connections.
//   Errors are reported, not hidden: there is no reconnect and no queueing.
//
// Server replies are addressed by the connection id that Recv stores in
// ClientInfo::fd_ (an id, not an OS socket, so a recycled socket number can
// never receive another peer's reply).
//
// All socket calls go through ctp::lbm::sock (socket_posix.cc / socket_win.cc).
//
// TCP transports run on the pipelined I/O engine (socket_engine.h) unless
// CLIO_SOCK_ENGINE=0: I/O threads read every connection in parallel and
// flush queued output, Recv pops complete frames, and Send never waits on the
// network. Unix-socket (ipc) transports keep the inline path below, because
// their callers block on the socket through an EventManager.

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "clio_ctp/util/logging.h"
#include "lightbeam.h"
#include "posix_socket.h"
#include "socket_engine.h"
#include "socket_frame.h"

namespace ctp::lbm {

/** One connection: its socket, a send lock and its frame reassembly state. */
struct SocketConn {
  sock::socket_t fd = sock::kInvalidSocket;  ///< connected socket
  int id = 0;                                ///< id handed out in ClientInfo
  std::mutex send_mu;                        ///< one frame on the wire at a time
  bool broken = false;                       ///< a send failed (under send_mu)
  sock::FrameReader reader;                  ///< receive side (under recv_mu_)
};

class SocketTransport : public Transport {
 public:
  explicit SocketTransport(TransportMode mode, const std::string& addr,
                           const std::string& protocol = "tcp", int port = 8193)
      : Transport(mode), addr_(addr), protocol_(protocol), port_(port) {
    type_ = TransportType::kSocket;
    sock::InitSocketLib();
    if (mode == TransportMode::kClient) {
      sock::socket_t fd = sock::Connect(addr_, port_, protocol_);
      if (fd == sock::kInvalidSocket) {
        throw std::runtime_error("SocketTransport: failed to connect to " +
                                 addr_ + ":" + std::to_string(port_));
      }
      if (UseEngine()) {
        engine_ = true;
        inbox_ = std::make_shared<sock::Inbox>();
        econn_ = std::make_shared<sock::EngineConn>();
        econn_->fd = fd;
        econn_->tcp = protocol_ != "ipc";
        econn_->inbox = inbox_;
        sock::IoEngine::Instance().AddConn(econn_);
        return;
      }
      sock::SetNonBlocking(fd, true);
      conn_ = std::make_shared<SocketConn>();
      conn_->fd = fd;
      HLOG(kDebug, "SocketTransport(client) connected to {}:{}", addr_, port_);
    } else {
      listen_fd_ = sock::Listen(addr_, port_, protocol_);
      if (listen_fd_ == sock::kInvalidSocket) {
        throw std::runtime_error("SocketTransport: failed to listen on " +
                                 addr_ + ":" + std::to_string(port_));
      }
      if (UseEngine()) {
        engine_ = true;
        inbox_ = std::make_shared<sock::Inbox>();
        elisten_ = sock::IoEngine::Instance().AddListener(
            listen_fd_, inbox_, protocol_ != "ipc");
        listen_fd_ = sock::kInvalidSocket;  // owned by elisten_ now
        return;
      }
      sock::SetNonBlocking(listen_fd_, true);
      HLOG(kDebug, "SocketTransport(server) listening on {}:{}", addr_, port_);
    }
  }

  ~SocketTransport() {
    if (engine_) {
      auto& eng = sock::IoEngine::Instance();
      if (elisten_) eng.RemoveListener(elisten_);
      if (econn_) eng.RemoveConn(econn_);
      for (auto& c : inbox_->TakeAll()) eng.RemoveConn(c);
      return;
    }
    if (IsClient()) {
      CloseConn(conn_);
      return;
    }
    for (auto& kv : conns_) CloseConn(kv.second);
    if (em_) em_->RemoveEvent(static_cast<int>(listen_fd_));
    sock::Close(listen_fd_);
    if (protocol_ == "ipc") sock::UnlinkPath(addr_.c_str());
  }

  Bulk Expose(const ctp::ipc::FullPtr<char>& ptr, size_t data_size,
              u32 flags) {
    Bulk bulk;
    bulk.data = ptr;
    bulk.size = data_size;
    bulk.flags = ctp::bitfield32_t(flags);
    return bulk;
  }

  /** Free the bulk buffers Recv allocated (tagged with the
   *  (UINT32_MAX-1, UINT32_MAX-1) sentinel). Buffers the caller swapped in
   *  (e.g. a CTP MallocAllocator FullPtr, offset 16 B into its malloc region)
   *  are not ours to free; the task destructor reclaims them. */
  void ClearRecvHandles(LbmMeta<>& meta) {
    for (auto& bulk : meta.recv) {
      if (bulk.data.ptr_ && bulk.data.shm_.alloc_id_ == RecvBulkAllocId()) {
        std::free(bulk.data.ptr_);
        bulk.data.ptr_ = nullptr;
      }
    }
  }

  std::string GetAddress() const { return addr_; }

  /** Check if the server is still alive via a connect probe. */
  bool IsServerAlive(const LbmContext& ctx = LbmContext()) const {
    (void)ctx;
    return sock::IsServerAlive(addr_, port_, protocol_);
  }

  /** Register every socket with em so em.Wait() wakes on incoming data (and,
   *  for a server, on new connections). Recv does not depend on it. */
  void RegisterEventManager(EventManager& em) {
    em_ = &em;
    if (engine_) return;  // engine transports wake via PollRecv
    if (IsClient()) {
      em.AddEvent(static_cast<int>(conn_->fd), kDefaultReadEvent, nullptr);
      return;
    }
    em.AddEvent(static_cast<int>(listen_fd_), kDefaultReadEvent, nullptr);
    std::lock_guard<std::mutex> lk(conns_mu_);
    for (auto& kv : conns_) {
      em.AddEvent(static_cast<int>(kv.second->fd), kDefaultReadEvent, nullptr);
    }
  }

  /** Detach without RemoveEvent: the caller is about to destroy the
   *  EventManager itself, so touching it here would be a use-after-free. */
  void UnregisterEventManager() { em_ = nullptr; }

  /** Send one task as a single frame: whole, or not at all (see file
   *  comment). Server mode sends to the connection id in
   *  meta.client_info_.fd_ (set by Recv).
   *  @return 0 on success, EHOSTUNREACH if that connection is gone, or the
   *          socket error of a failed write (the connection is then shut
   *          down) */
  template <typename MetaT>
  int Send(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    (void)ctx;
    std::shared_ptr<SocketConn> conn;
    std::shared_ptr<sock::EngineConn> econn;
    if (engine_) {
      econn = IsClient() ? econn_ : inbox_->Find(meta.client_info_.fd_);
      if (!econn) return EHOSTUNREACH;
    } else {
      conn = IsClient() ? conn_ : FindConn(meta.client_info_.fd_);
      if (!conn) return EHOSTUNREACH;
    }
    std::vector<uint64_t> lens;
    meta.send_bulks = 0;
    for (size_t i = 0; i < meta.send.size(); ++i) {
      if (!meta.send[i].flags.Any(BULK_XFER)) continue;
      lens.push_back(meta.send[i].size);
      meta.send_bulks++;
    }
    std::vector<char> meta_buf;
    {
      ctp::ipc::GlobalSerialize<std::vector<char>> ar(meta_buf);
      ar(meta);
      ar.Finalize();
    }
    std::vector<char> hdr;
    sock::EncodeFrameHeader(static_cast<uint32_t>(meta_buf.size()), lens,
                            &hdr);
    std::vector<sock::IoBuffer> iov;
    iov.reserve(2 + lens.size());
    iov.push_back({hdr.data(), hdr.size()});
    iov.push_back({meta_buf.data(), meta_buf.size()});
    for (size_t i = 0; i < meta.send.size(); ++i) {
      if (!meta.send[i].flags.Any(BULK_XFER)) continue;
      iov.push_back({meta.send[i].data.ptr_, meta.send[i].size});
    }
    if (engine_) {
      return sock::IoEngine::Instance().Send(econn, iov.data(), iov.size());
    }
    return SendFrame(conn, &iov);
  }

  /** Block up to timeout_ms until a frame may be ready: buffered bytes, data
   *  on any connection, or (server) a new connection to accept.
   *  @return >0 if Recv should be called, 0 on timeout */
  int PollRecv(int timeout_ms) {
    if (engine_) return inbox_->Wait(timeout_ms) ? 1 : 0;
    std::vector<sock::PollFd> pfds;
    {
      std::lock_guard<std::mutex> lk(recv_mu_);
      if (AnyBuffered()) return 1;
    }
    BuildPollSet(&pfds, nullptr);
    if (pfds.empty()) return 0;
    return sock::PollFds(pfds.data(), static_cast<int>(pfds.size()),
                         timeout_ms) > 0
               ? 1
               : 0;
  }

  /** Return the next complete task, if any.
   *  @return info.rc 0 with the task in meta; EAGAIN if none is complete;
   *          -1 if a peer closed (info.fd_ = its connection id) or sent a
   *          frame that does not decode */
  template <typename MetaT>
  ClientInfo Recv(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    (void)ctx;
    ClientInfo info;
    sock::Frame frame;
    int conn_id = -1;
    if (engine_) {
      info.rc = inbox_->Pop(&conn_id, &frame) ? 0 : EAGAIN;
    } else {
      info.rc = NextFrame(&frame, &conn_id);
    }
    info.fd_ = conn_id;
    if (info.rc != 0) return info;
    meta.client_info_.fd_ = conn_id;
    info.rc = DecodeFrame(&frame, meta);
    return info;
  }

 private:
  /** TCP transports use the I/O engine unless CLIO_SOCK_ENGINE=0. */
  bool UseEngine() const {
    return protocol_ != "ipc" && sock::IoEngine::Enabled();
  }

  /** Allocator id marking a bulk buffer Recv malloc'd (freed by
   *  ClearRecvHandles). Distinct from AllocatorId::GetNull(). */
  static ctp::ipc::AllocatorId RecvBulkAllocId() {
    return ctp::ipc::AllocatorId(UINT32_MAX - 1, UINT32_MAX - 1);
  }

  /** Deserialize a frame's metadata into meta and hand its bulks over, in
   *  order, to the BULK_XFER entries of meta.recv.
   *  @return 0, or -1 if the metadata does not decode or does not match the
   *          bulks in the frame */
  template <typename MetaT>
  int DecodeFrame(sock::Frame* frame, MetaT& meta) {
    try {
      ctp::ipc::GlobalDeserialize<std::vector<char>> ar(frame->meta);
      ar(meta);
    } catch (const std::exception& e) {
      HLOG(kError, "SocketTransport: metadata does not decode - {} (len={})",
           e.what(), frame->meta.size());
      return -1;
    }
    size_t k = 0;
    for (const auto& send_bulk : meta.send) {
      Bulk recv_bulk;
      recv_bulk.size = send_bulk.size;
      recv_bulk.flags = send_bulk.flags;
      recv_bulk.data = ctp::ipc::FullPtr<char>::GetNull();
      if (send_bulk.flags.Any(BULK_XFER)) {
        if (k >= frame->bulks.size() ||
            frame->bulks[k].second != send_bulk.size) {
          HLOG(kError, "SocketTransport: frame bulks do not match metadata");
          return -1;
        }
        char* buf = frame->bulks[k].first;
        frame->bulks[k].first = nullptr;  // now owned by meta.recv
        ++k;
        recv_bulk.data.ptr_ = buf;
        recv_bulk.data.shm_.alloc_id_ = RecvBulkAllocId();
        recv_bulk.data.shm_.off_ = reinterpret_cast<size_t>(buf);
      }
      meta.recv.push_back(recv_bulk);
    }
    return k == frame->bulks.size() ? 0 : -1;
  }

  /** Write one frame on conn under its send lock (iov: header, meta, bulks).
   *  A failed write shuts the connection down. */
  int SendFrame(const std::shared_ptr<SocketConn>& conn,
                std::vector<sock::IoBuffer>* iov) {
    // sock::SendV takes at most 64 buffers: merge a frame with more parts.
    constexpr size_t kMaxParts = 64;
    std::vector<char> merged;
    if (iov->size() > kMaxParts) {
      for (const auto& part : *iov) {
        const char* p = static_cast<const char*>(part.base);
        merged.insert(merged.end(), p, p + part.len);
      }
      iov->assign(1, {merged.data(), merged.size()});
    }
    std::lock_guard<std::mutex> lk(conn->send_mu);
    if (conn->broken || conn->fd == sock::kInvalidSocket) return EPIPE;
    if (sock::SendV(conn->fd, iov->data(), static_cast<int>(iov->size())) >=
        0) {
      return 0;
    }
    int err = sock::GetError();
    conn->broken = true;
    sock::Shutdown(conn->fd);  // the peer drops the partial frame
    HLOG(kWarning, "SocketTransport: send to {} failed ({}); connection closed",
         addr_, sock::GetErrorString());
    return err != 0 ? err : EPIPE;
  }

  /** Find the next complete frame on any connection (never blocks).
   *  @return 0 with *out filled; EAGAIN; or -1 if a connection closed
   *          (*conn_id names it) */
  int NextFrame(sock::Frame* out, int* conn_id) {
    std::lock_guard<std::mutex> lk(recv_mu_);
    if (IsClient()) {
      *conn_id = conn_->id;
      return ReadConn(conn_, out);
    }
    std::vector<std::shared_ptr<SocketConn>> order;
    std::vector<sock::PollFd> pfds;
    BuildPollSet(&pfds, &order);
    // Bytes already buffered first: poll() cannot see them.
    for (auto& c : order) {
      if (!c->reader.HasBuffered()) continue;
      int rc = ReadConn(c, out);
      if (rc != EAGAIN) {
        *conn_id = c->id;
        if (rc == -1) DropConn(c);
        return rc;
      }
    }
    if (sock::PollFds(pfds.data(), static_cast<int>(pfds.size()), 0) <= 0) {
      return EAGAIN;
    }
    if (pfds[0].revents != 0) AcceptPending();
    return ReadReady(pfds, order, out, conn_id);
  }

  /** Read every connection poll reported ready, starting after the one that
   *  delivered last (round-robin), until one frame completes. */
  int ReadReady(const std::vector<sock::PollFd>& pfds,
                const std::vector<std::shared_ptr<SocketConn>>& order,
                sock::Frame* out, int* conn_id) {
    const size_t n = order.size();
    for (size_t k = 0; k < n; ++k) {
      size_t i = (rr_next_ + k) % n;
      if (pfds[i + 1].revents == 0) continue;
      int rc = ReadConn(order[i], out);
      if (rc == EAGAIN) continue;
      *conn_id = order[i]->id;
      rr_next_ = i + 1;
      if (rc == -1) DropConn(order[i]);
      return rc;
    }
    return EAGAIN;
  }

  /** Drive one connection's reassembly. @return 0 frame, EAGAIN, -1 closed */
  int ReadConn(const std::shared_ptr<SocketConn>& c, sock::Frame* out) {
    if (c->fd == sock::kInvalidSocket) return -1;
    int r = c->reader.Read(c->fd, out);
    return r == 1 ? 0 : (r == 0 ? EAGAIN : -1);
  }

  /** Poll set: server = [listen, conns...] (order receives the conns in the
   *  same order); client = [conn]. */
  void BuildPollSet(std::vector<sock::PollFd>* pfds,
                    std::vector<std::shared_ptr<SocketConn>>* order) {
    if (IsClient()) {
      if (conn_->fd != sock::kInvalidSocket) {
        pfds->push_back({conn_->fd, sock::kPollIn, 0});
      }
      return;
    }
    pfds->push_back({listen_fd_, sock::kPollIn, 0});
    std::lock_guard<std::mutex> lk(conns_mu_);
    for (auto& kv : conns_) {
      pfds->push_back({kv.second->fd, sock::kPollIn, 0});
      if (order) order->push_back(kv.second);
    }
  }

  /** True if any connection holds bytes not yet parsed (under recv_mu_). */
  bool AnyBuffered() {
    if (IsClient()) return conn_->reader.HasBuffered();
    std::lock_guard<std::mutex> lk(conns_mu_);
    for (auto& kv : conns_) {
      if (kv.second->reader.HasBuffered()) return true;
    }
    return false;
  }

  /** Accept every pending connection. */
  void AcceptPending() {
    while (true) {
      sock::socket_t fd = sock::Accept(listen_fd_);
      if (fd == sock::kInvalidSocket) return;
      if (protocol_ != "ipc") sock::SetTcpNoDelay(fd);
      sock::SetRecvBuf(fd, 4 * 1024 * 1024);
      sock::SetNonBlocking(fd, true);
      auto c = std::make_shared<SocketConn>();
      c->fd = fd;
      {
        std::lock_guard<std::mutex> lk(conns_mu_);
        c->id = next_conn_id_++;
        if (next_conn_id_ <= 0) next_conn_id_ = 1;  // stay positive on wrap
        conns_[c->id] = c;
      }
      if (em_) em_->AddEvent(static_cast<int>(fd), kDefaultReadEvent, nullptr);
    }
  }

  /** Server: the live connection with this id, or null. */
  std::shared_ptr<SocketConn> FindConn(int id) {
    std::lock_guard<std::mutex> lk(conns_mu_);
    auto it = conns_.find(id);
    return it == conns_.end() ? nullptr : it->second;
  }

  /** Server: forget a closed connection and close its socket. */
  void DropConn(const std::shared_ptr<SocketConn>& c) {
    {
      std::lock_guard<std::mutex> lk(conns_mu_);
      conns_.erase(c->id);
    }
    CloseConn(c);
  }

  /** Close a connection's socket once no send is using it. Unregister from
   *  the EventManager first so a recycled socket number never inherits the
   *  stale registration. */
  void CloseConn(const std::shared_ptr<SocketConn>& c) {
    if (!c) return;
    std::lock_guard<std::mutex> lk(c->send_mu);
    if (c->fd == sock::kInvalidSocket) return;
    if (em_) em_->RemoveEvent(static_cast<int>(c->fd));
    sock::Close(c->fd);
    c->fd = sock::kInvalidSocket;
    c->broken = true;
  }

  std::string addr_;
  std::string protocol_;
  int port_;
  sock::socket_t listen_fd_ = sock::kInvalidSocket;  ///< server: listener
  std::shared_ptr<SocketConn> conn_;                 ///< client: connection
  std::mutex conns_mu_;                              ///< guards conns_ / ids
  std::unordered_map<int, std::shared_ptr<SocketConn>> conns_;  ///< server
  int next_conn_id_ = 1;
  std::mutex recv_mu_;    ///< one Recv / reassembly pass at a time
  size_t rr_next_ = 0;    ///< round-robin start for ReadReady
  EventManager* em_ = nullptr;

  // Engine mode (TCP): frames arrive in inbox_; connections live there.
  bool engine_ = false;
  std::shared_ptr<sock::Inbox> inbox_;
  std::shared_ptr<sock::EngineConn> econn_;        ///< client connection
  std::shared_ptr<sock::EngineListener> elisten_;  ///< server listener
};

}  // namespace ctp::lbm
