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
#if CTP_ENABLE_THALLIUM
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <thallium.hpp>

#include "clio_ctp/util/logging.h"
#include "lightbeam.h"
#include "posix_socket.h"
#include "thallium_engine.h"

namespace ctp::lbm {

/** @return Name of the RPC every ThalliumTransport listener registers. */
inline const char *ThalliumLbmRpcName() { return "ctp_lbm_send"; }

/**
 * Allocator id tagging a receive buffer that this transport std::malloc'd.
 * It is the same sentinel SocketTransport uses, so LoadTaskArchive::bulk
 * copies the payload into a task-owned buffer and ClearRecvHandles frees the
 * malloc'd one (see context-runtime/src/task_archive.cc).
 * @return The sentinel allocator id.
 */
inline ctp::ipc::AllocatorId ThalliumRecvAllocId() {
  return ctp::ipc::AllocatorId(UINT32_MAX - 1, UINT32_MAX - 1);
}

/** std::free deleter for malloc'd receive buffers. */
struct ThalliumFree {
  /** @param p Buffer returned by std::malloc (may be null). */
  void operator()(char *p) const { std::free(p); }
};
using ThalliumBuf = std::unique_ptr<char, ThalliumFree>;

/** One BULK_XFER segment of a message, as it crosses the wire. */
struct ThalliumSeg {
  const char *src_ = nullptr; /**< sender: the caller's payload (not owned) */
  uint64_t size_ = 0;         /**< payload bytes */
  bool inline_ = true;        /**< true: bytes ride in the RPC; false: pulled */
  ThalliumBuf buf_;           /**< receiver: owned copy of the payload */
};

/**
 * RPC argument: the serialized lightbeam metadata plus every BULK_XFER
 * segment's size and, for segments below the bulk threshold, its bytes.
 * Segments at or above the threshold are not in this message; the receiver
 * pulls them through the tl::bulk handle passed as the second RPC argument.
 *
 * The sender serializes straight from the caller's buffers (no intermediate
 * copy) and the receiver reads each inline segment straight into the buffer
 * it hands to Recv().
 */
class ThalliumWire {
 public:
  const char *meta_src_ = nullptr; /**< sender: serialized metadata */
  uint64_t meta_len_ = 0;          /**< metadata bytes */
  std::vector<char> meta_;         /**< receiver: serialized metadata */
  std::vector<ThalliumSeg> segs_;  /**< BULK_XFER segments, in send order */

  /**
   * Encode the message (sender side).
   * @param ar Thallium output archive.
   */
  template <typename A>
  void save(A &ar) const {
    ar(meta_len_);
    ar.write(meta_src_, meta_len_);
    uint32_t count = static_cast<uint32_t>(segs_.size());
    ar(count);
    for (const ThalliumSeg &seg : segs_) {
      uint8_t is_inline = seg.inline_ ? 1 : 0;
      ar(seg.size_, is_inline);
      if (seg.inline_ && seg.size_ > 0) ar.write(seg.src_, seg.size_);
    }
  }

  /**
   * Decode the message (receiver side). Inline segments are read into
   * freshly malloc'd buffers; pulled segments are left without a buffer.
   * @param ar Thallium input archive.
   */
  template <typename A>
  void load(A &ar) {
    ar(meta_len_);
    meta_.resize(meta_len_);
    ar.read(meta_.data(), meta_len_);
    uint32_t count = 0;
    ar(count);
    segs_.resize(count);
    for (ThalliumSeg &seg : segs_) {
      uint8_t is_inline = 0;
      ar(seg.size_, is_inline);
      seg.inline_ = is_inline != 0;
      if (!seg.inline_) continue;
      seg.buf_.reset(
          static_cast<char *>(std::malloc(seg.size_ > 0 ? seg.size_ : 1)));
      if (!seg.buf_) throw std::bad_alloc();
      if (seg.size_ > 0) ar.read(seg.buf_.get(), seg.size_);
    }
  }

  /** @return Metadata plus payload bytes this message occupies in a queue. */
  size_t Bytes() const {
    size_t total = meta_len_;
    for (const ThalliumSeg &seg : segs_) total += seg.size_;
    return total;
  }
};

/** A complete received message waiting for Recv(). */
struct ThalliumRecvEntry {
  std::vector<char> meta_;        /**< serialized lightbeam metadata */
  std::vector<ThalliumSeg> segs_; /**< payload buffers, in send order */
  size_t bytes_ = 0;              /**< bytes charged against the queue */
};

/**
 * Bounded queue between the RPC handlers (Argobots ULTs) and the thread that
 * calls Recv()/PollRecv() (a plain thread). Handlers reserve room before
 * pulling a payload, so at most max_msgs messages / max_bytes bytes are
 * buffered per listener (one message is always admitted to an empty queue,
 * whatever its size). Critical sections are a few pointer moves; nothing
 * blocks while holding the lock.
 */
class ThalliumRecvQueue {
 public:
  /**
   * @param max_msgs Bound on queued plus in-flight messages.
   * @param max_bytes Bound on queued plus in-flight bytes.
   */
  ThalliumRecvQueue(size_t max_msgs, size_t max_bytes)
      : max_msgs_(max_msgs), max_bytes_(max_bytes) {}

  /**
   * Reserve room for one message.
   * @param bytes Size of the message.
   * @param closed Set to true if the queue was closed (listener destroyed).
   * @return true if the reservation was made.
   */
  bool TryReserve(size_t bytes, bool &closed) {
    std::lock_guard<std::mutex> lock(mtx_);
    closed = closed_;
    if (closed_) return false;
    if (q_.size() + reserved_ >= max_msgs_) return false;
    if (bytes_ != 0 && bytes_ + bytes > max_bytes_) return false;
    ++reserved_;
    bytes_ += bytes;
    return true;
  }

  /**
   * Give back a reservation whose message will not be queued.
   * @param bytes Size passed to TryReserve.
   */
  void Unreserve(size_t bytes) {
    std::lock_guard<std::mutex> lock(mtx_);
    --reserved_;
    bytes_ -= bytes;
  }

  /**
   * Queue a message whose room was reserved and wake a waiting receiver.
   * @param entry The message; entry.bytes_ must equal the reserved size.
   * @return false if the queue was closed meanwhile (entry is dropped).
   */
  bool Push(ThalliumRecvEntry &&entry) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      --reserved_;
      if (closed_) {
        bytes_ -= entry.bytes_;
        return false;
      }
      q_.emplace_back(std::move(entry));
    }
    cv_.notify_one();
    return true;
  }

  /**
   * Take the oldest message, if any.
   * @param entry Receives the message.
   * @return true if a message was returned.
   */
  bool Pop(ThalliumRecvEntry &entry) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (q_.empty()) return false;
    entry = std::move(q_.front());
    q_.pop_front();
    bytes_ -= entry.bytes_;
    return true;
  }

  /**
   * Block until a message is queued, the queue is closed or the timeout
   * expires.
   * @param timeout_ms Longest time to wait, in milliseconds.
   * @return 1 if a message is ready, 0 otherwise.
   */
  int WaitReadable(int timeout_ms) {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                 [this]() { return closed_ || !q_.empty(); });
    return q_.empty() ? 0 : 1;
  }

  /** Refuse further messages, drop the queued ones and wake any waiter. */
  void Close() {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      closed_ = true;
      q_.clear();
    }
    cv_.notify_all();
  }

 private:
  std::mutex mtx_;
  std::condition_variable cv_;
  std::deque<ThalliumRecvEntry> q_;
  size_t max_msgs_;
  size_t max_bytes_;
  size_t reserved_ = 0; /**< admitted messages still being pulled */
  size_t bytes_ = 0;    /**< bytes queued plus reserved */
  bool closed_ = false;
};

/**
 * Lightbeam transport over Thallium (Mercury RPC + bulk transfer).
 *
 * Server mode is a listener: it registers the send RPC under a provider id
 * equal to its port (so several listeners share the one engine/address) and
 * answers the TCP address exchange on that port. Client mode is a connection
 * to one listener; `addr` is either a host (resolved through the address
 * exchange at host:port) or a full Mercury address containing "://".
 *
 * Send() is synchronous: it returns once the receiver has pulled the payload
 * and queued the message, so the caller's buffers only need to live for the
 * duration of the call and messages from one sending thread arrive in order.
 * It may be called concurrently from any threads, including plain OS threads
 * that are not Argobots execution streams (Argobots >= 1.1 lets external
 * threads block on the eventual Margo waits on; Mercury's forward and bulk
 * calls are thread-safe).
 */
class ThalliumTransport : public Transport {
 public:
  /**
   * @param mode kServer for a listener, kClient for a connection.
   * @param addr Server: host/IP this node listens on. Client: peer host, or a
   *        full Mercury address ("proto://...") to skip the address exchange.
   * @param protocol Unused (the Mercury protocol comes from
   *        CLIO_LBM_THALLIUM_PROTOCOL); kept for factory compatibility.
   * @param port Listener port: provider id and address-exchange TCP port.
   *        0 = default provider, no address exchange (direct addresses only).
   * @throws std::exception if the engine cannot start, the port cannot be
   *         bound (server) or the peer cannot be resolved (client).
   */
  explicit ThalliumTransport(TransportMode mode, const std::string &addr,
                             const std::string &protocol = "tcp", int port = 0)
      : Transport(mode),
        addr_(addr),
        port_(port),
        direct_(addr.find("://") != std::string::npos),
        bulk_threshold_(ThalliumConfig::Get().bulk_threshold_),
        timeout_ms_(ThalliumConfig::Get().rpc_timeout_ms_) {
    (void)protocol;
    type_ = TransportType::kThallium;
    if (port < 0 || port >= 0xFFFF) {
      throw std::runtime_error("ThalliumTransport: port out of range: " +
                               std::to_string(port));
    }
    if (mode == TransportMode::kServer) {
      InitServer();
    } else {
      InitClient();
    }
  }

  /** Stop listening (server) and release the connection (client). */
  ~ThalliumTransport() {
    if (!IsServer()) return;
    if (port_ != 0) ThalliumEngine::Unpublish(port_);
    recv_q_->Close();
    if (!ThalliumEngine::IsUp()) return;
    try {
      rpc_.deregister();
    } catch (const std::exception &e) {
      HLOG(kWarning, "ThalliumTransport: deregister failed: {}", e.what());
    }
  }

  // ----- Shared APIs -----

  /**
   * Describe a buffer for a later Send. Nothing is registered here: memory
   * is registered with the fabric inside Send(), only for segments at or
   * above the bulk threshold and only for the duration of that call.
   * @param ptr Buffer.
   * @param data_size Buffer size in bytes.
   * @param flags BULK_XFER or BULK_EXPOSE.
   * @return The bulk descriptor.
   */
  Bulk Expose(const ctp::ipc::FullPtr<char> &ptr, size_t data_size, u32 flags) {
    Bulk bulk;
    bulk.data = ptr;
    bulk.size = data_size;
    bulk.flags = ctp::bitfield32_t(flags);
    return bulk;
  }

  /**
   * Free the payload buffers Recv() allocated for a message.
   * @param meta Metadata previously filled by Recv().
   */
  void ClearRecvHandles(LbmMeta<> &meta) {
    for (auto &bulk : meta.recv) {
      if (bulk.data.ptr_ &&
          bulk.data.shm_.alloc_id_ == ThalliumRecvAllocId()) {
        std::free(bulk.data.ptr_);
        bulk.data.ptr_ = nullptr;
      }
    }
  }

  /**
   * @return Server: this process's Mercury address. Client: the peer's
   *         Mercury address if resolved, else the address it was given.
   */
  std::string GetAddress() const {
    if (IsServer()) return ThalliumEngine::SelfAddress();
    std::lock_guard<std::mutex> lock(ep_mtx_);
    return peer_addr_.empty() ? addr_ : peer_addr_;
  }

  /** No-op: receivers block in PollRecv() instead of an EventManager. */
  void RegisterEventManager(EventManager &em) { (void)em; }

  /**
   * Set the size at or above which a segment is pulled by bulk transfer
   * instead of travelling inside the RPC (sender side, this connection only).
   * @param bytes New threshold; 0 pulls every non-empty segment.
   */
  void SetBulkThreshold(size_t bytes) { bulk_threshold_ = bytes; }

  /**
   * Check whether the peer listener answers.
   * @param ctx Unused.
   * @return Server: true. Client: whether the peer's address-exchange port
   *         accepts a connection (or, for a direct address, lookup succeeds).
   */
  bool IsServerAlive(const LbmContext &ctx = LbmContext()) const {
    (void)ctx;
    if (IsServer()) return true;
    if (!direct_) return sock::IsServerAlive(addr_, port_, "tcp");
    try {
      (void)ThalliumEngine::Get().lookup(addr_);
      return true;
    } catch (...) {
      return false;
    }
  }

  /**
   * Block until a message can be received or the timeout expires.
   * @param timeout_ms Longest time to wait, in milliseconds.
   * @return 1 if Recv() will return a message, 0 otherwise.
   */
  int PollRecv(int timeout_ms) {
    if (!IsServer()) return 0;
    return recv_q_->WaitReadable(timeout_ms);
  }

  // ----- Templated Send / Recv -----

  /**
   * Send one message and wait until the receiver has queued it.
   * @param meta Lightbeam metadata; meta.send lists the bulk segments.
   * @param ctx Unused.
   * @return 0 on success; EAGAIN if the receiver's queue stayed full (the
   *         message was not delivered, retry later); ETIMEDOUT if the peer
   *         did not answer within the RPC timeout; another non-zero value if
   *         the peer is unreachable or failed to take the message.
   */
  template <typename MetaT>
  int Send(MetaT &meta, const LbmContext &ctx = LbmContext()) {
    (void)ctx;
    if (!IsClient()) {
      HLOG(kError, "ThalliumTransport::Send called on a listener");
      return -1;
    }
    std::vector<char> meta_buf;
    {
      ctp::ipc::GlobalSerialize<std::vector<char>> ar(meta_buf);
      ar(meta);
      ar.Finalize();
    }
    ThalliumWire wire;
    wire.meta_src_ = meta_buf.data();
    wire.meta_len_ = meta_buf.size();
    std::vector<std::pair<void *, size_t>> pulled;
    if (!BuildSegments(meta, wire, pulled)) return -1;
    try {
      std::shared_ptr<tl::provider_handle> ph = GetHandle();
      // Registered only while the RPC is in flight; the receiver responds
      // after its pull completes, so the handle outlives every remote read.
      tl::bulk bulk;
      if (!pulled.empty()) {
        bulk = ThalliumEngine::Get().expose(pulled, tl::bulk_mode::read_only);
      }
      int rc = Forward(*ph, wire, bulk);
      if (rc != 0 && rc != EAGAIN && Invalidate()) {
        HLOG(kWarning, "ThalliumTransport::Send to {}:{} refused: rc={}",
             addr_, port_, rc);
      }
      return rc;
    } catch (const tl::timeout &) {
      if (Invalidate()) {
        HLOG(kWarning,
             "ThalliumTransport::Send to {}:{} timed out after {} ms", addr_,
             port_, timeout_ms_);
      }
      return ETIMEDOUT;
    } catch (const std::exception &e) {
      if (Invalidate()) {
        HLOG(kWarning, "ThalliumTransport::Send to {}:{} failed: {}", addr_,
             port_, e.what());
      }
      return -1;
    }
  }

  /**
   * Take the next complete message, without blocking.
   * @param meta Receives the metadata; meta.recv gets one entry per entry of
   *        meta.send, with payload buffers attached to the BULK_XFER ones.
   * @param ctx Unused.
   * @return ClientInfo with rc 0 on success, EAGAIN if nothing is queued,
   *         -1 if a queued message could not be decoded.
   */
  template <typename MetaT>
  ClientInfo Recv(MetaT &meta, const LbmContext &ctx = LbmContext()) {
    (void)ctx;
    ClientInfo info;
    ThalliumRecvEntry entry;
    if (!IsServer() || !recv_q_->Pop(entry)) {
      info.rc = EAGAIN;
      return info;
    }
    try {
      ctp::ipc::GlobalDeserialize<std::vector<char>> ar(entry.meta_);
      ar(meta);
      if (ar.Failed()) {  // short or foreign metadata (#1353)
        HLOG(kError, "ThalliumTransport::Recv: malformed metadata dropped");
        info.rc = -1;
        return info;
      }
    } catch (const std::exception &e) {
      HLOG(kError, "ThalliumTransport::Recv: bad metadata: {}", e.what());
      info.rc = -1;
      return info;
    }
    size_t xfer = 0;
    for (const auto &send_bulk : meta.send) {
      if (send_bulk.flags.Any(BULK_XFER)) ++xfer;
    }
    if (xfer != entry.segs_.size()) {
      HLOG(kError,
           "ThalliumTransport::Recv: metadata lists {} BULK_XFER segments "
           "but {} arrived",
           xfer, entry.segs_.size());
      info.rc = -1;
      return info;
    }
    AttachSegments(meta, entry);
    info.rc = 0;
    return info;
  }

 private:
  /** Start listening: register the RPC and publish the address. */
  void InitServer() {
    const ThalliumConfig &cfg = ThalliumConfig::Get();
    tl::engine &eng = ThalliumEngine::Get(addr_);
    recv_q_ = std::make_shared<ThalliumRecvQueue>(cfg.queue_msgs_,
                                                  cfg.queue_bytes_);
    std::shared_ptr<ThalliumRecvQueue> q = recv_q_;
    margo_instance_id mid = eng.get_margo_instance();
    auto handler = [q, mid](const tl::request &req, ThalliumWire &wire,
                            tl::bulk &remote) {
      req.respond(HandleSend(*q, mid, req, wire, remote));
    };
    {
      std::lock_guard<std::mutex> lock(ThalliumEngine::DefineMutex());
      rpc_ = eng.define(ThalliumLbmRpcName(), handler,
                        static_cast<uint16_t>(port_));
    }
    if (port_ != 0) {
      try {
        ThalliumEngine::Publish(port_);
      } catch (...) {
        rpc_.deregister();
        throw;
      }
    }
    HLOG(kInfo, "ThalliumTransport(server) {}:{} at {}", addr_, port_,
         ThalliumEngine::SelfAddress());
  }

  /** Connect: bind the RPC and resolve the peer's Mercury address. */
  void InitClient() {
    {
      tl::engine &eng = ThalliumEngine::Get();
      std::lock_guard<std::mutex> lock(ThalliumEngine::DefineMutex());
      rpc_ = eng.define(ThalliumLbmRpcName());
    }
    (void)GetHandle();
    HLOG(kDebug, "ThalliumTransport(client) {}:{} -> {}", addr_, port_,
         peer_addr_);
  }

  /**
   * RPC handler body (runs in an Argobots ULT on a handler ES). Reserves
   * queue room, pulls the bulk payload, queues the message.
   * @param q The listener's receive queue.
   * @param mid Margo instance.
   * @param req The request (source endpoint for the pull).
   * @param wire Decoded message; its buffers move into the queue.
   * @param remote Sender's bulk handle (null if nothing is pulled).
   * @return 0, or an errno-style code sent back as Send()'s result.
   */
  static int HandleSend(ThalliumRecvQueue &q, margo_instance_id mid,
                        const tl::request &req, ThalliumWire &wire,
                        tl::bulk &remote) {
    tl::engine eng(mid);
    size_t bytes = wire.Bytes();
    int rc = Admit(q, eng, bytes);
    if (rc != 0) return rc;
    rc = PullBulk(eng, req, wire, remote);
    if (rc != 0) {
      q.Unreserve(bytes);
      return rc;
    }
    ThalliumRecvEntry entry;
    entry.meta_ = std::move(wire.meta_);
    entry.segs_ = std::move(wire.segs_);
    entry.bytes_ = bytes;
    return q.Push(std::move(entry)) ? 0 : ESHUTDOWN;
  }

  /**
   * Reserve queue room for a message, waiting while the queue is full. The
   * wait is a Margo timer sleep: the ULT yields, so neither its execution
   * stream nor the progress loop is blocked, and nothing spins.
   * @param q The listener's receive queue.
   * @param eng Engine (for the timer).
   * @param bytes Message size.
   * @return 0 once reserved; EAGAIN if still full after the configured wait;
   *         ESHUTDOWN if the listener is closing.
   */
  static int Admit(ThalliumRecvQueue &q, const tl::engine &eng, size_t bytes) {
    static constexpr double kStepMs = 1.0;
    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::milliseconds(ThalliumConfig::Get().queue_wait_ms_);
    while (true) {
      bool closed = false;
      if (q.TryReserve(bytes, closed)) return 0;
      if (closed) return ESHUTDOWN;
      if (std::chrono::steady_clock::now() >= deadline) return EAGAIN;
      tl::thread::sleep(eng, kStepMs);
    }
  }

  /**
   * Pull every non-inline segment from the sender in one bulk transfer
   * (RDMA read on fabrics that support it) into freshly malloc'd buffers.
   * @param eng Engine.
   * @param req The request (source endpoint).
   * @param wire Message; buffers are attached to its non-inline segments.
   * @param remote Sender's bulk handle.
   * @return 0 on success, ENOMEM / EPROTO / EIO on failure.
   */
  static int PullBulk(tl::engine &eng, const tl::request &req,
                      ThalliumWire &wire, tl::bulk &remote) {
    std::vector<std::pair<void *, size_t>> segs;
    size_t total = 0;
    for (ThalliumSeg &seg : wire.segs_) {
      if (seg.inline_) continue;
      if (seg.size_ == 0) return EPROTO;
      seg.buf_.reset(static_cast<char *>(std::malloc(seg.size_)));
      if (!seg.buf_) return ENOMEM;
      segs.emplace_back(seg.buf_.get(), seg.size_);
      total += seg.size_;
    }
    if (segs.empty()) return 0;
    if (remote.is_null() || remote.size() != total) return EPROTO;
    try {
      tl::bulk local = eng.expose(segs, tl::bulk_mode::write_only);
      size_t got = remote.on(req.get_endpoint()) >> local;
      return got == total ? 0 : EIO;
    } catch (const std::exception &e) {
      HLOG(kError, "ThalliumTransport: bulk pull of {} bytes failed: {}",
           total, e.what());
      return EIO;
    }
  }

  /**
   * Split meta.send's BULK_XFER segments into inline and pulled ones.
   * @param meta Metadata being sent.
   * @param wire Receives one ThalliumSeg per BULK_XFER segment.
   * @param pulled Receives the (pointer, size) of each segment to expose.
   * @return false if a segment has bytes but no buffer.
   */
  template <typename MetaT>
  bool BuildSegments(MetaT &meta, ThalliumWire &wire,
                     std::vector<std::pair<void *, size_t>> &pulled) const {
    for (const auto &b : meta.send) {
      if (!b.flags.Any(BULK_XFER)) continue;
      if (b.size > 0 && b.data.ptr_ == nullptr) {
        HLOG(kError, "ThalliumTransport::Send: BULK_XFER segment of {} bytes "
             "has no buffer", b.size);
        return false;
      }
      ThalliumSeg seg;
      seg.src_ = b.data.ptr_;
      seg.size_ = b.size;
      seg.inline_ = b.size == 0 || b.size < bulk_threshold_;
      if (!seg.inline_) {
        pulled.emplace_back(static_cast<void *>(b.data.ptr_), b.size);
      }
      wire.segs_.emplace_back(std::move(seg));
    }
    return true;
  }

  /**
   * Issue the RPC and wait for the receiver's result.
   * @param ph Peer listener.
   * @param wire Message.
   * @param bulk Handle for the pulled segments (null if none).
   * @return The receiver's result code.
   * @throws tl::timeout on timeout, another exception on a transport error.
   */
  int Forward(const tl::provider_handle &ph, const ThalliumWire &wire,
              const tl::bulk &bulk) {
    if (timeout_ms_ > 0) {
      return rpc_.on(ph).timed(std::chrono::milliseconds(timeout_ms_), wire,
                               bulk);
    }
    return rpc_.on(ph)(wire, bulk);
  }

  /**
   * Hand a received message's payload buffers to the metadata.
   * @param meta Deserialized metadata; meta.recv is appended to.
   * @param entry The message; ownership of its buffers moves to meta.recv
   *        (released later by ClearRecvHandles).
   */
  template <typename MetaT>
  void AttachSegments(MetaT &meta, ThalliumRecvEntry &entry) {
    size_t idx = 0;
    for (const auto &send_bulk : meta.send) {
      Bulk recv_bulk;
      recv_bulk.size = send_bulk.size;
      recv_bulk.flags = send_bulk.flags;
      recv_bulk.data = ctp::ipc::FullPtr<char>::GetNull();
      if (send_bulk.flags.Any(BULK_XFER)) {
        ThalliumSeg &seg = entry.segs_[idx++];
        char *buf = seg.buf_.release();
        recv_bulk.size = seg.size_;
        recv_bulk.data.ptr_ = buf;
        recv_bulk.data.shm_.alloc_id_ = ThalliumRecvAllocId();
        recv_bulk.data.shm_.off_ = reinterpret_cast<size_t>(buf);
      }
      meta.recv.push_back(recv_bulk);
    }
  }

  /**
   * Get the peer's provider handle, resolving and looking up its address if
   * there is none (first use, or after a failed send dropped it).
   * @return The handle.
   * @throws std::runtime_error if the peer cannot be resolved or looked up.
   */
  std::shared_ptr<tl::provider_handle> GetHandle() {
    std::lock_guard<std::mutex> lock(ep_mtx_);
    if (ph_) return ph_;
    std::string target = addr_;
    if (!direct_) {
      std::string why;
      target = ThalliumEngine::Resolve(addr_, port_, stale_, &why);
      if (target.empty()) {
        throw std::runtime_error("ThalliumTransport: no address from " +
                                 addr_ + ":" + std::to_string(port_) +
                                 ": " + why);
      }
    }
    try {
      tl::endpoint ep = ThalliumEngine::Get().lookup(target);
      ph_ = std::make_shared<tl::provider_handle>(
          ep, static_cast<uint16_t>(port_));
    } catch (const std::exception &e) {
      stale_ = true;
      throw std::runtime_error("ThalliumTransport: lookup of " + target +
                               " failed: " + e.what());
    }
    peer_addr_ = target;
    stale_ = false;
    return ph_;
  }

  /**
   * Drop the cached peer handle after a failed send, so the next send asks
   * the peer for its address again (it may have restarted with a new one).
   * @return true if the connection was healthy until now, i.e. this is the
   *         first failure since the last success (callers log only then, so
   *         a peer that stays down does not flood the log).
   */
  bool Invalidate() {
    std::lock_guard<std::mutex> lock(ep_mtx_);
    ph_.reset();
    bool first = !stale_;
    stale_ = true;
    return first;
  }

  std::string addr_;
  int port_;
  bool direct_;            /**< addr_ is a full Mercury address */
  size_t bulk_threshold_;
  int timeout_ms_;
  tl::remote_procedure rpc_;
  std::shared_ptr<ThalliumRecvQueue> recv_q_;  /**< server mode only */
  mutable std::mutex ep_mtx_;                  /**< guards the fields below */
  std::shared_ptr<tl::provider_handle> ph_;    /**< client: peer handle */
  std::string peer_addr_;                      /**< client: resolved address */
  bool stale_ = false;                         /**< last send/lookup failed */
};

}  // namespace ctp::lbm
#endif  // CTP_ENABLE_THALLIUM
