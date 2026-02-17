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
#if HSHM_ENABLE_ZMQ
#include <unistd.h>
#include <zmq.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>

#include "hermes_shm/util/logging.h"
#include "lightbeam.h"

namespace hshm::lbm {

/** No-op free callback for zmq_msg_init_data zero-copy sends */
static inline void zmq_noop_free(void *data, void *hint) {
  (void)data;
  (void)hint;
}

/** Free zmq_msg_t handles stored in Bulk::desc from zero-copy recv */
static inline void ClearZmqRecvHandles(LbmMeta &meta) {
  for (auto &bulk : meta.recv) {
    if (bulk.desc) {
      zmq_msg_t *msg = static_cast<zmq_msg_t*>(bulk.desc);
      zmq_msg_close(msg);
      delete msg;
      bulk.desc = nullptr;
    }
  }
}

class ZeroMqTransport : public Transport {
 private:
  static void* GetSharedContext() {
    static void* shared_ctx = nullptr;
    static std::mutex ctx_mutex;

    std::lock_guard<std::mutex> lock(ctx_mutex);
    if (!shared_ctx) {
      shared_ctx = zmq_ctx_new();
      zmq_ctx_set(shared_ctx, ZMQ_IO_THREADS, 2);
      HLOG(kInfo, "[ZeroMqTransport] Created shared context with 2 I/O threads");
    }
    return shared_ctx;
  }

 public:
  explicit ZeroMqTransport(TransportMode mode, const std::string& addr,
                           const std::string& protocol = "tcp", int port = 8192)
      : Transport(mode),
        addr_(addr),
        protocol_(protocol),
        port_(port) {
    type_ = TransportType::kZeroMq;

    std::string full_url;
    if (protocol_ == "ipc") {
      full_url = "ipc://" + addr_;
    } else {
      full_url = protocol_ + "://" + addr_ + ":" + std::to_string(port_);
    }

    if (mode == TransportMode::kClient) {
      // DEALER socket for client
      ctx_ = GetSharedContext();
      owns_ctx_ = false;
      socket_ = zmq_socket(ctx_, ZMQ_DEALER);

      // Set identity for ROUTER identification
      // Use hostname + PID to ensure uniqueness across Docker containers
      // (where multiple processes may have the same PID in different namespaces)
      char hostname_buf[64] = {};
      gethostname(hostname_buf, sizeof(hostname_buf) - 1);
      uint32_t pid = static_cast<uint32_t>(getpid());
      std::string identity = std::string(hostname_buf) + ":" +
                              std::to_string(pid);
      zmq_setsockopt(socket_, ZMQ_IDENTITY, identity.data(),
                      identity.size());

      HLOG(kDebug, "ZeroMqTransport(DEALER) connecting to URL: {}", full_url);

      int immediate = 0;
      zmq_setsockopt(socket_, ZMQ_IMMEDIATE, &immediate, sizeof(immediate));

      int timeout = 5000;
      zmq_setsockopt(socket_, ZMQ_SNDTIMEO, &timeout, sizeof(timeout));

      int sndbuf = 4 * 1024 * 1024;
      zmq_setsockopt(socket_, ZMQ_SNDBUF, &sndbuf, sizeof(sndbuf));

      int rcvbuf = 4 * 1024 * 1024;
      zmq_setsockopt(socket_, ZMQ_RCVBUF, &rcvbuf, sizeof(rcvbuf));

      // ZMTP heartbeat: detect dead connections within seconds
      int hb_ivl = 1000;    // Send ZMTP PING every 1 second
      zmq_setsockopt(socket_, ZMQ_HEARTBEAT_IVL, &hb_ivl, sizeof(hb_ivl));
      int hb_timeout = 3000; // Consider dead after 3s of no traffic
      zmq_setsockopt(socket_, ZMQ_HEARTBEAT_TIMEOUT, &hb_timeout, sizeof(hb_timeout));
      int hb_ttl = 3000;     // Tell remote peer: drop me if no traffic for 3s
      zmq_setsockopt(socket_, ZMQ_HEARTBEAT_TTL, &hb_ttl, sizeof(hb_ttl));

      int rc = zmq_connect(socket_, full_url.c_str());
      if (rc == -1) {
        std::string err = "ZeroMqTransport(DEALER) failed to connect to URL '" +
                          full_url + "': " + zmq_strerror(zmq_errno());
        zmq_close(socket_);
        throw std::runtime_error(err);
      }

      zmq_pollitem_t poll_item = {socket_, 0, ZMQ_POLLOUT, 0};
      int poll_timeout_ms = 5000;
      int poll_rc = zmq_poll(&poll_item, 1, poll_timeout_ms);

      if (poll_rc < 0) {
        HLOG(kError, "[ZeroMqTransport] Poll failed for {}: {}", full_url,
             zmq_strerror(zmq_errno()));
      } else if (poll_rc == 0) {
        HLOG(kWarning,
             "[ZeroMqTransport] Poll timeout - connection to {} may not be ready",
             full_url);
      }

      HLOG(kDebug, "ZeroMqTransport(DEALER) connected to {} (pid={})", full_url, pid);
    } else {
      // ROUTER socket for server
      ctx_ = zmq_ctx_new();
      owns_ctx_ = true;
      zmq_ctx_set(ctx_, ZMQ_IO_THREADS, 2);
      socket_ = zmq_socket(ctx_, ZMQ_ROUTER);

      // Set mandatory routing - reject messages to unknown identities
      int mandatory = 1;
      zmq_setsockopt(socket_, ZMQ_ROUTER_MANDATORY, &mandatory, sizeof(mandatory));

      int rcvbuf = 4 * 1024 * 1024;
      zmq_setsockopt(socket_, ZMQ_RCVBUF, &rcvbuf, sizeof(rcvbuf));

      int sndbuf = 4 * 1024 * 1024;
      zmq_setsockopt(socket_, ZMQ_SNDBUF, &sndbuf, sizeof(sndbuf));

      // ZMTP heartbeat: detect dead client connections
      int hb_ivl = 1000;
      zmq_setsockopt(socket_, ZMQ_HEARTBEAT_IVL, &hb_ivl, sizeof(hb_ivl));
      int hb_timeout = 3000;
      zmq_setsockopt(socket_, ZMQ_HEARTBEAT_TIMEOUT, &hb_timeout, sizeof(hb_timeout));
      int hb_ttl = 3000;
      zmq_setsockopt(socket_, ZMQ_HEARTBEAT_TTL, &hb_ttl, sizeof(hb_ttl));

      HLOG(kDebug, "ZeroMqTransport(ROUTER) binding to URL: {}", full_url);
      int rc = zmq_bind(socket_, full_url.c_str());
      if (rc == -1) {
        std::string err = "ZeroMqTransport(ROUTER) failed to bind to URL '" +
                          full_url + "': " + zmq_strerror(zmq_errno());
        zmq_close(socket_);
        zmq_ctx_destroy(ctx_);
        throw std::runtime_error(err);
      }
      HLOG(kDebug, "ZeroMqTransport(ROUTER) bound successfully to {}", full_url);
    }
  }

  ~ZeroMqTransport() override {
    HLOG(kDebug, "ZeroMqTransport destructor - closing socket to {}:{}", addr_,
         port_);

    int linger = 5000;
    zmq_setsockopt(socket_, ZMQ_LINGER, &linger, sizeof(linger));

    zmq_close(socket_);
    if (owns_ctx_) {
      zmq_ctx_destroy(ctx_);
    }
    HLOG(kDebug, "ZeroMqTransport destructor - socket closed");
  }

  Bulk Expose(const hipc::FullPtr<char>& ptr, size_t data_size,
              u32 flags) override {
    Bulk bulk;
    bulk.data = ptr;
    bulk.size = data_size;
    bulk.flags = hshm::bitfield32_t(flags);
    return bulk;
  }

  template <typename MetaT>
  int Send(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    // Compute send_bulks before serialization so receiver knows how many
    meta.send_bulks = 0;
    for (size_t i = 0; i < meta.send.size(); ++i) {
      if (meta.send[i].flags.Any(BULK_XFER)) {
        meta.send_bulks++;
      }
    }

    std::ostringstream oss(std::ios::binary);
    {
      cereal::BinaryOutputArchive ar(oss);
      ar(meta);
    }
    std::string meta_str = oss.str();
    size_t write_bulk_count = meta.send_bulks;

    // ROUTER mode: prepend identity frame + empty delimiter
    if (IsServer() && !meta.client_info_.identity_.empty()) {
      // Send identity frame
      int rc = zmq_send(socket_, meta.client_info_.identity_.data(),
                        meta.client_info_.identity_.size(), ZMQ_SNDMORE);
      if (rc == -1) {
        HLOG(kError, "ZeroMqTransport::Send(ROUTER) - identity frame FAILED: {}",
             zmq_strerror(zmq_errno()));
        return zmq_errno();
      }
      // Send empty delimiter frame
      rc = zmq_send(socket_, "", 0, ZMQ_SNDMORE);
      if (rc == -1) {
        HLOG(kError, "ZeroMqTransport::Send(ROUTER) - delimiter frame FAILED: {}",
             zmq_strerror(zmq_errno()));
        return zmq_errno();
      }
    } else if (IsClient()) {
      // DEALER mode: send empty delimiter frame
      int rc = zmq_send(socket_, "", 0, ZMQ_SNDMORE);
      if (rc == -1) {
        HLOG(kError, "ZeroMqTransport::Send(DEALER) - delimiter frame FAILED: {}",
             zmq_strerror(zmq_errno()));
        return zmq_errno();
      }
    }

    int base_flags = 0;
    int flags = base_flags;
    if (write_bulk_count > 0) {
      flags |= ZMQ_SNDMORE;
    }

    int rc = zmq_send(socket_, meta_str.data(), meta_str.size(), flags);
    if (rc == -1) {
      HLOG(kError, "ZeroMqTransport::Send - meta FAILED: {}",
           zmq_strerror(zmq_errno()));
      return zmq_errno();
    }

    size_t sent_count = 0;
    for (size_t i = 0; i < meta.send.size(); ++i) {
      if (!meta.send[i].flags.Any(BULK_XFER)) {
        continue;
      }

      flags = base_flags;
      sent_count++;
      if (sent_count < write_bulk_count) {
        flags |= ZMQ_SNDMORE;
      }

      zmq_msg_t msg;
      zmq_msg_init_data(&msg, meta.send[i].data.ptr_, meta.send[i].size,
                         zmq_noop_free, nullptr);
      rc = zmq_msg_send(&msg, socket_, flags);
      if (rc == -1) {
        HLOG(kError, "ZeroMqTransport::Send - bulk {} FAILED: {}", i,
             zmq_strerror(zmq_errno()));
        zmq_msg_close(&msg);
        return zmq_errno();
      }
    }
    return 0;
  }

  template <typename MetaT>
  ClientInfo Recv(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    ClientInfo info;
    info.rc = RecvMetadata(meta, ctx);
    if (info.rc != 0) return info;
    // Copy identity from recv into ClientInfo
    info.identity_ = meta.client_info_.identity_;
    // Set up recv entries from send descriptors
    for (const auto& send_bulk : meta.send) {
      Bulk recv_bulk;
      recv_bulk.size = send_bulk.size;
      recv_bulk.flags = send_bulk.flags;
      recv_bulk.data = hipc::FullPtr<char>::GetNull();
      meta.recv.push_back(recv_bulk);
    }
    info.rc = RecvBulks(meta, ctx);
    return info;
  }

 private:
  template <typename MetaT>
  int RecvMetadata(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    (void)ctx;

    // ROUTER mode: receive identity frame first
    if (IsServer()) {
      zmq_msg_t identity_msg;
      zmq_msg_init(&identity_msg);
      int rc = zmq_msg_recv(&identity_msg, socket_, ZMQ_DONTWAIT);
      if (rc == -1) {
        int err = zmq_errno();
        zmq_msg_close(&identity_msg);
        return err;
      }
      // Store identity in meta for targeted Send responses
      meta.client_info_.identity_ = std::string(
          static_cast<char*>(zmq_msg_data(&identity_msg)),
          zmq_msg_size(&identity_msg));
      zmq_msg_close(&identity_msg);

      // Receive and discard empty delimiter frame
      zmq_msg_t delim_msg;
      zmq_msg_init(&delim_msg);
      rc = zmq_msg_recv(&delim_msg, socket_, 0);
      zmq_msg_close(&delim_msg);
      if (rc == -1) {
        return zmq_errno();
      }
    } else {
      // DEALER mode: receive and discard empty delimiter frame
      zmq_msg_t delim_msg;
      zmq_msg_init(&delim_msg);
      int rc = zmq_msg_recv(&delim_msg, socket_, ZMQ_DONTWAIT);
      if (rc == -1) {
        int err = zmq_errno();
        zmq_msg_close(&delim_msg);
        return err;
      }
      zmq_msg_close(&delim_msg);
    }

    // Receive metadata frame
    zmq_msg_t msg;
    zmq_msg_init(&msg);
    int rc = zmq_msg_recv(&msg, socket_, 0);

    if (rc == -1) {
      int err = zmq_errno();
      zmq_msg_close(&msg);
      return err;
    }

    size_t msg_size = zmq_msg_size(&msg);
    try {
      std::string meta_str(static_cast<char*>(zmq_msg_data(&msg)), msg_size);
      std::istringstream iss(meta_str, std::ios::binary);
      cereal::BinaryInputArchive ar(iss);
      ar(meta);
    } catch (const std::exception& e) {
      HLOG(kFatal,
           "ZeroMQ RecvMetadata: Deserialization failed - {} (msg_size={})",
           e.what(), msg_size);
      zmq_msg_close(&msg);
      return -1;
    }
    zmq_msg_close(&msg);
    return 0;
  }

  template <typename MetaT>
  int RecvBulks(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    (void)ctx;
    size_t recv_count = 0;
    for (size_t i = 0; i < meta.recv.size(); ++i) {
      if (!meta.recv[i].flags.Any(BULK_XFER)) {
        continue;
      }
      recv_count++;
      int flags = (recv_count < meta.send_bulks) ? ZMQ_RCVMORE : 0;

      if (meta.recv[i].data.ptr_) {
        zmq_msg_t zmq_msg;
        zmq_msg_init(&zmq_msg);
        int rc = zmq_msg_recv(&zmq_msg, socket_, flags);
        if (rc == -1) {
          int err = zmq_errno();
          zmq_msg_close(&zmq_msg);
          return err;
        }
        memcpy(meta.recv[i].data.ptr_,
               zmq_msg_data(&zmq_msg), meta.recv[i].size);
        zmq_msg_close(&zmq_msg);
      } else {
        zmq_msg_t *zmq_msg = new zmq_msg_t;
        zmq_msg_init(zmq_msg);
        int rc = zmq_msg_recv(zmq_msg, socket_, flags);
        if (rc == -1) {
          int err = zmq_errno();
          zmq_msg_close(zmq_msg);
          delete zmq_msg;
          return err;
        }
        char *zmq_data = static_cast<char*>(zmq_msg_data(zmq_msg));
        meta.recv[i].data.ptr_ = zmq_data;
        meta.recv[i].data.shm_.alloc_id_ = hipc::AllocatorId::GetNull();
        meta.recv[i].data.shm_.off_ = reinterpret_cast<size_t>(zmq_data);
        meta.recv[i].desc = zmq_msg;
      }
    }
    return 0;
  }

 public:
  void ClearRecvHandles(LbmMeta& meta) override {
    ClearZmqRecvHandles(meta);
  }

  void RegisterEventManager(EventManager &em) override {
    int fd = GetFd();
    if (fd >= 0) {
      em.AddEvent(fd, EPOLLIN);
    }
  }

  std::string GetAddress() const override { return addr_; }

  int GetFd() const override {
    int fd;
    size_t fd_size = sizeof(fd);
    zmq_getsockopt(socket_, ZMQ_FD, &fd, reinterpret_cast<::size_t *>(&fd_size));
    return fd;
  }

 private:
  std::string addr_;
  std::string protocol_;
  int port_;
  void* ctx_;
  bool owns_ctx_;
  void* socket_;
};

}  // namespace hshm::lbm

#endif  // HSHM_ENABLE_ZMQ
