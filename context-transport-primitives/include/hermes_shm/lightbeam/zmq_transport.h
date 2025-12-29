#pragma once
#if HSHM_ENABLE_ZMQ
#include <unistd.h>
#include <zmq.h>

#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

#include "hermes_shm/util/logging.h"
#include "lightbeam.h"

// Cereal serialization for Bulk
// Note: data is transferred separately via bulk transfer mechanism, not
// serialized here
namespace cereal {
template <class Archive>
void serialize(Archive& ar, hshm::lbm::Bulk& bulk) {
  ar(bulk.size, bulk.flags);
}

template <class Archive>
void serialize(Archive& ar, hshm::lbm::LbmMeta& meta) {
  ar(meta.send, meta.recv, meta.send_bulks, meta.recv_bulks);
}
}  // namespace cereal

namespace hshm::lbm {

// Lightbeam context flags for Send operations
constexpr uint32_t LBM_SYNC =
    0x1; /**< Synchronous send (wait for completion) */

/**
 * Context for lightbeam Send operations
 * Controls send behavior (sync vs async)
 */
struct LbmContext {
  uint32_t flags; /**< Combination of LBM_* flags */

  LbmContext() : flags(0) {}

  explicit LbmContext(uint32_t f) : flags(f) {}

  bool IsSync() const { return flags & LBM_SYNC; }
};

class ZeroMqClient : public Client {
 private:
  /**
   * Get or create the shared ZeroMQ context for all clients
   * Uses a static local variable for thread-safe singleton initialization
   */
  static void* GetSharedContext() {
    static void* shared_ctx = nullptr;
    static std::mutex ctx_mutex;

    std::lock_guard<std::mutex> lock(ctx_mutex);
    if (!shared_ctx) {
      shared_ctx = zmq_ctx_new();
      // Set I/O threads to 2 for better throughput
      zmq_ctx_set(shared_ctx, ZMQ_IO_THREADS, 2);
      HLOG(kInfo, "[ZeroMqClient] Created shared context with 2 I/O threads");
    }
    return shared_ctx;
  }

 public:
  explicit ZeroMqClient(const std::string& addr,
                        const std::string& protocol = "tcp", int port = 8192)
      : addr_(addr),
        protocol_(protocol),
        port_(port),
        ctx_(GetSharedContext()),
        owns_ctx_(false),
        socket_(zmq_socket(ctx_, ZMQ_PUSH)) {
    std::string full_url =
        protocol_ + "://" + addr_ + ":" + std::to_string(port_);
    HLOG(kDebug, "ZeroMqClient connecting to URL: {}", full_url);

    // Disable ZMQ_IMMEDIATE - let messages queue until connection is
    // established With ZMQ_IMMEDIATE=1, messages may be dropped if no peer is
    // immediately available
    int immediate = 0;
    zmq_setsockopt(socket_, ZMQ_IMMEDIATE, &immediate, sizeof(immediate));

    // Set a reasonable send timeout (5 seconds)
    int timeout = 5000;
    zmq_setsockopt(socket_, ZMQ_SNDTIMEO, &timeout, sizeof(timeout));

    int rc = zmq_connect(socket_, full_url.c_str());
    if (rc == -1) {
      std::string err = "ZeroMqClient failed to connect to URL '" + full_url +
                        "': " + zmq_strerror(zmq_errno());
      zmq_close(socket_);
      throw std::runtime_error(err);
    }

    // Wait for socket to become writable (connection established)
    // zmq_connect is asynchronous, so we use poll to verify readiness
    zmq_pollitem_t poll_item = {socket_, 0, ZMQ_POLLOUT, 0};
    int poll_timeout_ms = 5000;  // 5 second timeout for connection
    int poll_rc = zmq_poll(&poll_item, 1, poll_timeout_ms);

    if (poll_rc < 0) {
      HLOG(kError, "[ZeroMqClient] Poll failed for {}: {}", full_url,
           zmq_strerror(zmq_errno()));
    } else if (poll_rc == 0) {
      HLOG(kWarning,
           "[ZeroMqClient] Poll timeout - connection to {} may not be ready",
           full_url);
    } else if (poll_item.revents & ZMQ_POLLOUT) {
      HLOG(kInfo, "[ZeroMqClient] Socket ready for writing to {}", full_url);
    }

    HLOG(kDebug, "ZeroMqClient connected to {} (poll_rc={})", full_url,
         poll_rc);
  }

  ~ZeroMqClient() override {
    HLOG(kDebug, "ZeroMqClient destructor - closing socket to {}:{}", addr_,
         port_);

    // Set linger to ensure any remaining messages are sent
    int linger = 5000;
    zmq_setsockopt(socket_, ZMQ_LINGER, &linger, sizeof(linger));

    zmq_close(socket_);
    // Don't destroy the shared context - it's shared across all clients
    HLOG(kDebug, "ZeroMqClient destructor - socket closed");
  }

  // Base Expose implementation - accepts hipc::FullPtr
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
    // Serialize metadata (includes both send and recv vectors)
    std::ostringstream oss(std::ios::binary);
    {
      cereal::BinaryOutputArchive ar(oss);
      ar(meta);
    }
    std::string meta_str = oss.str();

    // Use pre-computed send_bulks count for ZMQ_SNDMORE handling
    size_t write_bulk_count = meta.send_bulks;

    // IMPORTANT: Always use blocking send for distributed messaging
    // ZMQ_DONTWAIT with newly-created connections causes messages to be lost
    // because the connection may not be established when send is called
    int base_flags = 0;  // Use blocking sends

    // Send metadata - use ZMQ_SNDMORE only if there are WRITE bulks to follow
    int flags = base_flags;
    if (write_bulk_count > 0) {
      flags |= ZMQ_SNDMORE;
    }

    int rc = zmq_send(socket_, meta_str.data(), meta_str.size(), flags);
    if (rc == -1) {
      HLOG(kError, "ZeroMqClient::Send - FAILED: {}",
           zmq_strerror(zmq_errno()));
      return zmq_errno();
    }

    // Send only bulks marked with BULK_XFER
    size_t sent_count = 0;
    for (size_t i = 0; i < meta.send.size(); ++i) {
      if (!meta.send[i].flags.Any(BULK_XFER)) {
        continue;  // Skip bulks not marked for WRITE
      }

      flags = base_flags;
      sent_count++;
      if (sent_count < write_bulk_count) {
        flags |= ZMQ_SNDMORE;
      }

      rc = zmq_send(socket_, meta.send[i].data.ptr_, meta.send[i].size, flags);
      if (rc == -1) {
        HLOG(kError, "ZeroMqClient::Send - bulk {} FAILED: {}", i,
             zmq_strerror(zmq_errno()));
        return zmq_errno();
      }
    }
    return 0;  // Success
  }

 private:
  std::string addr_;
  std::string protocol_;
  int port_;
  void* ctx_;
  bool owns_ctx_;  // Whether this client owns the context (should destroy on
                   // cleanup)
  void* socket_;
};

class ZeroMqServer : public Server {
 public:
  explicit ZeroMqServer(const std::string& addr,
                        const std::string& protocol = "tcp", int port = 8192)
      : addr_(addr),
        protocol_(protocol),
        port_(port),
        ctx_(zmq_ctx_new()),
        socket_(zmq_socket(ctx_, ZMQ_PULL)) {
    std::string full_url =
        protocol_ + "://" + addr_ + ":" + std::to_string(port_);
    HLOG(kDebug, "ZeroMqServer binding to URL: {}", full_url);
    int rc = zmq_bind(socket_, full_url.c_str());
    if (rc == -1) {
      std::string err = "ZeroMqServer failed to bind to URL '" + full_url +
                        "': " + zmq_strerror(zmq_errno());
      zmq_close(socket_);
      zmq_ctx_destroy(ctx_);
      throw std::runtime_error(err);
    }
    HLOG(kDebug, "ZeroMqServer bound successfully to {} (socket={})", full_url,
         reinterpret_cast<uintptr_t>(socket_));
  }

  ~ZeroMqServer() override {
    zmq_close(socket_);
    zmq_ctx_destroy(ctx_);
  }

  // Base Expose implementation - accepts hipc::FullPtr
  Bulk Expose(const hipc::FullPtr<char>& ptr, size_t data_size,
              u32 flags) override {
    Bulk bulk;
    bulk.data = ptr;
    bulk.size = data_size;
    bulk.flags = hshm::bitfield32_t(flags);
    return bulk;
  }

  /**
   * Receive and deserialize metadata from the network
   * @param meta The metadata structure to populate
   * @return 0 on success, EAGAIN if no message, -1 on deserialization error
   */
  template <typename MetaT>
  int RecvMetadata(MetaT& meta) {
    // Receive metadata message (non-blocking)
    zmq_msg_t msg;
    zmq_msg_init(&msg);
    int rc = zmq_msg_recv(&msg, socket_, ZMQ_DONTWAIT);

    if (rc == -1) {
      int err = zmq_errno();
      zmq_msg_close(&msg);
      return err;
    }

    // Deserialize metadata
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
      return -1;  // Deserialization error
    }
    zmq_msg_close(&msg);
    return 0;  // Success
  }

  /**
   * Receive bulk data into pre-allocated buffers
   * Uses meta.send_bulks (from sender's metadata) to know exact count
   * @param meta The metadata with recv buffers already populated
   * @return 0 on success, errno on failure
   */
  template <typename MetaT>
  int RecvBulks(MetaT& meta) {
    size_t recv_count = 0;
    for (size_t i = 0; i < meta.recv.size(); ++i) {
      if (!meta.recv[i].flags.Any(BULK_XFER)) {
        continue;
      }
      recv_count++;
      // Use ZMQ_RCVMORE if more bulks remain
      int flags = (recv_count < meta.send_bulks) ? ZMQ_RCVMORE : 0;
      int rc = zmq_recv(socket_, meta.recv[i].data.ptr_, meta.recv[i].size, flags);
      if (rc == -1) {
        return zmq_errno();
      }
    }
    return 0;  // Success
  }

  std::string GetAddress() const override { return addr_; }

  /**
   * Get the file descriptor for the ZeroMQ socket
   * Can be used with epoll for efficient event-driven I/O
   * @return File descriptor for the socket
   */
  int GetFd() const {
    int fd;
    size_t fd_size = sizeof(fd);
    zmq_getsockopt(socket_, ZMQ_FD, &fd, &fd_size);
    return fd;
  }

 private:
  std::string addr_;
  std::string protocol_;
  int port_;
  void* ctx_;
  void* socket_;
};

// --- Base Class Template Implementations ---
// These delegate to the derived class implementations
template <typename MetaT>
int Client::Send(MetaT& meta, const LbmContext& ctx) {
  // Forward to ZeroMqClient implementation with provided context
  return static_cast<ZeroMqClient*>(this)->Send(meta, ctx);
}

template <typename MetaT>
int Server::RecvMetadata(MetaT& meta) {
  return static_cast<ZeroMqServer*>(this)->RecvMetadata(meta);
}

template <typename MetaT>
int Server::RecvBulks(MetaT& meta) {
  return static_cast<ZeroMqServer*>(this)->RecvBulks(meta);
}

// --- TransportFactory Implementations ---
inline std::unique_ptr<Client> TransportFactory::GetClient(
    const std::string& addr, Transport t, const std::string& protocol,
    int port) {
  if (t == Transport::kZeroMq) {
    return std::make_unique<ZeroMqClient>(addr, protocol, port);
  }
  throw std::runtime_error("Unsupported transport type");
}

inline std::unique_ptr<Client> TransportFactory::GetClient(
    const std::string& addr, Transport t, const std::string& protocol, int port,
    const std::string& domain) {
  if (t == Transport::kZeroMq) {
    return std::make_unique<ZeroMqClient>(addr, protocol, port);
  }
  throw std::runtime_error("Unsupported transport type");
}

inline std::unique_ptr<Server> TransportFactory::GetServer(
    const std::string& addr, Transport t, const std::string& protocol,
    int port) {
  if (t == Transport::kZeroMq) {
    return std::make_unique<ZeroMqServer>(addr, protocol, port);
  }
  throw std::runtime_error("Unsupported transport type");
}

inline std::unique_ptr<Server> TransportFactory::GetServer(
    const std::string& addr, Transport t, const std::string& protocol, int port,
    const std::string& domain) {
  if (t == Transport::kZeroMq) {
    return std::make_unique<ZeroMqServer>(addr, protocol, port);
  }
  throw std::runtime_error("Unsupported transport type");
}

}  // namespace hshm::lbm

#endif  // HSHM_ENABLE_ZMQ