#pragma once
#if HSHM_ENABLE_ZMQ
#include <zmq.h>

#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <memory>
#include <sstream>
#include <unistd.h>

#include "lightbeam.h"

// Cereal serialization for Bulk
namespace cereal {
template <class Archive>
void serialize(Archive& ar, hshm::lbm::Bulk& bulk) {
  ar(bulk.size, bulk.flags);
}

template <class Archive>
void serialize(Archive& ar, hshm::lbm::LbmMeta& meta) {
  ar(meta.send, meta.recv);
}
}  // namespace cereal

namespace hshm::lbm {

// Lightbeam context flags for Send operations
constexpr uint32_t LBM_SYNC = 0x1;   /**< Synchronous send (wait for completion) */

/**
 * Context for lightbeam Send operations
 * Controls send behavior (sync vs async)
 */
struct LbmContext {
  uint32_t flags;              /**< Combination of LBM_* flags */

  LbmContext() : flags(0) {}

  explicit LbmContext(uint32_t f) : flags(f) {}

  bool IsSync() const { return flags & LBM_SYNC; }
};

class ZeroMqClient : public Client {
 public:
  explicit ZeroMqClient(const std::string& addr,
                        const std::string& protocol = "tcp", int port = 8192)
      : addr_(addr),
        protocol_(protocol),
        port_(port),
        ctx_(zmq_ctx_new()),
        socket_(zmq_socket(ctx_, ZMQ_PUSH)) {
    std::string full_url =
        protocol_ + "://" + addr_ + ":" + std::to_string(port_);
    int rc = zmq_connect(socket_, full_url.c_str());
    if (rc == -1) {
      std::string err = "ZeroMqClient failed to connect to URL '" + full_url +
                        "': " + zmq_strerror(zmq_errno());
      zmq_close(socket_);
      zmq_ctx_destroy(ctx_);
      throw std::runtime_error(err);
    }
  }

  ~ZeroMqClient() override {
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

  // Expose from raw pointer - calls base implementation
  Bulk Expose(const char* data, size_t data_size, u32 flags) override {
    hipc::FullPtr<char> ptr(const_cast<char*>(data), hipc::ShmPtr());
    return Expose(ptr, data_size, flags);
  }

  // Expose from hipc::Pointer - calls base implementation
  Bulk Expose(const hipc::Pointer& shm_ptr, size_t data_size, u32 flags) override {
    // For hipc::Pointer, we don't have the raw pointer, so we can't create a valid FullPtr
    // This case needs to be handled differently - the pointer needs to be resolved first
    hipc::FullPtr<char> ptr(nullptr, shm_ptr);
    return Expose(ptr, data_size, flags);
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

    // Count bulks marked for WRITE
    size_t write_bulk_count = 0;
    for (const auto& bulk : meta.send) {
      if (bulk.flags.Any(BULK_XFER)) {
        write_bulk_count++;
      }
    }

    // Determine send flags based on context
    // Use ZMQ_DONTWAIT only if LBM_SYNC is not set
    int base_flags = ctx.IsSync() ? 0 : ZMQ_DONTWAIT;

    // Send metadata - use ZMQ_SNDMORE only if there are WRITE bulks to follow
    int flags = base_flags;
    if (write_bulk_count > 0) {
      flags |= ZMQ_SNDMORE;
    }

    int rc = zmq_send(socket_, meta_str.data(), meta_str.size(), flags);
    if (rc == -1) {
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
    int rc = zmq_bind(socket_, full_url.c_str());
    if (rc == -1) {
      std::string err = "ZeroMqServer failed to bind to URL '" + full_url +
                        "': " + zmq_strerror(zmq_errno());
      zmq_close(socket_);
      zmq_ctx_destroy(ctx_);
      throw std::runtime_error(err);
    }
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

  // Expose from raw pointer - calls base implementation
  Bulk Expose(char* data, size_t data_size, u32 flags) override {
    hipc::FullPtr<char> ptr(data, hipc::ShmPtr());
    return Expose(ptr, data_size, flags);
  }

  // Expose from hipc::Pointer - calls base implementation
  Bulk Expose(const hipc::Pointer& shm_ptr, size_t data_size, u32 flags) override {
    // For hipc::Pointer, we don't have the raw pointer, so we can't create a valid FullPtr
    // This case needs to be handled differently - the pointer needs to be resolved first
    hipc::FullPtr<char> ptr(nullptr, shm_ptr);
    return Expose(ptr, data_size, flags);
  }

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
    try {
      std::string meta_str(static_cast<char*>(zmq_msg_data(&msg)),
                           zmq_msg_size(&msg));
      std::istringstream iss(meta_str, std::ios::binary);
      cereal::BinaryInputArchive ar(iss);
      ar(meta);
    } catch (const std::exception& e) {
      zmq_msg_close(&msg);
      return -1;  // Deserialization error
    }

    zmq_msg_close(&msg);
    return 0;  // Success
  }

  template <typename MetaT>
  int RecvBulks(MetaT& meta) {
    // Count bulks marked with BULK_XFER (only these will be received)
    size_t write_bulk_count = 0;
    for (const auto& bulk : meta.recv) {
      if (bulk.flags.Any(BULK_XFER)) {
        write_bulk_count++;
      }
    }

    // If no WRITE bulks, return immediately
    if (write_bulk_count == 0) {
      return 0;
    }

    // Receive only bulks marked with BULK_XFER
    size_t recv_count = 0;
    for (size_t i = 0; i < meta.recv.size(); ++i) {
      if (!meta.recv[i].flags.Any(BULK_XFER)) {
        continue;  // Skip bulks not marked for WRITE
      }

      int rc = zmq_recv(socket_, meta.recv[i].data.ptr_, meta.recv[i].size, 0);
      if (rc == -1) {
        return zmq_errno();
      }
      recv_count++;

      // Check if there are more message parts
      int more = 0;
      size_t more_size = sizeof(more);
      zmq_getsockopt(socket_, ZMQ_RCVMORE, &more, &more_size);

      // If this is the last expected WRITE bulk but more parts exist, it's an
      // error
      if (recv_count == write_bulk_count && more) {
        return -1;  // More parts than expected
      }

      // If we expect more WRITE bulks but no more parts, it's incomplete
      if (recv_count < write_bulk_count && !more) {
        return -1;  // Fewer parts than expected
      }
    }

    return 0;  // Success
  }

  std::string GetAddress() const override { return addr_; }

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