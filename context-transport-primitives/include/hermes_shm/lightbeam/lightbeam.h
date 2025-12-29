#pragma once
// Common types, interfaces, and factory for lightbeam transports.
// Users must include the appropriate transport header (zmq_transport.h)
// before using the factory for that transport.
#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "hermes_shm/memory/allocator/allocator.h"
#include "hermes_shm/types/bitfield.h"

namespace hshm::lbm {

// --- Bulk Flags ---
#define BULK_EXPOSE \
  BIT_OPT(hshm::u32, 0)                  // Bulk metadata sent, no data transfer
#define BULK_XFER BIT_OPT(hshm::u32, 1)  // Bulk marked for data transmission

// --- Types ---
struct Bulk {
  hipc::FullPtr<char> data;
  size_t size;
  hshm::bitfield32_t flags;  // BULK_EXPOSE or BULK_XFER
  void* desc = nullptr;      // For RDMA memory registration
  void* mr = nullptr;        // For RDMA memory region handle (fid_mr*)
  // Note: Cereal serialization is defined as non-member function in zmq_transport.h
};

// --- Metadata Base Class ---
class LbmMeta {
 public:
  std::vector<Bulk>
      send;  // Sender's bulk descriptors (can have BULK_EXPOSE or BULK_XFER)
  std::vector<Bulk>
      recv;  // Receiver's bulk descriptors (copy of send with local pointers)
  size_t send_bulks = 0;  // Count of BULK_XFER entries in send vector
  size_t recv_bulks = 0;  // Count of BULK_XFER entries in recv vector
};

// --- Interfaces ---
class Client {
 public:
  virtual ~Client() = default;

  // Expose from hipc::FullPtr
  virtual Bulk Expose(const hipc::FullPtr<char>& ptr, size_t data_size,
                      u32 flags) = 0;

  template <typename MetaT>
  int Send(MetaT& meta, const struct LbmContext& ctx);
};

class Server {
 public:
  virtual ~Server() = default;

  // Expose from hipc::FullPtr
  virtual Bulk Expose(const hipc::FullPtr<char>& ptr, size_t data_size,
                      u32 flags) = 0;

  /**
   * Receive and deserialize metadata from the network
   * @param meta The metadata structure to populate
   * @return 0 on success, EAGAIN if no message, -1 on deserialization error
   */
  template <typename MetaT>
  int RecvMetadata(MetaT& meta);

  /**
   * Receive bulk data into pre-allocated buffers
   * @param meta The metadata with recv buffers already populated
   * @return 0 on success, errno on failure
   */
  template <typename MetaT>
  int RecvBulks(MetaT& meta);

  virtual std::string GetAddress() const = 0;
};

// --- Transport Enum ---
enum class Transport { kZeroMq };

// --- Factory ---
class TransportFactory {
 public:
  // Users must include the correct transport header before calling these.
  static std::unique_ptr<Client> GetClient(const std::string& addr, Transport t,
                                           const std::string& protocol = "",
                                           int port = 0);
  static std::unique_ptr<Client> GetClient(const std::string& addr, Transport t,
                                           const std::string& protocol,
                                           int port, const std::string& domain);
  static std::unique_ptr<Server> GetServer(const std::string& addr, Transport t,
                                           const std::string& protocol = "",
                                           int port = 0);
  static std::unique_ptr<Server> GetServer(const std::string& addr, Transport t,
                                           const std::string& protocol,
                                           int port, const std::string& domain);
};

}  // namespace hshm::lbm