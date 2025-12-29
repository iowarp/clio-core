/**
 * Task archive implementations for bulk transfer support
 * Contains SaveTaskArchive and LoadTaskArchive bulk() method implementations
 */

#include "chimaera/task_archives.h"
#include "chimaera/ipc_manager.h"
#include "hermes_shm/util/logging.h"

namespace chi {

/**
 * SaveTaskArchive bulk transfer implementation
 * Adds bulk descriptor to send vector with proper Expose handling
 * @param ptr Shared memory pointer to the data
 * @param size Size of the data in bytes
 * @param flags Transfer flags (BULK_XFER or BULK_EXPOSE)
 */
void SaveTaskArchive::bulk(hipc::ShmPtr<> ptr, size_t size, uint32_t flags) {
  hipc::FullPtr<char> full_ptr = CHI_IPC->ToFullPtr(ptr).template Cast<char>();
  hshm::lbm::Bulk bulk;
  bulk.data = full_ptr;
  bulk.size = size;
  bulk.flags.bits_ = flags;

  // If lbm_client is provided, automatically call Expose for RDMA registration
  if (lbm_client_) {
    bulk = lbm_client_->Expose(bulk.data, bulk.size, bulk.flags.bits_);
  }

  send.push_back(bulk);

  // Track count of BULK_XFER entries for proper ZMQ_SNDMORE handling
  if (flags & BULK_XFER) {
    send_bulks++;
  }
}

/**
 * LoadTaskArchive bulk transfer implementation
 * Handles both SerializeIn and SerializeOut modes
 * @param ptr Reference to shared memory pointer (output parameter for SerializeIn)
 * @param size Size of the data in bytes
 * @param flags Transfer flags (BULK_XFER or BULK_EXPOSE)
 */
void LoadTaskArchive::bulk(hipc::ShmPtr<> &ptr, size_t size, uint32_t flags) {
  HLOG(kDebug, "[LoadTaskArchive::bulk] Called with size={}, flags={}, msg_type_={}",
       size, flags, static_cast<int>(msg_type_));

  if (msg_type_ == MsgType::kSerializeIn) {
    // SerializeIn mode (input) - Get pointer from recv vector at current index
    // The task itself doesn't have a valid pointer during deserialization,
    // so we look into the recv vector and use the FullPtr at the current index
    if (current_bulk_index_ < recv.size()) {
      // Cast FullPtr<char>'s shm_ to ShmPtr<>
      ptr = recv[current_bulk_index_].data.shm_.template Cast<void>();
      current_bulk_index_++;
      HLOG(kDebug, "[LoadTaskArchive::bulk] SerializeIn - used recv[{}]", current_bulk_index_ - 1);
    } else {
      // Error: not enough bulk transfers in recv vector
      ptr = hipc::ShmPtr<>::GetNull();
      HLOG(kError, "[LoadTaskArchive::bulk] SerializeIn - recv vector empty or exhausted");
    }
  } else if (msg_type_ == MsgType::kSerializeOut) {
    // SerializeOut mode (output) - Expose the existing pointer using lbm_server
    // and append to recv vector for later retrieval
    HLOG(kDebug, "[LoadTaskArchive::bulk] SerializeOut - lbm_server_={}", (void*)lbm_server_);
    if (lbm_server_) {
      hipc::FullPtr<char> buffer = CHI_IPC->ToFullPtr(ptr).template Cast<char>();
      HLOG(kDebug, "[LoadTaskArchive::bulk] SerializeOut - buffer.ptr_={}", (void*)buffer.ptr_);
      hshm::lbm::Bulk bulk = lbm_server_->Expose(buffer, size, flags);
      recv.push_back(bulk);

      // Track count of BULK_XFER entries for proper ZMQ_RCVMORE handling
      if (flags & BULK_XFER) {
        recv_bulks++;
      }

      HLOG(kDebug, "[LoadTaskArchive::bulk] SerializeOut - added to recv, now has {} entries", recv.size());
    } else {
      // Error: lbm_server not set for output mode
      ptr = hipc::ShmPtr<>::GetNull();
      HLOG(kError, "[LoadTaskArchive::bulk] SerializeOut - lbm_server_ is null!");
    }
  }
  // kHeartbeat has no bulk transfers
}

} // namespace chi
