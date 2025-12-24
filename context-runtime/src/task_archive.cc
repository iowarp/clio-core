/**
 * Task archive implementations for bulk transfer support
 * Contains SaveTaskArchive and LoadTaskArchive bulk() method implementations
 */

#include "chimaera/task_archives.h"
#include "chimaera/ipc_manager.h"

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
}

/**
 * LoadTaskArchive bulk transfer implementation
 * Handles both SerializeIn and SerializeOut modes
 * @param ptr Reference to shared memory pointer (output parameter for SerializeIn)
 * @param size Size of the data in bytes
 * @param flags Transfer flags (BULK_XFER or BULK_EXPOSE)
 */
void LoadTaskArchive::bulk(hipc::ShmPtr<> &ptr, size_t size, uint32_t flags) {
  if (msg_type_ == MsgType::kSerializeIn) {
    // SerializeIn mode (input) - Get pointer from recv vector at current index
    // The task itself doesn't have a valid pointer during deserialization,
    // so we look into the recv vector and use the FullPtr at the current index
    if (current_bulk_index_ < recv.size()) {
      // Cast FullPtr<char>'s shm_ to ShmPtr<>
      ptr = recv[current_bulk_index_].data.shm_.template Cast<void>();
      current_bulk_index_++;
    } else {
      // Error: not enough bulk transfers in recv vector
      ptr = hipc::ShmPtr<>::GetNull();
    }
  } else if (msg_type_ == MsgType::kSerializeOut) {
    // SerializeOut mode (output) - Expose the existing pointer using lbm_server
    // and append to recv vector for later retrieval
    if (lbm_server_) {
      hipc::FullPtr<char> buffer = CHI_IPC->ToFullPtr(ptr).template Cast<char>();
      hshm::lbm::Bulk bulk = lbm_server_->Expose(buffer, size, flags);
      recv.push_back(bulk);
    } else {
      // Error: lbm_server not set for output mode
      ptr = hipc::ShmPtr<>::GetNull();
    }
  }
  // kHeartbeat has no bulk transfers
}

} // namespace chi
