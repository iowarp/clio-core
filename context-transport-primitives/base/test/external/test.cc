//
// Created by lukemartinlogan on 6/6/23.
//

#include "clio_ctp/clio_ctp.h"

int main() {
  std::string shm_url = "test_serializers";
  ctp::ipc::AllocatorId alloc_id(1, 0);
  auto mem_mngr = CTP_MEMORY_MANAGER;
  mem_mngr->UnregisterAllocator(alloc_id);
  mem_mngr->UnregisterBackend(ctp::ipc::MemoryBackendId::GetRoot());
  mem_mngr->CreateBackend<ctp::ipc::PosixShmMmap>(
      ctp::ipc::MemoryBackendId::GetRoot(), ctp::Unit<size_t>::Megabytes(100),
      shm_url);
  mem_mngr->CreateAllocator<ctp::ipc::StackAllocator>(ctp::ipc::MemoryBackendId::GetRoot(),
                                                  alloc_id, 0);
}
