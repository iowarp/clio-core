#include "hermes_shm/memory/allocator/malloc_allocator.h"

namespace hshm::ipc {

// Explicit template instantiations
template class _MallocAllocator<false>;
template class _MallocAllocator<true>;

}  // namespace hshm::ipc
