/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HSHM_DATA_STRUCTURES_IPC_SHM_CONTAINER_H_
#define HSHM_DATA_STRUCTURES_IPC_SHM_CONTAINER_H_

#include "hermes_shm/memory/allocator/allocator.h"
#include "hermes_shm/constants/macros.h"
#include <type_traits>

namespace hshm::ipc {

/**
 * Base class for shared-memory containers.
 *
 * Provides a mechanism to store and retrieve an allocator pointer within
 * shared memory structures using offset pointers.
 *
 * @tparam AllocT The allocator type
 */
template<typename AllocT>
class ShmContainer {
 public:
  OffsetPtr<void> this_;  /**< Offset to allocator pointer */

  /**
   * Default constructor
   */
  HSHM_CROSS_FUN
  ShmContainer() : this_(OffsetPtr<void>::GetNull()) {}

  /**
   * Constructor that stores allocator pointer
   *
   * @param alloc The allocator pointer to store
   */
  HSHM_INLINE_CROSS_FUN
  explicit ShmContainer(AllocT *alloc) {
    if (alloc) {
      this_ = OffsetPtr<void>((size_t)this - (size_t)alloc);
    } else {
      this_ = OffsetPtr<void>::GetNull();
    }
  }

  /**
   * Retrieve the allocator pointer from the stored offset
   *
   * @return The allocator pointer
   */
  HSHM_INLINE_CROSS_FUN
  AllocT* GetAllocator() const {
    if (this_.IsNull()) {
      return nullptr;
    }
    return reinterpret_cast<AllocT*>((char*)this - this_.load());
  }
};

/**
 * Helper template to detect if a type has allocator_type member using SFINAE
 *
 * This template uses SFINAE to safely detect at compile-time if a type
 * has an allocator_type member.
 */
namespace {
  template<typename T, typename = void>
  struct HasAllocatorType : std::false_type {};

  template<typename T>
  struct HasAllocatorType<T, std::void_t<typename T::allocator_type>> : std::true_type {};

  template<typename T, typename = void>
  struct IsShmContainerHelper : std::false_type {};

  template<typename T>
  struct IsShmContainerHelper<T, std::enable_if_t<HasAllocatorType<T>::value &&
    std::is_base_of_v<ShmContainer<typename T::allocator_type>, T>>> : std::true_type {};
}

/**
 * Macro to detect if a type inherits from ShmContainer
 *
 * This macro uses SFINAE to safely detect at compile-time if a type
 * is derived from ShmContainer, avoiding compile errors on primitive types.
 */
#define IS_SHM_CONTAINER(T) (IsShmContainerHelper<T>::value)

}  // namespace hshm::ipc

#endif  // HSHM_DATA_STRUCTURES_IPC_SHM_CONTAINER_H_
