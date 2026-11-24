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

#ifndef CTP_DATA_STRUCTURES_IPC_SHM_CONTAINER_H_
#define CTP_DATA_STRUCTURES_IPC_SHM_CONTAINER_H_

#include "clio_ctp/memory/allocator/allocator.h"
#include "clio_ctp/constants/macros.h"
#include <type_traits>

namespace ctp::ipc {

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
  CTP_CROSS_FUN
  ShmContainer() : this_(OffsetPtr<void>::GetNull()) {}

  /**
   * Constructor that stores allocator pointer
   *
   * @param alloc The allocator pointer to store
   */
  CTP_INLINE_CROSS_FUN
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
  CTP_INLINE_CROSS_FUN
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

}  // namespace ctp::ipc

#endif  // CTP_DATA_STRUCTURES_IPC_SHM_CONTAINER_H_
