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

#include "clio_runtime/dynamic_container.h"

#include "clio_runtime/container.h"
#include "clio_runtime/module_manager.h"
#include "clio_runtime/singletons.h"

namespace clio::run {

// ---------------------------------------------------------------------------
// ContainerHold
// ---------------------------------------------------------------------------

ContainerHold::ContainerHold(const std::string &chimod_name,
                             const PoolId &pool_id,
                             const std::string &pool_name) {
  ModuleManager *module_manager = CLIO_MODULE_MANAGER;
  if (module_manager != nullptr) {
    ptr_ = module_manager->CreateContainer(chimod_name, pool_id, pool_name);
  }
}

void ContainerHold::Destroy(const std::string &chimod_name) {
  if (ptr_ != nullptr) {
    ModuleManager *module_manager = CLIO_MODULE_MANAGER;
    if (module_manager != nullptr) {
      module_manager->DestroyContainer(chimod_name, ptr_);
    }
    ptr_ = nullptr;
  }
}

// ---------------------------------------------------------------------------
// DynamicContainer
// ---------------------------------------------------------------------------

DynamicContainer::DynamicContainer(const std::string &chimod_name,
                                   const PoolId &pool_id,
                                   const std::string &pool_name)
    : hold_(chimod_name, pool_id, pool_name) {}

TaskResume DynamicContainer::Migrate(u32 dest_node) {
  CLIO_TASK_BODY_BEGIN
  // Dormant online-migration path: no concurrent migration is assumed, so there
  // is nothing to drain — just drive the container's own Migrate.
  if (hold_) {
    hold_->Migrate(dest_node);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END

  // ---------------------------------------------------------------------------
  // Original reader-draining implementation, preserved for when online
  // migration is re-enabled. It relied on the (now removed) shared
  // ContainerPtr + ctp::RwLock: the writer took the write side, which waits for
  // all live reader handles to drain before the container's state is moved, so
  // no reader is ever live mid-migration. While the writer waits IsWriteLocked()
  // (IsPlugged()) is true, so RouteLocal retries new tasks elsewhere.
  //
  //   if (!ptr_.IsNull()) {
  //     ctp::ScopedRwWriteLock wlock(ptr_->Lock(), 0);
  //     if (ptr_->get() != nullptr) {
  //       ptr_->get()->Migrate(dest_node);
  //     }
  //   }
  // ---------------------------------------------------------------------------
}

TaskResume DynamicContainer::Recover() {
  CLIO_TASK_BODY_BEGIN
  // Dormant online-upgrade/recover path. Kept so re-enabling online container
  // replacement later is a localized change; currently a no-op.
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END

  // ---------------------------------------------------------------------------
  // Original online-upgrade implementation (was DynamicContainer::Upgrade), the
  // version-swap counterpart to Migrate. Same drain-via-write-lock guarantee:
  // the writer waits for all live readers to drain, then atomically swaps the
  // underlying Container* so in-flight handles never observe a torn swap.
  //
  //   if (!ptr_.IsNull()) {
  //     ctp::ScopedRwWriteLock wlock(ptr_->Lock(), 0);
  //     ptr_->Set(next);   // `next` was the replacement Container*
  //   }
  // ---------------------------------------------------------------------------
}

}  // namespace clio::run
