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

/**
 * Where a CTE blob lives: its placement hash, its owner container, and the
 * container that serves it while the owner's node is down (failover to the
 * first live successor, which the replication chimod's remote_copies keeps a
 * shadow copy on). Shared by the core's routing and the interposers.
 */
#ifndef CLIO_CTE_CORE_BLOB_PLACEMENT_H_
#define CLIO_CTE_CORE_BLOB_PLACEMENT_H_

#include <functional>
#include <string>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_tasks.h>

namespace clio::cte::core {

/**
 * The 32-bit hash that places a blob (owner = hash % num_containers).
 * @param tag_id blob's tag
 * @param blob_name blob name
 * @return hash
 */
inline clio::run::u32 BlobHash(const TagId &tag_id,
                               const std::string &blob_name) {
  std::hash<std::string> string_hasher;
  std::hash<clio::run::u32> u32_hasher;
  clio::run::u32 h = static_cast<clio::run::u32>(u32_hasher(tag_id.major_));
  h ^= static_cast<clio::run::u32>(u32_hasher(tag_id.minor_)) + 0x9e3779b9 +
       (h << 6) + (h >> 2);
  h ^= static_cast<clio::run::u32>(string_hasher(blob_name)) + 0x9e3779b9 +
       (h << 6) + (h >> 2);
  return h;
}

/**
 * Containers in a pool (0 if unknown here).
 * @param pool_id pool
 * @return container count
 */
inline clio::run::u32 PoolContainers(clio::run::PoolId pool_id) {
  auto *pm = CLIO_POOL_MANAGER;
  const clio::run::PoolInfo *info = pm->GetPoolInfo(pool_id);
  return info != nullptr ? info->num_containers_ : 0;
}

/**
 * Whether a container's node is alive.
 * @param pool_id pool
 * @param container container id
 * @return true if alive
 */
inline bool ContainerNodeAlive(clio::run::PoolId pool_id,
                               clio::run::u32 container) {
  auto *pm = CLIO_POOL_MANAGER;
  auto *ipc = CLIO_IPC;
  return ipc->IsAlive(pm->GetContainerNodeId(pool_id, container));
}

/**
 * The container serving the owner's blobs now: the owner while its node is
 * alive, else the first container after it whose node is.
 * @param pool_id pool
 * @param owner owner container
 * @return serving container
 */
inline clio::run::u32 FailoverContainer(clio::run::PoolId pool_id,
                                        clio::run::u32 owner) {
  const clio::run::u32 n = PoolContainers(pool_id);
  if (n <= 1 || ContainerNodeAlive(pool_id, owner)) return owner;
  for (clio::run::u32 i = 1; i < n; ++i) {
    const clio::run::u32 c = (owner + i) % n;
    if (ContainerNodeAlive(pool_id, c)) return c;
  }
  return owner;
}

}  // namespace clio::cte::core

#endif  // CLIO_CTE_CORE_BLOB_PLACEMENT_H_
