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

#ifndef CLIO_CTE_WRITE_THRU_CACHE_WRITE_THRU_CACHE_TASKS_H_
#define CLIO_CTE_WRITE_THRU_CACHE_WRITE_THRU_CACHE_TASKS_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/admin/admin_tasks.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/cache_base/cache_config.h>

namespace clio::cte::write_thru_cache {

/** Well-known default pool id/name of the write-through cache. */
static constexpr clio::run::PoolId kWriteThruCachePoolId(566, 0);
static constexpr const char *kWriteThruCachePoolName = "clio_cte_write_thru_cache";

/** Library-name traits for this module's config type. */
struct WriteThruTraits {
  static constexpr const char *kLibName = "clio_cte_write_thru_cache";
};

/** Module config: the shared policy cache params (see cache_config.h). */
using WriteThruCacheConfig =
    clio::cte::cache_base::PolicyCacheConfig<WriteThruTraits>;

/** Standard pool creation. */
using CreateTask = clio::run::admin::GetOrCreatePoolTask<WriteThruCacheConfig>;

}  // namespace clio::cte::write_thru_cache

#endif  // CLIO_CTE_WRITE_THRU_CACHE_WRITE_THRU_CACHE_TASKS_H_
