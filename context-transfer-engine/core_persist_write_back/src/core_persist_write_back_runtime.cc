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

#include <clio_cte/core_persist_write_back/core_persist_write_back_runtime.h>

#include <string>
#include <vector>

namespace clio::cte::core_persist_write_back {

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    // The task is NOT a CreateTask, whatever the parameter says: compose
    // hands this ChiMod a ComposeTask<PoolConfig>, a direct caller its own
    // instantiation. All of them ARE a CreatePoolFields, so every read and
    // write goes through that (see CreatePoolFields::GetParamsAs). Calling a
    // member through `task` is undefined behaviour, which UBSan reports.
    auto &fields = task.template Cast<clio::run::admin::CreatePoolFields>();
    const WriteBackCacheConfig params = fields->GetParamsAs<WriteBackCacheConfig>();
    if (params.volatile_pool_.IsNull()) {
      HLOG(kError, "clio_cte_core_persist_write_back: next_volatile is required");
      fields->SetReturnCode(1);
      CLIO_CO_RETURN;
    }
    InitCache(params, fields->new_pool_id_);
    fields->SetReturnCode(0);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// Shared container dispatch (Run / Save / Load / NewTask / Aggregate).
#include <clio_cte/cache_base/cache_lib_exec.inc>

}  // namespace clio::cte::core_persist_write_back

// ChiMod entry points (alloc/new/name/destroy) for the module manager.
CLIO_TASK_CC(clio::cte::core_persist_write_back::Runtime)
