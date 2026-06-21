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

#ifndef CLIO_CAE_CORE_STRING_DATA_ASSIMILATOR_H_
#define CLIO_CAE_CORE_STRING_DATA_ASSIMILATOR_H_

#include <clio_cae/core/factory/base_assimilator.h>
#include <memory>
#include <string>

namespace clio::cte::core {
class Client;
}  // namespace clio::cte::core

namespace clio::cae::core {

/**
 * StringDataAssimilator - Assimilates an in-memory data payload carried on
 * the AssimilationCtx itself (no filesystem / network fetch).
 *
 * URL form:   src = "string::<blob_name>"
 *             dst = "iowarp::<tag_name>"
 *             ctx.src_data holds the raw bytes to store.
 *
 * Behavior: creates (or gets) the tag derived from `dst`, then stores a
 * single blob named `<blob_name>` whose contents are `ctx.src_data`. This
 * is the simple path agentic loops use to push generated Python objects
 * (already serialized to bytes by the caller) into Clio.
 */
class StringDataAssimilator : public BaseAssimilator {
 public:
  explicit StringDataAssimilator(
      std::shared_ptr<clio::cte::core::Client> cte_client);

  clio::run::TaskResume Schedule(const AssimilationCtx& ctx,
                           int& error_code) override;

 private:
  std::shared_ptr<clio::cte::core::Client> cte_client_;
};

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_STRING_DATA_ASSIMILATOR_H_
