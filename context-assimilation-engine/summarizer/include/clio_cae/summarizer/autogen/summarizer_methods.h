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

#ifndef CLIO_CAE_SUMMARIZER_AUTOGEN_METHODS_H_
#define CLIO_CAE_SUMMARIZER_AUTOGEN_METHODS_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/autogen/core_methods.h>
#include <string>
#include <vector>

/**
 * Method ids for the summarizer chimod. Hand-maintained (see clio_mod.yaml
 * for why this module is not fed to `clio_run repo refresh`); same
 * interposition rules as the cte cache / replication / indexer chimods: the
 * interposed core verb carries the CORE's id and task struct, every other
 * core id is forwarded verbatim by the dispatch defaults in
 * src/autogen/summarizer_lib_exec.cc, and any module verb of our own would
 * have to live ABOVE the core's id space (>= 100).
 */
namespace clio::cae::summarizer {

namespace Method {
GLOBAL_CROSS_CONST clio::run::u32 kCreate = 0;
GLOBAL_CROSS_CONST clio::run::u32 kDestroy = 1;
GLOBAL_CROSS_CONST clio::run::u32 kMonitor = 9;

// The one interposed core verb — same id, same task struct as the core.
// Summarization hangs off the write path: the blob forwards down the chain
// first, then a matching rule fires the LLM call.
GLOBAL_CROSS_CONST clio::run::u32 kPutBlob = clio::cte::core::Method::kPutBlob;

// No summarizer-specific verbs beyond Create/Destroy/Monitor. Sized to
// cover the whole core id space this container dispatches (forwarded ids
// index the per-method load model too), with the space above the core's
// reserved for future module verbs — same convention as the cache chimod.
GLOBAL_CROSS_CONST clio::run::u32 kMaxMethodId = 101;

inline const std::vector<std::string> &GetMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(kMaxMethodId);
    v[kCreate] = "Create";
    v[kDestroy] = "Destroy";
    v[kMonitor] = "Monitor";
    v[kPutBlob] = "PutBlob";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace clio::cae::summarizer

#endif  // CLIO_CAE_SUMMARIZER_AUTOGEN_METHODS_H_
