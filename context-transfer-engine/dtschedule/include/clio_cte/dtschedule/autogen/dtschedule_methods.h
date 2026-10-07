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

#ifndef CLIO_CTE_DTSCHEDULE_AUTOGEN_DTSCHEDULE_METHODS_H_
#define CLIO_CTE_DTSCHEDULE_AUTOGEN_DTSCHEDULE_METHODS_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/autogen/core_methods.h>
#include <string>
#include <vector>

/**
 * Method ids for the dtschedule chimod. Hand-maintained (same pattern as
 * replication, compressor, and CTE core). Keep in sync with clio_mod.yaml
 * and the switch cases in autogen/dtschedule_lib_exec.cc.
 *
 * This chimod INTERPOSES on the CTE core's own task interface: a
 * clio::cte::core::Client pointed at this pool keeps working unchanged.
 * Core verbs (kPutBlob, kGetBlob, etc.) MUST carry the core's method ids,
 * and other core methods are forwarded verbatim to the next pool by the
 * dispatch defaults. dtschedule's own verbs are numbered ABOVE the core's
 * method space.
 */
namespace clio::cte::dtschedule {

namespace Method {
GLOBAL_CROSS_CONST clio::run::u32 kCreate = 0;
GLOBAL_CROSS_CONST clio::run::u32 kDestroy = 1;
GLOBAL_CROSS_CONST clio::run::u32 kMonitor = 9;

// Interposed core data verbs — same ids, same task structs as the core.
GLOBAL_CROSS_CONST clio::run::u32 kPutBlob = clio::cte::core::Method::kPutBlob;
GLOBAL_CROSS_CONST clio::run::u32 kGetBlob = clio::cte::core::Method::kGetBlob;
GLOBAL_CROSS_CONST clio::run::u32 kGetBlobSize =
    clio::cte::core::Method::kGetBlobSize;
GLOBAL_CROSS_CONST clio::run::u32 kMultiPutBlob =
    clio::cte::core::Method::kMultiPutBlob;

// dtschedule-specific methods (above the core's id space)
GLOBAL_CROSS_CONST clio::run::u32 kPollNodeLoad = 110;
GLOBAL_CROSS_CONST clio::run::u32 kRegisterConsumer = 111;
GLOBAL_CROSS_CONST clio::run::u32 kCompressAt = 112;
GLOBAL_CROSS_CONST clio::run::u32 kGetDecisionStats = 113;
GLOBAL_CROSS_CONST clio::run::u32 kSetKnobs = 114;
GLOBAL_CROSS_CONST clio::run::u32 kSampleLoad = 115;

GLOBAL_CROSS_CONST clio::run::u32 kMaxMethodId = 116;

/**
 * Get human-readable method names for logging and debugging.
 *
 * @return Vector of method names indexed by method id
 */
inline const std::vector<std::string>& GetMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(kMaxMethodId);
    v[0] = "Create";
    v[1] = "Destroy";
    v[9] = "Monitor";
    v[kPutBlob] = "PutBlob";
    v[kGetBlob] = "GetBlob";
    v[kGetBlobSize] = "GetBlobSize";
    v[kMultiPutBlob] = "MultiPutBlob";
    v[110] = "PollNodeLoad";
    v[111] = "RegisterConsumer";
    v[112] = "CompressAt";
    v[113] = "GetDecisionStats";
    v[114] = "SetKnobs";
    v[115] = "SampleLoad";
    return v;
  }();
  return names;
}

}  // namespace Method

}  // namespace clio::cte::dtschedule

#endif  // CLIO_CTE_DTSCHEDULE_AUTOGEN_DTSCHEDULE_METHODS_H_
