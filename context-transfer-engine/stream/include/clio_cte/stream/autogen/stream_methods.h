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

#ifndef CLIO_CTE_STREAM_AUTOGEN_STREAM_METHODS_H_
#define CLIO_CTE_STREAM_AUTOGEN_STREAM_METHODS_H_

#include <string>
#include <vector>

#include <clio_runtime/clio_runtime.h>

namespace clio::cte::stream {

namespace Method {
GLOBAL_CROSS_CONST clio::run::u32 kCreate = 0;
GLOBAL_CROSS_CONST clio::run::u32 kDestroy = 1;
GLOBAL_CROSS_CONST clio::run::u32 kMonitor = 9;
GLOBAL_CROSS_CONST clio::run::u32 kSizeOp = 10;
GLOBAL_CROSS_CONST clio::run::u32 kAppend = 11;
GLOBAL_CROSS_CONST clio::run::u32 kFlush = 12;
GLOBAL_CROSS_CONST clio::run::u32 kSequence = 13;
GLOBAL_CROSS_CONST clio::run::u32 kPlan = 14;
GLOBAL_CROSS_CONST clio::run::u32 kMaxMethodId = 15;

/** @return method-id -> name table (monitoring / logs). */
inline const std::vector<std::string> &GetMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(kMaxMethodId);
    v[kCreate] = "Create";
    v[kDestroy] = "Destroy";
    v[kMonitor] = "Monitor";
    v[kSizeOp] = "SizeOp";
    v[kAppend] = "Append";
    v[kFlush] = "Flush";
    v[kSequence] = "Sequence";
    v[kPlan] = "Plan";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace clio::cte::stream

#endif  // CLIO_CTE_STREAM_AUTOGEN_STREAM_METHODS_H_
