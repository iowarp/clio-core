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

#ifndef CLIO_CTE_INDEXER_AUTOGEN_METHODS_H_
#define CLIO_CTE_INDEXER_AUTOGEN_METHODS_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/autogen/core_methods.h>
#include <string>
#include <vector>

/**
 * Method ids for the indexer chimod. Hand-maintained; same interposition
 * rules as the cache/replication/compressor chimods: the interposed core
 * verbs carry the CORE's ids and task structs, every other core id is
 * forwarded verbatim by the dispatch defaults, and module verbs live ABOVE
 * the core's id space.
 */
namespace clio::cte::indexer {

namespace Method {
GLOBAL_CROSS_CONST clio::run::u32 kCreate = 0;
GLOBAL_CROSS_CONST clio::run::u32 kDestroy = 1;
GLOBAL_CROSS_CONST clio::run::u32 kMonitor = 9;

// Interposed core verbs — same ids, same task structs as the core.
// The mutating data verbs are intercepted to keep the index current;
// kSemanticSearch TERMINATES here (the core no longer implements it).
GLOBAL_CROSS_CONST clio::run::u32 kPutBlob = clio::cte::core::Method::kPutBlob;
GLOBAL_CROSS_CONST clio::run::u32 kDelBlob = clio::cte::core::Method::kDelBlob;
GLOBAL_CROSS_CONST clio::run::u32 kDelTag = clio::cte::core::Method::kDelTag;
GLOBAL_CROSS_CONST clio::run::u32 kSemanticSearch =
    clio::cte::core::Method::kSemanticSearch;
GLOBAL_CROSS_CONST clio::run::u32 kTruncateBlob =
    clio::cte::core::Method::kTruncateBlob;
GLOBAL_CROSS_CONST clio::run::u32 kRenameTag =
    clio::cte::core::Method::kRenameTag;
GLOBAL_CROSS_CONST clio::run::u32 kMultiPutBlob =
    clio::cte::core::Method::kMultiPutBlob;

// Module verbs live ABOVE the core's id space (>= 100, interposer rule).
// kIndexSweep drives the asynchronous index drain (periodic task);
// kReindexScan is the explicit on-demand backfill of existing data.
GLOBAL_CROSS_CONST clio::run::u32 kIndexSweep = 100;
GLOBAL_CROSS_CONST clio::run::u32 kReindexScan = 101;
GLOBAL_CROSS_CONST clio::run::u32 kMaxMethodId = 102;

inline const std::vector<std::string>& GetMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(kMaxMethodId);
    v[0] = "Create";
    v[1] = "Destroy";
    v[9] = "Monitor";
    v[kPutBlob] = "PutBlob";
    v[kDelBlob] = "DelBlob";
    v[kDelTag] = "DelTag";
    v[kSemanticSearch] = "SemanticSearch";
    v[kTruncateBlob] = "TruncateBlob";
    v[kRenameTag] = "RenameTag";
    v[kMultiPutBlob] = "MultiPutBlob";
    v[kIndexSweep] = "IndexSweep";
    v[kReindexScan] = "ReindexScan";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace clio::cte::indexer

#endif  // CLIO_CTE_INDEXER_AUTOGEN_METHODS_H_
