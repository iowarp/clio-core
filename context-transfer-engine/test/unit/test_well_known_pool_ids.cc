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
 * The well-known CTE ChiMod pool ids must be distinct (issue #1186).
 *
 * Each module hard-codes the id its clients and the default config use; two
 * modules sharing one (checkpoint and stream both held 565.0) silently
 * resolve one module's get-or-create to the other's container. Checked at
 * compile time, over whichever optional modules this build includes.
 */

#include <clio_cte/core/core_tasks.h>
#ifdef CLIO_HAVE_FILESYSTEM
#include <clio_cte/filesystem/filesystem_tasks.h>
#endif
#ifdef CLIO_HAVE_REPLICATION
#include <clio_cte/replication/replication_tasks.h>
#endif
#ifdef CLIO_HAVE_CACHE
#include <clio_cte/cache/cache_tasks.h>
#endif
#ifdef CLIO_HAVE_INDEXER
#include <clio_cte/indexer/indexer_tasks.h>
#endif
#ifdef CLIO_HAVE_STREAM
#include <clio_cte/stream/stream_tasks.h>
#endif
#ifdef CLIO_HAVE_CHECKPOINT
#include <clio_cte/checkpoint/checkpoint_tasks.h>
#endif

#include <cstdio>

namespace {

/** Every well-known id this build knows about. */
constexpr clio::run::PoolId kWellKnownIds[] = {
    clio::cte::core::kCtePoolId,
#ifdef CLIO_HAVE_FILESYSTEM
    clio::cte::filesystem::kCfsPoolId,
#endif
#ifdef CLIO_HAVE_REPLICATION
    clio::cte::replication::kReplicationPoolId,
#endif
#ifdef CLIO_HAVE_CACHE
    clio::cte::cache::kCachePoolId,
#endif
#ifdef CLIO_HAVE_INDEXER
    clio::cte::indexer::kIndexerPoolId,
#endif
#ifdef CLIO_HAVE_STREAM
    clio::cte::stream::kStreamPoolId,
#endif
#ifdef CLIO_HAVE_CHECKPOINT
    clio::cte::checkpoint::kCheckpointPoolId,
#endif
};

/**
 * Whether no two entries of kWellKnownIds are equal.
 * @return true if every id is distinct
 */
constexpr bool AllDistinct() {
  constexpr size_t n = sizeof(kWellKnownIds) / sizeof(kWellKnownIds[0]);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      if (kWellKnownIds[i].major_ == kWellKnownIds[j].major_ &&
          kWellKnownIds[i].minor_ == kWellKnownIds[j].minor_) {
        return false;
      }
    }
  }
  return true;
}

static_assert(AllDistinct(),
              "two CTE ChiMods claim the same well-known pool id (#1186)");

}  // namespace

int main() {
  std::printf("%zu well-known CTE pool ids, all distinct\n",
              sizeof(kWellKnownIds) / sizeof(kWellKnownIds[0]));
  return 0;
}
