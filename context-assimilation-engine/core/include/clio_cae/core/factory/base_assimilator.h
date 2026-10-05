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

#ifndef CLIO_CAE_CORE_BASE_ASSIMILATOR_H_
#define CLIO_CAE_CORE_BASE_ASSIMILATOR_H_

#include <clio_cae/core/factory/assimilation_ctx.h>
#include <clio_cae/core/factory/page_map.h>
#include <clio_runtime/task.h>

namespace clio::cae::core {

/**
 * BaseAssimilator - Abstract interface for data assimilators
 * Concrete implementations handle different data sources (file, URL, etc.)
 *
 * NOTE: Schedule is a coroutine that must be co_awaited from runtime code.
 * The error code is returned via output parameter since coroutines return TaskResume.
 */
class BaseAssimilator {
 public:
  virtual ~BaseAssimilator() = default;

  /**
   * Schedule assimilation tasks based on the provided context
   * This is a coroutine that uses co_await for async CTE operations.
   * @param ctx Assimilation context with source, destination, and metadata
   * @param error_code Output: 0 on success, non-zero error code on failure
   * @return TaskResume for coroutine suspension/resumption
   */
  virtual clio::run::TaskResume Schedule(const AssimilationCtx& ctx, int& error_code) = 0;

  /**
   * PAGIFY: the assimilator's page-space layout for `ctx`.
   *
   * A consumer (the gpu_vector) addresses data as page 0, 1, 2, ... and
   * knows nothing about blob names or how the source was chopped up.
   * Only the assimilator knows that mapping, so it publishes it here as a
   * PageMap, from which ToPage/FromPage follow mechanically -- including
   * PARTIAL pages (a blob covering part of one) and MIXED pages (several
   * blobs sharing one).
   *
   * Default: an empty map, meaning "this assimilator does not support
   * paged access" -- the pagify interposer then declines and the caller
   * keeps whatever addressing it used before.
   */
  virtual PageMap BuildPageMap(const AssimilationCtx& ctx,
                               clio::run::u64 page_size) {
    (void)ctx;
    (void)page_size;
    return PageMap();
  }

  /** Slices composing `page_id`, per the map above. */
  std::vector<PageSlice> ToPage(const PageMap& map,
                                clio::run::u64 page_id) const {
    return map.ToPage(page_id);
  }

  /** Pages holding [off, off+size) of `blob`, per the map above. */
  std::vector<PageRef> FromPage(const PageMap& map, const std::string& blob,
                                clio::run::u64 off,
                                clio::run::u64 size) const {
    return map.FromPage(blob, off, size);
  }
};

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_BASE_ASSIMILATOR_H_
