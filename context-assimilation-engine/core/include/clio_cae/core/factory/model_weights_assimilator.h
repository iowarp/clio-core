/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_CAE_CORE_MODEL_WEIGHTS_ASSIMILATOR_H_
#define CLIO_CAE_CORE_MODEL_WEIGHTS_ASSIMILATOR_H_

#include <memory>
#include <string>

#include "clio_cae/core/factory/base_assimilator.h"

// Forward-declare the CTE client to avoid pulling clio_cte headers (and their
// Method namespace) into this header.
namespace clio::cte::core {
class Client;
}

namespace clio::cae::core {

/**
 * ModelWeightsAssimilator — assimilate an AI model's weight file into a single
 * CTE tag laid out so it can be mapped directly by the GPU `gpu_vector`.
 *
 * Protocol: `modelweights::<path>`  ->  `iowarp::<tag>`
 *
 * Layout contract with clio::cte::gpu_vector::Vector<T>
 * ----------------------------------------------------
 * The vector reads/writes each cache page as a CTE blob named
 *     <tag>_b<block>_pi<page>
 * (see gpu_vector.h — per-block blob name `<tag>_b<b>`; the runtime appends
 *  `_pi<page_idx>`). This assimilator writes the whole weight byte stream as
 * fixed-size `page_size` pages under exactly that naming, so a
 * `Vector<T>("<tag>", nblocks, ..., page_size)` constructed with matching
 * geometry maps the tag and streams the weights on demand — no file re-read.
 *
 * Page/block assignment (contiguous stripes):
 *     num_pages       = ceil(total_size / page_size)
 *     pages_per_block = ceil(num_pages / nblocks)
 *     page p -> block  = p / pages_per_block,  blob "<tag>_b<block>_pi<p>"
 * The trailing partial page is zero-padded to page_size so every blob is a
 * full page (the vector always GetBlobs page_size bytes).
 *
 * A human/tooling-readable "manifest" blob records the geometry so the mapping
 * side can reconstruct matching Vector parameters. The vector ignores it (it
 * only ever requests `_b<block>_pi<page>` blobs).
 *
 * Geometry is passed via the `format` field:
 *     format: "modelweights"                       (defaults: page=2MiB, nblocks=1)
 *     format: "modelweights;page=2097152;nblocks=8"
 */
class ModelWeightsAssimilator : public BaseAssimilator {
 public:
  explicit ModelWeightsAssimilator(
      std::shared_ptr<clio::cte::core::Client> cte_client);

  clio::run::TaskResume Schedule(const AssimilationCtx& ctx,
                                 int& error_code) override;

  /**
   * PAGIFY (GGUF): the model file laid out linearly across the page space.
   *
   * Assimilation writes fixed-size page blobs, so the map is one extent
   * per page -- but expressing it as EXTENTS rather than "page N == blob
   * b0_pi<N>" is what lets the same consumer read a layout where a page
   * mixes several blobs, or holds only part of one, without knowing which
   * layout it got. The final page is short whenever the file is not a
   * multiple of the page size, which is already the partial case.
   */
  PageMap BuildPageMap(const AssimilationCtx& ctx,
                       clio::run::u64 page_size) override;

 private:
  std::string GetUrlProtocol(const std::string& url);
  std::string GetUrlPath(const std::string& url);
  size_t GetFileSize(const std::string& file_path);
  // Parse "key=value;key=value" tail of the format string. Returns `def` if
  // `key` is absent or unparseable.
  static size_t ParseFormatParam(const std::string& format,
                                 const std::string& key, size_t def);

  std::shared_ptr<clio::cte::core::Client> cte_client_;
};

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_MODEL_WEIGHTS_ASSIMILATOR_H_
