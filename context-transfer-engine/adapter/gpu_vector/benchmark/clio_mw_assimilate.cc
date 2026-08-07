/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

// Host-only (plain C++, NOT nvcc) helper that drives the CAE
// ModelWeightsAssimilator. The CAE task headers (ctp::priv::basic_string, the
// argpack serializers) do not compile under nvcc, so the assimilation call is
// isolated in this translation unit; the CUDA round-trip TU calls it via the
// forward declaration below.

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_cae/core/core_client.h>
#include <clio_cae/core/constants.h>
#include <clio_cae/core/factory/assimilation_ctx.h>

#include <cstdio>
#include <string>
#include <vector>

// Assimilate `path` into CTE tag `tag` using the modelweights CAE type, laid
// out for a gpu_vector with (page_size, nblocks). Returns the assimilation
// result code (0 == success). Blocks until all page blobs are written.
int mw_assimilate(const std::string& path, const std::string& tag,
                  unsigned long long page_size, unsigned nblocks) {
  clio::cae::core::Client cae(clio::cae::core::kCaePoolId);
  clio::cae::core::AssimilationCtx ctx;
  ctx.src = "modelweights::" + path;
  ctx.dst = "iowarp::" + tag;
  ctx.format = "modelweights;page=" + std::to_string(page_size) +
               ";nblocks=" + std::to_string(nblocks);
  std::vector<clio::cae::core::AssimilationCtx> ctxs{ctx};

  std::fprintf(stderr, "[MW] assimilating: src='%s' dst='%s' format='%s'\n",
               ctx.src.c_str(), ctx.dst.c_str(), ctx.format.c_str());
  auto omni = cae.AsyncParseOmni(ctxs);
  omni.Wait();
  std::fprintf(stderr, "[MW] ParseOmni: scheduled=%d result_code=%d\n",
               (int)omni->num_tasks_scheduled_, (int)omni->result_code_);
  return (int)omni->result_code_;
}
