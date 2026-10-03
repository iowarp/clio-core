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
 * OMNI YAML ingest, split out of util/clio_cae.cc so it can be linked into a
 * unit test. The clio_cae binary is a thin main() around this: everything it
 * does before talking to the runtime is here, which is the part worth testing.
 */

#ifndef CLIO_CAE_CORE_OMNI_LOADER_H_
#define CLIO_CAE_CORE_OMNI_LOADER_H_

#include <clio_cae/core/factory/assimilation_ctx.h>

#include <string>
#include <vector>

namespace clio::cae::core {

/**
 * Parse an OMNI YAML file into one AssimilationCtx per `transfers` entry.
 *
 * Throws std::runtime_error when the file cannot be read, when `transfers` is
 * missing or is not a sequence, or when an entry omits src/dst/format. The
 * message names the offending transfer by its 1-based position.
 */
std::vector<AssimilationCtx> LoadOmni(const std::string& omni_path);

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_OMNI_LOADER_H_
