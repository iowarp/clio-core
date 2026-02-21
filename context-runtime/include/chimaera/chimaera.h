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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_CHIMAERA_H_
#define CHIMAERA_INCLUDE_CHIMAERA_CHIMAERA_H_

/**
 * Main header file for Chimaera distributed task execution framework
 *
 * This header provides the primary interface for both runtime and client
 * applications using the Chimaera framework.
 */

#include "chimaera/pool_query.h"
#include "chimaera/singletons.h"
#include "chimaera/task.h"
#include "chimaera/task_archives.h"
#include "chimaera/types.h"
#include "chimaera/worker.h"

namespace chi {

/**
 * Chimaera initialization mode
 */
enum class ChimaeraMode {
  kClient,   /**< Client mode - connects to existing runtime */
  kServer,   /**< Server mode - starts runtime components */
  kRuntime = kServer  /**< Alias for kServer */
};

/**
 * Global initialization functions
 */

/**
 * Initialize Chimaera with specified mode
 *
 * @param mode Initialization mode (kClient or kServer/kRuntime)
 * @param default_with_runtime Default behavior if CHI_WITH_RUNTIME env var not set
 *        If true, will start runtime in addition to client initialization
 *        If false, will only initialize client components
 * @return true if initialization successful, false otherwise
 *
 * Environment variable:
 *   CHI_WITH_RUNTIME=1 - Start runtime regardless of mode
 *   CHI_WITH_RUNTIME=0 - Don't start runtime (client only)
 *   If not set, uses default_with_runtime parameter
 */
bool CHIMAERA_INIT(ChimaeraMode mode, bool default_with_runtime = false);

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_CHIMAERA_H_