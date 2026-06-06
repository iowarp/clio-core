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
 * Main CLIO Runtime initialization and global functions
 */

#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/container.h"
#include "clio_runtime/work_orchestrator.h"
#include <cstdlib>
#include <cstring>

namespace clio::run {

bool ClioInitImpl(ChimaeraMode mode, bool default_with_runtime,
                  bool is_restart) {
  // Static guard to prevent double initialization
  static bool s_initialized = false;
  if (s_initialized) {
    return true;  // Already initialized, return success
  }

  auto* runtime_manager = CLIO_RUNTIME_MANAGER;
  runtime_manager->is_restart_ = is_restart;

  // Check environment variable CHI_WITH_RUNTIME
  bool with_runtime = default_with_runtime;
  const char* env_val = chi::env::GetCompat("WITH_RUNTIME");
  if (env_val != nullptr) {
    with_runtime = (std::strcmp(env_val, "1") == 0 ||
                   std::strcmp(env_val, "true") == 0 ||
                   std::strcmp(env_val, "TRUE") == 0);
  }

  // Determine what to initialize based on mode and with_runtime flag
  bool init_runtime = false;
  bool init_client = false;

  if (mode == ChimaeraMode::kServer || mode == ChimaeraMode::kRuntime) {
    // Server/Runtime mode: always start runtime
    init_runtime = true;
    init_client = true;  // Runtime also needs client components
  } else {
    // Client mode
    init_client = true;
    init_runtime = with_runtime;
  }

  // Initialize runtime first if needed
  if (init_runtime) {
    if (!runtime_manager->ServerInit()) {
      return false;
    }
  }

  // Initialize client components
  if (init_client) {
    if (!runtime_manager->ClientInit()) {
      return false;
    }
  }

  // Register atexit handler so CLIO_RUNTIME_FINALIZE runs before static
  // destructors.  The CLIO Runtime singleton is heap-allocated (GetGlobalPtrVar)
  // so its destructor is never called automatically.  Without this the ZMQ
  // DEALER socket stays open and zmq_ctx_destroy blocks forever at exit.
  std::atexit(CLIO_RUNTIME_FINALIZE);

  // Mark as initialized on success
  s_initialized = true;
  return true;
}

void CLIO_RUNTIME_FINALIZE() {
  static bool s_finalized = false;
  if (s_finalized) {
    return;
  }
  s_finalized = true;
  auto *mgr = CLIO_RUNTIME_MANAGER;
  if (mgr) {
    // Server first: stop worker threads that may still be sending IPC
    mgr->ServerFinalize();
    // Client second: close DEALER socket and join recv thread
    mgr->ClientFinalize();
  }
}

}  // namespace clio::run
