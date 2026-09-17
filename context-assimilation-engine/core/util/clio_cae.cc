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
 * clio_cae - Ingest an OMNI YAML file for CAE processing.
 *
 * Reads the OMNI file and calls ParseOmni to schedule assimilation tasks.
 * Usage: clio_cae <omni_file_path>
 *
 * Installed as `clio_cae`; the legacy name `clio_cae_omni` is preserved
 * as an install-time symlink for backward compat.
 *
 * The parsing itself lives in core/src/omni_loader.cc so it can be unit
 * tested without a live runtime; what remains here is the part that cannot
 * be (client init, IPC, the ParseOmni round-trip). Because this file
 * defines main() no test can link it, so it is coverage-excluded: see the
 * lists in codecov.yml, CI/calculate_coverage.sh and the CTestCustom block
 * in the top-level CMakeLists.txt, which are kept in sync.
 */

#include <clio_ctp/util/logging.h>
#include <clio_cae/core/constants.h>
#include <clio_cae/core/core_client.h>
#include <clio_cae/core/omni_loader.h>

#include <string>
#include <vector>

void PrintUsage(const char* program_name) {
  HIPRINT("Usage: {} <omni_file_path>", program_name);
  HIPRINT("  omni_file_path - Path to the OMNI YAML file to ingest");
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    PrintUsage(argv[0]);
    return 1;
  }

  std::string omni_file_path(argv[1]);

  try {
    // Initialize CLIO Runtime client
    if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
      HLOG(kError, "Error: Failed to initialize Clio client");
      return 1;
    }

    // Verify CLIO Runtime IPC is available
    auto* ipc_manager = CLIO_IPC;
    if (!ipc_manager) {
      HLOG(kError, "Error: Clio IPC not initialized. Is the runtime running?");
      return 1;
    }

    // Load OMNI file and parse transfers
    std::vector<clio::cae::core::AssimilationCtx> contexts =
        clio::cae::core::LoadOmni(omni_file_path);

    // Connect to CAE core container using the standard pool ID
    clio::cae::core::Client client(clio::cae::core::kCaePoolId);

    HLOG(kInfo, "Calling ParseOmni...");

    // Call ParseOmni with vector of contexts
    auto parse_task = client.AsyncParseOmni(contexts);
    parse_task.Wait();
    clio::run::u32 result = parse_task->GetReturnCode();
    clio::run::u32 num_tasks_scheduled = parse_task->num_tasks_scheduled_;

    if (result != 0) {
      HLOG(kError, "Error: ParseOmni failed with result code {}", result);
      return 1;
    }

    HLOG(kSuccess, "ParseOmni completed successfully!");
    HLOG(kInfo, "  Tasks scheduled: {}", num_tasks_scheduled);

    return 0;

  } catch (const std::exception& e) {
    HLOG(kError, "Error: {}", e.what());
    return 1;
  }
}
