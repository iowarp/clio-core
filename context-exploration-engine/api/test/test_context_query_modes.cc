/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * test_context_query_modes.cc - Mode coverage for the ContextInterface APIs.
 *
 * test_context_query.cc covers the plain regex path; this test drives the
 * remaining dispatch modes of ContextQuery / ContextRetrieve (temporal
 * bounds, BM25 semantic prompt, result caps and batch sizes) plus the
 * ContextBundle empty-input, ContextSplice, and ContextDestroy paths.
 * The queries run against an empty store: the point is executing each
 * mode's full code path, results are expected to be empty.
 *
 * Environment Variables:
 * - INIT_CLIO: If set to "1", initializes CLIO Runtime runtime
 */

#include <clio_cee/api/context_interface.h>
#include <clio_ctp/introspect/system_info.h>
#include <clio_ctp/util/logging.h>
#include <clio_runtime/clio_runtime.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

void test_query_modes() {
  HLOG(kInfo, "TEST: ContextQuery modes");
  iowarp::ContextInterface ctx;

  // Temporal mode: nonzero time bounds.
  std::vector<std::string> temporal =
      ctx.ContextQuery(".*", ".*", 16, "", 1, 2000000000ULL);
  HLOG(kInfo, "temporal query -> {} results", temporal.size());

  // Temporal mode with only a lower bound.
  std::vector<std::string> lower_only =
      ctx.ContextQuery(".*", ".*", 0, "", 1, 0);
  HLOG(kInfo, "temporal lower-bound query -> {} results", lower_only.size());

  // Semantic (BM25) mode: non-empty prompt.
  std::vector<std::string> semantic =
      ctx.ContextQuery(".*", ".*", 5, "find the calibration data");
  HLOG(kInfo, "semantic query -> {} results", semantic.size());

  // Regex mode with a result cap.
  std::vector<std::string> capped = ctx.ContextQuery(".*", ".*", 3);
  HLOG(kInfo, "capped regex query -> {} results", capped.size());

  HLOG(kSuccess, "PASSED: ContextQuery modes");
}

void test_retrieve_modes() {
  HLOG(kInfo, "TEST: ContextRetrieve modes");
  iowarp::ContextInterface ctx;

  // Regex retrieve (default mode).
  std::vector<std::string> regex_ctx = ctx.ContextRetrieve(".*", ".*");
  HLOG(kInfo, "regex retrieve -> {} chunks", regex_ctx.size());

  // Semantic retrieve with custom cap / context size / batch size.
  std::vector<std::string> semantic_ctx = ctx.ContextRetrieve(
      ".*", ".*", 4, 1024 * 1024, 2, "summarize the experiment");
  HLOG(kInfo, "semantic retrieve -> {} chunks", semantic_ctx.size());

  // Temporal retrieve.
  std::vector<std::string> temporal_ctx = ctx.ContextRetrieve(
      ".*", ".*", 4, 1024 * 1024, 2, "", 1, 2000000000ULL);
  HLOG(kInfo, "temporal retrieve -> {} chunks", temporal_ctx.size());

  HLOG(kSuccess, "PASSED: ContextRetrieve modes");
}

void test_bundle_splice_destroy() {
  HLOG(kInfo, "TEST: ContextBundle/Splice/Destroy edges");
  iowarp::ContextInterface ctx;

  // Empty bundle: early-return success.
  std::vector<clio::cae::core::AssimilationCtx> empty_bundle;
  int rc = ctx.ContextBundle(empty_bundle);
  HLOG(kInfo, "empty bundle rc={}", rc);

  // Splice on an empty store: nothing matches, still a full pass.
  int splice_rc = ctx.ContextSplice("spliced_ctx", "no_such_tag.*", ".*");
  HLOG(kInfo, "splice rc={}", splice_rc);

  // Destroy: empty list and a missing context name.
  int destroy_empty_rc = ctx.ContextDestroy({});
  HLOG(kInfo, "destroy empty rc={}", destroy_empty_rc);
  int destroy_missing_rc =
      ctx.ContextDestroy({"no_such_context_a", "no_such_context_b"});
  HLOG(kInfo, "destroy missing rc={}", destroy_missing_rc);

  HLOG(kSuccess, "PASSED: ContextBundle/Splice/Destroy edges");
}

int main() {
  HLOG(kInfo, "========================================");
  HLOG(kInfo, "ContextInterface mode coverage tests");
  HLOG(kInfo, "========================================");

  try {
    const char* init_clio = std::getenv("INIT_CLIO");
    if (init_clio && std::strcmp(init_clio, "1") == 0) {
      HLOG(kInfo, "Initializing Clio (INIT_CLIO=1)...");
      clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
      HLOG(kSuccess, "Clio initialized");
    }

    auto* ipc_manager = CLIO_IPC;
    if (!ipc_manager) {
      HLOG(kError, "Clio IPC not initialized. Is the runtime running?");
      ctp::SystemInfo::TerminateProcessNow(1);
      return 1;
    }

    test_query_modes();
    test_retrieve_modes();
    test_bundle_splice_destroy();

    HLOG(kSuccess, "All tests PASSED!");
    ctp::SystemInfo::TerminateProcessNow(0);
    return 0;
  } catch (const std::exception& e) {
    HLOG(kError, "Test FAILED with exception: {}", e.what());
    ctp::SystemInfo::TerminateProcessNow(1);
    return 1;
  }
}
