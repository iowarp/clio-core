/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

// Linux-only END-TO-END check for the drive-failure prediction path.
//
// Unlike the cross-platform contract test, this exercises a *live* prediction:
// the driver script (run_prediction_e2e.sh) starts the real Python FastAPI
// server with the trained LightGBM models, exports CLIO_PREDICT_URL pointing at
// it, then runs this binary. We POST an HDD SMART snapshot through the same
// production code path (SystemInfo::PredictDriveFailure -> Poco::Net) and
// assert the server returned a genuine inference result, not the graceful
// fallback.
//
// Exit codes: 0 = pass, 1 = server reachable but response wrong, 2 = harness
// misconfiguration (no CLIO_PREDICT_URL). The script is responsible for the
// SKIP decision when prerequisites (Linux, models, python deps) are absent.

#include <cstdio>
#include <cstdlib>
#include <string>

#include <clio_ctp/introspect/system_info.h>

int main() {
  const char *url = std::getenv("CLIO_PREDICT_URL");
  if (url == nullptr || *url == '\0') {
    std::fprintf(stderr,
                 "E2E misconfigured: CLIO_PREDICT_URL is unset (the driver "
                 "script must point it at the live server)\n");
    return 2;
  }

  // A representative HDD SMART snapshot. Unknown features default to 0 on the
  // server side, so a partial reading still yields a valid inference.
  const std::string metrics =
      "{\"smart_5_raw\": 0, \"smart_187_raw\": 0, \"smart_197_raw\": 0, "
      "\"smart_198_raw\": 0}";

  const std::string res =
      ctp::SystemInfo::PredictDriveFailure("hdd", metrics, "e2e_hdd_drive");
  std::fprintf(stderr, "live prediction response: %s\n", res.c_str());

  // The graceful-degradation payloads ("{}" / {"error": ...}) mean we never
  // reached a working server — a hard failure for an E2E that set one up.
  if (res == "{}" || res.find("\"error\"") != std::string::npos ||
      res.find("unreachable") != std::string::npos) {
    std::fprintf(stderr,
                 "FAIL: got a fallback/error payload, not a live prediction\n");
    return 1;
  }

  // A real /predict/auto response carries the model's probability and which
  // model produced it. Both present == the inference round-tripped.
  const bool has_prob = res.find("failure_probability") != std::string::npos;
  const bool has_model = res.find("model_used") != std::string::npos;
  if (!has_prob || !has_model) {
    std::fprintf(stderr,
                 "FAIL: response missing failure_probability/model_used "
                 "(has_prob=%d has_model=%d)\n",
                 has_prob, has_model);
    return 1;
  }

  std::fprintf(stderr,
               "PASS: live server returned a real prediction "
               "(failure_probability + model_used present)\n");
  return 0;
}
