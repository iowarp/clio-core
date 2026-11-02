/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

// Cross-platform contract tests for the drive-failure prediction path
// (SystemInfo::DeriveDriveType + SystemInfo::PredictDriveFailure).
//
// These deliberately do NOT require a running prediction server: they assert
// the *graceful-degradation contract* that must hold on every supported OS,
// whether or not Poco is compiled in and whether or not a server is reachable:
//
//   * DeriveDriveType routes pool names to "hdd"/"ssd" deterministically.
//   * PredictDriveFailure always returns a JSON object, never throws, and is
//     bounded in time even when the endpoint is unreachable.
//
// The live end-to-end check (real Python server + models) lives in the
// Linux-only integration test; it is not portable to the Windows/macOS matrix.

#include <catch2/catch_all.hpp>

#include <chrono>
#include <cstdlib>
#include <string>

#include <clio_ctp/introspect/system_info.h>

namespace {

// Portable "set environment variable" for the test process.
void SetEnv(const char *key, const char *value) {
#ifdef _WIN32
  _putenv_s(key, value);
#else
  setenv(key, value, /*overwrite=*/1);
#endif
}

// A JSON object is the minimum shape the contract guarantees: non-empty and
// wrapped in braces. (A full parse would drag in a JSON lib the non-Poco
// matrix entries don't otherwise need.)
bool IsJsonObjectEnvelope(const std::string &s) {
  return !s.empty() && s.front() == '{' && s.back() == '}';
}

}  // namespace

TEST_CASE("DeriveDriveType routes pool names deterministically",
          "[introspect][prediction]") {
  using ctp::SystemInfo;

  // Spinning disk: any name containing "hdd", case-insensitive.
  REQUIRE(SystemInfo::DeriveDriveType("hdd_pool") == "hdd");
  REQUIRE(SystemInfo::DeriveDriveType("node0_hdd_tier0") == "hdd");
  REQUIRE(SystemInfo::DeriveDriveType("My_HDD_Tier") == "hdd");
  REQUIRE(SystemInfo::DeriveDriveType("HDD") == "hdd");

  // Everything else (including explicit ssd, nvme, and unlabelled) -> "ssd".
  REQUIRE(SystemInfo::DeriveDriveType("ssd_pool") == "ssd");
  REQUIRE(SystemInfo::DeriveDriveType("nvme0n1") == "ssd");
  REQUIRE(SystemInfo::DeriveDriveType("scratch") == "ssd");
  REQUIRE(SystemInfo::DeriveDriveType("") == "ssd");

  // "hdd" as a strict substring still matches (documents current behaviour).
  REQUIRE(SystemInfo::DeriveDriveType("shddx") == "hdd");
}

TEST_CASE("PredictDriveFailure degrades gracefully when unreachable",
          "[introspect][prediction]") {
  // Point the client at a closed local port so the connect fails fast and
  // deterministically (no dependency on host.docker.internal resolving).
  SetEnv("CLIO_PREDICT_URL", "http://127.0.0.1:1/predict/auto");

  const auto start = std::chrono::steady_clock::now();
  const std::string result =
      ctp::SystemInfo::PredictDriveFailure("hdd", "{}", "hdd_pool");
  const auto elapsed = std::chrono::steady_clock::now() - start;

  INFO("PredictDriveFailure returned: " << result);

  // Always a JSON object, never an empty/garbage/half-written buffer.
  REQUIRE(IsJsonObjectEnvelope(result));

  // Bounded: the 5s Poco timeout (plus slack) must cap the call; a hang here
  // is the failure mode we care most about. Well under the timeout when the
  // port is refused outright.
  REQUIRE(elapsed < std::chrono::seconds(15));

#if CTP_HAS_POCO
  // With the HTTP client compiled in, an unreachable endpoint surfaces as an
  // {"error": ...} object rather than the empty "{}" default.
  REQUIRE(result.find("error") != std::string::npos);
#else
  // Without Poco the call is compiled out to the neutral default.
  REQUIRE(result == "{}");
#endif
}

TEST_CASE("PredictDriveFailure tolerates varied inputs without throwing",
          "[introspect][prediction]") {
  SetEnv("CLIO_PREDICT_URL", "http://127.0.0.1:1/predict/auto");

  // Empty metrics, populated metrics, and both drive types must all yield a
  // JSON envelope with no exception escaping.
  const std::string a = ctp::SystemInfo::PredictDriveFailure("ssd", "", "ssd0");
  const std::string b = ctp::SystemInfo::PredictDriveFailure(
      "hdd", R"({"temperature": 42, "reallocated_sectors": 3})", "hdd0");

  REQUIRE(IsJsonObjectEnvelope(a));
  REQUIRE(IsJsonObjectEnvelope(b));
}
