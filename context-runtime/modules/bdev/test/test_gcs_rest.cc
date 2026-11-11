/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

/**
 * Round-trip tests for the Google Cloud Storage REST client behind the kGcs
 * bdev transport (gcs_rest.h).
 *
 * They talk to whatever GCS_ENDPOINT names. CI points it at fake-gcs-server
 * (an open-source GCS emulator, run as a service container), so the tests need
 * no Google account and no credentials. Without GCS_ENDPOINT they skip, so a
 * plain `ctest` never reaches out to the real storage.googleapis.com.
 */

#include "clio_runtime/bdev/transports/gcs_rest.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "simple_test.h"

namespace gcs = clio::run::bdev::gcs;

namespace {

/** True when a GCS endpoint (normally the emulator) has been provided. */
bool EndpointAvailable() {
  const char *ep = std::getenv("GCS_ENDPOINT");
  return ep != nullptr && *ep != '\0';
}

/** A bucket name unique to this process, so reruns never collide. */
std::string UniqueBucket(const std::string &stem) {
  auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  return "clio-" + stem + "-" + std::to_string(now % 1000000007);
}

/** Deterministic, position-dependent bytes so an offset slip is visible. */
std::vector<char> Pattern(size_t len, uint32_t seed) {
  std::vector<char> v(len);
  uint32_t x = seed * 2654435761u + 1u;
  for (size_t i = 0; i < len; ++i) {
    x = x * 1664525u + 1013904223u;
    v[i] = static_cast<char>(x >> 24);
  }
  return v;
}

gcs::GcsRestClient MakeClient(const std::string &stem,
                              const std::string &prefix) {
  return gcs::GcsRestClient(
      gcs::GcsRestClient::ConfigFromEnv(UniqueBucket(stem), prefix));
}

}  // namespace

TEST_CASE("gcs_rest_key_for_offset", "[gcs_rest]") {
  gcs::GcsRestClient plain(gcs::GcsRestClient::ConfigFromEnv("b", ""));
  REQUIRE(plain.KeyForOffset(4096) == "block_4096");
  gcs::GcsRestClient nested(gcs::GcsRestClient::ConfigFromEnv("b", "clio/t1"));
  REQUIRE(nested.KeyForOffset(0) == "clio/t1/block_0");
}

TEST_CASE("gcs_rest_ensure_bucket_is_idempotent", "[gcs_rest]") {
  if (!EndpointAvailable()) {
    INFO("GCS_ENDPOINT unset; run against fake-gcs-server. Skipping.");
    return;
  }
  gcs::GcsRestClient client = MakeClient("ensure", "");
  gcs::GcsResult first = client.EnsureBucket();
  REQUIRE(first.error.empty());
  // A second create of the same bucket answers 409, which must count as success.
  gcs::GcsResult again = client.EnsureBucket();
  REQUIRE(again.error.empty());
}

TEST_CASE("gcs_rest_put_get_delete_round_trip", "[gcs_rest]") {
  if (!EndpointAvailable()) {
    INFO("GCS_ENDPOINT unset; run against fake-gcs-server. Skipping.");
    return;
  }
  // The prefix contains '/', so the object name must travel as %2F.
  gcs::GcsRestClient client = MakeClient("roundtrip", "clio/roundtrip");
  REQUIRE(client.EnsureBucket().error.empty());

  SECTION("a block-sized object survives the round trip byte for byte");
  const size_t kLen = 64 * 1024;
  std::vector<char> out = Pattern(kLen, 7);
  const std::string key = client.KeyForOffset(1048576);
  gcs::GcsResult put = client.PutObject(key, out.data(), out.size());
  REQUIRE(put.error.empty());
  REQUIRE(put.ok());

  std::vector<char> in(kLen, 0);
  size_t got = 0;
  gcs::GcsResult get = client.GetObject(key, in.data(), in.size(), &got);
  REQUIRE(get.error.empty());
  REQUIRE(get.ok());
  REQUIRE_FALSE(get.not_found);
  REQUIRE(got == kLen);
  REQUIRE(std::memcmp(out.data(), in.data(), kLen) == 0);

  SECTION("a 1 MiB object, the size of a filesystem page, round trips");
  const size_t kBig = 1 << 20;
  std::vector<char> big = Pattern(kBig, 11);
  const std::string big_key = client.KeyForOffset(2 * kBig);
  REQUIRE(client.PutObject(big_key, big.data(), big.size()).ok());
  std::vector<char> big_in(kBig, 0);
  size_t big_got = 0;
  REQUIRE(client.GetObject(big_key, big_in.data(), big_in.size(), &big_got).ok());
  REQUIRE(big_got == kBig);
  REQUIRE(std::memcmp(big.data(), big_in.data(), kBig) == 0);

  SECTION("a missing object is a sparse miss, not an error");
  size_t miss_got = 1;
  gcs::GcsResult miss = client.GetObject(client.KeyForOffset(999 * kBig),
                                         in.data(), in.size(), &miss_got);
  REQUIRE(miss.not_found);
  REQUIRE(miss_got == 0);

  SECTION("delete removes it, and the next read is a sparse miss");
  REQUIRE(client.DeleteObject(key).error.empty());
  size_t after = 1;
  gcs::GcsResult gone = client.GetObject(key, in.data(), in.size(), &after);
  REQUIRE(gone.not_found);
  REQUIRE(after == 0);
}

SIMPLE_TEST_MAIN()
