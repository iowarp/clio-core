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

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "clio_ctp/introspect/system_info.h"
#include "clio_llm/lmcache/lmcache_store.h"

namespace clio_llm::lmcache {

/**
 * Narrow test-only access to LMCacheStore direct-read configuration state.
 */
struct LMCacheStoreTestAccess {
  /**
   * Parse the direct-read environment variable through production code.
   *
   * @return The production parser result.
   */
  static bool DirectReadEnabledFromEnv() {
    return LMCacheStore::DirectReadEnabledFromEnv();
  }

  /**
   * Read the setting latched by an LMCacheStore instance.
   *
   * @param store Store to inspect.
   * @return The store's latched direct-read setting.
   */
  static bool DirectReadEnabled(const LMCacheStore &store) {
    return store.direct_read_enabled_;
  }
};

}  // namespace clio_llm::lmcache

namespace {

/**
 * Temporarily set one process environment variable and restore it on exit.
 */
class ScopedEnvironmentVariable {
 public:
  /**
   * Save the current value and install a test value.
   *
   * @param name Environment variable name.
   * @param value Temporary value, or null to make the variable unset.
   */
  ScopedEnvironmentVariable(const char *name, const char *value) : name_(name) {
    const char *current = std::getenv(name);
    if (current != nullptr) {
      previous_value_ = current;
    }
    if (value != nullptr) {
      ctp::SystemInfo::Setenv(name_.c_str(), value, 1);
    } else {
      ctp::SystemInfo::Unsetenv(name_.c_str());
    }
  }

  /**
   * Restore the environment variable state from construction.
   */
  ~ScopedEnvironmentVariable() {
    if (previous_value_.has_value()) {
      ctp::SystemInfo::Setenv(name_.c_str(), *previous_value_, 1);
    } else {
      ctp::SystemInfo::Unsetenv(name_.c_str());
    }
  }

  ScopedEnvironmentVariable(const ScopedEnvironmentVariable &) = delete;
  ScopedEnvironmentVariable &operator=(const ScopedEnvironmentVariable &) =
      delete;

 private:
  std::string name_;
  std::optional<std::string> previous_value_;
};

/**
 * Initialize a test LMCacheStore or skip when CTE runtime is unavailable.
 *
 * @param tag_name Unique tag name for this test case.
 * @return Initialized store.
 */
clio_llm::lmcache::LMCacheStore MakeStore(const std::string &tag_name) {
  if (std::getenv("CLIO_LMCACHE_TEST_RUNTIME") == nullptr) {
    SKIP(
        "Set CLIO_LMCACHE_TEST_RUNTIME=1 to run CTE runtime integration tests");
  }

  clio_llm::lmcache::LMCacheStore store;
  if (!store.Init("", tag_name, "local")) {
    SKIP("CTE runtime not available");
  }
  return store;
}

/**
 * Convert raw bytes into a string_view-compatible std::string.
 *
 * @param bytes Input byte vector.
 * @return String preserving embedded NUL and non-UTF8 bytes.
 */
std::string ByteString(const std::vector<std::uint8_t> &bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

/**
 * Build the expected CLIOKV1 record bytes.
 *
 * @param metadata Metadata JSON bytes.
 * @param payload Payload bytes.
 * @return Encoded record bytes.
 */
std::vector<std::uint8_t> BuildRecord(std::string_view metadata,
                                      std::string_view payload) {
  std::vector<std::uint8_t> record(24 + metadata.size() + payload.size());
  const char magic[8] = {'C', 'L', 'I', 'O', 'K', 'V', '1', '\0'};
  std::memcpy(record.data(), magic, sizeof(magic));
  record[11] = 1;
  const auto metadata_size = static_cast<std::uint32_t>(metadata.size());
  record[12] = static_cast<std::uint8_t>((metadata_size >> 24) & 0xff);
  record[13] = static_cast<std::uint8_t>((metadata_size >> 16) & 0xff);
  record[14] = static_cast<std::uint8_t>((metadata_size >> 8) & 0xff);
  record[15] = static_cast<std::uint8_t>(metadata_size & 0xff);
  const auto payload_size = static_cast<std::uint64_t>(payload.size());
  for (std::size_t i = 0; i < 8; ++i) {
    record[16 + i] =
        static_cast<std::uint8_t>((payload_size >> (56 - (8 * i))) & 0xff);
  }
  std::memcpy(record.data() + 24, metadata.data(), metadata.size());
  std::memcpy(record.data() + 24 + metadata.size(), payload.data(),
              payload.size());
  return record;
}

}  // namespace

TEST_CASE("LMCacheStore default state is closed", "[lmcache][unit]") {
  clio_llm::lmcache::LMCacheStore store;
  const std::vector<std::string> names = {"anything"};
  const std::vector<std::string> payloads = {"payload"};
  std::vector<std::uint8_t> destination(8);

  REQUIRE_FALSE(store.IsReady());
  REQUIRE_FALSE(store.Exists("anything"));
  REQUIRE_FALSE(store.Size("anything").has_value());
  REQUIRE_FALSE(store.GetBytes("anything").has_value());
  REQUIRE_FALSE(store.Delete("anything"));
  REQUIRE(store.ExistsMany(names) == std::vector<bool>{false});
  REQUIRE_FALSE(store.SizeMany(names)[0].has_value());
  REQUIRE_FALSE(store.GetMany(names)[0].has_value());
  REQUIRE(store.PutMany(names, payloads) == std::vector<bool>{false});
  REQUIRE(store.PutManyRecords(names, payloads, {payloads[0].data()},
                               {payloads[0].size()}) ==
          std::vector<bool>{false});
  REQUIRE(store.GetManyInto(names, {destination.data()},
                            {destination.size()}) == std::vector<bool>{false});
  REQUIRE(store.GetManyRangesInto(names, {0}, {destination.size()},
                                  {destination.data()},
                                  {destination.size()}) ==
          std::vector<bool>{false});
  REQUIRE_FALSE(store.ReadRecordInfos(names)[0].has_value());
  store.Close();
  REQUIRE_FALSE(store.IsReady());
}

TEST_CASE("LMCacheStore parses direct-read environment values",
          "[lmcache][unit]") {
  using clio_llm::lmcache::LMCacheStoreTestAccess;
  const std::vector<const char *> false_settings = {
      nullptr, "", "0", "false", "FALSE", "garbage"};
  for (const char *setting : false_settings) {
    ScopedEnvironmentVariable direct_read_setting("CLIO_LMCACHE_DIRECT_READ",
                                                  setting);
    REQUIRE_FALSE(LMCacheStoreTestAccess::DirectReadEnabledFromEnv());
  }

  for (const char *setting : {"1", "true", "TrUe"}) {
    ScopedEnvironmentVariable direct_read_setting("CLIO_LMCACHE_DIRECT_READ",
                                                  setting);
    REQUIRE(LMCacheStoreTestAccess::DirectReadEnabledFromEnv());
  }
}

TEST_CASE("LMCacheStore round-trips non-UTF8 bytes", "[lmcache][integration]") {
  auto store = MakeStore("test_lmcache_bytes");
  const std::vector<std::uint8_t> input = {0x00, 0xff, 0x10, 0x80, 0x7f};

  REQUIRE(store.PutBytes("binary", ByteString(input)));
  REQUIRE(store.Exists("binary"));
  REQUIRE(store.Size("binary").value_or(0) == input.size());

  auto output = store.GetBytes("binary");
  REQUIRE(output.has_value());
  REQUIRE(*output == input);
}

TEST_CASE("LMCacheStore reports missing keys as misses",
          "[lmcache][integration]") {
  auto store = MakeStore("test_lmcache_missing");

  REQUIRE_FALSE(store.Exists("missing"));
  REQUIRE_FALSE(store.Size("missing").has_value());
  REQUIRE_FALSE(store.GetBytes("missing").has_value());
}

TEST_CASE("LMCacheStore delete removes a blob", "[lmcache][integration]") {
  auto store = MakeStore("test_lmcache_delete");
  const std::string payload = "payload";

  REQUIRE(store.PutBytes("to_delete", payload));
  REQUIRE(store.Exists("to_delete"));
  REQUIRE(store.Delete("to_delete"));
  REQUIRE_FALSE(store.Exists("to_delete"));
}

TEST_CASE("LMCacheStore handles large LMCache-like payloads",
          "[lmcache][integration]") {
  auto store = MakeStore("test_lmcache_large");
  std::vector<std::uint8_t> input(4 * 1024 * 1024);
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<std::uint8_t>(i % 251);
  }

  REQUIRE(store.PutBytes("large", ByteString(input)));
  auto output = store.GetBytes("large");
  REQUIRE(output.has_value());
  REQUIRE(*output == input);
}

TEST_CASE("LMCacheStore batches put, size, exists, and get",
          "[lmcache][integration]") {
  auto store = MakeStore("test_lmcache_batch");
  const std::vector<std::string> names = {"batch_a", "batch_b", "batch_c"};
  const std::vector<std::string> payloads = {
      ByteString(std::vector<std::uint8_t>{0, 1, 2}),
      ByteString(std::vector<std::uint8_t>{3, 4, 5, 6}),
      ByteString(std::vector<std::uint8_t>{7, 8})};

  REQUIRE(store.PutMany(names, payloads, 2) ==
          std::vector<bool>{true, true, true});
  REQUIRE(store.ExistsMany({"batch_a", "missing", "batch_c"}, 2) ==
          std::vector<bool>{true, false, true});

  const auto sizes = store.SizeMany(names, 2);
  REQUIRE(sizes.size() == payloads.size());
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    REQUIRE(sizes[i].has_value());
    REQUIRE(*sizes[i] == payloads[i].size());
  }

  const auto values = store.GetMany({"batch_a", "missing", "batch_c"}, 2);
  REQUIRE(values.size() == 3);
  REQUIRE(values[0].has_value());
  REQUIRE(*values[0] == std::vector<std::uint8_t>({0, 1, 2}));
  REQUIRE_FALSE(values[1].has_value());
  REQUIRE(values[2].has_value());
  REQUIRE(*values[2] == std::vector<std::uint8_t>({7, 8}));
}

TEST_CASE("LMCacheStore batches get into caller buffers",
          "[lmcache][integration]") {
  auto store = MakeStore("test_lmcache_batch_into");
  const std::vector<std::string> names = {"into_a", "into_b"};
  const std::vector<std::string> payloads = {
      ByteString(std::vector<std::uint8_t>{9, 8, 7}),
      ByteString(std::vector<std::uint8_t>{6, 5, 4, 3})};

  REQUIRE(store.PutMany(names, payloads, 2) == std::vector<bool>{true, true});

  std::vector<std::uint8_t> first(payloads[0].size());
  std::vector<std::uint8_t> second(payloads[1].size());
  const auto sizes = store.SizeMany(names, 2);
  const std::vector<void *> destinations = {first.data(), second.data()};
  const std::vector<std::size_t> destination_sizes = {first.size(),
                                                      second.size()};

  REQUIRE(store.GetManyInto(names, destinations, destination_sizes, sizes, 2) ==
          std::vector<bool>{true, true});
  REQUIRE(first == std::vector<std::uint8_t>({9, 8, 7}));
  REQUIRE(second == std::vector<std::uint8_t>({6, 5, 4, 3}));
}

TEST_CASE("LMCacheStore preserves record encoding for batched record writes",
          "[lmcache][integration]") {
  auto store = MakeStore("test_lmcache_batch_records");
  const std::vector<std::string> names = {"record_a", "record_b"};
  const std::vector<std::string> metadata = {
      R"({"canonical_key":"record_a","payload_length":3})",
      R"({"canonical_key":"record_b","payload_length":4})"};
  const std::vector<std::string> payloads = {
      ByteString(std::vector<std::uint8_t>{0, 255, 1}),
      ByteString(std::vector<std::uint8_t>{2, 3, 4, 5})};
  const std::vector<const void *> payload_ptrs = {payloads[0].data(),
                                                  payloads[1].data()};
  const std::vector<std::size_t> payload_sizes = {payloads[0].size(),
                                                  payloads[1].size()};

  REQUIRE(store.PutManyRecords(names, metadata, payload_ptrs, payload_sizes,
                               1) == std::vector<bool>{true, true});

  const auto values = store.GetMany(names, 2);
  REQUIRE(values.size() == 2);
  REQUIRE(values[0].has_value());
  REQUIRE(values[1].has_value());
  REQUIRE(*values[0] == BuildRecord(metadata[0], payloads[0]));
  REQUIRE(*values[1] == BuildRecord(metadata[1], payloads[1]));
}

TEST_CASE("LMCacheStore reads record info and payload ranges into callers",
          "[lmcache][integration]") {
  auto store = MakeStore("test_lmcache_record_ranges");
  const std::vector<std::string> names = {"range_record_a", "range_record_b"};
  const std::vector<std::string> metadata = {
      R"({"canonical_key":"range_record_a","payload_length":3})",
      R"({"canonical_key":"range_record_b","payload_length":4})"};
  const std::vector<std::string> payloads = {
      ByteString(std::vector<std::uint8_t>{10, 11, 12}),
      ByteString(std::vector<std::uint8_t>{13, 14, 15, 16})};
  const std::vector<const void *> payload_ptrs = {payloads[0].data(),
                                                  payloads[1].data()};
  const std::vector<std::size_t> payload_sizes = {payloads[0].size(),
                                                  payloads[1].size()};

  REQUIRE(store.PutManyRecords(names, metadata, payload_ptrs, payload_sizes,
                               2) == std::vector<bool>{true, true});

  const auto infos = store.ReadRecordInfos({"range_record_a", "missing",
                                            "range_record_b"}, 2);
  REQUIRE(infos.size() == 3);
  REQUIRE(infos[0].has_value());
  REQUIRE_FALSE(infos[1].has_value());
  REQUIRE(infos[2].has_value());
  REQUIRE(infos[0]->metadata_json == metadata[0]);
  REQUIRE(infos[0]->payload_offset == 24 + metadata[0].size());
  REQUIRE(infos[0]->payload_size == payloads[0].size());
  REQUIRE(infos[2]->metadata_json == metadata[1]);
  REQUIRE(infos[2]->payload_offset == 24 + metadata[1].size());
  REQUIRE(infos[2]->payload_size == payloads[1].size());

  std::vector<std::uint8_t> first(payloads[0].size());
  std::vector<std::uint8_t> second(payloads[1].size());
  const std::vector<void *> destinations = {first.data(), second.data()};
  const std::vector<std::size_t> destination_sizes = {first.size(),
                                                      second.size()};
  REQUIRE(store.GetManyRangesInto(
              names, {infos[0]->payload_offset, infos[2]->payload_offset},
              {infos[0]->payload_size, infos[2]->payload_size}, destinations,
              destination_sizes, 2) == std::vector<bool>{true, true});
  REQUIRE(first == std::vector<std::uint8_t>({10, 11, 12}));
  REQUIRE(second == std::vector<std::uint8_t>({13, 14, 15, 16}));
}

TEST_CASE("LMCacheStore range reads round trip with direct reads toggled",
          "[lmcache][integration]") {
  using clio_llm::lmcache::LMCacheStoreTestAccess;
  for (const char *setting : {"", "false", "1", "TrUe", "unsupported"}) {
    ScopedEnvironmentVariable direct_read_setting("CLIO_LMCACHE_DIRECT_READ",
                                                  setting);
    auto store = MakeStore(std::string("test_lmcache_direct_read_") + setting);
    const std::string payload = "direct-read-toggle-payload";
    const bool expected_enabled =
        std::string(setting) == "1" || std::string(setting) == "TrUe";

    REQUIRE(LMCacheStoreTestAccess::DirectReadEnabled(store) ==
            expected_enabled);
    REQUIRE(store.PutBytes("toggle_record", payload));
    std::vector<char> destination(payload.size());
    REQUIRE(store.GetManyRangesInto({"toggle_record"}, {0}, {payload.size()},
                                    {destination.data()}, {destination.size()},
                                    1) == std::vector<bool>{true});
    REQUIRE(std::string(destination.data(), destination.size()) == payload);
  }
}

TEST_CASE("LMCacheStore latches and moves direct-read configuration",
          "[lmcache][integration]") {
  using clio_llm::lmcache::LMCacheStore;
  using clio_llm::lmcache::LMCacheStoreTestAccess;
  ScopedEnvironmentVariable direct_read_setting("CLIO_LMCACHE_DIRECT_READ",
                                                "true");
  const std::string tag_name = "test_lmcache_direct_read_latching";
  auto store = MakeStore(tag_name);
  REQUIRE(LMCacheStoreTestAccess::DirectReadEnabled(store));

  ctp::SystemInfo::Setenv("CLIO_LMCACHE_DIRECT_READ", "false", 1);
  REQUIRE(store.Init("", tag_name, "local"));
  REQUIRE(LMCacheStoreTestAccess::DirectReadEnabled(store));

  LMCacheStore moved_store = std::move(store);
  REQUIRE(LMCacheStoreTestAccess::DirectReadEnabled(moved_store));
  REQUIRE_FALSE(LMCacheStoreTestAccess::DirectReadEnabled(store));

  moved_store.Close();
  REQUIRE_FALSE(LMCacheStoreTestAccess::DirectReadEnabled(moved_store));
  REQUIRE(moved_store.Init("", tag_name, "local"));
  REQUIRE_FALSE(LMCacheStoreTestAccess::DirectReadEnabled(moved_store));
}
