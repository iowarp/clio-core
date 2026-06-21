/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * STDIO ADAPTER UNIT TESTS
 *
 * Exercise the WRP CTE STDIO adapter (libclio_cte_stdio.so) under the
 * new clio:: prefix gating. Each test opens a file with fopen() on a
 * clio::-prefixed path so the adapter intercepts; bare paths fall
 * through to libc and act as a negative control.
 */

#include <catch2/catch_all.hpp>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <unistd.h>
#include <vector>

#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/bdev/bdev_client.h"
#include "clio_cte/core/core_client.h"

namespace stdfs = std::filesystem;

namespace {
constexpr size_t kSmall = 4096;
const std::string kBackend = "/tmp/clio_cte_stdio_test.dat";
const std::string kClio = "clio::" + kBackend;

bool initializeRuntime() {
  static bool initialized = false;
  if (initialized) return true;

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    INFO("Clio init failed");
    return false;
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    INFO("CTE init failed");
    return false;
  }
  // Give the CTE core pool a file-backed storage target.
  auto *cte_client = CLIO_CTE_CLIENT;
  clio::run::PoolId bdev_pool_id(951, 0);
  clio::run::bdev::Client bdev_client(bdev_pool_id);
  auto create_task = bdev_client.AsyncCreate(
      clio::run::PoolQuery::Dynamic(), kBackend, bdev_pool_id,
      clio::run::bdev::BdevType::kFile);
  create_task.Wait();
  auto reg_task = cte_client->AsyncRegisterTarget(
      kBackend, clio::run::bdev::BdevType::kFile, kSmall * 64,
      clio::run::PoolQuery::Local(), bdev_pool_id);
  reg_task.Wait();
  if (reg_task->GetReturnCode() != 0) {
    INFO("Failed to register target, rc=" << reg_task->GetReturnCode());
    return false;
  }
  initialized = true;
  return true;
}
}  // namespace

TEST_CASE("STDIO Adapter: fopen/fwrite/fread/fclose round-trips",
          "[stdio][adapter]") {
  REQUIRE(initializeRuntime());
  stdfs::remove(kBackend);

  // Write.
  FILE *fp = fopen(kClio.c_str(), "w");
  REQUIRE(fp != nullptr);
  std::vector<char> write_data(kSmall);
  for (size_t i = 0; i < kSmall; ++i) write_data[i] = char(i & 0xff);
  REQUIRE(fwrite(write_data.data(), 1, kSmall, fp) == kSmall);
  REQUIRE(fclose(fp) == 0);

  // Read.
  fp = fopen(kClio.c_str(), "r");
  REQUIRE(fp != nullptr);
  std::vector<char> read_data(kSmall, 0);
  REQUIRE(fread(read_data.data(), 1, kSmall, fp) == kSmall);
  REQUIRE(read_data == write_data);
  REQUIRE(fclose(fp) == 0);

  stdfs::remove(kBackend);
}

TEST_CASE("STDIO Adapter: fseek + ftell + rewind + fflush",
          "[stdio][adapter][seek]") {
  REQUIRE(initializeRuntime());
  stdfs::remove(kBackend);

  FILE *fp = fopen(kClio.c_str(), "w+");
  REQUIRE(fp != nullptr);

  // Write 'A' x 1024 then 'B' x 1024.
  std::vector<char> a(1024, 'A'), b(1024, 'B');
  REQUIRE(fwrite(a.data(), 1, 1024, fp) == 1024);
  REQUIRE(fwrite(b.data(), 1, 1024, fp) == 1024);
  REQUIRE(fflush(fp) == 0);

  // ftell should report 2048.
  REQUIRE(ftell(fp) == 2048);

  // Seek to offset 1024, read should yield 1024 'B's.
  REQUIRE(fseek(fp, 1024, SEEK_SET) == 0);
  std::vector<char> r(1024, 0);
  REQUIRE(fread(r.data(), 1, 1024, fp) == 1024);
  REQUIRE(r == b);

  // rewind back, read 1024 'A's.
  rewind(fp);
  std::fill(r.begin(), r.end(), 0);
  REQUIRE(fread(r.data(), 1, 1024, fp) == 1024);
  REQUIRE(r == a);

  REQUIRE(fclose(fp) == 0);
  stdfs::remove(kBackend);
}

TEST_CASE("STDIO Adapter: clio:: prefix is required for interception",
          "[stdio][adapter][prefix]") {
  REQUIRE(initializeRuntime());
  const std::string backend = "/tmp/clio_cte_stdio_prefix.dat";
  stdfs::remove(backend);

  SECTION("bare path goes to libc, fd is small") {
    FILE *fp = fopen(backend.c_str(), "w");
    REQUIRE(fp != nullptr);
    REQUIRE(fileno(fp) < 8192);
    REQUIRE(fclose(fp) == 0);
    stdfs::remove(backend);
  }

  SECTION("clio:: path is intercepted, fd is CTE-issued") {
    const std::string clio = "clio::" + backend;
    FILE *fp = fopen(clio.c_str(), "w");
    REQUIRE(fp != nullptr);
    // Under interception, the FILE* wraps a CTE fd >= 8192.
    REQUIRE(fileno(fp) >= 8192);
    REQUIRE(fclose(fp) == 0);
    stdfs::remove(backend);
  }
}

TEST_CASE("STDIO Adapter: fopen of missing clio:: file with mode \"r\" fails",
          "[stdio][adapter][errors]") {
  REQUIRE(initializeRuntime());
  errno = 0;
  FILE *fp = fopen("clio::/tmp/clio_cte_stdio_missing.dat", "r");
  REQUIRE(fp == nullptr);
  REQUIRE(errno != 0);
}
