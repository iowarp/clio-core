/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

// Unit tests for SystemInfo::ReadMemoryLimitFile, the cgroup memory-limit
// parser behind GetProcessMemoryBudget. CI runners usually have no readable
// cgroup limit file, so the parsing branches are driven here with temp files.

#include <catch2/catch_all.hpp>

#include <filesystem>
#include <fstream>
#include <string>

#include <clio_ctp/introspect/system_info.h>

namespace {

/**
 * Write @p contents to a fresh file in the temp directory.
 * @param name file name inside the temp directory
 * @param contents bytes to write
 * @return full path of the file
 */
std::string WriteTempFile(const std::string &name,
                          const std::string &contents) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / name;
  std::ofstream out(path, std::ios::trunc);
  out << contents;
  return path.string();
}

}  // namespace

TEST_CASE("ReadMemoryLimitFile parses a numeric limit", "[sysinfo]") {
  const std::string path = WriteTempFile("ctp_memlimit_num", "1073741824\n");
  REQUIRE(ctp::SystemInfo::ReadMemoryLimitFile(path) == 1073741824ULL);
  std::filesystem::remove(path);
}

TEST_CASE("ReadMemoryLimitFile treats unlimited values as 0", "[sysinfo]") {
  // cgroup v2 spelling.
  const std::string v2 = WriteTempFile("ctp_memlimit_max", "max\n");
  REQUIRE(ctp::SystemInfo::ReadMemoryLimitFile(v2) == 0);
  std::filesystem::remove(v2);

  // cgroup v1 "unlimited" sentinel (PAGE_COUNTER_MAX * page size).
  const std::string v1 =
      WriteTempFile("ctp_memlimit_v1", "9223372036854771712\n");
  REQUIRE(ctp::SystemInfo::ReadMemoryLimitFile(v1) == 0);
  std::filesystem::remove(v1);
}

TEST_CASE("ReadMemoryLimitFile rejects unreadable or malformed files",
          "[sysinfo]") {
  REQUIRE(ctp::SystemInfo::ReadMemoryLimitFile(
              "/nonexistent/ctp_memlimit_missing") == 0);

  const std::string empty = WriteTempFile("ctp_memlimit_empty", "");
  REQUIRE(ctp::SystemInfo::ReadMemoryLimitFile(empty) == 0);
  std::filesystem::remove(empty);

  const std::string junk = WriteTempFile("ctp_memlimit_junk", "unlimited\n");
  REQUIRE(ctp::SystemInfo::ReadMemoryLimitFile(junk) == 0);
  std::filesystem::remove(junk);

  const std::string zero = WriteTempFile("ctp_memlimit_zero", "0\n");
  REQUIRE(ctp::SystemInfo::ReadMemoryLimitFile(zero) == 0);
  std::filesystem::remove(zero);
}
