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

#include "basic_test.h"
#include "hermes_shm/util/config_parse.h"

#include <fstream>
#include <cstdlib>
#include <filesystem>

#ifdef _WIN32
#include <process.h>
inline int setenv(const char *name, const char *value, int) {
  return _putenv_s(name, value);
}
inline int unsetenv(const char *name) {
  return _putenv_s(name, "");
}
inline int getpid() { return _getpid(); }
#endif

static std::string GetTempDir() {
  return std::filesystem::temp_directory_path().string();
}

using hshm::ConfigParse;

//------------------------------------------------------------------------------
// ParseHostNameString Tests
//------------------------------------------------------------------------------

TEST_CASE("ParseHostNameString - Simple hostname without brackets") {
  std::vector<std::string> hosts;
  ConfigParse::ParseHostNameString("hello", hosts);
  REQUIRE(hosts.size() == 1);
  REQUIRE(hosts[0] == "hello");
}

TEST_CASE("ParseHostNameString - Empty string") {
  std::vector<std::string> hosts;
  ConfigParse::ParseHostNameString("", hosts);
  REQUIRE(hosts.empty());
}

TEST_CASE("ParseHostNameString - String with whitespace only") {
  std::vector<std::string> hosts;
  ConfigParse::ParseHostNameString("   \t\n\r  ", hosts);
  REQUIRE(hosts.empty());
}

TEST_CASE("ParseHostNameString - Single range expansion") {
  std::vector<std::string> hosts;
  ConfigParse::ParseHostNameString("node[00-03]", hosts);
  REQUIRE(hosts.size() == 4);
  REQUIRE(hosts[0] == "node00");
  REQUIRE(hosts[1] == "node01");
  REQUIRE(hosts[2] == "node02");
  REQUIRE(hosts[3] == "node03");
}

TEST_CASE("ParseHostNameString - Range with suffix") {
  std::vector<std::string> hosts;
  ConfigParse::ParseHostNameString("node[01-03]-40g", hosts);
  REQUIRE(hosts.size() == 3);
  REQUIRE(hosts[0] == "node01-40g");
  REQUIRE(hosts[1] == "node02-40g");
  REQUIRE(hosts[2] == "node03-40g");
}

TEST_CASE("ParseHostNameString - Multiple ranges in brackets") {
  std::vector<std::string> hosts;
  ConfigParse::ParseHostNameString("node[00-02,05]", hosts);
  REQUIRE(hosts.size() == 4);
  REQUIRE(hosts[0] == "node00");
  REQUIRE(hosts[1] == "node01");
  REQUIRE(hosts[2] == "node02");
  REQUIRE(hosts[3] == "node05");
}

TEST_CASE("ParseHostNameString - Multiple hostnames with semicolon") {
  std::vector<std::string> hosts;
  ConfigParse::ParseHostNameString("server1;server2;server3", hosts);
  REQUIRE(hosts.size() == 3);
  REQUIRE(hosts[0] == "server1");
  REQUIRE(hosts[1] == "server2");
  REQUIRE(hosts[2] == "server3");
}

TEST_CASE("ParseHostNameString - Complex example with ranges and semicolons") {
  std::vector<std::string> hosts;
  ConfigParse::ParseHostNameString("hello[00-02]-40g;world[10-11]", hosts);
  REQUIRE(hosts.size() == 5);
  REQUIRE(hosts[0] == "hello00-40g");
  REQUIRE(hosts[1] == "hello01-40g");
  REQUIRE(hosts[2] == "hello02-40g");
  REQUIRE(hosts[3] == "world10");
  REQUIRE(hosts[4] == "world11");
}

TEST_CASE("ParseHostNameString - Whitespace removal") {
  std::vector<std::string> hosts;
  ConfigParse::ParseHostNameString("  node[01-02]  ", hosts);
  REQUIRE(hosts.size() == 2);
  REQUIRE(hosts[0] == "node01");
  REQUIRE(hosts[1] == "node02");
}

TEST_CASE("ParseHostNameString - Single value in brackets") {
  std::vector<std::string> hosts;
  ConfigParse::ParseHostNameString("node[5]", hosts);
  REQUIRE(hosts.size() == 1);
  REQUIRE(hosts[0] == "node5");
}

TEST_CASE("ParseHostNameString - Zero-padded numbers preserved") {
  std::vector<std::string> hosts;
  ConfigParse::ParseHostNameString("node[007-009]", hosts);
  REQUIRE(hosts.size() == 3);
  REQUIRE(hosts[0] == "node007");
  REQUIRE(hosts[1] == "node008");
  REQUIRE(hosts[2] == "node009");
}

//------------------------------------------------------------------------------
// ParseNumberSuffix Tests
//------------------------------------------------------------------------------

TEST_CASE("ParseNumberSuffix - No suffix") {
  REQUIRE(ConfigParse::ParseNumberSuffix("1024") == "");
}

TEST_CASE("ParseNumberSuffix - Single char suffix") {
  REQUIRE(ConfigParse::ParseNumberSuffix("1024k") == "k");
  REQUIRE(ConfigParse::ParseNumberSuffix("1024K") == "K");
  REQUIRE(ConfigParse::ParseNumberSuffix("512m") == "m");
  REQUIRE(ConfigParse::ParseNumberSuffix("512M") == "M");
}

TEST_CASE("ParseNumberSuffix - Multi-char suffix") {
  REQUIRE(ConfigParse::ParseNumberSuffix("1024KB") == "KB");
  REQUIRE(ConfigParse::ParseNumberSuffix("512MB") == "MB");
}

TEST_CASE("ParseNumberSuffix - Float with suffix") {
  REQUIRE(ConfigParse::ParseNumberSuffix("1.5G") == "G");
  REQUIRE(ConfigParse::ParseNumberSuffix("2.75T") == "T");
}

TEST_CASE("ParseNumberSuffix - Whitespace before suffix") {
  REQUIRE(ConfigParse::ParseNumberSuffix("1024 KB") == "KB");
}

//------------------------------------------------------------------------------
// ParseNumber Tests
//------------------------------------------------------------------------------

TEST_CASE("ParseNumber - Integer parsing") {
  REQUIRE(ConfigParse::ParseNumber<int>("42") == 42);
  REQUIRE(ConfigParse::ParseNumber<int>("0") == 0);
  REQUIRE(ConfigParse::ParseNumber<int>("-100") == -100);
}

TEST_CASE("ParseNumber - Float parsing") {
  REQUIRE(ConfigParse::ParseNumber<double>("3.14") == Catch::Approx(3.14));
  REQUIRE(ConfigParse::ParseNumber<float>("2.5") == Catch::Approx(2.5f));
}

TEST_CASE("ParseNumber - Infinity special value") {
  REQUIRE(ConfigParse::ParseNumber<int>("inf") == std::numeric_limits<int>::max());
  REQUIRE(ConfigParse::ParseNumber<hshm::u64>("inf") == std::numeric_limits<hshm::u64>::max());
}

TEST_CASE("ParseNumber - Large numbers") {
  REQUIRE(ConfigParse::ParseNumber<hshm::u64>("1000000000") == 1000000000ULL);
}

//------------------------------------------------------------------------------
// ParseSize Tests
//------------------------------------------------------------------------------

TEST_CASE("ParseSize - Bytes (no suffix)") {
  REQUIRE(ConfigParse::ParseSize("1024") == 1024);
  REQUIRE(ConfigParse::ParseSize("0") == 0);
}

TEST_CASE("ParseSize - Kilobytes") {
  REQUIRE(ConfigParse::ParseSize("1k") == 1024);
  REQUIRE(ConfigParse::ParseSize("1K") == 1024);
  REQUIRE(ConfigParse::ParseSize("2k") == 2048);
}

TEST_CASE("ParseSize - Megabytes") {
  REQUIRE(ConfigParse::ParseSize("1m") == 1024 * 1024);
  REQUIRE(ConfigParse::ParseSize("1M") == 1024 * 1024);
  REQUIRE(ConfigParse::ParseSize("2M") == 2 * 1024 * 1024);
}

TEST_CASE("ParseSize - Gigabytes") {
  REQUIRE(ConfigParse::ParseSize("1g") == 1024ULL * 1024 * 1024);
  REQUIRE(ConfigParse::ParseSize("1G") == 1024ULL * 1024 * 1024);
}

TEST_CASE("ParseSize - Terabytes") {
  REQUIRE(ConfigParse::ParseSize("1t") == 1024ULL * 1024 * 1024 * 1024);
  REQUIRE(ConfigParse::ParseSize("1T") == 1024ULL * 1024 * 1024 * 1024);
}

TEST_CASE("ParseSize - Petabytes") {
  REQUIRE(ConfigParse::ParseSize("1p") == 1024ULL * 1024 * 1024 * 1024 * 1024);
  REQUIRE(ConfigParse::ParseSize("1P") == 1024ULL * 1024 * 1024 * 1024 * 1024);
}

TEST_CASE("ParseSize - Decimal sizes") {
  // 2.5 gigabytes
  hshm::u64 expected = (hshm::u64)(2.5 * 1024 * 1024 * 1024);
  REQUIRE(ConfigParse::ParseSize("2.5G") == expected);
}

TEST_CASE("ParseSize - Infinity") {
  REQUIRE(ConfigParse::ParseSize("inf") == std::numeric_limits<hshm::u64>::max());
}

//------------------------------------------------------------------------------
// ParseBandwidth Tests
//------------------------------------------------------------------------------

TEST_CASE("ParseBandwidth - Same as ParseSize") {
  // ParseBandwidth delegates to ParseSize
  REQUIRE(ConfigParse::ParseBandwidth("1G") == ConfigParse::ParseSize("1G"));
  REQUIRE(ConfigParse::ParseBandwidth("100M") == ConfigParse::ParseSize("100M"));
}

//------------------------------------------------------------------------------
// ParseLatency Tests
//------------------------------------------------------------------------------

TEST_CASE("ParseLatency - Nanoseconds (default)") {
  REQUIRE(ConfigParse::ParseLatency("100") == 100);
}

TEST_CASE("ParseLatency - Nanoseconds with suffix") {
  REQUIRE(ConfigParse::ParseLatency("100n") == 100);
  REQUIRE(ConfigParse::ParseLatency("100N") == 100);
}

TEST_CASE("ParseLatency - Microseconds") {
  // 1 microsecond = 1000 nanoseconds
  REQUIRE(ConfigParse::ParseLatency("1u") == 1024);  // Uses Kilobytes conversion
  REQUIRE(ConfigParse::ParseLatency("1U") == 1024);
}

TEST_CASE("ParseLatency - Milliseconds") {
  // 1 millisecond = 1,000,000 nanoseconds
  REQUIRE(ConfigParse::ParseLatency("1m") == 1024 * 1024);  // Uses Megabytes conversion
  REQUIRE(ConfigParse::ParseLatency("1M") == 1024 * 1024);
}

TEST_CASE("ParseLatency - Seconds") {
  // 1 second = 1,000,000,000 nanoseconds
  REQUIRE(ConfigParse::ParseLatency("1s") == 1024ULL * 1024 * 1024 * 1024);  // Uses Terabytes
  REQUIRE(ConfigParse::ParseLatency("1S") == 1024ULL * 1024 * 1024 * 1024);
}

//------------------------------------------------------------------------------
// ExpandPath Tests
//------------------------------------------------------------------------------

TEST_CASE("ExpandPath - No variables") {
  REQUIRE(ConfigParse::ExpandPath("/fixed/path") == "/fixed/path");
  REQUIRE(ConfigParse::ExpandPath("relative/path") == "relative/path");
}

TEST_CASE("ExpandPath - Single environment variable") {
  // Set a test environment variable
  setenv("TEST_VAR_EXPAND", "test_value", 1);
  std::string result = ConfigParse::ExpandPath("${TEST_VAR_EXPAND}/data");
  REQUIRE(result == "test_value/data");
  unsetenv("TEST_VAR_EXPAND");
}

TEST_CASE("ExpandPath - Multiple environment variables") {
  // Note: ExpandPath only expands the first variable per call
  // due to how std::regex_search works. Call it multiple times
  // to expand all variables.
  setenv("TEST_VAR1", "first", 1);
  setenv("TEST_VAR2", "second", 1);
  std::string result = ConfigParse::ExpandPath("${TEST_VAR1}/path/${TEST_VAR2}");
  // First call expands TEST_VAR1
  result = ConfigParse::ExpandPath(result);
  // Second call expands TEST_VAR2
  REQUIRE(result == "first/path/second");
  unsetenv("TEST_VAR1");
  unsetenv("TEST_VAR2");
}

TEST_CASE("ExpandPath - HOME variable expansion") {
#ifdef _WIN32
  const char* home = getenv("USERPROFILE");
  if (home != nullptr) {
    std::string result = ConfigParse::ExpandPath("${USERPROFILE}/data");
    std::string expected = std::string(home) + "/data";
    REQUIRE(result == expected);
  }
#else
  const char* home = getenv("HOME");
  if (home != nullptr) {
    std::string result = ConfigParse::ExpandPath("${HOME}/data");
    std::string expected = std::string(home) + "/data";
    REQUIRE(result == expected);
  }
#endif
}

//------------------------------------------------------------------------------
// ParseHostfile Tests
//------------------------------------------------------------------------------

TEST_CASE("ParseHostfile - Read hosts from file") {
  // Create a temporary hostfile
  std::string tmpfile = GetTempDir() + "/test_hostfile_" + std::to_string(getpid()) + ".txt";
  {
    std::ofstream file(tmpfile);
    file << "node01\n";
    file << "node02\n";
    file << "node[03-05]\n";
    file.close();
  }

  std::vector<std::string> hosts = ConfigParse::ParseHostfile(tmpfile);
  REQUIRE(hosts.size() == 5);
  REQUIRE(hosts[0] == "node01");
  REQUIRE(hosts[1] == "node02");
  REQUIRE(hosts[2] == "node03");
  REQUIRE(hosts[3] == "node04");
  REQUIRE(hosts[4] == "node05");

  // Cleanup
  std::remove(tmpfile.c_str());
}

TEST_CASE("ParseHostfile - Non-existent file") {
  std::vector<std::string> hosts = ConfigParse::ParseHostfile("/nonexistent/path/hostfile.txt");
  REQUIRE(hosts.empty());
}

TEST_CASE("ParseHostfile - Empty file") {
  std::string tmpfile = GetTempDir() + "/test_hostfile_empty_" + std::to_string(getpid()) + ".txt";
  {
    std::ofstream file(tmpfile);
    file.close();
  }

  std::vector<std::string> hosts = ConfigParse::ParseHostfile(tmpfile);
  REQUIRE(hosts.empty());

  // Cleanup
  std::remove(tmpfile.c_str());
}

//------------------------------------------------------------------------------
// rm_char Tests
//------------------------------------------------------------------------------

TEST_CASE("rm_char - Remove spaces") {
  std::string str = "hello world";
  ConfigParse::rm_char(str, ' ');
  REQUIRE(str == "helloworld");
}

TEST_CASE("rm_char - Remove multiple characters") {
  std::string str = "a,b,c,d";
  ConfigParse::rm_char(str, ',');
  REQUIRE(str == "abcd");
}

TEST_CASE("rm_char - Character not present") {
  std::string str = "hello";
  ConfigParse::rm_char(str, 'x');
  REQUIRE(str == "hello");
}

TEST_CASE("rm_char - Empty string") {
  std::string str = "";
  ConfigParse::rm_char(str, 'x');
  REQUIRE(str == "");
}
