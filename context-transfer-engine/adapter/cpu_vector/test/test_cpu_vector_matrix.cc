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
 * cpu_vector across the storage stack gpu_vector depends on, without a GPU.
 *
 * One process is one cell of a matrix chosen by environment variables (one
 * runtime per process, so each cell is its own ctest entry):
 *
 *   CLIO_CVM_CODEC       CTE compression library id; 0 = raw pages
 *   CLIO_CVM_TIERS       "single": one RAM tier holding everything;
 *                        "spill":  a RAM tier far smaller than the data
 *                                  above a file tier, so pages must spill
 *   CLIO_CVM_COMPRESSOR  1: compose the compressor module at 512.0 in front
 *                        of the core at 513.0 (the shipped layout), and route
 *                        the vector through it; 0: the core alone at 512.0
 *   CLIO_CVM_PORT        runtime port (default 9471)
 *
 * Every cell writes through a cache far smaller than the vector (out of core),
 * reads it back through the cache and through Download, checks a resident
 * vector, and checks a generational flush/fetch, all byte-exact. A codec or a
 * pool layout that silently stores nothing, refuses fetches, or places pages
 * on no tier fails here instead of in a GPU benchmark.
 */

#include <clio_cte/core/core_client.h>
#include <clio_cte/cpu_vector/cpu_vector.h>
#include <clio_runtime/clio_runtime.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace fs = std::filesystem;
namespace cv = clio::cte::cpu_vector;
using clio::run::u32;
using clio::run::u64;

namespace {

/**
 * Read a CLIO_<name> environment variable.
 * @param name variable name without the CLIO_ prefix
 * @param def  value when unset or empty
 * @return the value, or def
 */
std::string EnvOr(const char *name, const std::string &def) {
  const char *e = clio::run::env::GetCompat(name);
  return (e != nullptr && *e != '\0') ? std::string(e) : def;
}

/** Where the config and the file tier go: CLIO_TEST_DATA_DIR if it names a
 *  real directory, else the system temp directory. */
std::string TestDataDir() {
  std::string d = EnvOr("TEST_DATA_DIR", "");
  while (!d.empty() && (d.back() == ' ' || d.back() == '\t')) d.pop_back();
  std::error_code ec;
  if (!d.empty() && fs::is_directory(d, ec)) return d;
  return fs::temp_directory_path(ec).string();
}

/** The matrix cell this process runs. */
struct Cell {
  int codec = 0;
  bool spill = false;
  bool compressor = false;
  std::string port = "9471";
  /** Short name, unique per cell, for files and tags. */
  std::string Name() const {
    return "cvm_c" + std::to_string(codec) + (spill ? "_spill" : "_single") +
           (compressor ? "_cmp" : "_core");
  }
};

Cell ReadCell() {
  Cell c;
  c.codec = std::atoi(EnvOr("CVM_CODEC", "0").c_str());
  c.spill = EnvOr("CVM_TIERS", "single") == "spill";
  c.compressor = EnvOr("CVM_COMPRESSOR", "0") == "1";
  c.port = EnvOr("CVM_PORT", "9471");
  return c;
}

/**
 * Write the runtime config for a cell.
 *
 * Tier scores are at or below the vector's blob score (0.5) so every tier is
 * reachable by placement; in "spill" the 4 MB RAM tier is preferred and the
 * file tier takes the rest.
 *
 * @param c    the cell
 * @param dir  directory for the config and the file tier
 * @return path of the config file
 */
std::string WriteConfig(const Cell &c, const std::string &dir) {
  const std::string cfg = dir + "/" + c.Name() + ".yaml";
  const std::string file_tier = dir + "/" + c.Name() + "_tier.dat";
  std::error_code ec;
  fs::remove(file_tier, ec);
  std::ofstream f(cfg);
  REQUIRE(f.is_open());
  f << "networking:\n  port: " << c.port << "\n"
    << "runtime:\n  num_threads: 4\n  queue_depth: 1024\n"
    << "compose:\n";
  const std::string core_id = c.compressor ? "513.0" : "512.0";
  if (c.compressor) {
    f << "  - mod_name: clio_cte_compressor\n"
      << "    pool_name: cte_compressor\n    pool_query: local\n"
      << "    pool_id: \"512.0\"\n    next_pool_id: \"513.0\"\n";
  }
  f << "  - mod_name: clio_cte_core\n"
    << "    pool_name: cte_core\n    pool_query: local\n"
    << "    pool_id: \"" << core_id << "\"\n"
    << "    storage:\n"
    << "      - path: \"ram::" << c.Name() << "_ram\"\n"
    << "        bdev_type: \"ram\"\n"
    << "        capacity_limit: \"" << (c.spill ? "4MB" : "256MB") << "\"\n"
    << "        score: 0.5\n";
  if (c.spill) {
    f << "      - path: \"" << file_tier << "\"\n"
      << "        bdev_type: \"file\"\n"
      << "        capacity_limit: \"256MB\"\n"
      << "        score: 0.2\n";
  }
  f << "    dpe:\n      dpe_type: \"max_bw\"\n";
  return cfg;
}

/** Runtime started once per process, for the cell in the environment. */
struct Fixture {
  Cell cell;
  clio::run::PoolId storage_pool;
  Fixture() {
    cell = ReadCell();
    const std::string cfg = WriteConfig(cell, TestDataDir());
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", cfg, 1);
    REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    // The vector's puts and gets go to the pool at 512.0 either way: the
    // compressor when it is composed, the core otherwise.
    storage_pool = clio::run::PoolId(512, 0);
  }
};

Fixture &Env() {
  static Fixture f;
  return f;
}

constexpr u64 kPageBytes = 64 * 1024;
constexpr u64 kEpp = kPageBytes / sizeof(u64);

/**
 * Deterministic page content: half of each page is a slowly varying ramp
 * (compressible), half mixes in a multiplicative hash (not), so a codec both
 * shrinks pages and has to round-trip bytes it cannot predict.
 * @param i  element index
 * @param salt  distinguishes vectors and generations
 * @return the element's value
 */
u64 Value(u64 i, u64 salt) {
  if ((i % kEpp) < kEpp / 2) return (i / 64) + salt;
  return (i + salt) * 0x9E3779B97F4A7C15ull;
}

/**
 * Write every page through the cache, one page at a time, flushing as it goes.
 * @param v     the vector
 * @param n     element count
 * @param salt  content salt
 * @param gen   generation stamped on the flush (0 = none)
 */
void WriteAll(cv::Vector<u64> &v, u64 n, u64 salt, u64 gen) {
  for (u64 off = 0; off < n; off += kEpp) {
    const u64 cnt = (off + kEpp <= n) ? kEpp : (n - off);
    v.Fetch(0, off, cnt);
    {
      auto h = v.HoldPage(off, cnt, /*write=*/true);
      REQUIRE(h);
      for (u64 i = off; i < off + cnt; ++i) h[i] = Value(i, salt);
    }
    v.Flush(gen, off, cnt);
    v.UnpinRange(off, cnt);
  }
}

/**
 * Read every page back through the cache and compare.
 * @param v     the vector
 * @param n     element count
 * @param salt  content salt the pages were written with
 * @param gen   generation to fetch at (0 = none)
 */
void CheckAll(cv::Vector<u64> &v, u64 n, u64 salt, u64 gen) {
  for (u64 off = 0; off < n; off += kEpp) {
    const u64 cnt = (off + kEpp <= n) ? kEpp : (n - off);
    v.Fetch(gen, off, cnt);
    {
      auto h = v.HoldPage(off, cnt);
      REQUIRE(h);
      for (u64 i = off; i < off + cnt; ++i) REQUIRE(h[i] == Value(i, salt));
    }
    v.UnpinRange(off, cnt);
  }
}

}  // namespace

TEST_CASE("cpu_vector matrix: out-of-core write, read back, download",
          "[cpu_vector][matrix]") {
  auto &env = Env();
  // 16 MB through a 4-frame (256 KB) cache: every page is evicted and
  // refetched, and in "spill" the RAM tier holds only a quarter of it.
  const u64 pages = 256;
  const u64 n = pages * kEpp;
  cv::Vector<u64> v(env.cell.Name() + "_ooc", kPageBytes, n, 2, 2,
                    env.storage_pool, env.cell.codec, 1);
  v.EnableStats();
  WriteAll(v, n, 11, 0);
  CheckAll(v, n, 11, 0);

  std::vector<u64> out(n, 0);
  REQUIRE(v.Download(out.data(), n) == n);
  for (u64 i = 0; i < n; ++i) REQUIRE(out[i] == Value(i, 11));

  const auto s = v.ReadStats();
  REQUIRE(s.evicts > 0);
  REQUIRE(s.get_errors == 0);
  REQUIRE(s.put_errors == 0);
}

TEST_CASE("cpu_vector matrix: resident vector", "[cpu_vector][matrix]") {
  auto &env = Env();
  const u64 pages = 16;
  const u64 n = pages * kEpp + 123;  // a partial last page
  cv::Vector<u64> v(env.cell.Name() + "_res", kPageBytes, n, 4, 8,
                    env.storage_pool, env.cell.codec, 1);
  v.EnableStats();
  WriteAll(v, n, 29, 0);
  CheckAll(v, n, 29, 0);
  const auto s = v.ReadStats();
  REQUIRE(s.get_errors == 0);
  REQUIRE(s.put_errors == 0);
}

TEST_CASE("cpu_vector matrix: generational flush releases generational fetch",
          "[cpu_vector][matrix]") {
  auto &env = Env();
  const u64 n = 8 * kEpp;
  const std::string tag = env.cell.Name() + "_gen";
  cv::Vector<u64> w(tag, kPageBytes, n, 2, 4, env.storage_pool,
                    env.cell.codec, 1);
  cv::Vector<u64> r(tag, kPageBytes, n, 2, 4, env.storage_pool,
                    env.cell.codec, 1);
  WriteAll(w, n, 41, 1);
  CheckAll(r, n, 41, 1);
  WriteAll(w, n, 53, 2);
  CheckAll(r, n, 53, 2);
}

SIMPLE_TEST_MAIN()
