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
 * test_csv_assim.cc - csv:: transfers round-tripped through CTE.
 *
 * Writes a CSV whose columns cover every stored type (int64 large enough to
 * span two 1.5 MB chunks, float64 with empty fields, quoted text), submits
 * csv transfers with different dataset_filters through ParseOmni on an
 * in-process runtime, and checks in CTE that:
 *   - each selected column has tag <dst>/<header> with the expected
 *     "description" and chunk_N blobs equal to the column's element image;
 *   - each rejected column has nothing under that transfer's prefix;
 *   - an include list matching no column fails with -9, and a malformed
 *     file fails with -5.
 * Expected images are built here from the generator's values, not by the
 * code under test.
 */

#include <clio_cae/core/constants.h>
#include <clio_cae/core/core_client.h>
#include <clio_cae/core/factory/assimilation_ctx.h>
#include <clio_cte/core/core_client.h>
#include <clio_ctp/util/logging.h>
#include <clio_runtime/clio_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

/** 200k int64 = 1.6 MB: two chunk blobs at the 1.5 MB chunk size. */
constexpr size_t kRows = 200000;

const std::vector<std::string> kCities = {"Kansas City, MO", "St. \"Lou\"",
                                          "Rolla"};
/** Longest city in bytes: the char column's element width. */
constexpr size_t kCityWidth = 15;

/** Header order in the generated file. */
const std::vector<std::string> kColumns = {"id", "solar_GHI", "solar_Minute",
                                           "city"};

/** solar_GHI of row i; every 1000th row (from 7) is an empty field. */
double Ghi(size_t i) {
  if (i % 1000 == 7) return std::numeric_limits<double>::quiet_NaN();
  return 0.5 * static_cast<double>(i % 2000);
}

/** Quote a field per RFC 4180. */
std::string Quote(const std::string& s) {
  std::string q = "\"";
  for (char c : s) q += (c == '"') ? std::string("\"\"") : std::string(1, c);
  return q + "\"";
}

bool GenerateFile(const std::string& path) {
  std::ofstream f(path, std::ios::binary);
  f << "id,solar_GHI,solar_Minute,city\r\n";
  char num[64];
  for (size_t i = 0; i < kRows; ++i) {
    double g = Ghi(i);
    if (g != g) {
      num[0] = '\0';  // empty field -> NaN
    } else {
      std::snprintf(num, sizeof(num), "%.1f", g);
    }
    f << i << ',' << num << ",30," << Quote(kCities[i % kCities.size()])
      << "\r\n";
  }
  return f.good();
}

template <typename T>
std::vector<char> Image(const std::vector<T>& v) {
  std::vector<char> b(v.size() * sizeof(T));
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

/** Expected (description, bytes) of a column of the generated file. */
std::pair<std::string, std::vector<char>> Expected(const std::string& col) {
  const std::string n = std::to_string(kRows);
  if (col == "id" || col == "solar_Minute") {
    std::vector<int64_t> v(kRows);
    for (size_t i = 0; i < kRows; ++i) v[i] = col == "id" ? (int64_t)i : 30;
    return {"tensor<int64, " + n + ">", Image(v)};
  }
  if (col == "solar_GHI") {
    std::vector<double> v(kRows);
    for (size_t i = 0; i < kRows; ++i) v[i] = Ghi(i);
    return {"tensor<float64, " + n + ">", Image(v)};
  }
  std::vector<char> b(kRows * kCityWidth, '\0');
  for (size_t i = 0; i < kRows; ++i) {
    const std::string& c = kCities[i % kCities.size()];
    std::memcpy(b.data() + i * kCityWidth, c.data(), c.size());
  }
  return {"tensor<char, " + n + ", " + std::to_string(kCityWidth) + ">", b};
}

struct Transfer {
  const char* prefix;
  std::vector<std::string> include;
  std::vector<std::string> exclude;
  std::vector<std::string> expected;
};

const std::vector<Transfer> kTransfers = {
    {"csv_all", {}, {}, {"id", "solar_GHI", "solar_Minute", "city"}},
    {"csv_solar", {"/solar_*"}, {"/solar_Minute"}, {"solar_GHI"}},
    {"csv_pick", {"/id", "/c?ty"}, {}, {"id", "city"}},
};

bool ReadBlob(clio::cte::core::Client* cte, const clio::cte::core::TagId& tag,
              const std::string& name, std::vector<char>& out) {
  auto sz = cte->AsyncGetBlobSize(tag, name);
  sz.Wait();
  out.resize(sz->size_);
  if (out.empty()) return true;
  auto buf = CLIO_IPC->AllocateBuffer(out.size());
  auto get = cte->AsyncGetBlob(tag, name, 0, out.size(), 0,
                               buf.shm_.template Cast<void>());
  get.Wait();
  bool ok = get->GetReturnCode() == 0;
  if (ok) std::memcpy(out.data(), buf.ptr_, out.size());
  CLIO_IPC->FreeBuffer(buf);
  return ok;
}

/** Description text and concatenated chunk bytes of one tag. */
bool ReadTag(clio::cte::core::Client* cte, const std::string& tag_name,
             std::string* desc, std::vector<char>* data, size_t* nchunks) {
  auto tag = cte->AsyncGetOrCreateTag(tag_name);
  tag.Wait();
  auto blobs = cte->AsyncGetContainedBlobs(tag->tag_id_);
  blobs.Wait();
  std::vector<size_t> idx;
  bool has_desc = false;
  for (const auto& b : blobs->blob_names_) {
    if (b == "description") has_desc = true;
    if (b.rfind("chunk_", 0) == 0) idx.push_back(std::stoul(b.substr(6)));
  }
  std::sort(idx.begin(), idx.end());
  *nchunks = idx.size();
  data->clear();
  desc->clear();
  std::vector<char> buf;
  if (has_desc && ReadBlob(cte, tag->tag_id_, "description", buf)) {
    desc->assign(buf.begin(), buf.end());
  }
  for (size_t i : idx) {
    if (!ReadBlob(cte, tag->tag_id_, "chunk_" + std::to_string(i), buf)) {
      return false;
    }
    data->insert(data->end(), buf.begin(), buf.end());
  }
  return true;
}

int VerifyTransfer(clio::cte::core::Client* cte, const Transfer& t) {
  int failures = 0;
  for (const auto& col : kColumns) {
    bool sel = std::find(t.expected.begin(), t.expected.end(), col) !=
               t.expected.end();
    std::string desc;
    std::vector<char> data;
    size_t nchunks = 0;
    bool read =
        ReadTag(cte, std::string(t.prefix) + "/" + col, &desc, &data, &nchunks);
    bool ok;
    if (sel) {
      auto want = Expected(col);
      ok = read && desc == want.first && data == want.second;
      if (ok) {
        HLOG(kSuccess, "  [{}] {}: '{}', {} chunk(s), {} B -> OK", t.prefix,
             col, desc, nchunks, data.size());
      } else {
        HLOG(kError, "  [{}] {}: '{}', {} chunk(s), {} B -> MISMATCH", t.prefix,
             col, desc, nchunks, data.size());
      }
    } else {
      ok = read && desc.empty() && nchunks == 0;
      if (!ok) HLOG(kError, "  [{}] {}: rejected but stored", t.prefix, col);
    }
    failures += ok ? 0 : 1;
  }
  return failures;
}

clio::cae::core::AssimilationCtx Ctx(const std::string& file,
                                     const std::string& prefix,
                                     const std::vector<std::string>& inc,
                                     const std::vector<std::string>& exc) {
  clio::cae::core::AssimilationCtx ctx;
  ctx.src = "csv::" + file;
  ctx.dst = "iowarp::" + prefix;
  ctx.format = "csv";
  ctx.include_patterns = inc;
  ctx.exclude_patterns = exc;
  return ctx;
}

/** Run one ParseOmni call and return its result code. */
int Parse(const std::vector<clio::cae::core::AssimilationCtx>& ctxs) {
  clio::cae::core::Client cae(clio::cae::core::kCaePoolId);
  auto parse = cae.AsyncParseOmni(ctxs);
  parse.Wait();
  return parse->result_code_;
}

bool InitClients() {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    return false;
  }
  clio::cte::core::CLIO_CTE_CLIENT_INIT();
  CLIO_CAE_CLIENT_INIT();
  clio::cae::core::Client cae_client;
  clio::cae::core::CreateParams params;
  auto create =
      cae_client.AsyncCreate(clio::run::PoolQuery::Local(), "cae_main",
                             clio::cae::core::kCaePoolId, params);
  create.Wait();
  return create->GetReturnCode() == 0;
}

}  // namespace

int main(int, char**) {
  if (!InitClients()) {
    HLOG(kError, "runtime / CAE pool init failed");
    return 1;
  }
  const auto tmp = std::filesystem::temp_directory_path();
  const std::string file = (tmp / "cae_csv_assim.csv").generic_string();
  const std::string bad = (tmp / "cae_csv_assim_bad.csv").generic_string();
  if (!GenerateFile(file)) return 1;
  std::ofstream(bad, std::ios::binary) << "a,b\n1,2\n3\n";

  int failures = 0;
  std::vector<clio::cae::core::AssimilationCtx> ctxs;
  for (const auto& t : kTransfers) {
    ctxs.push_back(Ctx(file, t.prefix, t.include, t.exclude));
  }
  int rc = Parse(ctxs);
  HLOG(kInfo, "ParseOmni ({} csv transfers) rc={}", ctxs.size(), rc);
  if (rc != 0) {
    ++failures;
  } else {
    auto* cte = CLIO_CTE_CLIENT;
    for (const auto& t : kTransfers) failures += VerifyTransfer(cte, t);
  }
  rc = Parse({Ctx(file, "csv_nomatch", {"/nope*"}, {})});
  HLOG(kInfo, "include matching nothing -> rc={} (want -9)", rc);
  failures += rc == -9 ? 0 : 1;
  rc = Parse({Ctx(bad, "csv_bad", {}, {})});
  HLOG(kInfo, "malformed CSV -> rc={} (want -5)", rc);
  failures += rc == -5 ? 0 : 1;

  std::error_code ec;
  std::filesystem::remove(file, ec);
  std::filesystem::remove(bad, ec);
  if (failures == 0) {
    HLOG(kSuccess, "CSV ASSIMILATION ROUND TRIP PASSED");
    return 0;
  }
  HLOG(kError, "CSV ASSIMILATION ROUND TRIP FAILED ({} check(s))", failures);
  return 1;
}
