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
 * test_hdf5_dataset_filter.cc - dataset_filter round trip on a generated file.
 *
 * Writes a small HDF5 file whose dataset names exercise every glob feature
 * the assimilator's MatchGlobPattern supports (literal, '*' including across
 * '/', '?', "[a-b]", "[!a-b]", backslash escape, exclude-before-include),
 * submits one hdf5 transfer per pattern through ParseOmni on an in-process
 * runtime, and checks the outcome in CTE:
 *   - every dataset the table says is selected has a tag <dst>/<dataset>
 *     with a "description" blob and chunk_N blobs equal to its bytes;
 *   - every other dataset has no data under that transfer's prefix.
 *
 * The expected selections are listed explicitly rather than recomputed, so
 * the test does not share an implementation with the code under test. The
 * same source runs on POSIX (fnmatch) and Windows (portable matcher).
 */

#include <hdf5.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cae/core/core_client.h>
#include <clio_cae/core/constants.h>
#include <clio_cae/core/factory/assimilation_ctx.h>

#include <clio_cte/core/core_client.h>

#include <clio_ctp/util/logging.h>

namespace {

/** Seconds to wait for a selected dataset's blobs to land in CTE. */
constexpr double kLandTimeoutS = 60.0;

/** Forced chunk size so even these tiny datasets span several blobs. */
constexpr const char* kChunkSizeBytes = "256";

/** Datasets the assimilator must discover in the generated file. */
constexpr size_t kNumDatasetsInFile = 12;

/** One int32 dataset in the generated file. */
struct Dataset {
  const char* path;
  size_t count;
};

const std::vector<Dataset> kDatasets = {
    {"/das", 100},          {"/t", 10},     {"/t2", 12},
    {"/t10", 14},           {"/ch0", 4},    {"/ch1", 5},
    {"/ch3", 6},            {"/raw/RawData", 300},
    {"/raw/TimeArray", 20}, {"/aux/x", 3},  {"/star*name", 2},
    {"/starXname", 2},
};

/** One hdf5 transfer and the datasets its filter must select. */
struct Transfer {
  const char* tag_prefix;
  std::vector<std::string> include;
  std::vector<std::string> exclude;
  std::vector<std::string> expected;
};

const std::vector<Transfer> kTransfers = {
    {"flt_exact", {"/das"}, {}, {"/das"}},
    {"flt_star", {"/t*"}, {"/t2"}, {"/t", "/t10"}},
    {"flt_qmark", {"/ch?"}, {}, {"/ch0", "/ch1", "/ch3"}},
    {"flt_bracket", {"/ch[0-1]"}, {}, {"/ch0", "/ch1"}},
    {"flt_negate", {"/ch[!0-1]"}, {}, {"/ch3"}},
    {"flt_slash", {"*/RawData"}, {}, {"/raw/RawData"}},
    {"flt_group", {"/raw/*"}, {}, {"/raw/RawData", "/raw/TimeArray"}},
    {"flt_escape", {"/star\\*name"}, {}, {"/star*name"}},
    {"flt_exclude_only",
     {},
     {"/raw/*", "/ch*", "/t*", "/star*", "/aux/*"},
     {"/das"}},
};

/**
 * Deterministic contents of one dataset, so every dataset differs.
 * @param index Position of the dataset in kDatasets.
 * @param count Number of int32 elements.
 * @return The values written to the file.
 */
std::vector<int32_t> DatasetValues(size_t index, size_t count) {
  std::vector<int32_t> v(count);
  for (size_t i = 0; i < count; ++i) {
    v[i] = static_cast<int32_t>(index * 1000 + i * 7);
  }
  return v;
}

/**
 * Write the test file: int32 datasets under root and under two groups,
 * created through an LCPL so intermediate groups appear on demand.
 * @param path Output HDF5 file path.
 * @return true on success.
 */
bool GenerateFile(const std::string& path) {
  hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file < 0) {
    HLOG(kError, "H5Fcreate failed: {}", path);
    return false;
  }
  hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
  H5Pset_create_intermediate_group(lcpl, 1);
  bool ok = true;
  for (size_t i = 0; i < kDatasets.size() && ok; ++i) {
    hsize_t dims[1] = {kDatasets[i].count};
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t ds = H5Dcreate2(file, kDatasets[i].path, H5T_NATIVE_INT32, space,
                          lcpl, H5P_DEFAULT, H5P_DEFAULT);
    std::vector<int32_t> values = DatasetValues(i, kDatasets[i].count);
    ok = ds >= 0 && H5Dwrite(ds, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL,
                             H5P_DEFAULT, values.data()) >= 0;
    if (ds >= 0) {
      H5Dclose(ds);
    }
    H5Sclose(space);
  }
  H5Pclose(lcpl);
  H5Fclose(file);
  if (!ok) {
    HLOG(kError, "Failed to write a dataset into {}", path);
  }
  return ok;
}

/** Chunk blob names of a tag in numeric order, plus description presence. */
struct TagBlobs {
  std::vector<std::string> chunks;
  bool has_description = false;
  size_t chunk_bytes = 0;
};

/**
 * List a tag's blobs and total the chunk sizes.
 * @param cte CTE client.
 * @param tag_id Tag to inspect.
 * @return Chunk names in numeric order and the summed chunk size.
 */
TagBlobs ListTagBlobs(clio::cte::core::Client* cte,
                      const clio::cte::core::TagId& tag_id) {
  TagBlobs tb;
  auto blobs = cte->AsyncGetContainedBlobs(tag_id);
  blobs.Wait();
  for (const auto& name : blobs->blob_names_) {
    if (name == "description") {
      tb.has_description = true;
    } else if (name.rfind("chunk_", 0) == 0) {
      tb.chunks.push_back(name);
    }
  }
  std::sort(tb.chunks.begin(), tb.chunks.end(),
            [](const std::string& a, const std::string& b) {
              return std::stoul(a.substr(6)) < std::stoul(b.substr(6));
            });
  for (const auto& c : tb.chunks) {
    auto sz = cte->AsyncGetBlobSize(tag_id, c);
    sz.Wait();
    tb.chunk_bytes += sz->size_;
  }
  return tb;
}

/**
 * Read one blob through shared memory into a private buffer.
 * @param cte CTE client.
 * @param tag_id Tag holding the blob.
 * @param name Blob name.
 * @param out Receives the bytes.
 * @return true on success.
 */
bool ReadBlob(clio::cte::core::Client* cte,
              const clio::cte::core::TagId& tag_id, const std::string& name,
              std::vector<char>& out) {
  auto sz = cte->AsyncGetBlobSize(tag_id, name);
  sz.Wait();
  size_t size = sz->size_;
  out.resize(size);
  if (size == 0) {
    return true;
  }
  auto buf = CLIO_IPC->AllocateBuffer(size);
  ctp::ipc::ShmPtr<> shm = buf.shm_.template Cast<void>();
  auto get = cte->AsyncGetBlob(tag_id, name, 0, size, 0, shm);
  get.Wait();
  bool ok = get->GetReturnCode() == 0;
  if (ok) {
    std::memcpy(out.data(), buf.ptr_, size);
  }
  CLIO_IPC->FreeBuffer(buf);
  return ok;
}

/**
 * Poll until the tag holds a description blob and `expected` chunk bytes.
 * @param cte CTE client.
 * @param tag_id Tag to watch.
 * @param expected Total chunk bytes that must be present.
 * @return Milliseconds waited, or -1 on timeout.
 */
double WaitLanded(clio::cte::core::Client* cte,
                  const clio::cte::core::TagId& tag_id, size_t expected) {
  auto t0 = std::chrono::steady_clock::now();
  while (true) {
    TagBlobs tb = ListTagBlobs(cte, tag_id);
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0)
                    .count();
    if (tb.has_description && tb.chunk_bytes == expected) {
      return ms;
    }
    if (ms > kLandTimeoutS * 1000.0) {
      return -1.0;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
}

/**
 * Tag name the assimilator gives a dataset: "<prefix>/<path without '/'>".
 * @param prefix Transfer destination prefix.
 * @param path Dataset path with leading '/'.
 * @return The CTE tag name.
 */
std::string TagName(const std::string& prefix, const std::string& path) {
  return prefix + "/" + path.substr(1);
}

/**
 * Verify one selected dataset: wait for it to land, then compare the
 * concatenated chunk blobs with the values written to the file.
 * @param cte CTE client.
 * @param prefix Transfer destination prefix.
 * @param index Position of the dataset in kDatasets.
 * @return true if the CTE copy equals the dataset bytes.
 */
bool VerifySelected(clio::cte::core::Client* cte, const std::string& prefix,
                    size_t index) {
  const std::string path = kDatasets[index].path;
  std::vector<int32_t> values = DatasetValues(index, kDatasets[index].count);
  const size_t expected_bytes = values.size() * sizeof(int32_t);
  auto tag = cte->AsyncGetOrCreateTag(TagName(prefix, path));
  tag.Wait();
  clio::cte::core::TagId tag_id = tag->tag_id_;
  double waited = WaitLanded(cte, tag_id, expected_bytes);
  if (waited < 0) {
    HLOG(kError, "  [{}] {}: timed out waiting for {} bytes", prefix, path,
         expected_bytes);
    return false;
  }
  TagBlobs tb = ListTagBlobs(cte, tag_id);
  std::vector<char> got, chunk;
  for (const auto& c : tb.chunks) {
    if (!ReadBlob(cte, tag_id, c, chunk)) {
      HLOG(kError, "  [{}] {}: GetBlob failed for {}", prefix, path, c);
      return false;
    }
    got.insert(got.end(), chunk.begin(), chunk.end());
  }
  bool ok = got.size() == expected_bytes &&
            std::memcmp(got.data(), values.data(), expected_bytes) == 0;
  if (ok) {
    HLOG(kSuccess, "  [{}] {}: {} chunks, {} bytes, landed after {} ms -> OK",
         prefix, path, tb.chunks.size(), got.size(),
         static_cast<long long>(waited));
  } else {
    HLOG(kError, "  [{}] {}: {} chunks, {}/{} bytes -> MISMATCH", prefix,
         path, tb.chunks.size(), got.size(), expected_bytes);
  }
  return ok;
}

/**
 * Verify one rejected dataset: its tag must hold neither data nor a
 * description.
 * @param cte CTE client.
 * @param prefix Transfer destination prefix.
 * @param path Dataset path with leading '/'.
 * @return true if nothing was ingested for it.
 */
bool VerifyExcluded(clio::cte::core::Client* cte, const std::string& prefix,
                    const std::string& path) {
  auto tag = cte->AsyncGetOrCreateTag(TagName(prefix, path));
  tag.Wait();
  TagBlobs tb = ListTagBlobs(cte, tag->tag_id_);
  bool ok = tb.chunks.empty() && !tb.has_description;
  if (!ok) {
    HLOG(kError, "  [{}] {}: {} chunk blobs but the filter rejects it -> FAIL",
         prefix, path, tb.chunks.size());
  }
  return ok;
}

/**
 * Check every dataset of the file against one transfer's expected set.
 * @param cte CTE client.
 * @param t The transfer and its expected selection.
 * @return Number of datasets that failed.
 */
int VerifyTransfer(clio::cte::core::Client* cte, const Transfer& t) {
  int failures = 0;
  for (size_t i = 0; i < kDatasets.size(); ++i) {
    const std::string path = kDatasets[i].path;
    bool selected = std::find(t.expected.begin(), t.expected.end(), path) !=
                    t.expected.end();
    bool ok = selected ? VerifySelected(cte, t.tag_prefix, i)
                       : VerifyExcluded(cte, t.tag_prefix, path);
    failures += ok ? 0 : 1;
  }
  HLOG(kInfo, "[{}] include={} exclude={} -> {} selected, {} failure(s)",
       t.tag_prefix, t.include.size(), t.exclude.size(), t.expected.size(),
       failures);
  return failures;
}

/**
 * Build the ParseOmni input: one hdf5 transfer per table row.
 * @param file Source HDF5 path.
 * @return Assimilation contexts in table order.
 */
std::vector<clio::cae::core::AssimilationCtx> BuildContexts(
    const std::string& file) {
  std::vector<clio::cae::core::AssimilationCtx> out;
  for (const auto& t : kTransfers) {
    clio::cae::core::AssimilationCtx ctx;
    ctx.src = "hdf5::" + file;
    ctx.dst = std::string("iowarp::") + t.tag_prefix;
    ctx.format = "hdf5";
    ctx.depends_on = "";
    ctx.range_off = 0;
    ctx.range_size = 0;
    ctx.include_patterns = t.include;
    ctx.exclude_patterns = t.exclude;
    out.push_back(ctx);
  }
  return out;
}

/**
 * Bring up the runtime (per CLIO_WITH_RUNTIME), the CTE and CAE clients, and
 * the CAE pool.
 * @return true on success.
 */
bool InitClients() {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    HLOG(kError, "CLIO_INIT failed");
    return false;
  }
  clio::cte::core::CLIO_CTE_CLIENT_INIT();
  CLIO_CAE_CLIENT_INIT();
  clio::cae::core::Client cae_client;
  clio::cae::core::CreateParams params;
  auto create = cae_client.AsyncCreate(clio::run::PoolQuery::Local(),
                                       "cae_main",
                                       clio::cae::core::kCaePoolId, params);
  create.Wait();
  if (create->GetReturnCode() != 0) {
    HLOG(kError, "CAE pool create failed: {}", create->GetReturnCode());
    return false;
  }
  return true;
}

/**
 * Set an environment variable before the in-process runtime starts.
 * @param name Variable name.
 * @param value Value to give it.
 */
void SetEnv(const char* name, const std::string& value) {
#ifdef _WIN32
  _putenv_s(name, value.c_str());
#else
  setenv(name, value.c_str(), 1);
#endif
}

/**
 * Read the non-empty lines of a text file.
 * @param path File to read; a missing file reads as no lines.
 * @return The lines in file order.
 */
std::vector<std::string> ReadLines(const std::string& path) {
  std::ifstream in(path);
  std::vector<std::string> lines;
  for (std::string line; std::getline(in, line);) {
    if (!line.empty()) {
      lines.push_back(line);
    }
  }
  return lines;
}

/**
 * Find the telemetry record a transfer wrote.
 * @param records Lines of the CLIO_CAE_TELEMETRY file.
 * @param prefix Transfer destination prefix.
 * @param count Receives how many records name that destination.
 * @return The last such record, or "" if there is none.
 */
std::string FindRecord(const std::vector<std::string>& records,
                       const std::string& prefix, int* count) {
  const std::string key = "\"dst\":\"iowarp::" + prefix + "\"";
  std::string found;
  *count = 0;
  for (const auto& r : records) {
    if (r.find(key) != std::string::npos) {
      found = r;
      ++*count;
    }
  }
  return found;
}

/**
 * Check the CLIO_CAE_TELEMETRY file: one JSON record per transfer, carrying
 * its destination, how many datasets it discovered and selected, no errors,
 * and its patterns escaped as JSON strings.
 * @param path Telemetry file the runtime appended to.
 * @return Number of failed checks.
 */
int VerifyTelemetry(const std::string& path) {
  // A transfer writes its record as it finishes, which can be just after its
  // last blob lands, so wait for all of them.
  std::vector<std::string> records;
  auto t0 = std::chrono::steady_clock::now();
  while (true) {
    records = ReadLines(path);
    double s = std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t0)
                   .count();
    if (records.size() >= kTransfers.size() || s > kLandTimeoutS) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  int failures = 0;
  if (records.size() != kTransfers.size()) {
    HLOG(kError, "telemetry: {} record(s) in {}, expected {}", records.size(),
         path, kTransfers.size());
    ++failures;
  }
  for (const auto& t : kTransfers) {
    int count = 0;
    std::string r = FindRecord(records, t.tag_prefix, &count);
    const std::vector<std::string> fields = {
        "\"datasets_discovered\":" + std::to_string(kNumDatasetsInFile) + ",",
        "\"datasets_filtered\":" + std::to_string(t.expected.size()) + ",",
        "\"dataset_errors\":0,", "\"error_code\":0}", "\"total_ms\":"};
    bool ok = count == 1;
    for (const auto& f : fields) {
      ok = ok && r.find(f) != std::string::npos;
    }
    if (!ok) {
      HLOG(kError, "telemetry [{}]: {} record(s); last: {}", t.tag_prefix,
           count, r);
      ++failures;
    }
  }
  // The escape row's pattern holds a backslash, which JSON must double.
  int count = 0;
  std::string r = FindRecord(records, "flt_escape", &count);
  if (r.find("\"include_patterns\":[\"/star\\\\*name\"]") ==
      std::string::npos) {
    HLOG(kError, "telemetry [flt_escape]: pattern not JSON-escaped: {}", r);
    ++failures;
  }
  if (failures == 0) {
    HLOG(kSuccess, "telemetry: {} records, one per transfer -> OK",
         records.size());
  }
  return failures;
}

}  // namespace

int main(int, char**) {
  SetEnv("CLIO_VOL_CHUNK_SIZE", kChunkSizeBytes);
  // Every transfer appends one record here; the file is checked at the end.
  const std::filesystem::path telemetry =
      std::filesystem::temp_directory_path() / "cae_hdf5_dataset_filter.jsonl";
  std::error_code ec;
  std::filesystem::remove(telemetry, ec);
  SetEnv("CLIO_CAE_TELEMETRY", telemetry.generic_string());
  if (!InitClients()) {
    return 1;
  }
  std::filesystem::path file =
      std::filesystem::temp_directory_path() / "cae_hdf5_dataset_filter.h5";
  const std::string file_str = file.generic_string();
  if (!GenerateFile(file_str)) {
    return 1;
  }
  HLOG(kInfo, "Generated {} with {} datasets", file_str, kDatasets.size());

  clio::cae::core::Client cae_client(clio::cae::core::kCaePoolId);
  std::vector<clio::cae::core::AssimilationCtx> contexts =
      BuildContexts(file_str);
  auto t0 = std::chrono::steady_clock::now();
  auto parse = cae_client.AsyncParseOmni(contexts);
  parse.Wait();
  double parse_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
  HLOG(kInfo, "ParseOmni rc={} scheduled={} for {} transfers in {} ms",
       parse->GetReturnCode(), parse->num_tasks_scheduled_, contexts.size(),
       static_cast<long long>(parse_ms));
  int failures = 0;
  if (parse->GetReturnCode() != 0 || parse->num_tasks_scheduled_ == 0) {
    HLOG(kError, "ParseOmni did not schedule the transfers");
    failures = 1;
  } else {
    auto* cte = CLIO_CTE_CLIENT;
    for (const auto& t : kTransfers) {
      failures += VerifyTransfer(cte, t);
    }
    failures += VerifyTelemetry(telemetry.generic_string());
  }
  std::filesystem::remove(file, ec);
  std::filesystem::remove(telemetry, ec);
  if (failures == 0) {
    HLOG(kSuccess, "HDF5 DATASET FILTER ROUND TRIP PASSED");
    return 0;
  }
  HLOG(kError, "HDF5 DATASET FILTER ROUND TRIP FAILED ({} check(s))",
       failures);
  return 1;
}
