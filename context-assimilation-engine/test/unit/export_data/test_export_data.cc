/*
 * test_export_data.cc — integration test for CAE Runtime::ExportData, the .h5
 * exporter, driven against a LIVE CLIO/CTE runtime but with NO GPU.
 *
 * The ChunkedDatasetWriter unit test (cae_hdf5_export_test) already covers the
 * HDF5-writing mechanics in isolation. What it CANNOT reach is the ExportData
 * task coroutine itself: detecting the __meta blob, decoding it out of the store,
 * enumerating chunk blobs while skipping __meta, feeding the writer, and the
 * completeness check. Those run inside the server process on a real tag. This
 * test exercises exactly that seam.
 *
 * It seeds the store the same way a kvhdf5 GPU producer would — a __meta blob
 * plus one raw chunk blob per coordinate — but via host CTE PutBlob, so it needs
 * no GPU. That is faithful because the store format is identical regardless of
 * who wrote it; the GPU-specific half (GpuCteDataset::FromPath calling
 * WriteDatasetMeta) is a thin wrapper whose output IS this byte layout.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cae/core/core_client.h>
#include <clio_cae/core/constants.h>
#include <clio_cte/core/core_client.h>

#include <clio_cte/kvhdf5/meta_blob.h>

#include <hdf5.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using clio::cae::core::Client;
using kvhdf5::DType;
using kvhdf5::MetaBlob;

namespace {

int g_failures = 0;
void Check(bool cond, const std::string &what) {
  if (cond) {
    HLOG(kInfo, "  ok: {}", what);
  } else {
    HLOG(kError, "  FAIL: {}", what);
    ++g_failures;
  }
}

// 2-D f32: 8x6 elements in 4x3 chunks => 2x2 == 4 chunks.
constexpr uint64_t kRows = 8, kCols = 6, kCRows = 4, kCCols = 3;

float Expected(uint64_t i, uint64_t j) { return static_cast<float>(i * kCols + j); }

MetaBlob Make2dF32() {
  MetaBlob m;
  m.rank = 2;
  m.dtype = static_cast<uint32_t>(DType::kF32);
  m.elem_size = 4;
  m.dims[0] = kRows;        m.dims[1] = kCols;
  m.chunk_dims[0] = kCRows; m.chunk_dims[1] = kCCols;
  return m;
}

std::vector<float> MakeChunk(uint64_t ci, uint64_t cj) {
  std::vector<float> buf(kCRows * kCCols);
  for (uint64_t li = 0; li < kCRows; ++li)
    for (uint64_t lj = 0; lj < kCCols; ++lj)
      buf[li * kCCols + lj] = Expected(ci * kCRows + li, cj * kCCols + lj);
  return buf;
}

// Seed a tag with a __meta blob + all four chunk blobs, out of order.
void SeedDataset(const std::string &tag_name) {
  clio::cte::core::Tag tag(tag_name);

  const MetaBlob meta = Make2dF32();
  unsigned char wire[sizeof(MetaBlob)];
  kvhdf5::EncodeMeta(meta, wire);
  tag.PutBlob(kvhdf5::kMetaBlobName, reinterpret_cast<const char *>(wire),
              sizeof(wire));

  const std::pair<uint64_t, uint64_t> order[4] = {{1, 1}, {0, 1}, {1, 0}, {0, 0}};
  for (auto [ci, cj] : order) {
    const std::vector<float> chunk = MakeChunk(ci, cj);
    const std::string name = std::to_string(ci) + "_" + std::to_string(cj);
    tag.PutBlob(name, reinterpret_cast<const char *>(chunk.data()),
                chunk.size() * sizeof(float));
  }
}

// ---- Test 1: a self-describing tag exports as the real N-D dataset ----------
void TestNdExport(Client &cae) {
  HLOG(kInfo, "TestNdExport");
  const std::string tag_name = "export_test/grp/temperature";
  const std::string out = "/tmp/cae_export_data_nd.h5";
  std::remove(out.c_str());

  SeedDataset(tag_name);

  auto task = cae.AsyncExportData(tag_name, out, "hdf5");
  task.Wait();
  Check(task->result_code_ == 0,
        "ExportData result_code == 0 (got " + std::to_string(task->result_code_) + ")");
  Check(task->bytes_exported_ == kRows * kCols * sizeof(float),
        "bytes_exported == full dataset size");

  hid_t f = H5Fopen(out.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  Check(f >= 0, "H5Fopen exported file");
  if (f < 0) return;

  Check(H5Lexists(f, "export_test", H5P_DEFAULT) > 0, "group 'export_test' created");
  Check(H5Lexists(f, "export_test/grp", H5P_DEFAULT) > 0, "nested group created");

  hid_t d = H5Dopen2(f, tag_name.c_str(), H5P_DEFAULT);
  Check(d >= 0, "H5Dopen2 on the tag path");
  if (d >= 0) {
    hid_t s = H5Dget_space(d);
    hsize_t dims[2] = {0, 0};
    H5Sget_simple_extent_dims(s, dims, nullptr);
    Check(dims[0] == kRows && dims[1] == kCols, "dataset dims == 8x6");

    hid_t t = H5Dget_type(d);
    Check(H5Tequal(t, H5T_NATIVE_FLOAT) > 0, "dtype == native float");

    std::vector<float> got(kRows * kCols, -1.0f);
    Check(H5Dread(d, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, got.data()) >= 0,
          "H5Dread");
    bool ok = true;
    for (uint64_t i = 0; i < kRows && ok; ++i)
      for (uint64_t j = 0; j < kCols; ++j)
        if (got[i * kCols + j] != Expected(i, j)) {
          HLOG(kError, "  element ({},{}) == {}, expected {}", i, j,
               got[i * kCols + j], Expected(i, j));
          ++g_failures; ok = false; break;
        }
    if (ok) HLOG(kInfo, "  every element round-tripped through the store to the .h5");

    H5Tclose(t);
    H5Sclose(s);
    H5Dclose(d);
  }
  H5Fclose(f);
}

// ---- Test 2: a tag WITHOUT __meta falls back to the flat per-blob dump ------
void TestFallbackExport(Client &cae) {
  HLOG(kInfo, "TestFallbackExport");
  const std::string tag_name = "export_test_opaque_bag";
  const std::string out = "/tmp/cae_export_data_flat.h5";
  std::remove(out.c_str());

  {
    clio::cte::core::Tag tag(tag_name);
    const std::vector<uint8_t> a(16, 0x11), b(16, 0x22);
    tag.PutBlob("payload_a", reinterpret_cast<const char *>(a.data()), a.size());
    tag.PutBlob("payload_b", reinterpret_cast<const char *>(b.data()), b.size());
  }

  auto task = cae.AsyncExportData(tag_name, out, "hdf5");
  task.Wait();
  Check(task->result_code_ == 0, "fallback ExportData result_code == 0");

  hid_t f = H5Fopen(out.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  Check(f >= 0, "H5Fopen fallback file");
  if (f < 0) return;
  // Flat mode: one 1-D uint8 dataset per blob, named by the blob.
  Check(H5Lexists(f, "payload_a", H5P_DEFAULT) > 0, "flat dataset 'payload_a' exists");
  Check(H5Lexists(f, "payload_b", H5P_DEFAULT) > 0, "flat dataset 'payload_b' exists");
  hid_t d = H5Dopen2(f, "payload_a", H5P_DEFAULT);
  if (d >= 0) {
    hid_t t = H5Dget_type(d);
    Check(H5Tget_class(t) == H5T_INTEGER && H5Tget_size(t) == 1,
          "flat dataset is 1-byte integer");
    H5Tclose(t);
    H5Dclose(d);
  }
  H5Fclose(f);
}

// ---- Test 3: a PRESENT-but-undecodable __meta is a hard failure -------------
// (Not a silent degrade to the flat dump reported as success.)
void TestCorruptMetaFails(Client &cae) {
  HLOG(kInfo, "TestCorruptMetaFails");
  const std::string tag_name = "export_test_corrupt_meta";
  const std::string out = "/tmp/cae_export_data_corrupt.h5";
  std::remove(out.c_str());

  {
    clio::cte::core::Tag tag(tag_name);
    // A __meta blob of the right SIZE but garbage content (bad magic): present,
    // so the tag claims to be self-describing, but it cannot be decoded.
    std::vector<unsigned char> garbage(sizeof(MetaBlob), 0xEE);
    tag.PutBlob(kvhdf5::kMetaBlobName,
                reinterpret_cast<const char *>(garbage.data()), garbage.size());
    // A real chunk too, so the tag is non-empty.
    const std::vector<float> chunk = MakeChunk(0, 0);
    tag.PutBlob("0_0", reinterpret_cast<const char *>(chunk.data()),
                chunk.size() * sizeof(float));
  }

  auto task = cae.AsyncExportData(tag_name, out, "hdf5");
  task.Wait();
  Check(task->result_code_ != 0,
        "corrupt __meta => export FAILS (result_code " +
            std::to_string(task->result_code_) + " != 0), not a silent flat dump");
}

// ---- Test 4: binary export omits the internal __meta blob -------------------
void TestBinarySkipsMeta(Client &cae) {
  HLOG(kInfo, "TestBinarySkipsMeta");
  const std::string tag_name = "export_test_binary/grp/temperature";
  const std::string out = "/tmp/cae_export_data.bin";
  std::remove(out.c_str());

  SeedDataset(tag_name);  // writes __meta + 4 chunks

  auto task = cae.AsyncExportData(tag_name, out, "binary");
  task.Wait();
  Check(task->result_code_ == 0, "binary export result_code == 0");

  // Parse the documented record stream: name_len(u32) name data_len(u64) data.
  std::ifstream ifs(out, std::ios::binary);
  Check(ifs.good(), "reopen binary output");
  bool saw_meta = false;
  int chunk_records = 0;
  while (ifs.peek() != EOF) {
    uint32_t name_len = 0;
    ifs.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));
    if (!ifs) break;
    std::string name(name_len, '\0');
    ifs.read(name.data(), name_len);
    uint64_t data_len = 0;
    ifs.read(reinterpret_cast<char *>(&data_len), sizeof(data_len));
    ifs.seekg(static_cast<std::streamoff>(data_len), std::ios::cur);
    if (name == kvhdf5::kMetaBlobName) saw_meta = true;
    if (kvhdf5::IsChunkBlobName(name)) ++chunk_records;
  }
  Check(!saw_meta, "binary stream does NOT contain a __meta record");
  Check(chunk_records == 4, "binary stream contains all 4 chunk records");
}

}  // namespace

int main() {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    HLOG(kError, "CLIO_INIT failed");
    return 1;
  }
  clio::cte::core::CLIO_CTE_CLIENT_INIT();
  CLIO_CAE_CLIENT_INIT();

  Client cae;
  clio::cae::core::CreateParams params;
  auto create = cae.AsyncCreate(clio::run::PoolQuery::Local(), "test_export_pool",
                                clio::cae::core::kCaePoolId, params);
  create.Wait();

  TestNdExport(cae);
  TestFallbackExport(cae);
  TestCorruptMetaFails(cae);
  TestBinarySkipsMeta(cae);

  if (g_failures == 0) {
    HLOG(kInfo, "All ExportData integration tests passed.");
    return 0;
  }
  HLOG(kError, "{} check(s) FAILED.", g_failures);
  return 1;
}
