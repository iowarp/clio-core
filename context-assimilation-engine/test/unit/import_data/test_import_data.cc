/*
 * test_import_data.cc — integration test for CAE Runtime::ImportData, the .h5
 * importer, driven against a LIVE CLIO/CTE runtime but with NO GPU.
 *
 * The ChunkedDatasetReader unit test (cae_hdf5_import_test) already covers the
 * HDF5-reading mechanics in isolation. What it CANNOT reach is the ImportData
 * task coroutine itself: opening the dataset, publishing __meta, and streaming
 * each chunk into the store via CTE PutBlob on a real tag. This test exercises
 * exactly that seam, two ways:
 *
 *   1. Direct import: build a .h5 with the HDF5 API, ImportData it, then read
 *      the tag's blobs back through CTE and check every byte -- independent of
 *      the exporter, so an export bug cannot mask an import bug.
 *   2. Store -> file -> store round trip: seed a tag, ExportData it to a .h5,
 *      ImportData that file back into the SAME tag (dataset path == tag name),
 *      and confirm the chunk bytes are byte-identical to what was seeded. This
 *      is the closed loop the two features form together.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cae/core/core_client.h>
#include <clio_cae/core/constants.h>
#include <clio_cte/core/core_client.h>

#include <clio_cte/kvhdf5/meta_blob.h>

#include <hdf5.h>

#include <cstdint>
#include <cstdio>
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

// Build an .h5 with one chunked 2-D f32 dataset at `path`, each element stamped
// with its global index. `path` has no '/' so no intermediate groups are needed.
void MakeChunkedF32File(const std::string &file, const std::string &path) {
  std::vector<float> data(kRows * kCols);
  for (uint64_t i = 0; i < kRows; ++i)
    for (uint64_t j = 0; j < kCols; ++j)
      data[i * kCols + j] = Expected(i, j);

  hid_t f = H5Fcreate(file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hsize_t dims[2] = {kRows, kCols};
  hid_t space = H5Screate_simple(2, dims, nullptr);
  hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
  hsize_t chunk[2] = {kCRows, kCCols};
  H5Pset_chunk(dcpl, 2, chunk);
  hid_t d = H5Dcreate2(f, path.c_str(), H5T_IEEE_F32LE, space, H5P_DEFAULT, dcpl,
                       H5P_DEFAULT);
  H5Dwrite(d, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  H5Dclose(d);
  H5Pclose(dcpl);
  H5Sclose(space);
  H5Fclose(f);
}

std::vector<float> MakeChunk(uint64_t ci, uint64_t cj) {
  std::vector<float> buf(kCRows * kCCols);
  for (uint64_t li = 0; li < kCRows; ++li)
    for (uint64_t lj = 0; lj < kCCols; ++lj)
      buf[li * kCCols + lj] = Expected(ci * kCRows + li, cj * kCCols + lj);
  return buf;
}

MetaBlob Make2dF32() {
  MetaBlob m;
  m.rank = 2;
  m.dtype = static_cast<uint32_t>(DType::kF32);
  m.elem_size = 4;
  m.dims[0] = kRows;        m.dims[1] = kCols;
  m.chunk_dims[0] = kCRows; m.chunk_dims[1] = kCCols;
  return m;
}

// ---- Test 1: import a hand-built .h5 and verify the store contents ----------
void TestDirectImport(Client &cae) {
  HLOG(kInfo, "TestDirectImport");
  const std::string tag_name = "import_direct";  // flat: also the in-file path
  const std::string in = "/tmp/cae_import_direct.h5";
  std::remove(in.c_str());
  MakeChunkedF32File(in, tag_name);

  auto task = cae.AsyncImportData(tag_name, in, "hdf5");
  task.Wait();
  Check(task->result_code_ == 0,
        "ImportData result_code == 0 (got " + std::to_string(task->result_code_) + ")");
  Check(task->bytes_imported_ == kRows * kCols * sizeof(float),
        "bytes_imported == full dataset size");

  clio::cte::core::Tag tag(tag_name);

  // __meta must be present and describe the 8x6 / 4x3 f32 layout.
  MetaBlob meta{};
  tag.GetBlob(kvhdf5::kMetaBlobName, reinterpret_cast<char *>(&meta),
              sizeof(meta));
  Check(meta.Valid(), "__meta decodes as valid");
  Check(meta.rank == 2 && meta.dims[0] == kRows && meta.dims[1] == kCols,
        "__meta dims 8x6");
  Check(meta.chunk_dims[0] == kCRows && meta.chunk_dims[1] == kCCols,
        "__meta chunk 4x3");
  Check(meta.Dtype() == DType::kF32, "__meta dtype f32");

  // Every chunk blob must hold its slab of the global-index pattern.
  bool ok = true;
  for (uint64_t ci = 0; ci < 2 && ok; ++ci) {
    for (uint64_t cj = 0; cj < 2; ++cj) {
      const std::string name = std::to_string(ci) + "_" + std::to_string(cj);
      std::vector<float> got(kCRows * kCCols, -1.0f);
      tag.GetBlob(name, reinterpret_cast<char *>(got.data()),
                  got.size() * sizeof(float));
      const std::vector<float> want = MakeChunk(ci, cj);
      if (got != want) {
        HLOG(kError, "  chunk '{}' bytes mismatch", name);
        ++g_failures;
        ok = false;
        break;
      }
    }
  }
  if (ok) HLOG(kInfo, "  all 4 chunks imported byte-correct");
}

// ---- Test 2: store -> file -> store is the identity --------------------------
void TestRoundTrip(Client &cae) {
  HLOG(kInfo, "TestRoundTrip");
  const std::string tag_name = "roundtrip/grp/temperature";
  const std::string file = "/tmp/cae_import_roundtrip.h5";
  std::remove(file.c_str());

  // Seed the tag exactly as a kvhdf5 producer would: __meta + 4 chunks.
  clio::cte::core::Tag tag(tag_name);
  const MetaBlob meta = Make2dF32();
  unsigned char wire[sizeof(MetaBlob)];
  kvhdf5::EncodeMeta(meta, wire);
  tag.PutBlob(kvhdf5::kMetaBlobName, reinterpret_cast<const char *>(wire),
              sizeof(wire));
  std::vector<std::vector<float>> seeded(4);
  for (uint64_t ci = 0; ci < 2; ++ci) {
    for (uint64_t cj = 0; cj < 2; ++cj) {
      seeded[ci * 2 + cj] = MakeChunk(ci, cj);
      const std::string name = std::to_string(ci) + "_" + std::to_string(cj);
      tag.PutBlob(name, reinterpret_cast<const char *>(seeded[ci * 2 + cj].data()),
                  seeded[ci * 2 + cj].size() * sizeof(float));
    }
  }

  // Export to a file, then import that same file back into the same tag.
  auto ex = cae.AsyncExportData(tag_name, file, "hdf5");
  ex.Wait();
  Check(ex->result_code_ == 0, "round-trip Export result_code == 0");

  auto im = cae.AsyncImportData(tag_name, file, "hdf5");
  im.Wait();
  Check(im->result_code_ == 0, "round-trip Import result_code == 0");
  Check(im->bytes_imported_ == kRows * kCols * sizeof(float),
        "round-trip bytes_imported == full dataset");

  // The re-read chunks must be byte-identical to what was seeded: store -> file
  // -> store did not perturb a single byte.
  bool ok = true;
  for (uint64_t ci = 0; ci < 2 && ok; ++ci) {
    for (uint64_t cj = 0; cj < 2; ++cj) {
      const std::string name = std::to_string(ci) + "_" + std::to_string(cj);
      std::vector<float> got(kCRows * kCCols, -1.0f);
      tag.GetBlob(name, reinterpret_cast<char *>(got.data()),
                  got.size() * sizeof(float));
      if (got != seeded[ci * 2 + cj]) {
        HLOG(kError, "  round-trip chunk '{}' changed", name);
        ++g_failures;
        ok = false;
        break;
      }
    }
  }
  if (ok) HLOG(kInfo, "  store -> file -> store is byte-identical");
}

// ---- Test 3: bad inputs fail, they do not silently succeed ------------------
void TestBadInputs(Client &cae) {
  HLOG(kInfo, "TestBadInputs");
  auto missing = cae.AsyncImportData("import_missing", "/tmp/cae_no_such.h5",
                                     "hdf5");
  missing.Wait();
  Check(missing->result_code_ != 0, "import of a missing file FAILS");

  const std::string in = "/tmp/cae_import_badfmt.h5";
  MakeChunkedF32File(in, "import_badfmt");
  auto badfmt = cae.AsyncImportData("import_badfmt", in, "bogus");
  badfmt.Wait();
  Check(badfmt->result_code_ != 0, "unsupported format FAILS");
}

}  // namespace

int main() {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    HLOG(kError, "CLIO_INIT failed");
    return 1;
  }
  clio::cte::core::CLIO_CTE_CLIENT_INIT();
  CLIO_CAE_CLIENT_INIT();

  // Quiet HDF5's own error stack; ImportData reports through result codes.
  H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

  Client cae;
  clio::cae::core::CreateParams params;
  auto create = cae.AsyncCreate(clio::run::PoolQuery::Local(), "test_import_pool",
                                clio::cae::core::kCaePoolId, params);
  create.Wait();

  TestDirectImport(cae);
  TestRoundTrip(cae);
  TestBadInputs(cae);

  if (g_failures == 0) {
    HLOG(kInfo, "All ImportData integration tests passed.");
    return 0;
  }
  HLOG(kError, "{} check(s) FAILED.", g_failures);
  return 1;
}
