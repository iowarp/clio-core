// Tests for ChunkedDatasetReader (clio_cae/core/hdf5_import.h): the HDF5-reading
// half of ImportData.
//
// No CLIO runtime, no server, no GPU -- this builds .h5 files with the HDF5 C
// API and reads them back through the reader, checking the derived MetaBlob and
// the per-chunk bytes. That is the whole reason the reader was pulled out of the
// task coroutine: the fiddly parts (the dtype reverse-map, the chunking policy,
// per-chunk hyperslab arithmetic) are the most likely to be subtly wrong and
// inside the coroutine were untestable.
//
// The load-bearing checks write each element with its GLOBAL index and read the
// chunks back individually, so a wrong offset or a transposed coordinate lands
// data in the wrong place and the element-by-element comparison catches it.

#include <clio_cae/core/hdf5_import.h>
#include <clio_cte/kvhdf5/meta_blob.h>

#include <hdf5.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using clio::cae::core::ChunkedDatasetReader;
using kvhdf5::DType;

namespace {

int g_failures = 0;

void Check(bool cond, const std::string &what) {
  if (!cond) {
    std::fprintf(stderr, "  FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

// Create a rank-2 dataset at `path` in a fresh file. If `chunk0`/`chunk1` are
// non-zero the dataset is chunked with that geometry (optionally deflated);
// otherwise it is contiguous. Every element is stamped with its global
// row-major index so a misplaced chunk is detectable.
void MakeF32File(const std::string &file, const std::string &path,
                 hsize_t rows, hsize_t cols, hsize_t chunk0, hsize_t chunk1,
                 bool deflate) {
  std::vector<float> data(rows * cols);
  for (hsize_t i = 0; i < rows; ++i)
    for (hsize_t j = 0; j < cols; ++j)
      data[i * cols + j] = static_cast<float>(i * cols + j);

  hid_t f = H5Fcreate(file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hsize_t dims[2] = {rows, cols};
  hid_t space = H5Screate_simple(2, dims, nullptr);
  hid_t dcpl = H5P_DEFAULT;
  if (chunk0 != 0) {
    dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk[2] = {chunk0, chunk1};
    H5Pset_chunk(dcpl, 2, chunk);
    if (deflate) H5Pset_deflate(dcpl, 6);
  }
  // Materialize any '/'-separated groups in `path` (e.g. "/grp/temp").
  hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
  H5Pset_create_intermediate_group(lcpl, 1);
  hid_t d = H5Dcreate2(f, path.c_str(), H5T_IEEE_F32LE, space, lcpl, dcpl,
                       H5P_DEFAULT);
  H5Dwrite(d, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  H5Dclose(d);
  H5Pclose(lcpl);
  if (dcpl != H5P_DEFAULT) H5Pclose(dcpl);
  H5Sclose(space);
  H5Fclose(f);
}

float ExpectedF32(uint64_t i, uint64_t j, uint64_t cols) {
  return static_cast<float>(i * cols + j);
}

// Read every chunk of a 2-D f32 dataset and verify each element carries its
// global index. Works for any valid chunking (adopted or single-chunk).
void VerifyAllChunksF32(ChunkedDatasetReader &r, uint64_t cols) {
  const kvhdf5::MetaBlob &m = r.Meta();
  const size_t cb = static_cast<size_t>(m.ChunkBytes());
  std::vector<unsigned char> buf(cb);
  bool ok = true;
  for (uint64_t k = 0; k < m.ChunkCount() && ok; ++k) {
    std::string name;
    if (!r.ReadChunk(k, buf.data(), cb, &name)) {
      Check(false, "ReadChunk succeeded");
      ok = false;
      break;
    }
    uint64_t coord[kvhdf5::kMetaMaxDims];
    m.ChunkCoordFromLinearIndex(k, coord);
    std::string expect_name;
    kvhdf5::ChunkCoordToName(coord, m.rank, &expect_name);
    Check(name == expect_name, "chunk name == " + expect_name + " (got " + name + ")");

    const auto *f = reinterpret_cast<const float *>(buf.data());
    for (uint64_t li = 0; li < m.chunk_dims[0] && ok; ++li) {
      for (uint64_t lj = 0; lj < m.chunk_dims[1]; ++lj) {
        const uint64_t gi = coord[0] * m.chunk_dims[0] + li;
        const uint64_t gj = coord[1] * m.chunk_dims[1] + lj;
        if (f[li * m.chunk_dims[1] + lj] != ExpectedF32(gi, gj, cols)) {
          std::fprintf(stderr, "  element (%llu,%llu) mismatch\n",
                       (unsigned long long)gi, (unsigned long long)gj);
          ++g_failures;
          ok = false;
          break;
        }
      }
    }
  }
}

// ---- Chunked, evenly divides: adopt the file's chunking --------------------
void TestAdoptChunked() {
  std::fprintf(stderr, "TestAdoptChunked\n");
  const std::string f = "/tmp/cae_import_adopt.h5";
  MakeF32File(f, "/grp/temp", 8, 6, 4, 3, /*deflate=*/false);

  ChunkedDatasetReader r;
  Check(r.Open(f, "/grp/temp"), "Open adopt-chunked");
  const auto &m = r.Meta();
  Check(m.rank == 2, "rank == 2");
  Check(m.Dtype() == DType::kF32 && m.elem_size == 4, "dtype f32");
  Check(m.dims[0] == 8 && m.dims[1] == 6, "dims 8x6");
  Check(m.chunk_dims[0] == 4 && m.chunk_dims[1] == 3, "adopted chunk 4x3");
  Check(m.ChunkCount() == 4, "4 chunks");
  VerifyAllChunksF32(r, 6);
}

// ---- Contiguous: fall back to a single whole-dataset chunk -----------------
void TestContiguousSingleChunk() {
  std::fprintf(stderr, "TestContiguousSingleChunk\n");
  const std::string f = "/tmp/cae_import_contig.h5";
  MakeF32File(f, "/flat", 5, 7, 0, 0, /*deflate=*/false);

  ChunkedDatasetReader r;
  Check(r.Open(f, "/flat"), "Open contiguous");
  const auto &m = r.Meta();
  Check(m.chunk_dims[0] == 5 && m.chunk_dims[1] == 7, "single chunk == dims");
  Check(m.ChunkCount() == 1, "1 chunk");
  VerifyAllChunksF32(r, 7);
}

// ---- Chunked but non-dividing: also single whole-dataset chunk -------------
void TestNonDividingSingleChunk() {
  std::fprintf(stderr, "TestNonDividingSingleChunk\n");
  const std::string f = "/tmp/cae_import_nondiv.h5";
  MakeF32File(f, "/nd", 6, 6, 4, 4, /*deflate=*/false);  // 6 % 4 != 0

  ChunkedDatasetReader r;
  Check(r.Open(f, "/nd"), "Open non-dividing chunked");
  const auto &m = r.Meta();
  Check(m.chunk_dims[0] == 6 && m.chunk_dims[1] == 6, "fell back to single chunk");
  Check(m.ChunkCount() == 1, "1 chunk");
  VerifyAllChunksF32(r, 6);
}

// ---- Compressed chunks are decompressed on read ----------------------------
void TestDeflateChunked() {
  std::fprintf(stderr, "TestDeflateChunked\n");
  if (H5Zfilter_avail(H5Z_FILTER_DEFLATE) <= 0) {
    std::fprintf(stderr, "  (deflate filter unavailable; skipping)\n");
    return;
  }
  const std::string f = "/tmp/cae_import_deflate.h5";
  MakeF32File(f, "/z", 8, 6, 4, 3, /*deflate=*/true);

  ChunkedDatasetReader r;
  Check(r.Open(f, "/z"), "Open deflated");
  Check(r.Meta().chunk_dims[0] == 4 && r.Meta().chunk_dims[1] == 3,
        "adopted chunk 4x3 through filter");
  VerifyAllChunksF32(r, 6);  // must read back DECOMPRESSED native bytes
}

// ---- Integer/native dtype mapping ------------------------------------------
void TestIntDtypeMapping() {
  std::fprintf(stderr, "TestIntDtypeMapping\n");
  const std::string f = "/tmp/cae_import_int.h5";
  hid_t file = H5Fcreate(f.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hsize_t dims[1] = {4};
  hid_t space = H5Screate_simple(1, dims, nullptr);
  std::vector<uint16_t> data = {1, 2, 3, 4};
  hid_t d = H5Dcreate2(file, "/u16", H5T_STD_U16LE, space, H5P_DEFAULT,
                       H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(d, H5T_NATIVE_UINT16, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  H5Dclose(d);
  H5Sclose(space);
  H5Fclose(file);

  ChunkedDatasetReader r;
  Check(r.Open(f, "/u16"), "Open u16");
  Check(r.Meta().Dtype() == DType::kU16 && r.Meta().elem_size == 2,
        "mapped to kU16");
  unsigned char buf[8];
  std::string name;
  Check(r.ReadChunk(0, buf, 8, &name), "read u16 chunk");
  const auto *u = reinterpret_cast<const uint16_t *>(buf);
  Check(u[0] == 1 && u[3] == 4, "u16 values round-trip");
}

// ---- Unsupported element type is rejected ----------------------------------
void TestUnsupportedTypeRejected() {
  std::fprintf(stderr, "TestUnsupportedTypeRejected\n");
  const std::string f = "/tmp/cae_import_str.h5";
  hid_t file = H5Fcreate(f.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hsize_t dims[1] = {2};
  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t st = H5Tcopy(H5T_C_S1);
  H5Tset_size(st, H5T_VARIABLE);
  hid_t d = H5Dcreate2(file, "/s", st, space, H5P_DEFAULT, H5P_DEFAULT,
                       H5P_DEFAULT);
  H5Dclose(d);
  H5Tclose(st);
  H5Sclose(space);
  H5Fclose(file);

  ChunkedDatasetReader r;
  Check(!r.Open(f, "/s"), "variable-length string dataset is refused");
}

// ---- A missing dataset is refused ------------------------------------------
void TestMissingDatasetRejected() {
  std::fprintf(stderr, "TestMissingDatasetRejected\n");
  const std::string f = "/tmp/cae_import_adopt.h5";  // exists from an earlier test
  ChunkedDatasetReader r;
  Check(!r.Open(f, "/does/not/exist"), "absent dataset is refused");
  ChunkedDatasetReader r2;
  Check(!r2.Open("/tmp/cae_import_no_such_file.h5", "/x"), "absent file is refused");
}

}  // namespace

int main() {
  // Quiet HDF5's own error stack; we test return values, not stderr noise.
  H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

  TestAdoptChunked();
  TestContiguousSingleChunk();
  TestNonDividingSingleChunk();
  TestDeflateChunked();
  TestIntDtypeMapping();
  TestUnsupportedTypeRejected();
  TestMissingDatasetRejected();

  if (g_failures == 0) {
    std::fprintf(stderr, "All ChunkedDatasetReader tests passed.\n");
    return 0;
  }
  std::fprintf(stderr, "%d check(s) FAILED.\n", g_failures);
  return 1;
}
