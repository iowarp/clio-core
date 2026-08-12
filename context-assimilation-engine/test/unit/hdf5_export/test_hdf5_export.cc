// Tests for ChunkedDatasetWriter (clio_cae/core/hdf5_export.h): the HDF5-writing
// half of ExportData.
//
// No CLIO runtime, no server, no GPU -- this drives the writer with synthetic
// chunks and reads the resulting .h5 back with the HDF5 C API. That is the whole
// reason the writer was pulled out of the task coroutine: these mechanics
// (chunk-offset arithmetic, H5Dwrite_chunk's contract, intermediate-group
// creation, the dtype map) are the part most likely to be subtly wrong, and
// inside the coroutine they were untestable.
//
// The load-bearing test is "round-trips a 2-D f32 dataset": every element is
// stamped with its GLOBAL index and the chunks are written OUT OF ORDER, so a
// wrong chunk-offset formula cannot pass -- it would land data in the wrong place
// and the element-by-element comparison would catch it. A test that wrote chunks
// in order with uniform values would happily pass with the offsets transposed.

#include <clio_cae/core/hdf5_export.h>
#include <clio_cte/kvhdf5/meta_blob.h>

#include <hdf5.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using clio::cae::core::ChunkedDatasetWriter;
using kvhdf5::DType;
using kvhdf5::MetaBlob;

namespace {

int g_failures = 0;

void Check(bool cond, const std::string &what) {
  if (!cond) {
    std::fprintf(stderr, "  FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

// 2-D f32: 8x6 elements, 4x3 chunks => 2x2 == 4 chunks.
constexpr uint64_t kRows = 8, kCols = 6;
constexpr uint64_t kCRows = 4, kCCols = 3;

MetaBlob Make2dF32() {
  MetaBlob m;
  m.rank = 2;
  m.dtype = static_cast<uint32_t>(DType::kF32);
  m.elem_size = 4;
  m.dims[0] = kRows;       m.dims[1] = kCols;
  m.chunk_dims[0] = kCRows; m.chunk_dims[1] = kCCols;
  return m;
}

// The value we expect at global element (i, j). Distinct for every element, so a
// misplaced chunk is always detectable.
float Expected(uint64_t i, uint64_t j) {
  return static_cast<float>(i * kCols + j);
}

// Build chunk (ci, cj)'s payload: row-major within the chunk, each element
// carrying its GLOBAL index value.
std::vector<float> MakeChunk(uint64_t ci, uint64_t cj) {
  std::vector<float> buf(kCRows * kCCols);
  for (uint64_t li = 0; li < kCRows; ++li) {
    for (uint64_t lj = 0; lj < kCCols; ++lj) {
      buf[li * kCCols + lj] = Expected(ci * kCRows + li, cj * kCCols + lj);
    }
  }
  return buf;
}

void TestRoundTrip2dF32() {
  std::fprintf(stderr, "TestRoundTrip2dF32\n");
  const std::string path = "/tmp/cae_hdf5_export_2d.h5";
  const std::string dset_path = "results/snapshots/2026/temperature";
  std::remove(path.c_str());

  {
    ChunkedDatasetWriter w;
    Check(w.Create(path, dset_path, Make2dF32()), "Create");
    Check(w.ChunksExpected() == 4, "ChunksExpected == 4");

    // OUT OF ORDER on purpose: HDF5 places chunks by coordinate, and this is what
    // makes a bad offset formula fail loudly instead of accidentally passing.
    const std::pair<uint64_t, uint64_t> order[4] = {{1, 1}, {0, 1}, {1, 0}, {0, 0}};
    for (auto [ci, cj] : order) {
      const std::vector<float> chunk = MakeChunk(ci, cj);
      const std::string name = std::to_string(ci) + "_" + std::to_string(cj);
      Check(w.WriteChunk(name, chunk.data(), chunk.size() * sizeof(float)),
            "WriteChunk " + name);
    }
    Check(w.ChunksWritten() == 4, "ChunksWritten == 4");
    Check(w.Complete(), "Complete() == true (all chunks present)");
  }

  // ---- Read it back as a plain HDF5 consumer would ----
  hid_t f = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  Check(f >= 0, "H5Fopen");
  if (f < 0) return;

  // The tag's '/' segments must have become real groups.
  Check(H5Lexists(f, "results", H5P_DEFAULT) > 0, "group 'results' exists");
  Check(H5Lexists(f, "results/snapshots", H5P_DEFAULT) > 0,
        "group 'results/snapshots' exists");
  Check(H5Lexists(f, "results/snapshots/2026", H5P_DEFAULT) > 0,
        "group 'results/snapshots/2026' exists");

  hid_t d = H5Dopen2(f, dset_path.c_str(), H5P_DEFAULT);
  Check(d >= 0, "H5Dopen2 on the full path");
  if (d < 0) { H5Fclose(f); return; }

  // Shape.
  hid_t s = H5Dget_space(d);
  Check(H5Sget_simple_extent_ndims(s) == 2, "rank == 2");
  hsize_t dims[2] = {0, 0};
  H5Sget_simple_extent_dims(s, dims, nullptr);
  Check(dims[0] == kRows && dims[1] == kCols, "dims == 8x6");

  // Type: a real float, not opaque bytes.
  hid_t t = H5Dget_type(d);
  Check(H5Tequal(t, H5T_NATIVE_FLOAT) > 0, "dtype == H5T_NATIVE_FLOAT");

  // Chunked layout with the geometry we asked for.
  hid_t dcpl = H5Dget_create_plist(d);
  Check(H5Pget_layout(dcpl) == H5D_CHUNKED, "layout == H5D_CHUNKED");
  hsize_t chunk[2] = {0, 0};
  H5Pget_chunk(dcpl, 2, chunk);
  Check(chunk[0] == kCRows && chunk[1] == kCCols, "chunk dims == 4x3");

  // Contents: every element must be where it belongs.
  std::vector<float> got(kRows * kCols, -1.0f);
  Check(H5Dread(d, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                got.data()) >= 0, "H5Dread");
  bool all_ok = true;
  for (uint64_t i = 0; i < kRows && all_ok; ++i) {
    for (uint64_t j = 0; j < kCols; ++j) {
      const float want = Expected(i, j);
      if (got[i * kCols + j] != want) {
        std::fprintf(stderr,
                     "  FAIL: element (%llu,%llu) == %f, expected %f "
                     "(chunk placed at the wrong offset?)\n",
                     (unsigned long long)i, (unsigned long long)j,
                     got[i * kCols + j], want);
        ++g_failures;
        all_ok = false;
        break;
      }
    }
  }
  if (all_ok) std::fprintf(stderr, "  every element landed at its correct index\n");

  H5Pclose(dcpl);
  H5Tclose(t);
  H5Sclose(s);
  H5Dclose(d);
  H5Fclose(f);
}

void TestIncompleteIsAFailure() {
  std::fprintf(stderr, "TestIncompleteIsAFailure\n");
  const std::string path = "/tmp/cae_hdf5_export_partial.h5";
  std::remove(path.c_str());

  ChunkedDatasetWriter w;
  Check(w.Create(path, "a/b", Make2dF32()), "Create");

  // Write 3 of the 4 chunks. With FILL_TIME_NEVER the missing chunk is
  // uninitialized bytes, so this MUST NOT be reported as a success.
  const std::pair<uint64_t, uint64_t> order[3] = {{0, 0}, {0, 1}, {1, 0}};
  for (auto [ci, cj] : order) {
    const std::vector<float> chunk = MakeChunk(ci, cj);
    w.WriteChunk(std::to_string(ci) + "_" + std::to_string(cj), chunk.data(),
                 chunk.size() * sizeof(float));
  }
  Check(w.ChunksWritten() == 3, "ChunksWritten == 3");
  Check(!w.Complete(), "Complete() == FALSE when a chunk is missing");
}

void TestBadChunksRejected() {
  std::fprintf(stderr, "TestBadChunksRejected\n");
  const std::string path = "/tmp/cae_hdf5_export_bad.h5";
  std::remove(path.c_str());

  ChunkedDatasetWriter w;
  Check(w.Create(path, "d", Make2dF32()), "Create");

  const std::vector<float> good = MakeChunk(0, 0);
  const size_t good_bytes = good.size() * sizeof(float);

  Check(!w.WriteChunk("0_0", good.data(), good_bytes - 4),
        "wrong-size chunk rejected (would corrupt the dataset)");
  Check(!w.WriteChunk("__meta", good.data(), good_bytes),
        "reserved name rejected as a chunk");
  Check(!w.WriteChunk("0", good.data(), good_bytes),
        "wrong-rank chunk name rejected");
  Check(!w.WriteChunk("9_9", good.data(), good_bytes),
        "out-of-range chunk coord rejected");
  Check(!w.WriteChunk("0_0", nullptr, good_bytes), "null data rejected");
  Check(w.ChunksWritten() == 0, "no bad chunk was counted as written");
}

void TestDuplicateChunkRejected() {
  std::fprintf(stderr, "TestDuplicateChunkRejected\n");
  const std::string path = "/tmp/cae_hdf5_export_dup.h5";
  std::remove(path.c_str());

  ChunkedDatasetWriter w;
  Check(w.Create(path, "d", Make2dF32()), "Create");

  const std::vector<float> c00 = MakeChunk(0, 0);
  const size_t nbytes = c00.size() * sizeof(float);

  Check(w.WriteChunk("0_0", c00.data(), nbytes), "first write of 0_0 accepted");
  // Writing 0_0 again must be REJECTED, not silently counted — otherwise a
  // duplicate could push ChunksWritten() to the expected total while a different
  // coordinate is still missing, and Complete() would wrongly report success.
  Check(!w.WriteChunk("0_0", c00.data(), nbytes), "second write of 0_0 rejected");
  Check(w.ChunksWritten() == 1, "duplicate not counted (still 1 written)");

  // Fill the remaining three DISTINCT chunks; only then is it complete.
  const std::pair<uint64_t, uint64_t> rest[3] = {{0, 1}, {1, 0}, {1, 1}};
  for (auto [ci, cj] : rest) {
    const std::vector<float> chunk = MakeChunk(ci, cj);
    w.WriteChunk(std::to_string(ci) + "_" + std::to_string(cj), chunk.data(),
                 chunk.size() * sizeof(float));
  }
  Check(w.ChunksWritten() == 4, "four distinct chunks written");
  Check(w.Complete(), "Complete() true once all four distinct chunks present");
}

void TestOpaqueKeepsItsShape() {
  std::fprintf(stderr, "TestOpaqueKeepsItsShape\n");
  const std::string path = "/tmp/cae_hdf5_export_opaque.h5";
  std::remove(path.c_str());

  // Untyped 3-byte elements: 4x4 in 2x2 chunks. The point is that an opaque
  // dataset still exports with its true N-D SHAPE rather than a flat byte dump.
  MetaBlob m;
  m.rank = 2;
  m.dtype = static_cast<uint32_t>(DType::kOpaque);
  m.elem_size = 3;
  m.dims[0] = 4; m.dims[1] = 4;
  m.chunk_dims[0] = 2; m.chunk_dims[1] = 2;

  {
    ChunkedDatasetWriter w;
    Check(w.Create(path, "raw", m), "Create (opaque)");
    std::vector<uint8_t> chunk(2 * 2 * 3, 0xAB);
    for (uint64_t ci = 0; ci < 2; ++ci) {
      for (uint64_t cj = 0; cj < 2; ++cj) {
        Check(w.WriteChunk(std::to_string(ci) + "_" + std::to_string(cj),
                           chunk.data(), chunk.size()),
              "WriteChunk (opaque)");
      }
    }
    Check(w.Complete(), "Complete (opaque)");
  }

  hid_t f = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  Check(f >= 0, "H5Fopen (opaque)");
  if (f < 0) return;
  hid_t d = H5Dopen2(f, "raw", H5P_DEFAULT);
  Check(d >= 0, "H5Dopen2 (opaque)");
  if (d >= 0) {
    hid_t t = H5Dget_type(d);
    Check(H5Tget_class(t) == H5T_OPAQUE, "type class == H5T_OPAQUE");
    Check(H5Tget_size(t) == 3, "opaque element width == 3");

    hid_t s = H5Dget_space(d);
    hsize_t dims[2] = {0, 0};
    H5Sget_simple_extent_dims(s, dims, nullptr);
    Check(dims[0] == 4 && dims[1] == 4, "opaque dataset kept its 4x4 shape");

    H5Sclose(s);
    H5Tclose(t);
    H5Dclose(d);
  }
  H5Fclose(f);
}

void TestInvalidMetaRefused() {
  std::fprintf(stderr, "TestInvalidMetaRefused\n");
  ChunkedDatasetWriter w;
  MetaBlob bad = Make2dF32();
  bad.dims[0] = 9;  // 9 % 4 != 0 -- an edge-chunk layout the writer cannot honor
  Check(!w.Create("/tmp/cae_hdf5_export_never.h5", "x", bad),
        "Create refuses an invalid __meta");
}

}  // namespace

int main() {
  // HDF5 prints its own error stack on failure; we report our own diagnostics.
  TestRoundTrip2dF32();
  TestIncompleteIsAFailure();
  TestBadChunksRejected();
  TestDuplicateChunkRejected();
  TestOpaqueKeepsItsShape();
  TestInvalidMetaRefused();

  if (g_failures == 0) {
    std::fprintf(stderr, "\nAll ChunkedDatasetWriter tests passed.\n");
    return 0;
  }
  std::fprintf(stderr, "\n%d check(s) FAILED.\n", g_failures);
  return 1;
}
