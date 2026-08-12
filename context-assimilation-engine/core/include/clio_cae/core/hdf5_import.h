#pragma once

// Reading side of the .h5 import: turn a real HDF5 dataset into the kvhdf5
// store form (a __meta description + one raw N-D chunk per coordinate). The
// inverse of hdf5_export.h.
//
// Like the exporter, this is deliberately SEPARATE from the ImportData task
// coroutine and holds NO CLIO types -- it hands back a MetaBlob and, per chunk,
// a coordinate name plus raw native bytes. Two reasons, same as export:
//
//   1. Testability. The fiddly parts (the dtype reverse-map, the chunking
//      policy, per-chunk hyperslab arithmetic) are exercisable here with
//      synthetic .h5 files and no runtime, server, or GPU.
//   2. Reuse. A standalone CLI or a Python binding wants exactly this and none
//      of the task plumbing.
//
// WHY LOGICAL H5Dread (not raw H5Dread_chunk): the exporter could use
// H5Dwrite_chunk because it CREATED the file -- unfiltered, native-order,
// chunked. An imported file is arbitrary: it may be contiguous, compressed, or
// foreign-endian. Reading each chunk as a logical hyperslab into a native-typed
// buffer makes HDF5 do the decompression, the layout walk, and the endian
// conversion for us, and still keeps only ONE chunk resident. For a file we
// ourselves exported the read is native->native and unfiltered, i.e. a copy.

#include <clio_cte/kvhdf5/meta_blob.h>

#include <hdf5.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace clio::cae::core {

/**
 * Map an HDF5 datatype onto a kvhdf5 DType + element width. The inverse of
 * DTypeToH5.
 *
 * Classifies by class + size (+ sign for integers). H5T_OPAQUE becomes kOpaque
 * of its byte width. Every other class (compound, string, enum, array,
 * reference, vlen, bitfield, time) and any unsupported size is rejected -- the
 * kvhdf5 store has no representation for them.
 *
 * @param file_type an HDF5 datatype id (typically from H5Dget_type).
 * @return false, leaving the outputs untouched, if the type is not representable.
 */
inline bool H5ToDType(hid_t file_type, kvhdf5::DType *out_dtype,
                      uint64_t *out_elem_size) {
  if (file_type < 0 || out_dtype == nullptr || out_elem_size == nullptr) {
    return false;
  }
  const size_t size = H5Tget_size(file_type);
  if (size == 0) return false;

  switch (H5Tget_class(file_type)) {
    case H5T_FLOAT:
      if (size == 4) *out_dtype = kvhdf5::DType::kF32;
      else if (size == 8) *out_dtype = kvhdf5::DType::kF64;
      else return false;
      break;
    case H5T_INTEGER: {
      const H5T_sign_t sign = H5Tget_sign(file_type);
      if (sign == H5T_SGN_ERROR) return false;
      const bool u = (sign == H5T_SGN_NONE);
      switch (size) {
        case 1: *out_dtype = u ? kvhdf5::DType::kU8  : kvhdf5::DType::kI8;  break;
        case 2: *out_dtype = u ? kvhdf5::DType::kU16 : kvhdf5::DType::kI16; break;
        case 4: *out_dtype = u ? kvhdf5::DType::kU32 : kvhdf5::DType::kI32; break;
        case 8: *out_dtype = u ? kvhdf5::DType::kU64 : kvhdf5::DType::kI64; break;
        default: return false;
      }
      break;
    }
    case H5T_OPAQUE:
      *out_dtype = kvhdf5::DType::kOpaque;
      break;
    default:
      return false;
  }
  *out_elem_size = static_cast<uint64_t>(size);
  return true;
}

/**
 * Reads one HDF5 dataset out as kvhdf5 chunks.
 *
 * Usage: Open() once, then ReadChunk() for each index in [0, Meta().ChunkCount())
 * -- in any order -- into a caller-owned ChunkBytes() buffer. Move-only; the
 * dtor releases every HDF5 id it owns.
 *
 * CHUNKING POLICY. kvhdf5 forbids edge chunks (dims % chunk_dims == 0). So:
 *   - if the file dataset is chunked AND its chunk dims both fit within and
 *     evenly divide the dataset dims, that chunking is ADOPTED (a 1:1, streaming
 *     import -- this is what every exported file hits); otherwise
 *   - the whole dataset becomes a SINGLE chunk (chunk_dims == dims), which is
 *     valid for any shape. That one chunk is the entire dataset, so a very large
 *     contiguous dataset is resident in full for its single read; pre-chunk such
 *     inputs if that matters.
 */
class ChunkedDatasetReader {
 public:
  ChunkedDatasetReader() = default;
  ~ChunkedDatasetReader() { Close(); }

  ChunkedDatasetReader(const ChunkedDatasetReader &) = delete;
  ChunkedDatasetReader &operator=(const ChunkedDatasetReader &) = delete;

  /**
   * Open `dataset_path` inside `file_path` and derive its kvhdf5 layout.
   *
   * @return false (and a clean, closed object) if the file/dataset cannot be
   *         opened, the rank is 0 or exceeds kMetaMaxDims, the element type is
   *         not representable, or the derived layout fails MetaBlob::Valid().
   */
  bool Open(const std::string &file_path, const std::string &dataset_path) {
    file_ = H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_ < 0) return false;

    dset_ = H5Dopen2(file_, dataset_path.c_str(), H5P_DEFAULT);
    if (dset_ < 0) return false;

    file_space_ = H5Dget_space(dset_);
    if (file_space_ < 0) return false;

    const int rank = H5Sget_simple_extent_ndims(file_space_);
    if (rank <= 0 || static_cast<size_t>(rank) > kvhdf5::kMetaMaxDims) return false;

    hsize_t dims[kvhdf5::kMetaMaxDims];
    if (H5Sget_simple_extent_dims(file_space_, dims, nullptr) < 0) return false;

    const hid_t file_type = H5Dget_type(dset_);
    if (file_type < 0) return false;
    kvhdf5::DType dtype;
    uint64_t elem_size = 0;
    if (!H5ToDType(file_type, &dtype, &elem_size)) {
      H5Tclose(file_type);
      return false;
    }

    // Memory type to read INTO: opaque bytes pass through unchanged; a numeric
    // type is converted to this machine's native representation (this is what
    // silently fixes a foreign-endian file). Either way it is a fresh id we own.
    if (dtype == kvhdf5::DType::kOpaque) {
      mem_type_ = H5Tcopy(file_type);
    } else {
      mem_type_ = H5Tget_native_type(file_type, H5T_DIR_ASCEND);
    }
    H5Tclose(file_type);
    if (mem_type_ < 0) return false;

    // Decide the chunking (see the class comment).
    hsize_t chunk[kvhdf5::kMetaMaxDims];
    if (!ChooseChunking(rank, dims, chunk)) return false;

    meta_ = kvhdf5::MetaBlob{};
    meta_.rank = static_cast<uint32_t>(rank);
    meta_.dtype = static_cast<uint32_t>(dtype);
    meta_.elem_size = elem_size;
    for (int i = 0; i < rank; ++i) {
      meta_.dims[i] = static_cast<uint64_t>(dims[i]);
      meta_.chunk_dims[i] = static_cast<uint64_t>(chunk[i]);
    }
    if (!meta_.Valid()) return false;

    mem_space_ = H5Screate_simple(rank, chunk, nullptr);
    if (mem_space_ < 0) return false;

    open_ = true;
    return true;
  }

  /** The derived dataset description. Meaningful only after a successful Open(). */
  [[nodiscard]] const kvhdf5::MetaBlob &Meta() const { return meta_; }

  /**
   * Read the chunk at linear index `index` into `buf`, and set `*out_name` to
   * its coordinate blob name (e.g. "12_47").
   *
   * `buf_bytes` must equal Meta().ChunkBytes(); a mismatch is a caller error and
   * is refused so a short buffer can never be overrun.
   *
   * @return false if not open, the index is out of range, the buffer is wrong,
   *         the name would overflow, or HDF5 fails the read.
   */
  bool ReadChunk(uint64_t index, void *buf, size_t buf_bytes,
                 std::string *out_name) {
    if (!open_ || buf == nullptr || out_name == nullptr) return false;
    if (index >= meta_.ChunkCount()) return false;
    if (buf_bytes != static_cast<size_t>(meta_.ChunkBytes())) return false;

    uint64_t coord[kvhdf5::kMetaMaxDims];
    meta_.ChunkCoordFromLinearIndex(index, coord);
    if (!kvhdf5::ChunkCoordToName(coord, meta_.rank, out_name)) return false;

    hsize_t start[kvhdf5::kMetaMaxDims];
    hsize_t count[kvhdf5::kMetaMaxDims];
    for (uint32_t i = 0; i < meta_.rank; ++i) {
      start[i] = static_cast<hsize_t>(coord[i] * meta_.chunk_dims[i]);
      count[i] = static_cast<hsize_t>(meta_.chunk_dims[i]);
    }
    if (H5Sselect_hyperslab(file_space_, H5S_SELECT_SET, start, nullptr, count,
                            nullptr) < 0) {
      return false;
    }
    return H5Dread(dset_, mem_type_, mem_space_, file_space_, H5P_DEFAULT, buf) >=
           0;
  }

 private:
  // Fill `out_chunk` with the chunk geometry to import, per the class policy.
  // Always leaves an evenly-dividing, in-range chunking. Returns false only on
  // an impossible read of the dataset's own creation plist.
  bool ChooseChunking(int rank, const hsize_t *dims, hsize_t *out_chunk) {
    bool adopt = false;
    const hid_t dcpl = H5Dget_create_plist(dset_);
    if (dcpl < 0) return false;
    if (H5Pget_layout(dcpl) == H5D_CHUNKED) {
      hsize_t file_chunk[kvhdf5::kMetaMaxDims];
      if (H5Pget_chunk(dcpl, rank, file_chunk) == rank) {
        adopt = true;
        for (int i = 0; i < rank; ++i) {
          if (file_chunk[i] == 0 || file_chunk[i] > dims[i] ||
              dims[i] % file_chunk[i] != 0) {
            adopt = false;  // extendible or edge chunking -> cannot adopt
            break;
          }
        }
        if (adopt) {
          for (int i = 0; i < rank; ++i) out_chunk[i] = file_chunk[i];
        }
      }
    }
    H5Pclose(dcpl);
    if (!adopt) {
      // Single chunk covering the whole dataset. Always evenly divides.
      for (int i = 0; i < rank; ++i) out_chunk[i] = dims[i];
    }
    return true;
  }

  void Close() {
    if (mem_space_ >= 0)  { H5Sclose(mem_space_);  mem_space_ = -1; }
    if (file_space_ >= 0) { H5Sclose(file_space_); file_space_ = -1; }
    if (mem_type_ >= 0)   { H5Tclose(mem_type_);   mem_type_ = -1; }
    if (dset_ >= 0)       { H5Dclose(dset_);       dset_ = -1; }
    if (file_ >= 0)       { H5Fclose(file_);       file_ = -1; }
    open_ = false;
  }

  kvhdf5::MetaBlob meta_{};
  hid_t file_ = -1;
  hid_t dset_ = -1;
  hid_t file_space_ = -1;
  hid_t mem_space_ = -1;
  hid_t mem_type_ = -1;
  bool open_ = false;
};

}  // namespace clio::cae::core
