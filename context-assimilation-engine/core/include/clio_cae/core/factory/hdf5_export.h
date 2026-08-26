#pragma once

// Writing side of the .h5 export: turn a kvhdf5 dataset (a __meta description +
// its chunk blobs) into a real N-D chunked HDF5 dataset.
//
// This is deliberately SEPARATE from ExportData's task coroutine, and holds no
// CLIO types at all -- it takes a chunk's name and its bytes, nothing more. Two
// reasons:
//
//   1. Testability. The HDF5 mechanics here (chunk-offset arithmetic,
//      H5Dwrite_chunk's contract, intermediate-group creation, the dtype map)
//      are the part most likely to be subtly wrong, and inside a coroutine that
//      needs a live runtime, a GPU, and a server they are effectively untestable.
//      Out here, a unit test drives them with synthetic chunks and reads the
//      result back.
//   2. Reuse. Any future exporter (a standalone CLI, a Python binding) wants
//      exactly this and none of the task plumbing.
//
// WHY H5Dwrite_chunk: a kvhdf5 chunk blob IS a whole, unfiltered HDF5 chunk, with
// identical geometry. H5Dwrite_chunk drops those bytes straight into the file's
// chunk slot -- bypassing the chunk cache and the filter pipeline -- so the export
// is a 1:1 copy with no reassembly and never more than ONE chunk resident. A
// dataset far larger than RAM still exports.

#include <clio_cte/kvhdf5/meta_blob.h>

#include <hdf5.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace clio::cae::core {

/**
 * Map a kvhdf5 DType onto an HDF5 type id.
 *
 * The mapping lives HERE, in the exporter, by design: kvhdf5 does not link
 * libhdf5 (it is an HDF5-*like* API over the KV store and is compiled into device
 * code), so its DType enum is deliberately HDF5-agnostic. This is the only place
 * that has HDF5, so this is the only place that can translate.
 *
 * kOpaque ("untyped bytes of width elem_size") becomes a real H5T_OPAQUE of that
 * width, so a raw-byte dataset still exports with its true N-D SHAPE instead of
 * collapsing into a flat byte dump.
 *
 * @param own_type set to true iff the returned id was freshly created and must be
 *        H5Tclose'd. The H5T_NATIVE_* ids are HDF5 predefined constants and
 *        closing them is an error, so the caller cannot just always close.
 * @return a negative id if the dtype is unknown.
 */
inline hid_t DTypeToH5(kvhdf5::DType dtype, uint64_t elem_size, bool *own_type) {
  *own_type = false;
  switch (dtype) {
    case kvhdf5::DType::kI8:  return H5T_NATIVE_INT8;
    case kvhdf5::DType::kU8:  return H5T_NATIVE_UINT8;
    case kvhdf5::DType::kI16: return H5T_NATIVE_INT16;
    case kvhdf5::DType::kU16: return H5T_NATIVE_UINT16;
    case kvhdf5::DType::kI32: return H5T_NATIVE_INT32;
    case kvhdf5::DType::kU32: return H5T_NATIVE_UINT32;
    case kvhdf5::DType::kI64: return H5T_NATIVE_INT64;
    case kvhdf5::DType::kU64: return H5T_NATIVE_UINT64;
    case kvhdf5::DType::kF32: return H5T_NATIVE_FLOAT;
    case kvhdf5::DType::kF64: return H5T_NATIVE_DOUBLE;
    case kvhdf5::DType::kOpaque: {
      hid_t t = H5Tcreate(H5T_OPAQUE, static_cast<size_t>(elem_size));
      if (t < 0) return t;
      H5Tset_tag(t, "kvhdf5 opaque");
      *own_type = true;
      return t;
    }
    default: return -1;
  }
}

/**
 * Writes one kvhdf5 dataset into a fresh .h5 file.
 *
 * Usage: Create() once, WriteChunk() per chunk blob (in any order), then
 * Complete() to learn whether every chunk actually landed. Move-only; the dtor
 * releases every HDF5 id it owns.
 */
class ChunkedDatasetWriter {
 public:
  ChunkedDatasetWriter() = default;
  ~ChunkedDatasetWriter() { Close(); }

  ChunkedDatasetWriter(const ChunkedDatasetWriter &) = delete;
  ChunkedDatasetWriter &operator=(const ChunkedDatasetWriter &) = delete;

  /**
   * Create `file_path` and, inside it, a chunked dataset at `dataset_path`.
   *
   * `dataset_path` is the kvhdf5 tag name, which IS the dataset's full path
   * ("results/snapshots/2026/temperature"), so the group hierarchy comes free:
   * create_intermediate_group makes H5Dcreate2 materialize each '/'-separated
   * prefix as a real HDF5 group.
   *
   * @return false on any HDF5 failure or if `meta` is invalid.
   */
  bool Create(const std::string &file_path, const std::string &dataset_path,
              const kvhdf5::MetaBlob &meta) {
    if (!meta.Valid()) return false;
    meta_ = meta;

    type_ = DTypeToH5(meta.Dtype(), meta.elem_size, &own_type_);
    if (type_ < 0) return false;

    hsize_t dims[kvhdf5::kMetaMaxDims];
    hsize_t chunk[kvhdf5::kMetaMaxDims];
    for (uint32_t i = 0; i < meta.rank; ++i) {
      dims[i] = static_cast<hsize_t>(meta.dims[i]);
      chunk[i] = static_cast<hsize_t>(meta.chunk_dims[i]);
    }

    file_ = H5Fcreate(file_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_ < 0) return false;

    lcpl_ = H5Pcreate(H5P_LINK_CREATE);
    if (lcpl_ < 0) return false;
    H5Pset_create_intermediate_group(lcpl_, 1);

    dcpl_ = H5Pcreate(H5P_DATASET_CREATE);
    if (dcpl_ < 0) return false;
    H5Pset_chunk(dcpl_, static_cast<int>(meta.rank), chunk);
    // We overwrite every chunk, so the fill pass is pure waste. This is safe
    // ONLY because Complete() refuses to call the export a success unless every
    // chunk was written -- an unwritten chunk here is uninitialized file bytes.
    H5Pset_fill_time(dcpl_, H5D_FILL_TIME_NEVER);
    H5Pset_alloc_time(dcpl_, H5D_ALLOC_TIME_EARLY);

    space_ = H5Screate_simple(static_cast<int>(meta.rank), dims, nullptr);
    if (space_ < 0) return false;

    dset_ = H5Dcreate2(file_, dataset_path.c_str(), type_, space_, lcpl_, dcpl_,
                       H5P_DEFAULT);
    if (dset_ < 0) return false;

    // Track which chunk coordinates have actually been written, so Complete()
    // counts DISTINCT chunks. Without this a repeated coordinate would inflate a
    // scalar counter and mask a genuinely missing chunk (which, under
    // FILL_TIME_NEVER, is uninitialized file bytes).
    chunk_seen_.assign(meta_.ChunkCount(), false);
    return true;
  }

  /**
   * Write one chunk, addressed by its kvhdf5 blob name (a chunk coordinate,
   * e.g. "12_47"). `bytes` must be exactly ChunkBytes() -- a mismatch means the
   * blob is not the chunk __meta claims it is, and writing it would silently
   * corrupt the dataset, so it is refused.
   *
   * Chunks may arrive in any order; HDF5 places them by coordinate. Writing the
   * SAME coordinate twice is rejected (returns false) rather than counted twice,
   * so ChunksWritten() stays an honest count of distinct chunks.
   *
   * @return false if the name is not a rank-correct in-range chunk coordinate, it
   *         was already written, the size is wrong, or HDF5 rejects the write.
   */
  bool WriteChunk(const std::string &blob_name, const void *data, size_t bytes) {
    if (dset_ < 0 || data == nullptr) return false;
    if (bytes != static_cast<size_t>(meta_.ChunkBytes())) return false;

    uint64_t coord[kvhdf5::kMetaMaxDims];
    if (!kvhdf5::ParseChunkName(blob_name, meta_.rank, coord)) return false;

    // H5Dwrite_chunk's offset is the chunk origin in ELEMENT coordinates, not a
    // chunk index and not a byte offset.
    hsize_t off[kvhdf5::kMetaMaxDims];
    for (uint32_t i = 0; i < meta_.rank; ++i) {
      if (coord[i] >= meta_.dims[i] / meta_.chunk_dims[i]) return false;  // out of range
      off[i] = static_cast<hsize_t>(coord[i] * meta_.chunk_dims[i]);
    }

    // Reject a duplicate coordinate: it would overwrite an already-placed chunk
    // while advancing the count past a chunk that is still missing.
    const uint64_t idx = meta_.ChunkLinearIndex(coord);
    if (idx >= chunk_seen_.size() || chunk_seen_[idx]) return false;

    if (H5Dwrite_chunk(dset_, H5P_DEFAULT, /*filter_mask=*/0, off, bytes,
                       data) < 0) {
      return false;
    }
    chunk_seen_[idx] = true;
    ++chunks_written_;
    return true;
  }

  /** Chunks successfully written so far. */
  [[nodiscard]] uint64_t ChunksWritten() const { return chunks_written_; }

  /** Chunks the dataset must have for the file to be complete. */
  [[nodiscard]] uint64_t ChunksExpected() const { return meta_.ChunkCount(); }

  /**
   * Flush and close. Returns true only if EVERY chunk was written.
   *
   * A partial dataset is not a partial success -- with FILL_TIME_NEVER the chunks
   * nobody wrote are uninitialized bytes that read back as plausible-looking
   * data. The caller must treat false as a failed export.
   */
  bool Complete() {
    const bool full = (dset_ >= 0) && (chunks_written_ == ChunksExpected());
    Close();
    return full;
  }

 private:
  void Close() {
    if (dset_ >= 0)  { H5Dclose(dset_);  dset_ = -1; }
    if (space_ >= 0) { H5Sclose(space_); space_ = -1; }
    if (dcpl_ >= 0)  { H5Pclose(dcpl_);  dcpl_ = -1; }
    if (lcpl_ >= 0)  { H5Pclose(lcpl_);  lcpl_ = -1; }
    if (own_type_ && type_ >= 0) { H5Tclose(type_); }
    type_ = -1;
    own_type_ = false;
    if (file_ >= 0)  { H5Fclose(file_);  file_ = -1; }
  }

  kvhdf5::MetaBlob meta_{};
  hid_t file_ = -1;
  hid_t dset_ = -1;
  hid_t space_ = -1;
  hid_t dcpl_ = -1;
  hid_t lcpl_ = -1;
  hid_t type_ = -1;
  bool own_type_ = false;
  uint64_t chunks_written_ = 0;
  std::vector<bool> chunk_seen_;  // presence per linear chunk index (dedup)
};

}  // namespace clio::cae::core
