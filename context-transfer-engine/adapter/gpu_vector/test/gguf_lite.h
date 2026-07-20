/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Minimal GGUF reader + ggml block-geometry table, for tests that map real
 * model tensors through the GPU vector.
 *
 * WHY NOT LINK ggml. The gpu_vector tests are nvcc CUDA TUs; pulling ggml's
 * headers and target into them drags llama.cpp's whole build in (and ggml is
 * only conditionally available, gated on CLIO_CORE_ENABLE_LLAMA_SERVER). All
 * that is actually needed is the tensor table -- name, type, shape, and file
 * offset -- which is a short, stable, well-specified part of GGUF v2/v3. So it
 * is parsed directly here. This reader is deliberately READ-ONLY and partial:
 * it skips metadata values rather than interpreting them, except
 * general.alignment, which is required to locate the data section.
 *
 * Block geometry comes from ggml-common.h's own static_asserts
 * (K_SCALE_SIZE=12, QK_K=256, QK4_0=QK8_0=32) so tensor byte sizes are computed
 * the way ggml computes them rather than guessed.
 */

#ifndef CLIO_CTE_GPU_VECTOR_TEST_GGUF_LITE_H_
#define CLIO_CTE_GPU_VECTOR_TEST_GGUF_LITE_H_

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace gguf_lite {

/** A ggml tensor type's block geometry. */
struct Format {
  const char *name;
  uint32_t block_elems;  ///< weights per block (1 for scalar types)
  uint32_t block_bytes;  ///< encoded bytes per block
};

/** Indexed by ggml_type. Only the types these tests exercise are filled in;
 *  block_bytes == 0 marks "not modelled here" so callers can skip cleanly
 *  instead of computing a wrong size. */
inline const Format *FormatForGgmlType(uint32_t t) {
  // ggml_type ordering (ggml.h): 0 F32, 1 F16, 2 Q4_0, 3 Q4_1, 6 Q5_0, 7 Q5_1,
  // 8 Q8_0, 9 Q8_1, 10 Q2_K, 11 Q3_K, 12 Q4_K, 13 Q5_K, 14 Q6_K, 15 Q8_K,
  // 30 BF16.
  static const Format kNone{"?", 0, 0};
  static const Format kTable[] = {
      {"F32", 1, 4},      // 0
      {"F16", 1, 2},      // 1
      {"Q4_0", 32, 18},   // 2
      {"Q4_1", 32, 20},   // 3
      {"?", 0, 0},        // 4  (removed Q4_2)
      {"?", 0, 0},        // 5  (removed Q4_3)
      {"Q5_0", 32, 22},   // 6
      {"Q5_1", 32, 24},   // 7
      {"Q8_0", 32, 34},   // 8
      {"Q8_1", 32, 36},   // 9
      {"Q2_K", 256, 84},  // 10
      {"Q3_K", 256, 110}, // 11
      {"Q4_K", 256, 144}, // 12
      {"Q5_K", 256, 176}, // 13
      {"Q6_K", 256, 210}, // 14
      {"Q8_K", 256, 292}, // 15
  };
  if (t < sizeof(kTable) / sizeof(kTable[0])) {
    return kTable[t].block_bytes ? &kTable[t] : &kNone;
  }
  if (t == 30) {  // BF16
    static const Format kBf16{"BF16", 1, 2};
    return &kBf16;
  }
  return &kNone;
}

/** One tensor's entry from the GGUF tensor table. */
struct TensorInfo {
  std::string name;
  uint32_t ggml_type = 0;
  uint64_t nelem = 0;        ///< product of dims
  uint64_t nbytes = 0;       ///< ggml-computed encoded size
  uint64_t file_offset = 0;  ///< ABSOLUTE offset of the tensor's bytes
};

/**
 * Parse a GGUF file's tensor table. Returns false (with `err` set) rather than
 * throwing, so a test can skip cleanly on a malformed or absent file.
 */
inline bool ReadTensorTable(const std::string &path,
                            std::vector<TensorInfo> *out, std::string *err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { *err = "cannot open " + path; return false; }

  auto rd = [&f](void *dst, size_t n) -> bool {
    f.read(reinterpret_cast<char *>(dst), (std::streamsize) n);
    return (bool) f;
  };
  auto rd_u32 = [&](uint32_t *v) { return rd(v, 4); };
  auto rd_u64 = [&](uint64_t *v) { return rd(v, 8); };
  auto rd_str = [&](std::string *s) {
    uint64_t n = 0;
    if (!rd_u64(&n) || n > (1u << 20)) return false;
    s->resize((size_t) n);
    return n == 0 ? true : rd(&(*s)[0], (size_t) n);
  };

  char magic[4];
  if (!rd(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
    *err = "bad magic"; return false;
  }
  uint32_t version = 0;
  uint64_t n_tensors = 0, n_kv = 0;
  if (!rd_u32(&version) || !rd_u64(&n_tensors) || !rd_u64(&n_kv)) {
    *err = "truncated header"; return false;
  }
  if (version < 2 || version > 3) {
    *err = "unsupported GGUF version " + std::to_string(version);
    return false;
  }

  // Skip metadata, capturing only general.alignment (needed for data_start).
  uint32_t alignment = 32;
  // Value sizes by gguf_metadata_value_type; 0 means variable (string/array).
  auto scalar_size = [](uint32_t t) -> size_t {
    switch (t) {
      case 0: case 1: case 7: return 1;   // u8, i8, bool
      case 2: case 3: return 2;           // u16, i16
      case 4: case 5: case 6: return 4;   // u32, i32, f32
      case 10: case 11: case 12: return 8;  // u64, i64, f64
      default: return 0;                  // 8=string, 9=array
    }
  };
  // Recursive-free skip: arrays of arrays do not occur in practice, and a
  // nested array would be reported as an error rather than silently mis-parsed.
  for (uint64_t i = 0; i < n_kv; ++i) {
    std::string key;
    uint32_t vt = 0;
    if (!rd_str(&key) || !rd_u32(&vt)) { *err = "truncated kv"; return false; }
    if (vt == 8) {  // string
      std::string sv;
      if (!rd_str(&sv)) { *err = "truncated kv string"; return false; }
    } else if (vt == 9) {  // array
      uint32_t et = 0; uint64_t cnt = 0;
      if (!rd_u32(&et) || !rd_u64(&cnt)) { *err = "truncated array"; return false; }
      if (et == 8) {
        for (uint64_t k = 0; k < cnt; ++k) {
          std::string sv;
          if (!rd_str(&sv)) { *err = "truncated array string"; return false; }
        }
      } else if (et == 9) {
        *err = "nested arrays not supported"; return false;
      } else {
        size_t es = scalar_size(et);
        if (es == 0) { *err = "unknown array elem type"; return false; }
        f.seekg((std::streamoff) (es * cnt), std::ios::cur);
        if (!f) { *err = "truncated array data"; return false; }
      }
    } else {
      size_t vs = scalar_size(vt);
      if (vs == 0) { *err = "unknown kv type"; return false; }
      if (vs == 4 && key == "general.alignment") {
        if (!rd_u32(&alignment)) { *err = "truncated alignment"; return false; }
      } else {
        f.seekg((std::streamoff) vs, std::ios::cur);
        if (!f) { *err = "truncated kv value"; return false; }
      }
    }
  }

  // Tensor table.
  std::vector<TensorInfo> infos;
  infos.reserve((size_t) n_tensors);
  for (uint64_t i = 0; i < n_tensors; ++i) {
    TensorInfo ti;
    uint32_t n_dims = 0;
    if (!rd_str(&ti.name) || !rd_u32(&n_dims) || n_dims > 4) {
      *err = "truncated tensor info"; return false;
    }
    uint64_t nelem = 1;
    for (uint32_t d = 0; d < n_dims; ++d) {
      uint64_t dim = 0;
      if (!rd_u64(&dim)) { *err = "truncated dims"; return false; }
      nelem *= dim;
    }
    uint64_t rel_off = 0;
    if (!rd_u32(&ti.ggml_type) || !rd_u64(&rel_off)) {
      *err = "truncated tensor type/offset"; return false;
    }
    ti.nelem = nelem;
    ti.file_offset = rel_off;  // made absolute below
    const Format *fmt = FormatForGgmlType(ti.ggml_type);
    ti.nbytes = fmt->block_bytes
                    ? (nelem / fmt->block_elems) * fmt->block_bytes
                    : 0;  // 0 == geometry unknown; caller skips
    infos.push_back(std::move(ti));
  }

  if (alignment == 0) alignment = 32;
  const uint64_t here = (uint64_t) f.tellg();
  const uint64_t data_start = ((here + alignment - 1) / alignment) * alignment;
  for (auto &ti : infos) ti.file_offset += data_start;

  *out = std::move(infos);
  return true;
}

// ---------------------------------------------------------------------------
// Minimal GGUF *writer*, for building architecturally-real model files in a
// self-contained unit test (no multi-GB download). It emits a byte-exact GGUF
// v3 container -- header, a couple of KV entries, the tensor table, alignment
// padding, then the tensor data -- so the sibling ReadTensorTable parses it the
// same way it parses a file produced by llama.cpp's convert scripts. Tensor
// bytes are deterministic (a per-tensor keyed byte stream) so a reader can
// recompute and verify them without storing a golden copy.
// ---------------------------------------------------------------------------

/** A tensor to emit: name, ggml type, and up to 4 dims (row-major, ggml order:
 *  dims[0] is the fastest-varying / innermost). */
struct TensorSpec {
  std::string name;
  uint32_t ggml_type = 0;
  std::vector<uint64_t> dims;  ///< 1..4 entries
};

/** Deterministic byte for tensor `tkey` at byte index `i`. A cheap xorshift
 *  mix -- not crypto, just enough that a transposed/mis-offset page differs. */
inline uint8_t GgufFillByte(uint32_t tkey, uint64_t i) {
  uint64_t x = (static_cast<uint64_t>(tkey) << 40) ^ (i * 0x9E3779B97F4A7C15ull);
  x ^= x >> 33;
  x *= 0xFF51AFD7ED558CCDull;
  x ^= x >> 29;
  return static_cast<uint8_t>(x & 0xFF);
}

/**
 * Write `tensors` as a GGUF v3 file at `path`. Byte layout matches the reader
 * above. `arch` is written as general.architecture so the file is
 * self-describing. Returns false with `err` set on any I/O failure.
 */
inline bool WriteGguf(const std::string &path, const std::string &arch,
                      const std::vector<TensorSpec> &tensors, std::string *err) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) { *err = "cannot create " + path; return false; }

  auto wr = [&f](const void *src, size_t n) {
    f.write(reinterpret_cast<const char *>(src), (std::streamsize) n);
  };
  auto wr_u32 = [&](uint32_t v) { wr(&v, 4); };
  auto wr_u64 = [&](uint64_t v) { wr(&v, 8); };
  auto wr_str = [&](const std::string &s) {
    wr_u64((uint64_t) s.size());
    if (!s.empty()) wr(s.data(), s.size());
  };

  const uint32_t alignment = 32;

  // --- header ---
  wr("GGUF", 4);
  wr_u32(3);                       // version
  wr_u64((uint64_t) tensors.size());
  wr_u64(3);                       // n_kv (arch, alignment, one array)

  // --- KV: general.architecture (string) ---
  wr_str("general.architecture");
  wr_u32(8);                       // value type: string
  wr_str(arch);
  // --- KV: general.alignment (u32) ---
  wr_str("general.alignment");
  wr_u32(4);                       // value type: u32
  wr_u32(alignment);
  // --- KV: a small u32 array, to exercise the array-skip path ---
  wr_str("test.dims_preview");
  wr_u32(9);                       // value type: array
  wr_u32(4);                       // element type: u32
  wr_u64(2);                       // count
  wr_u32(0);
  wr_u32(0);

  // --- tensor table (offsets are relative to the aligned data section) ---
  std::vector<uint64_t> nbytes(tensors.size());
  uint64_t rel = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    const TensorSpec &t = tensors[i];
    wr_str(t.name);
    wr_u32((uint32_t) t.dims.size());
    uint64_t nelem = 1;
    for (uint64_t d : t.dims) { wr_u64(d); nelem *= d; }
    wr_u32(t.ggml_type);
    const Format *fmt = FormatForGgmlType(t.ggml_type);
    if (fmt->block_bytes == 0) { *err = "unmodelled ggml type in writer"; return false; }
    nbytes[i] = (nelem / fmt->block_elems) * fmt->block_bytes;
    wr_u64(rel);                   // relative offset
    rel += ((nbytes[i] + alignment - 1) / alignment) * alignment;  // pad each tensor
  }

  // --- pad from end of tensor table to the aligned data section ---
  uint64_t here = (uint64_t) f.tellp();
  uint64_t data_start = ((here + alignment - 1) / alignment) * alignment;
  static const char kZeros[64] = {0};
  auto pad_to = [&](uint64_t target) {
    uint64_t cur = (uint64_t) f.tellp();
    while (cur < target) {
      uint64_t n = target - cur;
      wr(kZeros, (size_t) (n < sizeof(kZeros) ? n : sizeof(kZeros)));
      cur += (n < sizeof(kZeros) ? n : sizeof(kZeros));
    }
  };
  pad_to(data_start);

  // --- tensor data, each padded up to `alignment`, matching the table ---
  std::vector<uint8_t> buf;
  for (size_t i = 0; i < tensors.size(); ++i) {
    const uint64_t nb = nbytes[i];
    buf.resize((size_t) nb);
    for (uint64_t b = 0; b < nb; ++b) buf[(size_t) b] = GgufFillByte((uint32_t) i, b);
    wr(buf.data(), (size_t) nb);
    const uint64_t padded = ((nb + alignment - 1) / alignment) * alignment;
    if (padded > nb) wr(kZeros, (size_t) (padded - nb));
  }

  f.flush();
  if (!f) { *err = "write failed for " + path; return false; }
  return true;
}

}  // namespace gguf_lite

#endif  // CLIO_CTE_GPU_VECTOR_TEST_GGUF_LITE_H_
