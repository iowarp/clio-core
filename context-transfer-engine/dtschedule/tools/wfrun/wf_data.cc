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
 * @file wf_data.cc
 * Data generation, payload windows and the compute kernel of
 * dtschedule_wfrun (see wf_data.h).
 */

#include "wf_data.h"

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <unordered_map>

namespace dtwf {

namespace {

/** Base block per file: later blocks are cheap rotations of it. */
constexpr size_t kBlock = 8u << 20;
/** Width of the float field in elements (rows of the 2-D field). */
constexpr size_t kWidth = 2048;
/** Float fields are snapped to this grid (limited stored precision). */
constexpr float kQuantum = 1.0f / 512.0f;
/** mixed_binary interleaves float and random runs of this length. */
constexpr size_t kStripe = 64u << 10;

/** Extension -> class, as dt_datagen.EXT_TABLE. */
const std::unordered_map<std::string, DataClass> &ExtTable() {
  static const std::unordered_map<std::string, DataClass> table = [] {
    std::unordered_map<std::string, DataClass> t;
    for (const char *e : {".fits", ".nc", ".h5", ".hdf5", ".bp", ".sac", ".dat",
                          ".stf", ".lht", ".bin", ".raw", ".f32"}) {
      t[e] = DataClass::kFloatField;
    }
    for (const char *e :
         {".fastq", ".fq", ".sfq", ".fa", ".fna", ".fasta", ".bfq"}) {
      t[e] = DataClass::kTextSeq;
    }
    for (const char *e : {".vcf", ".txt", ".tbl", ".csv", ".map", ".pileup",
                          ".out", ".sam", ".hdr", ".err", ".list", ".idx",
                          ".soil", ".weather", ".operation", ".ctrl"}) {
      t[e] = DataClass::kTextTable;
    }
    return t;
  }();
  return table;
}

/** splitmix64 generator: small, fast, deterministic. */
class Rng {
 public:
  /**
   * Seed the generator.
   * @param seed Seed
   */
  explicit Rng(uint64_t seed) : s_(seed * 0x9E3779B97F4A7C15ULL + 1) {}

  /**
   * Next 64 random bits.
   * @return Random value
   */
  uint64_t Next() {
    uint64_t z = (s_ += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }

  /**
   * Uniform double in [0, 1).
   * @return Random value
   */
  double Uniform() { return static_cast<double>(Next() >> 11) * 0x1.0p-53; }

  /**
   * Uniform integer in [lo, hi].
   * @param lo Lower bound
   * @param hi Upper bound (inclusive)
   * @return Random value
   */
  int Int(int lo, int hi) {
    return lo + static_cast<int>(Next() % static_cast<uint64_t>(hi - lo + 1));
  }

  /**
   * Standard normal sample (Box-Muller).
   * @return Random value
   */
  double Normal() {
    const double u = std::max(Uniform(), 1e-300), v = Uniform();
    return std::sqrt(-2.0 * std::log(u)) * std::cos(2.0 * M_PI * v);
  }

 private:
  uint64_t s_;  ///< State
};

/**
 * Smooth 2-D float32 field (Gaussian bumps + sinusoids + noise relative to
 * the field's std, quantized), as dt_datagen._float_field.
 * @param seed Seed
 * @param noise Relative noise amplitude
 * @param bytes Block size (a multiple of two rows)
 * @return The block
 */
std::vector<char> FloatBlock(uint32_t seed, double noise, size_t bytes) {
  Rng rng(seed);
  const size_t n = bytes / sizeof(float), h = n / kWidth;
  std::vector<float> f(n, 0.0f);
  std::vector<float> ex(kWidth), ey(h);
  for (int b = 0; b < 8; ++b) {
    const double cx = rng.Uniform(), cy = rng.Uniform();
    const double s = 0.03 + 0.12 * rng.Uniform(),
                 amp = 0.5 + 1.5 * rng.Uniform();
    for (size_t x = 0; x < kWidth; ++x) {
      const double d = static_cast<double>(x) / (kWidth - 1) - cx;
      ex[x] = static_cast<float>(amp * std::exp(-d * d / (2 * s * s)));
    }
    for (size_t y = 0; y < h; ++y) {
      const double d = static_cast<double>(y) / (h - 1) - cy;
      ey[y] = static_cast<float>(std::exp(-d * d / (2 * s * s)));
    }
    for (size_t y = 0; y < h; ++y) {
      for (size_t x = 0; x < kWidth; ++x) f[y * kWidth + x] += ex[x] * ey[y];
    }
  }
  for (int k = 0; k < 3; ++k) {
    const double fx = rng.Int(1, 11), fy = rng.Int(1, 11);
    const double ph = rng.Uniform() * 2 * M_PI, amp = 0.1 + 0.4 * rng.Uniform();
    for (size_t y = 0; y < h; ++y) {
      for (size_t x = 0; x < kWidth; ++x) {
        f[y * kWidth + x] += static_cast<float>(
            amp *
            std::sin(2 * M_PI * (fx * x / (kWidth - 1) + fy * y / (h - 1)) +
                     ph));
      }
    }
  }
  double sum = 0, sq = 0;
  for (float v : f) {
    sum += v;
    sq += static_cast<double>(v) * v;
  }
  const double mean = sum / n,
               sd = std::sqrt(std::max(sq / n - mean * mean, 1e-12));
  for (float &v : f) {
    if (noise > 0) v += static_cast<float>(noise * sd * rng.Normal());
    v = std::round(v / kQuantum) * kQuantum;
  }
  std::vector<char> out(bytes);
  std::memcpy(out.data(), f.data(), bytes);
  return out;
}

/**
 * FASTQ-like records (ACGT reads of 100-150 bases with random-walk quality
 * strings), as dt_datagen._text_seq.
 * @param seed Seed
 * @param bytes Block size
 * @return The block
 */
std::vector<char> SeqBlock(uint32_t seed, size_t bytes) {
  static const char kBases[] = "AAACCGGTTT";
  Rng rng(seed);
  std::string s;
  s.reserve(bytes + 512);
  char head[64];
  for (int i = 0; s.size() < bytes; ++i) {
    const int len = rng.Int(100, 150);
    snprintf(head, sizeof(head), "@SEQ_%08x_%07d len=%d\n", seed, i, len);
    s += head;
    for (int j = 0; j < len; ++j) s += kBases[rng.Next() % 10];
    s += "\n+\n";
    int q = 36;
    for (int j = 0; j < len; ++j) {
      q = std::min(40, std::max(2, q + rng.Int(-2, 2)));
      s += static_cast<char>(q + 33);
    }
    s += '\n';
  }
  return std::vector<char>(s.begin(), s.begin() + bytes);
}

/**
 * VCF/TSV-like rows (sorted chromosomes, increasing positions, small
 * alphabets), as dt_datagen._text_table.
 * @param seed Seed
 * @param bytes Block size
 * @return The block
 */
std::vector<char> TableBlock(uint32_t seed, size_t bytes) {
  static const char *kFilters[] = {"PASS", "PASS", "PASS", "LowQual", "."};
  static const char kAcgt[] = "ACGT";
  Rng rng(seed);
  std::string s = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n";
  s.reserve(bytes + 256);
  const size_t rows = bytes / 40 + 64, per_chrom = rows / 23 + 1;
  uint64_t pos = 0;
  char row[160];
  for (size_t i = 0; s.size() < bytes; ++i) {
    const size_t c = std::min<size_t>(i / per_chrom, 22);
    pos += rng.Int(1, 399);
    double qual = 0;
    for (int k = 0; k < 4; ++k)
      qual -= 12.0 * std::log(std::max(rng.Uniform(), 1e-12));
    const double u = rng.Uniform();
    snprintf(row, sizeof(row),
             "chr%s\t%llu\t.\t%c\t%c\t%.1f\t%s\tDP=%d;AF=%.3f\n",
             c == 22 ? "X" : std::to_string(c + 1).c_str(),
             static_cast<unsigned long long>(pos), kAcgt[rng.Next() % 4],
             kAcgt[rng.Next() % 4], qual, kFilters[rng.Next() % 5],
             rng.Int(5, 119), u * u * u);
    s += row;
  }
  return std::vector<char>(s.begin(), s.begin() + bytes);
}

/**
 * Half float field, half random bytes, interleaved in kStripe runs (like
 * already-compressed formats), as dt_datagen._mixed_binary.
 * @param seed Seed
 * @param noise Noise of the float half
 * @param bytes Block size (a multiple of two stripes)
 * @return The block
 */
std::vector<char> MixedBlock(uint32_t seed, double noise, size_t bytes) {
  std::vector<char> fl = FloatBlock(seed, noise, bytes), out(bytes);
  Rng rng(seed ^ 0x5A5A5A5Au);
  for (size_t s = 0; s < bytes / kStripe; ++s) {
    char *dst = out.data() + s * kStripe;
    if (s % 2 == 0) {
      std::memcpy(dst, fl.data() + (s / 2) * kStripe, kStripe);
    } else {
      for (size_t i = 0; i < kStripe; i += 8) {
        const uint64_t r = rng.Next();
        std::memcpy(dst + i, &r, 8);
      }
    }
  }
  return out;
}

/**
 * Base block size for a file: kBlock, or the file size rounded up to the
 * class's granule (two field rows, two stripes) for smaller files.
 * @param cls Data class
 * @param bytes File size
 * @return Block size
 */
size_t BlockSize(DataClass cls, size_t bytes) {
  size_t g = 4096;
  if (cls == DataClass::kFloatField) g = 2 * kWidth * sizeof(float);
  if (cls == DataClass::kMixedBinary) g = 2 * kStripe;
  return std::min(kBlock, std::max(g, (bytes + g - 1) / g * g));
}

/**
 * Copy block k of a file: the base block rotated (text: to a line start)
 * and, for float fields, raised by k quanta.
 * @param base Base block
 * @param cls Data class
 * @param k Block index
 * @param out Destination
 * @param bytes Bytes to copy (<= base size)
 */
void CopyBlock(const std::vector<char> &base, DataClass cls, size_t k,
               char *out, size_t bytes) {
  const size_t bs = base.size();
  size_t shift = 0;
  if (cls == DataClass::kFloatField) {
    shift = (k * 4099 * sizeof(float)) % bs;
  } else if (cls == DataClass::kMixedBinary) {
    shift = (k * 2 * kStripe) % bs;
  } else if (k > 0) {
    size_t at = (k * 7919 * 131) % bs;
    while (at < bs && base[at] != '\n') ++at;
    shift = at + 1 < bs ? at + 1 : 0;
  }
  const size_t first = std::min(bytes, bs - shift);
  std::memcpy(out, base.data() + shift, first);
  if (bytes > first) std::memcpy(out + first, base.data(), bytes - first);
  if (cls == DataClass::kFloatField && k > 0) {
    float *f = reinterpret_cast<float *>(out);
    const float drift = static_cast<float>(k) * kQuantum;
    for (size_t i = 0; i < bytes / sizeof(float); ++i) f[i] += drift;
  }
}

}  // namespace

DataClass ClassifyFile(const std::string &name) {
  const size_t slash = name.find_last_of('/');
  std::string base = slash == std::string::npos ? name : name.substr(slash + 1);
  const size_t dot = base.find_last_of('.');
  if (dot == std::string::npos) return DataClass::kMixedBinary;
  std::string ext = base.substr(dot);
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  auto it = ExtTable().find(ext);
  return it == ExtTable().end() ? DataClass::kMixedBinary : it->second;
}

const char *DataClassName(DataClass c) {
  switch (c) {
    case DataClass::kFloatField:
      return "float_field";
    case DataClass::kTextSeq:
      return "text_seq";
    case DataClass::kTextTable:
      return "text_table";
    default:
      return "mixed_binary";
  }
}

uint32_t SeedOf(const std::string &name) {
  const size_t slash = name.find_last_of('/');
  const std::string base =
      slash == std::string::npos ? name : name.substr(slash + 1);
  uint32_t crc = 0xFFFFFFFFu;
  for (unsigned char ch : base) {
    crc ^= ch;
    for (int k = 0; k < 8; ++k)
      crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
  }
  return ~crc;
}

void GenerateFile(const std::string &name, DataClass cls, double noise,
                  char *out, size_t bytes) {
  const uint32_t seed = SeedOf(name);
  const size_t bs = BlockSize(cls, bytes);
  std::vector<char> base;
  switch (cls) {
    case DataClass::kFloatField:
      base = FloatBlock(seed, noise, bs);
      break;
    case DataClass::kTextSeq:
      base = SeqBlock(seed, bs);
      break;
    case DataClass::kTextTable:
      base = TableBlock(seed, bs);
      break;
    default:
      base = MixedBlock(seed, noise, bs);
      break;
  }
  for (size_t off = 0, k = 0; off < bytes; off += bs, ++k) {
    CopyBlock(base, cls, k, out + off, std::min(bs, bytes - off));
  }
}

bool Payload::Open(const std::string &dir) {
  struct dirent **list = nullptr;
  const int n = scandir(dir.c_str(), &list, nullptr, alphasort);
  for (int i = 0; i < n; ++i) {
    const std::string path = dir + "/" + list[i]->d_name;
    struct stat st;
    if (list[i]->d_name[0] != '.' && stat(path.c_str(), &st) == 0 &&
        S_ISREG(st.st_mode) && st.st_size > 0) {
      files_.push_back(path);
    }
    free(list[i]);
  }
  free(list);
  return !files_.empty();
}

const std::vector<char> *Payload::Load(size_t idx) {
  for (auto it = cache_.begin(); it != cache_.end(); ++it) {
    if (it->first == idx) {
      cache_.splice(cache_.begin(), cache_, it);
      return &cache_.front().second;
    }
  }
  std::ifstream in(files_[idx], std::ios::binary | std::ios::ate);
  if (!in) return nullptr;
  std::vector<char> data(static_cast<size_t>(in.tellg()));
  in.seekg(0);
  if (!in.read(data.data(), static_cast<std::streamsize>(data.size())))
    return nullptr;
  cache_.emplace_front(idx, std::move(data));
  if (cache_.size() > 2) cache_.pop_back();
  return &cache_.front().second;
}

bool Payload::Window(const std::string &name, char *out, size_t bytes) {
  if (files_.empty()) return false;
  const uint32_t seed = SeedOf(name);
  const std::vector<char> *p = Load(seed % files_.size());
  if (p == nullptr || p->empty()) return false;
  const size_t plen = p->size();
  size_t off =
      (static_cast<size_t>(seed) * 4096u) % plen & ~static_cast<size_t>(4095);
  for (size_t done = 0; done < bytes;) {
    const size_t n = std::min(bytes - done, plen - off);
    std::memcpy(out + done, p->data() + off, n);
    done += n;
    off = 0;
  }
  return true;
}

double ComputeFor(const std::vector<std::pair<const char *, size_t>> &bufs,
                  double seconds) {
  if (seconds <= 0) return 0.0;
  static std::vector<char> scratch(1u << 20, 7);
  std::vector<std::pair<const char *, size_t>> in;
  for (const auto &b : bufs) {
    if (b.second >= 3) in.push_back(b);
  }
  if (in.empty()) in.emplace_back(scratch.data(), scratch.size());
  constexpr size_t kSlice = 256u << 10;
  const auto end =
      std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
  double a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  size_t bi = 0, off = 0;
  do {
    const auto *p = reinterpret_cast<const unsigned char *>(in[bi].first) + off;
    const size_t n = std::min(kSlice, in[bi].second - off);
    for (size_t i = 1; i + 4 < n; i += 4) {
      a0 = a0 * 0.999999 + (0.25 * p[i - 1] + 0.5 * p[i] + 0.25 * p[i + 1]);
      a1 = a1 * 0.999999 + (0.25 * p[i] + 0.5 * p[i + 1] + 0.25 * p[i + 2]);
      a2 = a2 * 0.999999 + (0.25 * p[i + 1] + 0.5 * p[i + 2] + 0.25 * p[i + 3]);
      a3 = a3 * 0.999999 + (0.25 * p[i + 2] + 0.5 * p[i + 3] + 0.25 * p[i + 4]);
    }
    off += n;
    if (off + 3 > in[bi].second) {
      off = 0;
      bi = (bi + 1) % in.size();
    }
  } while (std::chrono::steady_clock::now() < end);
  return a0 + a1 + a2 + a3;
}

}  // namespace dtwf
