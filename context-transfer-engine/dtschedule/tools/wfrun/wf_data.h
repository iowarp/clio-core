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
 * @file wf_data.h
 * Data and compute of dtschedule_wfrun's tasks: the data class of a file
 * (same extension table as tools/wfcommons/dt_datagen.py), deterministic
 * compressible content per file name (or windows of real payload files),
 * and the CPU-bound kernel a task runs for its scaled instance runtime.
 */

#ifndef DTSCHEDULE_WFRUN_WF_DATA_H_
#define DTSCHEDULE_WFRUN_WF_DATA_H_

#include <cstddef>
#include <cstdint>
#include <list>
#include <string>
#include <utility>
#include <vector>

namespace dtwf {

/** Data classes of dt_datagen.py. */
enum class DataClass { kFloatField, kTextSeq, kTextTable, kMixedBinary };

/** Number of data classes. */
constexpr int kNumDataClasses = 4;

/**
 * Data class of a file by its extension (dt_datagen.EXT_TABLE; unknown
 * extensions are mixed_binary).
 * @param name File name or path
 * @return Its class
 */
DataClass ClassifyFile(const std::string &name);

/**
 * Name of a data class as dt_datagen spells it.
 * @param c Class
 * @return e.g. "float_field"
 */
const char *DataClassName(DataClass c);

/**
 * CRC32 of a file's base name: the per-file seed (as dt_datagen.seed_for).
 * @param name File name or path
 * @return Seed
 */
uint32_t SeedOf(const std::string &name);

/**
 * Fill a buffer with a file's generated content. Bytes are a pure function
 * of (name, class, noise, size): an 8 MiB base block per file, later blocks
 * rotated (line-aligned for text) with a small drift for floats, so it
 * compresses like real data and any reader can regenerate it.
 * @param name File name (selects the seed)
 * @param cls Data class
 * @param noise Relative noise amplitude of float fields
 * @param out Destination
 * @param bytes File size
 */
void GenerateFile(const std::string &name, DataClass cls, double noise,
                  char *out, size_t bytes);

/** Real data files to cut file contents from instead of generating them. */
class Payload {
 public:
  /**
   * List the regular files of a directory (sorted by name).
   * @param dir Directory
   * @return true when it holds at least one non-empty file
   */
  bool Open(const std::string &dir);

  /**
   * Fill out with a window of the payload file chosen by the name's seed,
   * at a 4 KiB-aligned seed-derived offset, wrapping when the payload file
   * is smaller than the window.
   * @param name File name
   * @param out Destination
   * @param bytes Window size
   * @return true on success
   */
  bool Window(const std::string &name, char *out, size_t bytes);

 private:
  /**
   * Load a payload file (two most recent are cached).
   * @param idx Index into files_
   * @return The bytes, or nullptr on failure
   */
  const std::vector<char> *Load(size_t idx);

  std::vector<std::string> files_;  ///< Payload file paths
  std::list<std::pair<size_t, std::vector<char>>> cache_;  ///< Loaded files
};

/**
 * CPU-bound kernel: 3-point stencil with fused multiply-adds sweeping the
 * given buffers (wrapping around), until seconds have elapsed.
 * @param bufs Input buffers (an internal scratch buffer when empty)
 * @param seconds Wall time to compute for (<= 0: return immediately)
 * @return Accumulated value (keeps the work observable)
 */
double ComputeFor(const std::vector<std::pair<const char *, size_t>> &bufs,
                  double seconds);

}  // namespace dtwf

#endif  // DTSCHEDULE_WFRUN_WF_DATA_H_
