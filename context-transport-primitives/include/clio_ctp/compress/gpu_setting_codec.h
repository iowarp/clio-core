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

#ifndef CTP_COMPRESS_GPU_SETTING_CODEC_H_
#define CTP_COMPRESS_GPU_SETTING_CODEC_H_

#include <cstddef>
#include <memory>
#include <string>
#include "compress.h"

namespace ctp {

/**
 * @brief Number of fixed GPU lossless codec settings.
 *
 * This is the count of the frozen settings table used by the NeuroPress v2 model.
 */
constexpr int kGpuSettingCount = 45;

/**
 * @brief Get the canonical specification string for a GPU setting index.
 *
 * The specification is in the format recognized by gpu_codecs.cuh's ParseSpec,
 * with tokens sorted alphabetically by key.
 *
 * @param index Setting index (0 to 44)
 * @return Canonical spec string (e.g., "ans", "lz4 shuffle=byte type=int"),
 *         or nullptr if index is out of range
 */
const char* GpuSettingSpec(int index);

/**
 * @brief Look up the setting index from a specification string.
 *
 * The input string can have tokens in any order; they are canonicalized by
 * sorting before lookup. Returns -1 if the spec is unknown or malformed.
 *
 * @param spec Specification string; tokens can be in any order
 * @return Setting index (0 to 44), or -1 if unknown
 */
int GpuSettingIndex(const std::string& spec);

/**
 * @brief Check if a GPU setting is available in this build.
 *
 * Returns false when the build lacks a required library (e.g., ndzip, gpulz,
 * fpcompress, or nvcomp), or when the setting has not been registered. A
 * setting needing only CUDA (like "store") is available when CTP_ENABLE_CUDA
 * is on.
 *
 * @param index Setting index (0 to 44)
 * @return true if the setting can be used in this build, false otherwise
 */
bool GpuSettingAvailable(int index);

/** Number of per-thread scratch buffers GpuSettingScratch() keeps. */
constexpr int kGpuScratchSlots = 4;

/**
 * @brief A device buffer of at least `bytes`, owned by the calling thread and
 * reused across calls.
 *
 * For measuring codecs without host copies: compress into one slot,
 * decompress into another, and copy to the host only what must be kept. A
 * slot only grows (its contents are lost when it does) and lives for the
 * thread's lifetime. Slots are independent; the caller owns their meaning.
 *
 * @param slot  buffer index, 0 to kGpuScratchSlots - 1
 * @param bytes minimum size in bytes
 * @return device pointer, or nullptr on error or in a build without CUDA
 */
void* GpuSettingScratch(int slot, size_t bytes);

/**
 * @brief Copy bytes from device (or host) memory to host memory.
 * @param dst   host destination
 * @param src   device or host source
 * @param bytes byte count
 * @return true on success
 */
bool GpuSettingCopyToHost(void* dst, const void* src, size_t bytes);

/**
 * @brief One GPU codec selected by a frozen setting index.
 *
 * Compresses and decompresses with any of the 45 fixed lossless GPU codec
 * settings. The setting index is what the NeuroPress v2 model outputs and what
 * each blob will record. Compress/Decompress follow the Compressor contract
 * and accept HOST or DEVICE pointers for input and output. Timing includes
 * preprocessing (shuffle/unshuffle) via CodecKernelTimer.
 */
class GpuSettingCodec : public Compressor {
 public:
  /**
   * @brief Create a GPU settings codec for the given index.
   *
   * Cheap: the codec objects (nvcomp managers, streams, scratch) are built on
   * first use and cached per thread. An invalid or unavailable index makes
   * every call return false; nothing is thrown.
   *
   * @param index Setting index (0 to 44)
   */
  explicit GpuSettingCodec(int index);

  ~GpuSettingCodec() override;

  /**
   * @brief Compress data with the codec's selected setting.
   *
   * Host or device pointers for either buffer; host data is staged to the
   * device untimed. LastCodecKernelMs() afterwards holds the GPU time of the
   * shuffle (if any) plus the codec, as the v2 labels were measured.
   *
   * @param output Compressed output buffer (host or device)
   * @param output_size [in] output buffer capacity; [out] compressed bytes
   * @param input Raw input (host or device)
   * @param input_size Raw byte count
   * @return true if successful, false on any error (CUDA/codec failure)
   */
  bool Compress(void* output, size_t& output_size, void* input,
                size_t input_size) override;

  /**
   * @brief Decompress data with the codec's selected setting.
   *
   * Host or device pointers for either buffer. LastCodecKernelMs() afterwards
   * holds the GPU time of the codec plus the un-shuffle.
   *
   * @param output Decompressed output buffer (host or device)
   * @param output_size the exact decompressed size (the original size)
   * @param input Compressed input (host or device)
   * @param input_size Compressed byte count
   * @return true if successful, false on any error (CUDA/codec failure)
   */
  bool Decompress(void* output, size_t& output_size, void* input,
                  size_t input_size) override;

  /**
   * @brief Worst-case compressed size for n input bytes.
   *
   * Reported by the codec's Bound() method, accounting for any
   * preprocessing (e.g., shuffle).
   *
   * @param input_size Input byte count
   * @return Maximum possible compressed size, or 0 if unknown
   */
  size_t MaxCompressedSize(size_t input_size) override;

 private:
  int index_;  ///< setting index, 0..kGpuSettingCount-1
};

}  // namespace ctp

#endif  // CTP_COMPRESS_GPU_SETTING_CODEC_H_
