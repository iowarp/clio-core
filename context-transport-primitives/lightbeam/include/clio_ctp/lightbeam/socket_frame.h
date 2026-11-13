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

#pragma once
// Framing for SocketTransport. A frame carries one task: its serialized
// metadata and its bulk payloads. The header states every length up front, so
// a receiver can reassemble a frame from whatever bytes arrive, without
// knowing the task type, and hands out only complete frames:
//
//   [u32 magic][u32 meta_len][u32 nbulks][u32 reserved]   16 B, big-endian
//   [u64 bulk_len] x nbulks                               big-endian
//   [meta bytes][bulk 0 bytes][bulk 1 bytes]...
//
// A connection that breaks mid-frame loses that frame and nothing else.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "clio_ctp/lightbeam/posix_socket.h"

namespace ctp::lbm::sock {

constexpr uint32_t kFrameMagic = 0x4C42534Bu;   ///< "LBSK"
constexpr size_t kFrameHeaderSize = 16;          ///< fixed part of a header
constexpr uint32_t kFrameMaxBulks = 4096;        ///< sanity cap per frame
constexpr uint32_t kFrameMaxMeta = 1u << 30;     ///< sanity cap (1 GiB)

/** One complete frame. Bulk buffers are malloc'd and owned by the frame until
 *  taken (set the pointer to nullptr after taking it). */
struct Frame {
  std::vector<char> meta;                          ///< serialized metadata
  std::vector<std::pair<char*, uint64_t>> bulks;   ///< payloads, in order

  Frame() = default;
  Frame(Frame&& other) noexcept = default;
  Frame& operator=(Frame&& other) noexcept;
  Frame(const Frame&) = delete;
  Frame& operator=(const Frame&) = delete;
  ~Frame();

  /** Free every bulk buffer still owned by the frame and clear it. */
  void Clear();
};

/** Build a frame header for the given metadata length and bulk lengths.
 *  @param meta_len length of the serialized metadata
 *  @param bulk_lens length of each bulk, in send order
 *  @param out receives the header bytes (replaced) */
CTP_DLL void EncodeFrameHeader(uint32_t meta_len,
                               const std::vector<uint64_t>& bulk_lens,
                               std::vector<char>* out);

/** Reassembles frames from one non-blocking socket. Not thread-safe: one
 *  reader per connection, driven by one thread at a time. */
class FrameReader {
 public:
  FrameReader();

  /** Read what the socket has and complete at most one frame. Never blocks.
   *  @param fd the connection's non-blocking socket
   *  @param out receives the frame when one completes
   *  @return 1 if *out holds a complete frame, 0 if more data is needed,
   *          -1 if the connection closed or sent malformed data */
  CTP_DLL int Read(socket_t fd, Frame* out);

  /** True if bytes already read are waiting to be parsed (call Read again
   *  before blocking on the socket). */
  bool HasBuffered() const { return rpos_ < rlen_; }

 private:
  enum class Stage { kHeader, kLens, kMeta, kBulk };

  bool Parse(Frame* out);
  bool ParseHeader();
  bool ParseLens();
  int ReadDirect(socket_t fd);
  int Refill(socket_t fd);
  size_t Take(char* dst, size_t want);
  void Reset();

  std::vector<char> rbuf_;     ///< read-ahead buffer
  size_t rpos_ = 0;            ///< next unparsed byte in rbuf_
  size_t rlen_ = 0;            ///< bytes valid in rbuf_

  Stage stage_ = Stage::kHeader;
  std::vector<char> scratch_;  ///< header / lengths being assembled
  size_t have_ = 0;            ///< bytes of the current stage received
  uint32_t meta_len_ = 0;
  uint32_t nbulks_ = 0;
  std::vector<uint64_t> lens_;
  size_t bulk_idx_ = 0;
  Frame cur_;
  bool bad_ = false;           ///< malformed stream seen
};

}  // namespace ctp::lbm::sock
