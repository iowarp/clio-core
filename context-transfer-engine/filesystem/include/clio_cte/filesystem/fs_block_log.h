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
 * The log form of a directory block's CTE blob.
 *
 * A block's blob is a log of framed records, each frame being
 *
 *   [u32 magic = "CLG1"][u32 kind][u32 payload length][u32 checksum][payload]
 *
 * with the little-endian FsEnc scalars, the checksum being FNV-1a over the
 * payload, and the kind one of:
 *   - kImage:  payload = EncodeTableBlock(block, true), a whole block.
 *   - kDeltas: payload = EncodeTableDeltas(batch), one commit's changes.
 *
 * A commit either REPLACES the blob with one image frame (new, split, sealed
 * or failed blocks, and when the log outgrows dir_log_compact_bytes) or
 * APPENDS one delta frame. A read takes the last valid image as the base and
 * applies the delta frames after it in order, stopping at the first frame
 * that is truncated, corrupt, or does not follow the block's version. Bytes
 * past that point are a torn append or an unusable tail and are ignored.
 *
 * Earlier clio-fs wrote a bare EncodeTableBlock image with no frame. Every
 * bare image begins with kTableBlockMagic ("CIT1") or kOldDirBlockMagic
 * ("CDB1"), and the frame magic "CLG1" is neither, so a blob is in log form
 * exactly when it begins with kBlockFrameMagic. Bare images are read as before.
 */
#ifndef CLIO_CTE_FILESYSTEM_FS_BLOCK_LOG_H_
#define CLIO_CTE_FILESYSTEM_FS_BLOCK_LOG_H_

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/filesystem/fs_inode_table.h>

namespace clio::cte::filesystem {

/** Magic opening every frame of a block blob ("CLG1"). */
GLOBAL_CROSS_CONST clio::run::u32 kBlockFrameMagic = 0x31474C43u;
/** Bytes in a frame header: magic, kind, payload length, checksum. */
static constexpr size_t kBlockFrameHeaderBytes = 16;

/** What a frame's payload holds. */
enum class BlockFrameKind : clio::run::u32 {
  kImage = 1,   ///< EncodeTableBlock(block, true)
  kDeltas = 2,  ///< EncodeTableDeltas(batch)
};

/** A frame found by ParseBlockFrame. */
struct BlockFrame {
  BlockFrameKind kind_ = BlockFrameKind::kImage;
  size_t offset_ = 0;       ///< first byte of the frame in the blob
  size_t payload_len_ = 0;  ///< payload bytes (after the header)
  size_t total_len_ = 0;    ///< header plus payload
};

/** What a blob read saw, for DecodeBlockLog's caller. */
struct BlockLogInfo {
  bool legacy_ = false;     ///< a bare image of the earlier format
  size_t frames_ = 0;       ///< frames applied (images and delta batches)
  size_t valid_bytes_ = 0;  ///< bytes through the last frame applied
  size_t base_bytes_ = 0;   ///< bytes through the last image frame applied
  bool bad_tail_ = false;   ///< bytes past valid_bytes_ were not applied

  /** @return bytes of delta frames after the base image */
  size_t DeltaBytes() const { return valid_bytes_ - base_bytes_; }

  /**
   * Whether a home loading this blob should write one fresh image before it
   * appends: the blob is an old bare image, holds more than one frame, or
   * has an unusable tail. A single clean image needs nothing.
   * @return true when the next commit should compact the blob
   */
  bool NeedsImage() const { return legacy_ || frames_ != 1 || bad_tail_; }
};

/** One write a commit makes to a block's blob. */
struct BlockBlobWrite {
  bool append_ = false;            ///< append the frame; else replace the blob
  std::string bytes_;              ///< the frame to write (empty: no write)
  clio::run::u64 expect_end_ = 0;  ///< append: the blob length it must land at
};

/**
 * Frame a whole block as a replacing write.
 * @param b the block, as the home holds it (encoded durable)
 * @return a replace write holding one image frame
 */
BlockBlobWrite MakeImageWrite(const InodeTableBlock &b);

/**
 * Frame one commit's delta batch as an append write.
 * @param deltas the batch, in version order
 * @return an append write holding one deltas frame (expect_end_ unset)
 */
BlockBlobWrite MakeDeltaWrite(const std::vector<TableDelta> &deltas);

/**
 * Whether bytes begin with the frame magic (log form, not a bare image).
 * @param data bytes
 * @param len length
 * @return true if the first four bytes are kBlockFrameMagic
 */
bool HasBlockFrameMagic(const char *data, size_t len);

/**
 * FNV-1a over a byte range (the frame checksum; the same function the
 * record log uses, so a frame is checked the same way).
 * @param data bytes
 * @param len length
 * @return the 32-bit checksum
 */
clio::run::u32 BlockFrameChecksum(const char *data, size_t len);

/**
 * Append one frame to an encoding.
 * @param kind what the payload holds
 * @param payload the payload bytes
 * @param out destination (appended to)
 */
void AppendBlockFrame(BlockFrameKind kind, const std::string &payload,
                      std::string *out);

/**
 * Parse the frame starting at a byte offset: its magic, a known kind, a
 * payload that fits the buffer, and a matching checksum.
 * @param data blob bytes
 * @param len blob length
 * @param offset where the frame starts
 * @param frame receives the frame when true is returned
 * @return false if the frame is truncated, corrupt, or not a frame
 */
bool ParseBlockFrame(const char *data, size_t len, size_t offset,
                     BlockFrame *frame);

/**
 * Visit the valid frames of a blob in order, stopping at the first invalid
 * one or when the visitor declines a frame.
 * @param data blob bytes
 * @param len blob length
 * @param fn called with each valid frame and its payload; return false to
 *        stop before that frame
 * @return bytes through the last frame the visitor accepted
 */
size_t ForEachBlockFrame(
    const char *data, size_t len,
    const std::function<bool(const BlockFrame &, const char *)> &fn);

/**
 * Decode a block blob in either form. A log is read from its last valid
 * image frame, then each later delta frame is replayed onto it (ReplayTableDelta);
 * replay ends at the first frame that is invalid or does not follow the block,
 * which is logged. A bare image is decoded as before. Replayed state is
 * normalized to the durable form (pending rows dropped, leaving rows live,
 * unsealed), as the directory journal's replay does.
 * @param data blob bytes (trailing bytes are ignored)
 * @param len blob length
 * @param out receives the block
 * @param info if non-null, receives what was read
 * @return false if there is no decodable base image
 */
bool DecodeBlockLog(const char *data, size_t len, InodeTableBlock *out,
                    BlockLogInfo *info);

}  // namespace clio::cte::filesystem

#endif  // CLIO_CTE_FILESYSTEM_FS_BLOCK_LOG_H_
