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

#include <clio_cte/filesystem/fs_block_log.h>

#include <utility>

namespace clio::cte::filesystem {

namespace {

/** Frame kinds are checked against these; anything else is not a frame. */
bool KnownBlockFrameKind(clio::run::u32 kind) {
  return kind == static_cast<clio::run::u32>(BlockFrameKind::kImage) ||
         kind == static_cast<clio::run::u32>(BlockFrameKind::kDeltas);
}

/**
 * Re-encode a block in the durable form and decode it back, so replayed
 * state matches what an image would hold: pending rows dropped, leaving rows
 * live, unsealed.
 * @param b the block, replaced in place
 */
void NormalizeDurable(InodeTableBlock *b) {
  const std::string img = EncodeTableBlock(*b, true);
  InodeTableBlock nb;
  if (DecodeTableBlock(img.data(), img.size(), &nb)) *b = std::move(nb);
}

}  // namespace

BlockBlobWrite MakeImageWrite(const InodeTableBlock &b) {
  BlockBlobWrite w;
  AppendBlockFrame(BlockFrameKind::kImage, EncodeTableBlock(b, true),
                   &w.bytes_);
  return w;
}

BlockBlobWrite MakeDeltaWrite(const std::vector<TableDelta> &deltas) {
  BlockBlobWrite w;
  w.append_ = true;
  AppendBlockFrame(BlockFrameKind::kDeltas, EncodeTableDeltas(deltas),
                   &w.bytes_);
  return w;
}

bool HasBlockFrameMagic(const char *data, size_t len) {
  FsDec d(data, len);
  clio::run::u32 magic = 0;
  return d.U32(&magic) && magic == kBlockFrameMagic;
}

clio::run::u32 BlockFrameChecksum(const char *data, size_t len) {
  clio::run::u32 h = 2166136261u;
  for (size_t i = 0; i < len; ++i) {
    h ^= static_cast<unsigned char>(data[i]);
    h *= 16777619u;
  }
  return h;
}

void AppendBlockFrame(BlockFrameKind kind, const std::string &payload,
                      std::string *out) {
  FsEnc e(out);
  e.U32(kBlockFrameMagic);
  e.U32(static_cast<clio::run::u32>(kind));
  e.U32(static_cast<clio::run::u32>(payload.size()));
  e.U32(BlockFrameChecksum(payload.data(), payload.size()));
  out->append(payload);
}

bool ParseBlockFrame(const char *data, size_t len, size_t offset,
                     BlockFrame *frame) {
  if (offset > len || len - offset < kBlockFrameHeaderBytes) return false;
  FsDec d(data + offset, len - offset);
  clio::run::u32 magic = 0, kind = 0, payload_len = 0, sum = 0;
  if (!d.U32(&magic) || !d.U32(&kind) || !d.U32(&payload_len) ||
      !d.U32(&sum)) {
    return false;
  }
  if (magic != kBlockFrameMagic || !KnownBlockFrameKind(kind)) return false;
  const size_t room = len - offset - kBlockFrameHeaderBytes;
  if (payload_len > room) return false;  // truncated by a torn append
  const char *payload = data + offset + kBlockFrameHeaderBytes;
  if (BlockFrameChecksum(payload, payload_len) != sum) return false;
  frame->kind_ = static_cast<BlockFrameKind>(kind);
  frame->offset_ = offset;
  frame->payload_len_ = payload_len;
  frame->total_len_ = kBlockFrameHeaderBytes + payload_len;
  return true;
}

size_t ForEachBlockFrame(
    const char *data, size_t len,
    const std::function<bool(const BlockFrame &, const char *)> &fn) {
  size_t off = 0;
  BlockFrame f;
  while (ParseBlockFrame(data, len, off, &f)) {
    if (!fn(f, data + off + kBlockFrameHeaderBytes)) break;
    off += f.total_len_;
  }
  return off;
}

bool DecodeBlockLog(const char *data, size_t len, InodeTableBlock *out,
                    BlockLogInfo *info) {
  BlockLogInfo scan;
  if (!HasBlockFrameMagic(data, len)) {
    // A bare image of the earlier format; any padding past it is ignored.
    scan.legacy_ = true;
    scan.frames_ = 1;
    scan.valid_bytes_ = scan.base_bytes_ = len;
    const bool ok = DecodeTableBlock(data, len, out);
    if (info != nullptr) *info = scan;
    return ok;
  }
  bool have_base = false;
  bool applied_deltas = false;
  scan.valid_bytes_ = ForEachBlockFrame(
      data, len, [&](const BlockFrame &f, const char *p) {
        const std::string payload(p, f.payload_len_);
        if (f.kind_ == BlockFrameKind::kImage) {
          InodeTableBlock img;
          if (!DecodeTableBlock(payload.data(), payload.size(), &img)) {
            return false;
          }
          *out = std::move(img);
          have_base = true;
          scan.base_bytes_ = f.offset_ + f.total_len_;
        } else {
          if (!have_base) return false;  // deltas with no image to build on
          std::vector<TableDelta> ds;
          if (!DecodeTableDeltas(payload, &ds)) return false;
          for (const TableDelta &x : ds) {
            if (ReplayTableDelta(out, x) != ReplayStep::kMismatch) continue;
            HLOG(kWarning, "filesystem: block {}/{} log: a delta based on "
                 "version {} does not follow version {}; the log is unusable "
                 "past this point", out->dir_, out->index_, x.base_version_,
                 out->version_);
            return false;
          }
          applied_deltas = true;
        }
        ++scan.frames_;
        return true;
      });
  if (!have_base) return false;
  scan.bad_tail_ = scan.valid_bytes_ < len;
  if (applied_deltas) NormalizeDurable(out);
  if (info != nullptr) *info = scan;
  return true;
}

}  // namespace clio::cte::filesystem
