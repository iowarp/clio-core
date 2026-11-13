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

#include "clio_ctp/lightbeam/socket_frame.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>

namespace ctp::lbm::sock {

namespace {

/** Read-ahead buffer size: many small frames per recv() call. */
constexpr size_t kReadAheadBytes = 256 * 1024;
/** A bulk with at least this much left is read straight into its buffer. */
constexpr size_t kDirectReadMin = 64 * 1024;

/** Store a 32-bit value big-endian at p. */
void Put32(char* p, uint32_t v) {
  uint32_t n = HostToNet32(v);
  std::memcpy(p, &n, sizeof(n));
}

/** Load a big-endian 32-bit value from p. */
uint32_t Get32(const char* p) {
  uint32_t n;
  std::memcpy(&n, p, sizeof(n));
  return NetToHost32(n);
}

}  // namespace

// --- Frame ---

Frame& Frame::operator=(Frame&& other) noexcept {
  if (this != &other) {
    Clear();
    meta = std::move(other.meta);
    bulks = std::move(other.bulks);
    other.bulks.clear();
  }
  return *this;
}

Frame::~Frame() { Clear(); }

void Frame::Clear() {
  for (auto& b : bulks) {
    std::free(b.first);
    b.first = nullptr;
  }
  bulks.clear();
  meta.clear();
}

void EncodeFrameHeader(uint32_t meta_len,
                       const std::vector<uint64_t>& bulk_lens,
                       std::vector<char>* out) {
  const uint32_t n = static_cast<uint32_t>(bulk_lens.size());
  out->assign(kFrameHeaderSize + 8 * static_cast<size_t>(n), 0);
  char* p = out->data();
  Put32(p, kFrameMagic);
  Put32(p + 4, meta_len);
  Put32(p + 8, n);
  Put32(p + 12, 0);
  p += kFrameHeaderSize;
  for (uint32_t i = 0; i < n; ++i, p += 8) {
    Put32(p, static_cast<uint32_t>(bulk_lens[i] >> 32));
    Put32(p + 4, static_cast<uint32_t>(bulk_lens[i] & 0xFFFFFFFFu));
  }
}

// --- FrameReader ---

FrameReader::FrameReader() : rbuf_(kReadAheadBytes) { Reset(); }

int FrameReader::Read(socket_t fd, Frame* out) {
  while (!bad_) {
    if (Parse(out)) return 1;
    if (bad_) break;
    // A large bulk with nothing buffered is read straight into its buffer
    // (no copy); everything else goes through the read-ahead buffer.
    bool direct = stage_ == Stage::kBulk && bulk_idx_ < nbulks_ &&
                  rpos_ == rlen_ && lens_[bulk_idx_] - have_ >= kDirectReadMin;
    int rc = direct ? ReadDirect(fd) : Refill(fd);
    if (rc == EAGAIN) return 0;
    if (rc != 0) return -1;
  }
  return -1;
}

bool FrameReader::Parse(Frame* out) {
  while (true) {
    switch (stage_) {
      case Stage::kHeader:
        have_ += Take(scratch_.data() + have_, kFrameHeaderSize - have_);
        if (have_ < kFrameHeaderSize) return false;
        if (!ParseHeader()) return false;
        break;
      case Stage::kLens:
        have_ += Take(scratch_.data() + have_, scratch_.size() - have_);
        if (have_ < scratch_.size()) return false;
        if (!ParseLens()) return false;
        break;
      case Stage::kMeta:
        have_ += Take(cur_.meta.data() + have_, meta_len_ - have_);
        if (have_ < meta_len_) return false;
        stage_ = Stage::kBulk;
        have_ = 0;
        bulk_idx_ = 0;
        break;
      case Stage::kBulk: {
        if (bulk_idx_ >= nbulks_) {
          *out = std::move(cur_);
          Reset();
          return true;
        }
        auto& b = cur_.bulks[bulk_idx_];
        have_ += Take(b.first + have_, b.second - have_);
        if (have_ < b.second) return false;
        ++bulk_idx_;
        have_ = 0;
        break;
      }
    }
  }
}

bool FrameReader::ParseHeader() {
  const char* p = scratch_.data();
  meta_len_ = Get32(p + 4);
  nbulks_ = Get32(p + 8);
  if (Get32(p) != kFrameMagic || meta_len_ > kFrameMaxMeta ||
      nbulks_ > kFrameMaxBulks) {
    bad_ = true;
    return false;
  }
  cur_.meta.resize(meta_len_);
  have_ = 0;
  if (nbulks_ > 0) {
    scratch_.assign(8 * static_cast<size_t>(nbulks_), 0);
    stage_ = Stage::kLens;
  } else {
    stage_ = Stage::kMeta;
  }
  return true;
}

bool FrameReader::ParseLens() {
  lens_.resize(nbulks_);
  cur_.bulks.assign(nbulks_, {nullptr, 0});
  for (uint32_t i = 0; i < nbulks_; ++i) {
    const char* p = scratch_.data() + 8 * static_cast<size_t>(i);
    lens_[i] = (static_cast<uint64_t>(Get32(p)) << 32) | Get32(p + 4);
    // malloc(0) may return nullptr; keep every bulk pointer non-null.
    char* buf = static_cast<char*>(std::malloc(lens_[i] > 0 ? lens_[i] : 1));
    if (buf == nullptr) {
      bad_ = true;
      return false;
    }
    cur_.bulks[i] = {buf, lens_[i]};
  }
  have_ = 0;
  stage_ = Stage::kMeta;
  return true;
}

int FrameReader::ReadDirect(socket_t fd) {
  auto& b = cur_.bulks[bulk_idx_];
  size_t got = 0;
  int rc = RecvSome(fd, b.first + have_, b.second - have_, &got);
  if (rc == 0) have_ += got;
  return rc;
}

int FrameReader::Refill(socket_t fd) {
  if (rpos_ == rlen_) {
    rpos_ = rlen_ = 0;
  } else if (rpos_ > 0) {
    std::memmove(rbuf_.data(), rbuf_.data() + rpos_, rlen_ - rpos_);
    rlen_ -= rpos_;
    rpos_ = 0;
  }
  size_t got = 0;
  int rc = RecvSome(fd, rbuf_.data() + rlen_, rbuf_.size() - rlen_, &got);
  if (rc == 0) rlen_ += got;
  return rc;
}

size_t FrameReader::Take(char* dst, size_t want) {
  size_t n = std::min(want, rlen_ - rpos_);
  if (n > 0) {
    std::memcpy(dst, rbuf_.data() + rpos_, n);
    rpos_ += n;
  }
  return n;
}

void FrameReader::Reset() {
  stage_ = Stage::kHeader;
  scratch_.assign(kFrameHeaderSize, 0);
  have_ = 0;
  meta_len_ = 0;
  nbulks_ = 0;
  lens_.clear();
  bulk_idx_ = 0;
  cur_.Clear();
}

}  // namespace ctp::lbm::sock
