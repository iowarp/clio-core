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

#include "clio_runtime/safe_bdev/safe_bdev_journal.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>

namespace clio::run::safe_bdev {

namespace {

/**
 * pwrite all of buf.
 * @param fd file
 * @param buf bytes
 * @param len byte count
 * @param off file offset
 * @return true when every byte was written
 */
bool PwriteAll(int fd, const void *buf, size_t len, uint64_t off) {
  const char *p = static_cast<const char *>(buf);
  while (len > 0) {
    const ssize_t n = ::pwrite(fd, p, len, static_cast<off_t>(off));
    if (n <= 0) return false;
    p += n;
    len -= static_cast<size_t>(n);
    off += static_cast<uint64_t>(n);
  }
  return true;
}

/**
 * pread all of buf.
 * @param fd file
 * @param buf destination
 * @param len byte count
 * @param off file offset
 * @return true when every byte was read
 */
bool PreadAll(int fd, void *buf, size_t len, uint64_t off) {
  char *p = static_cast<char *>(buf);
  while (len > 0) {
    const ssize_t n = ::pread(fd, p, len, static_cast<off_t>(off));
    if (n <= 0) return false;
    p += n;
    len -= static_cast<size_t>(n);
    off += static_cast<uint64_t>(n);
  }
  return true;
}

}  // namespace

uint64_t StripeJournal::Checksum(const RecordHeader &h, const uint8_t *data,
                                 uint64_t len) {
  uint64_t x = 1469598103934665603ULL;  // FNV-1a 64
  auto mix = [&x](const uint8_t *p, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i) {
      x ^= p[i];
      x *= 1099511628211ULL;
    }
  };
  mix(reinterpret_cast<const uint8_t *>(&h.magic), sizeof(h.magic));
  mix(reinterpret_cast<const uint8_t *>(&h.key), sizeof(h.key));
  mix(reinterpret_cast<const uint8_t *>(&h.slot), sizeof(h.slot));
  mix(reinterpret_cast<const uint8_t *>(&h.col), sizeof(h.col));
  mix(data, len);
  return x;
}

bool StripeJournal::Open(const std::string &path, uint64_t chunk_len,
                         const std::set<uint64_t> &live_keys) {
  Close();
  if (path.empty()) return true;
  std::lock_guard<std::mutex> g(mu_);
  fd_ = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
  if (fd_ < 0) return false;
  path_ = path;
  chunk_len_ = chunk_len;
  Load(live_keys);
  // Nothing worth keeping: start the file over.
  if (recs_.empty() && end_ != 0) {
    if (::ftruncate(fd_, 0) == 0) end_ = 0;
  }
  return true;
}

void StripeJournal::Close() {
  std::lock_guard<std::mutex> g(mu_);
  if (fd_ >= 0) ::close(fd_);
  fd_ = -1;
  recs_.clear();
  loaded_keys_.clear();
  end_ = 0;
}

void StripeJournal::Load(const std::set<uint64_t> &live_keys) {
  const uint64_t rec_len = sizeof(RecordHeader) + chunk_len_;
  std::vector<uint8_t> payload(chunk_len_);
  uint64_t off = 0;
  scan_ = ScanStats();
  const off_t fsize = ::lseek(fd_, 0, SEEK_END);
  scan_.file_size = fsize > 0 ? static_cast<uint64_t>(fsize) : 0;
  scan_.why = "end of file";
  while (true) {
    RecordHeader h{};
    if (!PreadAll(fd_, &h, sizeof(h), off)) break;
    if (h.magic != kMagic) {
      scan_.why = "bad magic";
      break;
    }
    if (!PreadAll(fd_, payload.data(), chunk_len_, off + sizeof(h))) {
      scan_.why = "short payload";
      break;
    }
    if (Checksum(h, payload.data(), chunk_len_) != h.sum) {  // torn tail
      scan_.why = "bad checksum";
      break;
    }
    ++scan_.records;
    if (h.key > scan_.max_key) scan_.max_key = h.key;
    if (live_keys.count(h.key) != 0) {
      ++scan_.live;
      loaded_keys_.insert(h.key);
      Loc &cur = recs_[h.slot][h.col];  // keys start at 1: {0,0} is unset
      if (h.key >= cur.key) cur = Loc{h.key, off};
    }
    off += rec_len;
  }
  end_ = off;
  scan_.stopped_at = off;
}

bool StripeJournal::WriteAt(uint64_t off, uint64_t key, uint64_t slot,
                            uint32_t col, const uint8_t *data) {
  RecordHeader h{};
  h.magic = kMagic;
  h.key = key;
  h.slot = slot;
  h.col = col;
  h.sum = Checksum(h, data, chunk_len_);
  // Payload first, then the header: a torn record fails its checksum.
  return PwriteAll(fd_, data, chunk_len_, off + sizeof(h)) &&
         PwriteAll(fd_, &h, sizeof(h), off);
}

bool StripeJournal::Append(uint64_t key, uint64_t slot, uint32_t col,
                           const uint8_t *data) {
  std::lock_guard<std::mutex> g(mu_);
  if (fd_ < 0) return false;
  if (end_ >= kCompactBytes && !CompactLocked()) return false;
  const uint64_t off = end_;
  if (!WriteAt(off, key, slot, col, data)) return false;
  end_ = off + sizeof(RecordHeader) + chunk_len_;
  recs_[slot][col] = Loc{key, off};
  return true;
}

bool StripeJournal::CompactLocked() {
  const uint64_t rec_len = sizeof(RecordHeader) + chunk_len_;
  size_t live = 0;
  for (const auto &s : recs_) live += s.second.size();
  if (live * rec_len * 2 > end_) return true;  // mostly live: keep growing
  const std::string tmp = path_ + ".tmp";
  const int nfd = ::open(tmp.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (nfd < 0) return false;
  std::vector<uint8_t> payload(chunk_len_);
  uint64_t noff = 0;
  bool ok = true;
  std::map<uint64_t, std::map<uint32_t, Loc>> moved;
  for (const auto &s : recs_) {
    for (const auto &c : s.second) {
      RecordHeader h{};
      ok = PreadAll(fd_, &h, sizeof(h), c.second.off) &&
           PreadAll(fd_, payload.data(), chunk_len_,
                    c.second.off + sizeof(h)) &&
           PwriteAll(nfd, &h, sizeof(h), noff) &&
           PwriteAll(nfd, payload.data(), chunk_len_, noff + sizeof(h));
      if (!ok) break;
      moved[s.first][c.first] = Loc{c.second.key, noff};
      noff += rec_len;
    }
    if (!ok) break;
  }
  // The new file must be on disk before it replaces the old one.
  if (!ok || ::fdatasync(nfd) != 0 ||
      std::rename(tmp.c_str(), path_.c_str()) != 0) {
    ::close(nfd);
    ::unlink(tmp.c_str());
    return false;
  }
  ::close(fd_);
  fd_ = nfd;
  end_ = noff;
  recs_ = std::move(moved);
  return true;
}

bool StripeJournal::Sync() {
  std::lock_guard<std::mutex> g(mu_);
  return fd_ < 0 || ::fdatasync(fd_) == 0;
}

bool StripeJournal::Read(uint64_t slot, uint32_t col,
                         std::vector<uint8_t> &out) {
  std::lock_guard<std::mutex> g(mu_);
  if (fd_ < 0) return false;
  auto s = recs_.find(slot);
  if (s == recs_.end()) return false;
  auto c = s->second.find(col);
  if (c == s->second.end()) return false;
  RecordHeader h{};
  out.resize(chunk_len_);
  if (!PreadAll(fd_, &h, sizeof(h), c->second.off) ||
      !PreadAll(fd_, out.data(), chunk_len_, c->second.off + sizeof(h))) {
    return false;
  }
  return h.magic == kMagic && h.slot == slot && h.col == col &&
         h.key == c->second.key && Checksum(h, out.data(), chunk_len_) == h.sum;
}

void StripeJournal::DropSlot(uint64_t slot) {
  std::lock_guard<std::mutex> g(mu_);
  recs_.erase(slot);
}

void StripeJournal::DropKey(uint64_t slot, uint64_t key) {
  std::lock_guard<std::mutex> g(mu_);
  auto s = recs_.find(slot);
  if (s == recs_.end()) return;
  for (auto c = s->second.begin(); c != s->second.end();) {
    if (c->second.key == key) {
      c = s->second.erase(c);
    } else {
      ++c;
    }
  }
  if (s->second.empty()) recs_.erase(s);
}

bool StripeJournal::HasKey(uint64_t key) {
  std::lock_guard<std::mutex> g(mu_);
  return loaded_keys_.count(key) != 0;
}

bool StripeJournal::HasSlot(uint64_t slot) {
  std::lock_guard<std::mutex> g(mu_);
  return recs_.count(slot) != 0;
}

size_t StripeJournal::NumSlots() {
  std::lock_guard<std::mutex> g(mu_);
  return recs_.size();
}

}  // namespace clio::run::safe_bdev
