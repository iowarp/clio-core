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
#include <clio_cte/core/record_log.h>

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include <filesystem>

namespace clio::cte::core {

namespace {
constexpr clio::run::u32 kRecMagic = 0xC1F5E6A1u;
constexpr size_t kRecHeader = 4 + 4 + 1;  // magic, len, type
constexpr size_t kRecTrailer = 4;         // checksum
constexpr clio::run::u32 kMaxRecLen = 64u << 20;

/** FNV-1a over a byte range, continuing from `h`. */
clio::run::u32 Fnv1a(const char *p, size_t n, clio::run::u32 h = 2166136261u) {
  for (size_t i = 0; i < n; ++i) {
    h ^= static_cast<unsigned char>(p[i]);
    h *= 16777619u;
  }
  return h;
}

// Thin POSIX/CRT file wrappers. On Windows the log must be opened binary
// (text mode would rewrite the records' bytes) and not inherited.
#ifdef _WIN32
constexpr int kOsOpenFlags = _O_BINARY | _O_NOINHERIT;
#else
constexpr int kOsOpenFlags = O_CLOEXEC;
#endif

/**
 * Open a file descriptor.
 * @param path file to open
 * @param flags POSIX open flags; the platform's binary/cloexec bits are added
 * @return the descriptor, or -1
 */
int FileOpen(const std::string &path, int flags) {
#ifdef _WIN32
  return ::_open(path.c_str(), flags | kOsOpenFlags, _S_IREAD | _S_IWRITE);
#else
  return ::open(path.c_str(), flags | kOsOpenFlags, 0600);
#endif
}

/** Close a descriptor. @param fd descriptor to close */
void FileClose(int fd) {
#ifdef _WIN32
  ::_close(fd);
#else
  ::close(fd);
#endif
}

/**
 * Write once. @param fd descriptor @param p bytes @param n byte count
 * @return bytes written, or -1
 */
long long FileWrite(int fd, const char *p, size_t n) {
#ifdef _WIN32
  return ::_write(fd, p, static_cast<unsigned>(n));
#else
  return ::write(fd, p, n);
#endif
}

/**
 * Read once. @param fd descriptor @param p buffer @param n buffer size
 * @return bytes read, 0 at EOF, or -1
 */
long long FileRead(int fd, char *p, size_t n) {
#ifdef _WIN32
  return ::_read(fd, p, static_cast<unsigned>(n));
#else
  return ::read(fd, p, n);
#endif
}

/** Seek to the start. @param fd descriptor @return true on success */
bool FileRewind(int fd) {
#ifdef _WIN32
  return ::_lseeki64(fd, 0, SEEK_SET) >= 0;
#else
  return ::lseek(fd, 0, SEEK_SET) >= 0;
#endif
}

/** Truncate. @param fd descriptor @param size new length @return 0 or -1 */
int FileTruncate(int fd, size_t size) {
#ifdef _WIN32
  return ::_chsize_s(fd, static_cast<long long>(size)) == 0 ? 0 : -1;
#else
  return ::ftruncate(fd, static_cast<off_t>(size));
#endif
}

/** Flush to media. @param fd descriptor @return 0 or -1 */
int FileSync(int fd) {
#ifdef _WIN32
  return ::_commit(fd);
#else
  return ::fsync(fd);
#endif
}

/**
 * Make a create or rename in a file's directory durable. A no-op on Windows,
 * where a directory cannot be opened through the CRT and NTFS journals the
 * rename itself.
 * @param path file whose parent directory is synced
 */
void SyncParentDir(const std::string &path) {
#ifdef _WIN32
  (void)path;
#else
  const size_t slash = path.find_last_of('/');
  const std::string dir = slash == std::string::npos ? "." :
                          slash == 0 ? "/" : path.substr(0, slash);
  const int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (dfd < 0) return;
  if (::fsync(dfd) != 0) {
    HLOG(kError, "record log: fsync of directory {} failed: {}", dir,
         std::strerror(errno));
  }
  ::close(dfd);
#endif
}

/** write(2) the whole buffer, retrying short writes and EINTR. */
bool WriteAll(int fd, const char *p, size_t n) {
  while (n > 0) {
    long long w = FileWrite(fd, p, n);
    if (w < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    p += w;
    n -= static_cast<size_t>(w);
  }
  return true;
}

/** Read a whole file into memory (empty on failure). */
std::string Slurp(int fd) {
  std::string out;
  if (!FileRewind(fd)) return out;
  char buf[1 << 16];
  for (;;) {
    long long r = FileRead(fd, buf, sizeof(buf));
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) break;
    out.append(buf, static_cast<size_t>(r));
  }
  return out;
}
}  // namespace

RecordLog::~RecordLog() {
  if (fd_ >= 0) FileClose(fd_);
}

bool RecordLog::Open(const std::string &path) {
  std::lock_guard<std::mutex> g(mu_);
  path_ = path;
  std::error_code ec;
  std::filesystem::path parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent, ec);
  fd_ = FileOpen(path, O_RDWR | O_CREAT | O_APPEND);
  since_compact_ = 0;
  return fd_ >= 0;
}

void RecordLog::Frame(clio::run::u32 type, const std::string &payload,
                      std::string *out) {
  const clio::run::u32 magic = kRecMagic;
  const clio::run::u32 len = static_cast<clio::run::u32>(payload.size());
  const char t = static_cast<char>(type & 0xFF);
  clio::run::u32 sum = Fnv1a(&t, 1);
  sum = Fnv1a(payload.data(), payload.size(), sum);
  out->append(reinterpret_cast<const char *>(&magic), 4);
  out->append(reinterpret_cast<const char *>(&len), 4);
  out->push_back(t);
  out->append(payload);
  out->append(reinterpret_cast<const char *>(&sum), 4);
}

size_t RecordLog::Replay(
    const std::function<void(clio::run::u32, const std::string &)> &fn) {
  std::lock_guard<std::mutex> g(mu_);
  if (fd_ < 0) return 0;
  const std::string all = Slurp(fd_);
  size_t off = 0;
  size_t n = 0;
  while (off + kRecHeader + kRecTrailer <= all.size()) {
    clio::run::u32 magic = 0, len = 0, sum = 0;
    std::memcpy(&magic, all.data() + off, 4);
    std::memcpy(&len, all.data() + off + 4, 4);
    if (magic != kRecMagic || len > kMaxRecLen ||
        off + kRecHeader + len + kRecTrailer > all.size()) {
      break;
    }
    const char *body = all.data() + off + 8;  // type byte + payload
    std::memcpy(&sum, body + 1 + len, 4);
    if (Fnv1a(body, 1 + len) != sum) break;
    fn(static_cast<clio::run::u32>(static_cast<unsigned char>(body[0])),
       std::string(body + 1, len));
    off += kRecHeader + len + kRecTrailer;
    ++n;
  }
  if (off < all.size()) {
    // A torn or corrupt tail: everything after the last intact record is
    // unacknowledged work from a crash. Cut it so appends stay parseable.
    if (FileTruncate(fd_, off) != 0) {
      HLOG(kError, "record log: truncating torn tail of {} failed: {}",
           path_, std::strerror(errno));
    }
  }
  return n;
}

void RecordLog::Append(clio::run::u32 type, const std::string &payload) {
  std::string rec;
  rec.reserve(payload.size() + kRecHeader + kRecTrailer);
  Frame(type, payload, &rec);
  std::lock_guard<std::mutex> g(mu_);
  if (fd_ < 0) return;
  if (!WriteAll(fd_, rec.data(), rec.size())) {
    HLOG(kError, "record log: append to {} failed: {}", path_,
         std::strerror(errno));
    return;
  }
  since_compact_ += rec.size();
  unsynced_ = true;
}

bool RecordLog::Sync() {
  std::lock_guard<std::mutex> g(mu_);
  if (fd_ < 0) return true;
  if (!unsynced_) return true;
  if (FileSync(fd_) != 0) {
    HLOG(kError, "record log: fsync of {} failed: {}", path_,
         std::strerror(errno));
    return false;
  }
  unsynced_ = false;
  return true;
}

bool RecordLog::Unsynced() {
  std::lock_guard<std::mutex> g(mu_);
  return fd_ >= 0 && unsynced_;
}

bool RecordLog::Rewrite(
    const std::vector<std::pair<clio::run::u32, std::string>> &records) {
  std::string buf;
  for (const auto &r : records) Frame(r.first, r.second, &buf);
  std::lock_guard<std::mutex> g(mu_);
  if (path_.empty()) return false;
  const std::string tmp = path_ + ".compact";
  int tfd = FileOpen(tmp, O_WRONLY | O_CREAT | O_TRUNC);
  if (tfd < 0) return false;
  bool ok = WriteAll(tfd, buf.data(), buf.size()) && FileSync(tfd) == 0;
  FileClose(tfd);
#ifdef _WIN32
  // Windows cannot replace a file that is open; drop ours first. On a failed
  // replace the reopen below still reattaches to the untouched old log.
  if (ok && fd_ >= 0) {
    FileClose(fd_);
    fd_ = -1;
  }
#endif
  std::error_code ec;
  // std::filesystem::rename replaces an existing target on every platform
  // (::rename does not on Windows).
  if (ok) std::filesystem::rename(tmp, path_, ec);
  if (!ok || ec) {
    std::filesystem::remove(tmp, ec);
#ifdef _WIN32
    if (fd_ < 0) fd_ = FileOpen(path_, O_RDWR | O_APPEND);
#endif
    return false;
  }
  // The rename is durable only once the directory is: until then a power
  // loss brings back the old file, and records appended to the new one
  // (fsynced or not) are gone with it.
  SyncParentDir(path_);
  int nfd = FileOpen(path_, O_RDWR | O_APPEND);
  if (nfd < 0) return false;
  if (fd_ >= 0) FileClose(fd_);
  fd_ = nfd;
  since_compact_ = 0;
  unsynced_ = false;  // the new file was fsynced before the rename
  return true;
}

}  // namespace clio::cte::core
