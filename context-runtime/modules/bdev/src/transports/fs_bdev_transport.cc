/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <clio_runtime/bdev/transports/fs_bdev_transport.h>
#include <clio_ctp/introspect/system_info.h>
#include <clio_ctp/io/io_error.h>
#include <clio_ctp/util/config_parse.h>
#include <clio_ctp/util/msan.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/manager.h>
#include <clio_runtime/worker.h>
#include <clio_runtime/work_orchestrator.h>
#include <fcntl.h>
#ifdef __linux__
#include <sys/vfs.h>  // statfs (IsNetworkFs)
#endif
#ifndef _WIN32
#include <unistd.h>  // close (ReserveFileSpace)
#else
#include <io.h>  // _open, _commit, _close (SyncFileData)
#endif

namespace clio::run::bdev {

namespace {

// Open `file_path` through the best AsyncIO backend that actually works at
// runtime. AsyncIoFactory's kDefault selects the preferred backend at *compile*
// time (NIXL > io_uring > libaio > POSIX), but a backend can be compiled in yet
// be unavailable at runtime: under a container's default seccomp profile
// io_uring_queue_init() fails, returning a negative errno as its return value
// while leaving the global errno untouched, so IoUringAsyncIO::Open() reports
// failure (errno=0) even though the file itself opened fine. When the preferred
// backend cannot open the file, transparently fall back to POSIX AIO, which
// works in restricted environments (containers/CI). Returns an opened AsyncIO,
// or nullptr only if even POSIX AIO cannot open the file (a genuine error).
/**
 * True when `file_path` (or, before it exists, its directory) lives on a
 * FUSE, Lustre, NFS or other network filesystem, where kernel AIO does not
 * submit asynchronously.
 */
bool IsNetworkFs(const std::string &file_path) {
#ifndef __linux__
  (void)file_path;
  return false;
#else
  struct statfs sf;
  if (statfs(file_path.c_str(), &sf) != 0) {
    const auto slash = file_path.find_last_of('/');
    const std::string dir =
        slash == std::string::npos ? "." : file_path.substr(0, slash);
    if (statfs(dir.empty() ? "/" : dir.c_str(), &sf) != 0) return false;
  }
  switch (static_cast<unsigned long>(sf.f_type)) {
    case 0x65735546UL:  // FUSE (dfuse)
    case 0x0BD00BD0UL:  // Lustre
    case 0x6969UL:      // NFS
    case 0xFE534D42UL:  // SMB2
    case 0xFF534D42UL:  // CIFS
    case 0x47504653UL:  // GPFS
      return true;
    default:
      return false;
  }
#endif
}

std::unique_ptr<ctp::AsyncIO> OpenBackingFile(clio::run::u32 io_depth,
                                              const std::string &file_path,
                                              bool create = false) {
  // Only the bdev's Init may create the file. Every later open (worker
  // contexts, growth) reopens the SAME file; with O_CREAT a deleted backing
  // file was silently recreated empty and later writes and reads went to it
  // (#1210).
  const int flags = O_RDWR | (create ? O_CREAT : 0);
  // libaio's io_submit() runs a buffered (or FUSE/network) write
  // synchronously inside the syscall, so on dfuse, Lustre or NFS every write
  // blocked a runtime worker for the full network round trip (IO500 at 16
  // nodes: workers STALLED for seconds, metadata commits took 75 s). POSIX
  // AIO performs the I/O on helper threads, keeping workers free.
  // CLIO_BDEV_AIO=posix|default overrides the detection.
  bool use_posix = IsNetworkFs(file_path);
  if (const char *e = std::getenv("CLIO_BDEV_AIO")) {
    use_posix = std::string(e) == "posix";
  }
  if (use_posix) {
    auto pio = ctp::AsyncIoFactory::Get(io_depth, ctp::AsyncIoBackend::kPosixAio);
    if (pio && pio->Open(file_path, flags, 0644)) {
      return pio;
    }
  }
  auto io = ctp::AsyncIoFactory::Get(io_depth);
  if (io && io->Open(file_path, flags, 0644)) {
    return io;
  }
  // Preferred backend is unusable in this environment. IoUringAsyncIO::Open()
  // closes any fds it opened before returning false, so it is safe to discard
  // it and retry with POSIX AIO.
  io = ctp::AsyncIoFactory::Get(io_depth, ctp::AsyncIoBackend::kPosixAio);
  if (io && io->Open(file_path, flags, 0644)) {
    return io;
  }
  return nullptr;
}

/**
 * Reserve real disk blocks for [from, to) of the backing file. ftruncate
 * alone makes a SPARSE extent: it never fails for lack of space, so a full
 * disk showed up later as failed writes (EIO to the application) instead
 * of as a full device at allocation time.
 * @param file_path backing file
 * @param from start of the new extent
 * @param to end of the new extent
 * @return 0, or the errno (ENOSPC when the disk cannot hold it). A
 *         filesystem without fallocate support keeps the sparse extent (0).
 *
 * Uses the fallocate(2) system call, not posix_fallocate(3): where the
 * filesystem cannot preallocate (9p/drvfs, NFSv3, many FUSE filesystems)
 * glibc's posix_fallocate does not report EOPNOTSUPP, it EMULATES the
 * reservation by writing every block. That turned this call into writing the
 * whole extent -- 1 GiB for a default bdev, ~155 s on a Docker Desktop bind
 * mount -- synchronously on the runtime worker running the pool's Create or a
 * PutBlob growth step (issue #1180). fallocate(2) fails fast there instead.
 */
int ReserveFileSpace(const std::string &file_path, clio::run::u64 from,
                     clio::run::u64 to) {
#ifdef __linux__
  if (to <= from) return 0;
  const int fd = open(file_path.c_str(), O_RDWR);
  if (fd < 0) return errno;
  int rc = 0;
  while (fallocate(fd, 0, static_cast<off_t>(from),
                   static_cast<off_t>(to - from)) != 0) {
    if (errno == EINTR) continue;
    rc = errno;
    break;
  }
  close(fd);
  if (rc == EOPNOTSUPP || rc == ENOSYS || rc == EINVAL) return 0;
  return rc;
#else
  (void)file_path;
  (void)from;
  (void)to;
  return 0;
#endif
}

/**
 * Flush a file's data to persistent media.
 *
 * Linux has fdatasync. macOS has none, and its fsync stops at the drive's
 * volatile cache, so F_FULLFSYNC is what reaches media there (falling back
 * to fsync on filesystems that reject it). Windows flushes via _commit,
 * which needs a writable handle.
 * @param file_path the file to flush
 * @return 0, or the errno of the failing open or flush
 */
int SyncFileData(const std::string &file_path) {
#if defined(_WIN32)
  const int fd = _open(file_path.c_str(), _O_RDWR | _O_BINARY);
  if (fd < 0) return errno;
  const int rc = _commit(fd);
  const int err = errno;
  _close(fd);
  return rc == 0 ? 0 : err;
#else
  const int fd = ::open(file_path.c_str(), O_RDONLY);
  if (fd < 0) return errno;
#if defined(__linux__)
  const int rc = ::fdatasync(fd);
#elif defined(__APPLE__)
  int rc = ::fcntl(fd, F_FULLFSYNC);
  if (rc != 0) rc = ::fsync(fd);
#else
  const int rc = ::fsync(fd);
#endif
  const int err = errno;
  ::close(fd);
  return rc == 0 ? 0 : err;
#endif
}

/**
 * Striping requested through the environment, read once at Init:
 * CLIO_BDEV_STRIPE_FILES=<N> (default 1, clamped to [1, 64]) and
 * CLIO_BDEV_STRIPE_UNIT=<size> (default 1 MiB). A unit that is not a multiple
 * of the direct-I/O alignment (or 4 KiB) falls back to the default with a
 * warning, so every stripe boundary stays O_DIRECT-aligned.
 * @return the stripe layout for this device (num_files == 1 when unset)
 */
StripeLayout LoadStripeLayout() {
  constexpr clio::run::u64 kDefaultUnit = clio::run::u64(1) << 20;
  StripeLayout layout;
  layout.unit = kDefaultUnit;
  if (const char *n = std::getenv("CLIO_BDEV_STRIPE_FILES")) {
    const long parsed = std::strtol(n, nullptr, 10);
    layout.num_files =
        static_cast<clio::run::u32>(std::clamp<long>(parsed, 1, 64));
  }
  if (const char *u = std::getenv("CLIO_BDEV_STRIPE_UNIT")) {
    const clio::run::u64 want = ctp::ConfigParse::ParseSize(u);
    const clio::run::u64 align =
        std::max<clio::run::u64>(4096, ctp::DirectIoAlign());
    if (want > 0 && want % align == 0) {
      layout.unit = want;
    } else {
      HLOG(kWarning,
           "CLIO_BDEV_STRIPE_UNIT={} is not a positive multiple of {} bytes; "
           "using {} bytes",
           u, align, kDefaultUnit);
    }
  }
  return layout;
}

// Steady-clock time in nanoseconds, used to age the host free-space sample.
clio::run::u64 SteadyNowNs() {
  return static_cast<clio::run::u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count());
}

}  // namespace

bool WorkerIOContext::Init(const std::vector<std::string> &paths,
                           clio::run::u32 io_depth, clio::run::u32 worker_id) {
  if (is_initialized_) return true;

  for (const std::string &path : paths) {
    auto io = OpenBackingFile(io_depth, path);
    if (!io) {
      HLOG(kError, "Worker {} failed to open file {} ({})", worker_id, path,
           std::strerror(errno));
      Cleanup();  // release the files that did open
      return false;
    }
    async_ios_.push_back(std::move(io));
  }

  is_initialized_ = true;
  return true;
}

void WorkerIOContext::Cleanup() {
  for (auto &io : async_ios_) {
    if (io) io->Close();
  }
  async_ios_.clear();
  is_initialized_ = false;
}

bool FsBdevTransport::Init(const CreateParams& params,
                           const std::string& pool_name, Runtime* runtime) {
  // The pool name doubles as the backing file path.
  file_path_ = pool_name;
  io_depth_ = params.io_depth_;
  fail_marker_path_ = file_path_ + ".fail";
  fail_marker_checked_ns_.store(0, std::memory_order_relaxed);
  fail_marker_present_.store(false, std::memory_order_relaxed);

  stripe_ = LoadStripeLayout();
  if (stripe_.IsStriped()) {
    HLOG(kInfo, "bdev {}: striped over {} files ({}.s0 .. .s{}), stripe unit {} bytes",
         file_path_, stripe_.num_files, file_path_, stripe_.num_files - 1,
         stripe_.unit);
  }

  clio::run::u64 existing = 0;
  if (!OpenBackingFiles(&existing)) return false;

  // Device capacity (what the allocator hands out): the configured size when
  // one is given, else the existing file's size, else a 1GB default.
  //
  // #858 semantic change: a configured capacity is no longer clamped to a
  // smaller existing file. With lazy growth, "backing file smaller than
  // capacity" is the NORMAL state after any restart (the file only ever grew
  // as far as allocations reached), so the old min(existing, configured)
  // clamp would silently shrink the device to its previously-touched prefix
  // on every restart. The existing file is instead treated as the
  // already-backed prefix, and growth resumes from there.
  clio::run::u64 file_size;
  if (params.total_size_ > 0) {
    file_size = params.total_size_;
  } else if (existing > 0) {
    file_size = existing;
  } else {
    file_size = 1ULL << 30;
  }

  // #858: growth granularity for the lazy backing-file extension
  // (EnsureFileBacked). 0 disables stepping — each allocation extends the
  // file exactly as far as it needs.
  growth_unit_ = params.growth_unit_;

  if (existing == 0) {
    // Fresh (empty) backing file: truncate only the FIRST growth unit into
    // existence. The rest of the file materializes in EnsureFileBacked as
    // allocations cross the backed frontier. This keeps a 500GB
    // capacity_limit from claiming 500GB at compose time on filesystems that
    // allocate eagerly on a size set (NTFS SetEndOfFile; any FS without
    // sparse files) and gives ENOSPC a chance to surface at allocation time
    // instead of mid-write.
    clio::run::u64 initial = file_size;
    if (growth_unit_ > 0 && growth_unit_ < initial) {
      initial = growth_unit_;
    }
    // GrowBackingFile truncates and reserves; on failure it leaves the file
    // at size 0 (its undo), so a failed fresh start holds no data.
    if (!GrowBackingFile(0, initial)) {
      HLOG(kError, "Cannot reserve {} bytes for bdev file {}", initial,
           file_path_);
      return false;
    }
    file_backed_bytes_.store(initial, std::memory_order_relaxed);
  } else {
    // Existing file: its current extent is the already-backed prefix (capped
    // at the device capacity — a larger file than the device just has spare
    // tail the allocator will never address).
    file_backed_bytes_.store(existing < file_size ? existing : file_size,
                             std::memory_order_relaxed);
  }

  if (!InitializeWorkerIOContexts()) {
    // A worker that cannot open the file fails every I/O it is handed: refuse
    // the device rather than create one that errors intermittently.
    HLOG(kError, "Failed to open {} on every worker; refusing the device",
         file_path_);
    CleanupWorkerIOContexts();  // release the descriptors that did open
    return false;
  }

  clio::run::WorkOrchestrator *work_orchestrator = CLIO_WORK_ORCHESTRATOR;
  size_t num_workers = work_orchestrator ? work_orchestrator->GetWorkerCount() : 16;
  allocator_.Init(num_workers, file_size, params.alignment_);

  return OpenAllocLog(params);
}

std::vector<std::string> FsBdevTransport::BackingPaths() const {
  if (!stripe_.IsStriped()) return {file_path_};
  std::vector<std::string> paths;
  paths.reserve(stripe_.num_files);
  for (clio::run::u32 i = 0; i < stripe_.num_files; ++i) {
    paths.push_back(file_path_ + ".s" + std::to_string(i));
  }
  return paths;
}

bool FsBdevTransport::OpenBackingFiles(clio::run::u64 *existing) {
  const std::vector<std::string> paths = BackingPaths();
  file_ids_.clear();
  file_ids_.resize(paths.size());
  file_id_valid_ = true;
  int64_t min_size = -1;
  for (size_t i = 0; i < paths.size(); ++i) {
    // errno from the last open attempt. Without it this message says only
    // "it did not work", and a permission problem on a leftover file is
    // indistinguishable from a missing io_uring or a full disk -- which is
    // exactly the ambiguity that made a stale scratch file look like a broken
    // driver.
    auto io = OpenBackingFile(io_depth_, paths[i], /*create=*/true);
    if (!io) {
      HLOG(kError, "Failed to open bdev file: {} ({})", paths[i],
           strerror(errno));
      return false;
    }
    // #1210: remember which file this is; later by-path opens check it.
    const bool id_ok =
        ctp::SystemInfo::GetFileIdentity(paths[i], &file_ids_[i]);
    file_id_valid_ = file_id_valid_ && id_ok;
    // int64_t, not off_t: MSVC off_t is a 32-bit long, which wraps any
    // backing file of 2 GiB or more (#1059).
    const int64_t size = io->GetFileSize();
    io->Close();
    if (size < 0) {
      HLOG(kError, "Failed to get file size for: {}", paths[i]);
      return false;
    }
    if (min_size < 0 || size < min_size) min_size = size;
  }
  const auto smallest = static_cast<clio::run::u64>(min_size);
  // Striped: the prefix every file fully covers (a file torn by a crash can
  // only shrink the claim, never make a hole the allocator would address).
  *existing = stripe_.IsStriped() ? stripe_.LogicalPrefixFromFileSize(smallest)
                                  : smallest;
  return true;
}

bool FsBdevTransport::GrowStripedFiles(clio::run::u64 target) {
  if (!BackingFileIntact("grow")) return false;
  const clio::run::u64 per_file = stripe_.PerFileBytes(target);
  std::vector<std::pair<std::string, clio::run::u64>> grown;
  for (const std::string &path : BackingPaths()) {
    clio::run::u64 prev = 0;
    if (!ExtendStripeFile(path, per_file, &prev)) {
      UndoStripeGrowth(grown);
      return false;
    }
    if (prev < per_file) grown.emplace_back(path, prev);
  }
  return true;
}

bool FsBdevTransport::ExtendStripeFile(const std::string &path,
                                       clio::run::u64 to,
                                       clio::run::u64 *prev) {
  auto io = OpenBackingFile(io_depth_, path);
  if (!io) {
    HLOG(kError, "EnsureFileBacked: failed to open {} to grow it", path);
    return false;
  }
  const int64_t cur_signed = io->GetFileSize();
  if (cur_signed < 0) {
    io->Close();
    return false;
  }
  const auto cur = static_cast<clio::run::u64>(cur_signed);
  *prev = cur;
  if (to <= cur) {
    io->Close();
    return true;  // already long enough (another growth or a torn run)
  }
  bool ok = io->Truncate(static_cast<size_t>(to));
  if (ok) {
    const int rrc = ReserveFileSpace(path, cur, to);
    if (rrc != 0) {
      HLOG(kWarning, "EnsureFileBacked: no disk space to grow {} to {} "
           "bytes ({})", path, to, std::strerror(rrc));
      io->Truncate(static_cast<size_t>(cur));
      ok = false;
    }
  }
  io->Close();
  return ok;
}

void FsBdevTransport::UndoStripeGrowth(
    const std::vector<std::pair<std::string, clio::run::u64>> &grown) {
  for (const auto &[path, size] : grown) {
    auto io = OpenBackingFile(io_depth_, path);
    if (io) {
      io->Truncate(static_cast<size_t>(size));
      io->Close();
    }
  }
}

int FsBdevTransport::SyncBackingFiles() {
  for (const std::string &path : BackingPaths()) {
    const int rc = SyncFileData(path);
    if (rc != 0) return rc;
  }
  return 0;
}

clio::run::u64 FsBdevTransport::NextWriteSegment(clio::run::u64 fo,
                                                 clio::run::u64 left,
                                                 const StripeLayout &stripe) {
  // Up to three writes per block: an unaligned head and tail through the
  // page cache, and the DirectAlign()-aligned body between them bounced to
  // the O_DIRECT fd (see WantsBounce). A FUSE write from an unaligned user
  // buffer arrives as ~1 MiB + a few bytes, so without the split no byte
  // of it could go direct.
  const clio::run::u64 align = DirectAlign();
  const clio::run::u64 head =
      std::min<clio::run::u64>(left, (align - fo % align) % align);
  const clio::run::u64 body = ((left - head) / align) * align;
  clio::run::u64 seg = left;
  if (body >= kBounceMin) {
    seg = head != 0 ? head : body;
  }
  // A segment never crosses a stripe boundary, so it maps to one file.
  return std::min(seg, stripe.BytesToBoundary(fo));
}

FsBdevTransport::AlignedBuf FsBdevTransport::PrepareWriteSegment(
    char *src, clio::run::u64 seg, clio::run::u64 file_off, void **out_data) {
  *out_data = src;
  AlignedBuf bounce;
  if (WantsBounce(src, static_cast<size_t>(seg), file_off)) {
    bounce = TakeBounce(static_cast<size_t>(seg));
    std::memcpy(bounce.get(), src, static_cast<size_t>(seg));
    *out_data = bounce.get();
  }
  return bounce;
}

void FsBdevTransport::LogWriteFailure(int error_code,
                                      clio::run::u64 logical_off,
                                      clio::run::u64 seg, size_t file_index) {
  static std::atomic<clio::run::u64> write_errs{0};
  const clio::run::u64 ne = write_errs.fetch_add(1) + 1;
  if (ne <= 20 || (ne & (ne - 1)) == 0) {
    HLOG(kError,
         "file bdev write failed: errno {} ({}) offset {} size {} file {} "
         "(#{} so far)",
         error_code, std::strerror(error_code), logical_off, seg, file_index,
         ne);
  }
}

ctp::IoToken FsBdevTransport::SubmitReadSegment(ctp::AsyncIO *aio, char *dst,
                                                clio::run::u64 seg,
                                                clio::run::u64 file_off,
                                                AlignedBuf &rbounce) {
  // issue #1343: reads bounce like writes. Task buffers in shared memory
  // are not DirectAlign()-aligned, so an aligned read was still sent to the
  // buffered fd: served (and polluted) by the page cache, never the device.
  void *read_dst = dst;
  if (WantsBounce(dst, static_cast<size_t>(seg), file_off)) {
    rbounce = TakeBounce(static_cast<size_t>(seg));
    read_dst = rbounce.get();
  }
  ctp::IoToken token = aio->Read(read_dst, static_cast<size_t>(seg),
                                 static_cast<int64_t>(file_off));
  if (token == ctp::kInvalidIoToken && rbounce) {
    GiveBounce(std::move(rbounce), static_cast<size_t>(seg));
  }
  return token;
}

void FsBdevTransport::FinishBouncedRead(AlignedBuf &rbounce, char *dst,
                                        clio::run::u64 seg,
                                        const ctp::IoResult &result) {
  if (!rbounce) return;  // read went straight into the caller's buffer
  if (result.error_code == 0 && result.bytes_transferred > 0) {
    std::memcpy(dst, rbounce.get(),
                std::min(static_cast<size_t>(result.bytes_transferred),
                         static_cast<size_t>(seg)));
  }
  GiveBounce(std::move(rbounce), static_cast<size_t>(seg));
}

clio::run::u64 FsBdevTransport::ZeroFillUnbacked(char *dst,
                                                 clio::run::u64 got,
                                                 clio::run::u64 seg,
                                                 clio::run::u64 logical_off,
                                                 bool in_capacity) {
  if (got >= seg || !in_capacity) return got;
  // Inside the device but past the backed end of the (lazily grown) file:
  // never written, so it reads as zeros -- the same answer the file would
  // give had it been truncated to full capacity up front. A caller that
  // writes at fixed offsets without going through this allocator (a
  // safe_bdev member) otherwise saw a short read of a partly written
  // chunk and took it for a failing disk.
  std::memset(dst + got, 0, seg - got);
  static std::atomic<clio::run::u64> zero_fills{0};
  const clio::run::u64 nz = zero_fills.fetch_add(1) + 1;
  if (nz <= 20 || (nz & (nz - 1)) == 0) {
    // Which reads land past the backed end of a file (#1124 diagnosis).
    HLOG(kDebug, "bdev ReadBlocks: zero-fill #{} on {}: off {} len {} "
         "got {} backed {}", nz, file_path_, logical_off, seg, got,
         file_backed_bytes_.load());
  }
  return seg;
}

bool FsBdevTransport::OpenAllocLog(const CreateParams& params) {
  if (params.SkipsAllocLog()) {
    // Explicitly volatile: nothing recovers this bdev's allocations, so keep
    // no WAL and no sync thread. Remove a default log a previous non-volatile
    // run may have left, so a later run with persistence never replays it.
    std::error_code ec;
    std::filesystem::remove(file_path_ + ".alloc_log", ec);
    HLOG(kDebug, "bdev {}: persistence_level volatile, allocator WAL off",
         file_path_);
    return true;
  }
  // An explicit alloc_log always recovers. The default log recovers on every
  // recovering start (a plain `clio_run start`); after `start --fresh` no
  // metadata references the old bytes, so their allocations are garbage and
  // the log starts empty.
  const bool explicit_path = !params.alloc_log_path_.empty();
  const std::string path =
      explicit_path ? params.alloc_log_path_ : file_path_ + ".alloc_log";
  auto *manager = CLIO_RUNTIME_MANAGER;
  const bool recover =
      explicit_path || (manager != nullptr && manager->is_restart_);
  if (!recover) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
  if (!alloc_log_.Open(path, recover)) {
    HLOG(kError, "Cannot open allocator log {} for bdev {} ({})", path,
         file_path_, strerror(errno));
    return false;
  }
  has_alloc_log_ = true;
  const std::vector<LiveBlock> &live = alloc_log_.live(/*group_id=*/0);
  if (!live.empty()) {
    std::vector<std::pair<clio::run::u64, clio::run::u64>> ext;
    ext.reserve(live.size());
    for (const auto &b : live) {
      ext.emplace_back(b.offset, b.size);
    }
    allocator_.InitFromLive(ext);
    HLOG(kInfo, "bdev {}: recovered {} live blocks from {}; {} bytes free",
         file_path_, live.size(), path, allocator_.GetRemainingSize());
  }
  // Start from a compact log so it tracks the live set, not history.
  alloc_log_.Compact();
  sync_stop_ = false;
  sync_thread_ = std::thread(&FsBdevTransport::AllocLogSyncLoop, this);
  return true;
}

bool FsBdevTransport::FaultInjected() {
  const clio::run::u64 now_ns = static_cast<clio::run::u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count());
  clio::run::u64 last = fail_marker_checked_ns_.load(std::memory_order_relaxed);
  if (last != 0 && now_ns - last < kFailMarkerPollNs) {
    return fail_marker_present_.load(std::memory_order_relaxed);
  }
  // One caller per period re-checks; the others keep the cached answer.
  if (!fail_marker_checked_ns_.compare_exchange_strong(
          last, now_ns, std::memory_order_relaxed)) {
    return fail_marker_present_.load(std::memory_order_relaxed);
  }
  std::error_code ec;
  const bool present = std::filesystem::exists(fail_marker_path_, ec);
  const bool was = fail_marker_present_.exchange(present,
                                                 std::memory_order_relaxed);
  if (present != was) {
    HLOG(kWarning, "bdev {}: fault-injection marker {} {}; I/O now {}",
         file_path_, fail_marker_path_, present ? "present" : "removed",
         present ? "FAILS" : "works");
  }
  return present;
}

bool FsBdevTransport::Sync() {
  if (FaultInjected()) {
    HLOG(kError, "bdev Sync: {} failed (injected device fault)", file_path_);
    return false;
  }
  // Data first, then the allocator state that references it: a crash in
  // between leaves synced bytes in blocks the log may not show yet (the
  // CTE's own WAL still does), never a logged block whose bytes are lost.
  if (!BackingFileIntact("sync")) return false;
  // Group commit. Every CTE fsync syncs the whole multi-GB tier file, and
  // under tiering load hundreds of them queued up, each a full fdatasync
  // (4.9 s average, 26 s max on the DRAM-heavy mix). A caller's writes have
  // completed before it calls Sync, so ANY fdatasync that starts after this
  // call registered covers them: register a ticket, then either wait for a
  // sync that started after the ticket or start one that covers everyone
  // registered so far.
  std::unique_lock<std::mutex> lk(gc_mu_);
  const clio::run::u64 ticket = ++gc_next_;
  for (;;) {
    if (gc_done_ >= ticket) return gc_ok_through_ >= ticket;
    if (!gc_running_) break;
    gc_cv_.wait(lk);
  }
  gc_running_ = true;
  const clio::run::u64 covers = gc_next_;  // everyone registered so far
  lk.unlock();
  const int err = SyncBackingFiles();
  if (err == 0) FlushAllocLog();
  lk.lock();
  gc_running_ = false;
  gc_done_ = covers;
  if (err == 0) gc_ok_through_ = covers;
  gc_cv_.notify_all();
  lk.unlock();
  if (err != 0) {
    HLOG(kError, "bdev Sync: flushing {} failed ({})", file_path_,
         strerror(err));
    return false;
  }
  return true;
}

void FsBdevTransport::AllocLogSyncLoop() {
  std::unique_lock<std::mutex> lock(sync_mu_);
  while (!sync_stop_) {
    sync_cv_.wait_for(lock, std::chrono::milliseconds(kAllocLogSyncPeriodMs),
                      [this] { return sync_stop_; });
    lock.unlock();
    FlushAllocLog();
    lock.lock();
  }
}

void FsBdevTransport::StopAllocLogSync() {
  {
    std::lock_guard<std::mutex> lock(sync_mu_);
    sync_stop_ = true;
  }
  sync_cv_.notify_all();
  if (sync_thread_.joinable()) {
    sync_thread_.join();
  }
}

void FsBdevTransport::LogBlocks(const std::vector<Block>& blocks,
                                bool is_free) {
  if (!has_alloc_log_) return;
  for (const Block &b : blocks) {
    if (is_free) {
      alloc_log_.LogFree(/*group_id=*/0, b.offset_, b.size_, b.block_type_);
    } else {
      alloc_log_.LogAlloc(/*group_id=*/0, b.offset_, b.size_, b.block_type_);
    }
  }
  alloc_log_.Append();
}

void FsBdevTransport::FlushAllocLog() {
  if (!has_alloc_log_) return;
  // SyncAppended, not Flush: Flush holds the log mutex across the fsync, and
  // every worker logging an allocation or free (LogBlocks) waited on it for
  // the whole fsync -- AllocateBlocks/FreeBlocks stalled their workers for
  // up to ~10 s under tiering load. SyncAppended appends under the lock and
  // fsyncs a duplicate descriptor without it; the same records are durable
  // when it returns.
  alloc_log_.SyncAppended();
  // Compact once history dominates: the log is then bounded by ~2x the
  // live-block count instead of growing with every allocate/free.
  constexpr clio::run::u64 kMinCompactRecords = 4096;
  const clio::run::u64 live = alloc_log_.live_block_count();
  if (alloc_log_.records_on_disk() > std::max(kMinCompactRecords, 2 * live)) {
    alloc_log_.Compact();
  }
}

void FsBdevTransport::Destroy() {
  CleanupWorkerIOContexts();
  StopAllocLogSync();
  if (has_alloc_log_) {
    alloc_log_.Close();
  }
}

bool FsBdevTransport::AllocateBlocks(size_t size, int worker_id, std::vector<Block>& blocks) {
  if (!allocator_.AllocateBlocks(size, worker_id, blocks)) {
    return false;
  }
  // #858: back every returned block with real file before the caller can
  // touch it. Reads of allocated-but-unwritten blocks must land inside the
  // file (and return zeros) exactly as they did when the whole capacity was
  // truncated up front — a block past EOF would short-read instead.
  clio::run::u64 end = 0;
  for (const Block &b : blocks) {
    clio::run::u64 e = b.offset_ + b.size_;
    if (e > end) end = e;
  }
  if (!EnsureFileBacked(end)) {
    // The disk cannot actually provide the space the allocator promised.
    // Fail the allocation cleanly (the caller sees the same "no space"
    // result as an exhausted allocator) instead of handing out blocks whose
    // writes would EIO later.
    allocator_.FreeBlocks(worker_id, blocks);
    blocks.clear();
    return false;
  }
  // Logged BEFORE the caller can write into or reference the blocks.
  LogBlocks(blocks, /*is_free=*/false);
  return true;
}

bool FsBdevTransport::AllocateBlocksDeferred(size_t size, int worker_id,
                                             std::vector<Block>& blocks,
                                             clio::run::u64* need) {
  *need = 0;
  if (!allocator_.AllocateBlocks(size, worker_id, blocks)) {
    return false;
  }
  clio::run::u64 end = 0;
  for (const Block &b : blocks) {
    clio::run::u64 e = b.offset_ + b.size_;
    if (e > end) end = e;
  }
  if (end <= file_backed_bytes_.load(std::memory_order_acquire)) {
    // Inside the backed prefix: complete now (the common case).
    LogBlocks(blocks, /*is_free=*/false);
    return true;
  }
  *need = end;  // the caller grows the file off the worker, then finishes
  return true;
}

void FsBdevTransport::FinishAllocate(bool backed_ok, int worker_id,
                                     std::vector<Block>& blocks) {
  if (!backed_ok) {
    // Same as AllocateBlocks: a disk that cannot provide the promised space
    // fails the allocation cleanly instead of handing out blocks that EIO.
    allocator_.FreeBlocks(worker_id, blocks);
    blocks.clear();
    return;
  }
  LogBlocks(blocks, /*is_free=*/false);
}

bool FsBdevTransport::GrowBackingFile(clio::run::u64 backed,
                                      clio::run::u64 target) {
  if (stripe_.IsStriped()) return GrowStripedFiles(target);
  if (!BackingFileIntact("grow")) return false;
  auto io = OpenBackingFile(io_depth_, file_path_);
  if (!io) {
    HLOG(kError, "EnsureFileBacked: failed to open {} to grow it", file_path_);
    return false;
  }
  bool ok = io->Truncate(static_cast<size_t>(target));
  if (ok) {
    const int rrc = ReserveFileSpace(file_path_, backed, target);
    if (rrc != 0) {
      // Out of disk: undo the sparse growth so the file never claims space
      // it does not have; the allocation fails as a full device.
      HLOG(kWarning, "EnsureFileBacked: no disk space to grow {} to {} "
           "bytes ({})", file_path_, target, std::strerror(rrc));
      io->Truncate(static_cast<size_t>(backed));
      ok = false;
    }
  }
  io->Close();
  return ok;
}

bool FsBdevTransport::EnsureFileBacked(clio::run::u64 end_offset) {
  // Fast path: recycled or low blocks are already inside the backed prefix.
  if (end_offset <= file_backed_bytes_.load(std::memory_order_acquire)) {
    return true;
  }
  std::lock_guard<std::mutex> lock(grow_mu_);
  clio::run::u64 backed = file_backed_bytes_.load(std::memory_order_relaxed);
  if (end_offset <= backed) {
    return true;  // another thread grew past us while we waited
  }
  // Grow in growth-unit steps (capped at capacity) so a bump-allocation
  // stream costs one truncate per unit, not one per allocation.
  clio::run::u64 target = end_offset;
  if (growth_unit_ > 0) {
    target = ((end_offset + growth_unit_ - 1) / growth_unit_) * growth_unit_;
  }
  clio::run::u64 capacity = allocator_.GetCapacity();
  if (capacity > 0 && target > capacity) {
    target = capacity;
  }
  const clio::run::u64 now_ns = static_cast<clio::run::u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count());
  if (grow_fail_end_ != 0 && end_offset >= grow_fail_end_ &&
      now_ns - grow_fail_ns_ < kGrowRetryNs) {
    return false;  // the disk was just found full: fail fast
  }
  // A whole growth unit first; if the disk cannot hold it, just what this
  // allocation needs (the unit can be gigabytes more than what is left).
  bool ok = GrowBackingFile(backed, target);
  if (!ok && target > end_offset) {
    target = end_offset;
    ok = GrowBackingFile(backed, target);
  }
  if (!ok) {
    if (grow_fail_end_ == 0 || end_offset < grow_fail_end_ ||
        now_ns - grow_fail_ns_ >= kGrowRetryNs) {
      grow_fail_end_ = end_offset;
    }
    grow_fail_ns_ = now_ns;
    HLOG(kError,
         "EnsureFileBacked: failed to grow {} from {} to {} bytes — treating "
         "as out of space",
         file_path_, backed, end_offset);
    return false;
  }
  grow_fail_end_ = 0;
  HLOG(kDebug, "EnsureFileBacked: grew {} from {} to {} bytes", file_path_,
       backed, target);
  file_backed_bytes_.store(target, std::memory_order_release);
  return true;
}

void FsBdevTransport::SampleHostFreeBytes(
    std::shared_ptr<HostFreeCache> cache, std::string dir_str) {
  std::error_code ec;
  // parent_path() and space() both run in libstdc++.so, which MSan does not
  // see: the returned path and space_info read back as uninitialized.
  std::filesystem::path dir(dir_str);
  CTP_MSAN_UNPOISON_PATH(dir);
  std::filesystem::space_info si = std::filesystem::space(dir, ec);
  CTP_MSAN_UNPOISON_OBJ(si);
  CTP_MSAN_UNPOISON_OBJ(ec);
  // Unknown (e.g. no permission to stat): do not cap by it.
  const clio::run::u64 free_b =
      ec ? ~clio::run::u64(0) : static_cast<clio::run::u64>(si.available);
  // Timestamp at completion so the TTL measures from the fresh value.
  const clio::run::u64 done_ns = SteadyNowNs();
  cache->free_bytes.store(free_b, std::memory_order_relaxed);
  cache->sampled_ns.store(done_ns, std::memory_order_release);
  cache->refreshing.store(false, std::memory_order_release);
}

clio::run::u64 FsBdevTransport::HostFreeBytes() const {
  // Never call statfs on the caller's thread: on DAOS dfuse or Lustre it is
  // a network RPC that measured up to 11.7 s, stalling a runtime worker.
  // A stale sample is refreshed by a detached thread; callers get the last
  // value immediately, or "unknown" (no cap) before any value exists.
  HostFreeCache &cache = *host_free_cache_;
  const clio::run::u64 now_ns = SteadyNowNs();
  const clio::run::u64 at = cache.sampled_ns.load(std::memory_order_acquire);
  const clio::run::u64 cached =
      cache.free_bytes.load(std::memory_order_relaxed);
  if (at != 0 && now_ns - at < kHostFreeTtlNs) {
    return cached;
  }
  const clio::run::u64 fallback = at != 0 ? cached : ~clio::run::u64(0);
  // Single-flight: concurrent callers (every worker running GetStats) do not
  // each start a refresh; the winner launches one.
  bool expected = false;
  if (!cache.refreshing.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    return fallback;
  }
  // The thread captures a copy of the path and the shared cache, never
  // `this`, so it stays valid if the transport is destroyed first.
  std::string dir = std::filesystem::path(file_path_).parent_path().string();
  try {
    std::thread(&FsBdevTransport::SampleHostFreeBytes, host_free_cache_,
                std::move(dir))
        .detach();
  } catch (const std::system_error &) {
    // Could not start a thread: release the guard and report the fallback.
    cache.refreshing.store(false, std::memory_order_release);
  }
  return fallback;
}

clio::run::u64 FsBdevTransport::GetRemainingSize() const {
  const clio::run::u64 alloc_free = allocator_.GetRemainingSize();
  const clio::run::u64 cap = allocator_.GetCapacity();
  const clio::run::u64 used = cap > alloc_free ? cap - alloc_free : 0;
  const clio::run::u64 backed =
      file_backed_bytes_.load(std::memory_order_relaxed);
  // Free room inside the already-reserved prefix, then whatever the disk
  // can still add to the file.
  const clio::run::u64 in_backed = backed > used ? backed - used : 0;
  const clio::run::u64 host = HostFreeBytes();
  const clio::run::u64 can_back =
      host > ~clio::run::u64(0) - in_backed ? ~clio::run::u64(0)
                                            : in_backed + host;
  return std::min(alloc_free, can_back);
}

void FsBdevTransport::FreeBlocks(int worker_id, const std::vector<Block>& blocks) {
  // Logged BEFORE reuse is possible: a crash between the two leaves the
  // blocks allocated in the log (a leak), never free while still in use.
  LogBlocks(blocks, /*is_free=*/true);
  allocator_.FreeBlocks(worker_id, blocks);
}

bool FsBdevTransport::BackingFileIntact(const char *what) const {
  if (!file_id_valid_) return true;  // nothing recorded to compare with
  const std::vector<std::string> paths = BackingPaths();
  for (size_t i = 0; i < paths.size() && i < file_ids_.size(); ++i) {
    ctp::SystemInfo::FileIdentity now;
    if (ctp::SystemInfo::GetFileIdentity(paths[i], &now) &&
        now == file_ids_[i]) {
      continue;
    }
    HLOG(kError, "bdev {}: cannot {}: the backing file {} was deleted or "
         "replaced while the device was in use; refusing rather than using a "
         "different file", file_path_, what, paths[i]);
    return false;
  }
  return true;
}

bool FsBdevTransport::InitializeWorkerIOContexts() {
  clio::run::WorkOrchestrator *work_orchestrator = CLIO_WORK_ORCHESTRATOR;
  size_t num_workers = work_orchestrator ? work_orchestrator->GetWorkerCount() : 16;

  // Reserve slots for workers that do not exist yet. The worker pool GROWS at
  // runtime: WorkOrchestrator::SpawnAdditionalWorker() hands out ids past the
  // startup count (the #781/#785 lane-rescue path, and any scheduler that
  // sizes pools on demand). This vector used to be sized once at pool
  // creation, so GetWorkerIOContext(id) returned nullptr for every such worker
  // and its I/O silently failed — writes logged "WriteToFile called with
  // invalid I/O context" and reads returned 0 bytes, which the CTE read path
  // treats as a hole rather than an error. Sizing to the same elastic headroom
  // the lane table reserves keeps a context available for any id the
  // orchestrator can produce.
  //
  // Only the startup contexts are opened eagerly; the reserved tail is opened
  // lazily by GetWorkerIOContext the first time a spawned worker uses it, so
  // this costs a vector of empty structs, not file descriptors.
  const size_t reserved =
      num_workers + clio::run::WorkOrchestrator::ElasticHeadroom();
  io_contexts_.resize(reserved);
  bool success = true;
  if (!BackingFileIntact("open worker I/O contexts")) return false;
  const std::vector<std::string> paths = BackingPaths();
  for (size_t i = 0; i < num_workers; ++i) {
    if (!io_contexts_[i].Init(paths, io_depth_, static_cast<clio::run::u32>(i))) {
      success = false;
    }
  }
  return success;
}

void FsBdevTransport::CleanupWorkerIOContexts() {
  for (auto &ctx : io_contexts_) {
    ctx.Cleanup();
  }
}

WorkerIOContext *FsBdevTransport::GetWorkerIOContext(size_t worker_id) {
  if (worker_id >= io_contexts_.size()) {
    return nullptr;
  }
  WorkerIOContext *ctx = &io_contexts_[worker_id];
  if (!ctx->is_initialized_) {
    if (!BackingFileIntact("open a worker I/O context") ||
        !ctx->Init(BackingPaths(), io_depth_,
                   static_cast<clio::run::u32>(worker_id))) {
      return nullptr;
    }
  }
  return ctx;
}

FsBdevTransport::AlignedBuf FsBdevTransport::TakeBounce(size_t size) {
  {
    std::lock_guard<std::mutex> lock(bounce_mu_);
    for (size_t i = 0; i < bounce_pool_.size(); ++i) {
      if (bounce_pool_[i].first == size) {
        AlignedBuf buf = std::move(bounce_pool_[i].second);
        bounce_pool_[i] = std::move(bounce_pool_.back());
        bounce_pool_.pop_back();
        return buf;
      }
    }
  }
  // The aligned operator new[] called directly (not a new-expression): MSVC
  // reads `new (std::align_val_t(...)) char[n]` as PLACEMENT new and rejects
  // it (C2956). AlignedFree releases it with the matching operator delete[].
  return AlignedBuf(static_cast<char *>(
      ::operator new[](size, std::align_val_t(DirectAlign()))));
}

void FsBdevTransport::GiveBounce(AlignedBuf buf, size_t size) {
  if (size > kBounceKeepMax) return;
  std::lock_guard<std::mutex> lock(bounce_mu_);
  if (bounce_pool_.size() < kBouncePoolMax) {
    bounce_pool_.emplace_back(size, std::move(buf));
  }
}

clio::run::TaskResume FsBdevTransport::WriteBlocks(ctp::ipc::FullPtr<WriteTask> task) {
  CLIO_TASK_BODY_BEGIN
  clio::run::Worker *worker = CLIO_CUR_WORKER;
  size_t worker_id = worker ? worker->GetId() : 0;
  WorkerIOContext *io_ctx = GetWorkerIOContext(worker_id);

  auto *ipc_mgr = CLIO_IPC;
  ctp::ipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).Cast<char>();

  bool data_on_device = ctp::IsDevicePointer(data_ptr.ptr_);
  std::vector<char> staging;
  if (data_on_device) {
    staging.resize(task->length_);
    ctp::DeviceAwareMemcpy(staging.data(), data_ptr.ptr_, task->length_);
  }

  clio::run::u64 total_bytes_written = 0;
  clio::run::u64 data_offset = 0;
  task->io_error_ = static_cast<clio::run::u32>(ctp::IoError::kOk);

  if (FaultInjected()) {
    // TEST-ONLY: the device is "dead" (see fail_marker_path_).
    task->return_code_ = 4;
    task->io_error_ = static_cast<clio::run::u32>(ctp::IoError::kDeviceFault);
    task->bytes_written_ = 0;
    CLIO_CO_RETURN;
  }

  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];
    clio::run::u64 remaining = task->length_ - total_bytes_written;
    if (remaining == 0) break;
    clio::run::u64 block_write_size = std::min(remaining, block.size_);

    void *block_data = data_on_device
                           ? static_cast<void *>(staging.data() + data_offset)
                           : static_cast<void *>(data_ptr.ptr_ + data_offset);

    if (io_ctx == nullptr || !io_ctx->is_initialized_ || io_ctx->async_ios_.empty()) {
      HLOG(kError, "WriteToFile called with invalid I/O context");
      task->return_code_ = 1;
      task->bytes_written_ = total_bytes_written;
      CLIO_CO_RETURN;
    }

    // Each segment lies inside one stripe file (NextWriteSegment clamps at
    // stripe boundaries), so it is one AIO write to that file's context.
    clio::run::u64 seg_done = 0;
    while (seg_done < block_write_size) {
      const clio::run::u64 fo = block.offset_ + seg_done;
      const clio::run::u64 seg =
          NextWriteSegment(fo, block_write_size - seg_done, stripe_);
      const StripeLocation loc = stripe_.Locate(fo);
      char *seg_src = static_cast<char *>(block_data) + seg_done;
      void *seg_data = seg_src;
      AlignedBuf bounce =
          PrepareWriteSegment(seg_src, seg, loc.file_offset, &seg_data);

      ctp::AsyncIO *aio = io_ctx->async_ios_[loc.file_index].get();
      ctp::IoToken token = aio->Write(seg_data, static_cast<size_t>(seg),
                                      static_cast<int64_t>(loc.file_offset));
      if (token == ctp::kInvalidIoToken) {
        task->return_code_ = 2;
        task->bytes_written_ = total_bytes_written + seg_done;
        CLIO_CO_RETURN;
      }

      ctp::IoResult result;
      while (!aio->IsComplete(token, result)) {
        CLIO_CO_AWAIT(clio::run::yield(10.0));
      }
      if (bounce) {
        GiveBounce(std::move(bounce), static_cast<size_t>(seg));
      }

      if (result.error_code != 0) {
        LogWriteFailure(result.error_code, fo, seg, loc.file_index);
        task->return_code_ = 4;
        task->io_error_ =
            static_cast<clio::run::u32>(ctp::ClassifyErrno(result.error_code));
        task->bytes_written_ = total_bytes_written + seg_done;
        CLIO_CO_RETURN;
      }
      const clio::run::u64 got = std::min(
          static_cast<clio::run::u64>(result.bytes_transferred), seg);
      seg_done += got;
      if (got < seg) break;  // short write: account for what landed
    }

    total_bytes_written += seg_done;
    data_offset += seg_done;
  }

  task->return_code_ = 0;
  task->bytes_written_ = total_bytes_written;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume FsBdevTransport::ReadBlocks(ctp::ipc::FullPtr<ReadTask> task) {
  CLIO_TASK_BODY_BEGIN
  clio::run::Worker *worker = CLIO_CUR_WORKER;
  size_t worker_id = worker ? worker->GetId() : 0;
  WorkerIOContext *io_ctx = GetWorkerIOContext(worker_id);

  auto *ipc_mgr = CLIO_IPC;
  ctp::ipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).Cast<char>();

  bool data_on_device = ctp::IsDevicePointer(data_ptr.ptr_);
  std::vector<char> staging;
  if (data_on_device) {
    staging.resize(task->length_);
  }

  clio::run::u64 total_bytes_read = 0;
  clio::run::u64 data_offset = 0;
  task->io_error_ = static_cast<clio::run::u32>(ctp::IoError::kOk);

  if (FaultInjected()) {
    // TEST-ONLY: the device is "dead" (see fail_marker_path_).
    task->return_code_ = 4;
    task->io_error_ = static_cast<clio::run::u32>(ctp::IoError::kDeviceFault);
    task->bytes_read_ = 0;
    CLIO_CO_RETURN;
  }

  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];
    clio::run::u64 remaining = task->length_ - total_bytes_read;
    if (remaining == 0) break;
    clio::run::u64 block_read_size = std::min(remaining, block.size_);

    void *block_data = data_on_device
                           ? static_cast<void *>(staging.data() + data_offset)
                           : static_cast<void *>(data_ptr.ptr_ + data_offset);

    if (io_ctx == nullptr || !io_ctx->is_initialized_ || io_ctx->async_ios_.empty()) {
      task->return_code_ = 1;
      task->bytes_read_ = total_bytes_read;
      CLIO_CO_RETURN;
    }

    // Reads split only at stripe boundaries; short reads are zero-filled per
    // piece (ZeroFillUnbacked), which keeps the block-level answer unchanged.
    const bool in_capacity =
        block.offset_ + block_read_size <= allocator_.GetCapacity();
    clio::run::u64 seg_read = 0;
    while (seg_read < block_read_size) {
      const clio::run::u64 fo = block.offset_ + seg_read;
      const clio::run::u64 seg = std::min(block_read_size - seg_read,
                                          stripe_.BytesToBoundary(fo));
      const StripeLocation loc = stripe_.Locate(fo);
      char *seg_dst = static_cast<char *>(block_data) + seg_read;

      ctp::AsyncIO *aio = io_ctx->async_ios_[loc.file_index].get();
      AlignedBuf rbounce;
      ctp::IoToken token =
          SubmitReadSegment(aio, seg_dst, seg, loc.file_offset, rbounce);
      if (token == ctp::kInvalidIoToken) {
        task->return_code_ = 2;
        task->bytes_read_ = total_bytes_read + seg_read;
        CLIO_CO_RETURN;
      }

      ctp::IoResult result;
      while (!aio->IsComplete(token, result)) {
        CLIO_CO_AWAIT(clio::run::yield(10.0));
      }
      FinishBouncedRead(rbounce, seg_dst, seg, result);

      if (result.error_code != 0) {
        task->return_code_ = 4;
        task->io_error_ =
            static_cast<clio::run::u32>(ctp::ClassifyErrno(result.error_code));
        task->bytes_read_ = total_bytes_read + seg_read;
        CLIO_CO_RETURN;
      }

      const clio::run::u64 got = ZeroFillUnbacked(
          seg_dst,
          std::min(static_cast<clio::run::u64>(result.bytes_transferred), seg),
          seg, fo, in_capacity);
      seg_read += got;
      if (got < seg) break;  // past the backed end, outside the device
    }

    total_bytes_read += seg_read;
    data_offset += seg_read;
  }

  if (data_on_device && total_bytes_read > 0) {
    ctp::DeviceAwareMemcpy(data_ptr.ptr_, staging.data(), total_bytes_read);
  }

  task->return_code_ = 0;
  task->bytes_read_ = total_bytes_read;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

} // namespace clio::run::bdev
