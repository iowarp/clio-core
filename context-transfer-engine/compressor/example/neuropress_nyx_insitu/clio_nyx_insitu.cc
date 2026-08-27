/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file clio_nyx_insitu.cc
 * @brief Clio's half of the Nyx in-situ path: AMReX device memory -> NeuroPress.
 *
 * Nyx is an AMReX application with no library interface, so unlike the
 * LAMMPS-as-a-library example there is no way to reach into it from the
 * outside. The hand-over therefore goes the other way: Nyx calls out, through
 * a tiny dlopen'd C ABI (clio_nyx_insitu.h), from its own in-situ callback.
 * Everything Clio-shaped lives here; the Nyx patch is ~90 lines that walk a
 * MultiFab and call these five functions.
 *
 * WHAT ARRIVES HERE
 * -----------------
 * `fab.dataPtr(comp)` of the level-0 State_Type MultiFab. Measured with
 * cudaPointerGetAttributes on this build (see the README's Q1 section):
 *
 *   The_Arena: isDevice=1 isManaged=0 isHostAccessible=0
 *   comp=0 (density) ... type=2 (DEVICE) device=0 hostPointer=0
 *
 * so it is a plain cudaMalloc allocation, not managed/UVM and not pinned
 * host. That is the fact the whole design rests on, and it is re-checked per
 * blob rather than assumed: `RequireDevice()` below refuses anything whose
 * cudaPointerGetAttributes type is not cudaMemoryTypeDevice.
 *
 * GHOST CELLS AND WHAT A CHUNK IS
 * -------------------------------
 * A FArrayBox component is contiguous over the GROWN box (valid box plus
 * nGrow cells on every side; State_Type carries nGrow=1). The component
 * stride is fab.box().numPts(), which is 34^3 for a 32^3 valid box -- so the
 * "obvious" read of numPts(validbox) elements from dataPtr(comp) is NOT the
 * field, it is the first 32768 of 39304 ghost-inclusive cells. This adapter
 * takes the valid box explicitly, as a 3D sub-volume copy on the GPU, so a
 * blob is exactly the region of the domain the box owns. Ghost cells are
 * excluded because they are duplicates of a neighbour's interior (or, at an
 * output point, not necessarily filled at all): storing them would inflate
 * the ratio with redundancy nothing asked for and make the blob correspond to
 * no region of the simulation. NYX_CLIO_INSITU_GHOSTS=1 on the Nyx side asks
 * for the whole grown box instead, for anyone who wants to measure that.
 *
 * A blob is one (box, component, chunk): the valid-box field for one
 * component of one FArrayBox, cut into CLIO_NYX_CHUNK-byte chunks. The chunk
 * size defaults to 4 MiB, the same as ../neuropress_field_replay and the
 * LAMMPS examples, so ratios stay comparable across all of them.
 *
 * STREAMS
 * -------
 * Every device copy this file issues runs on its OWN cudaStreamNonBlocking
 * stream and is synchronised before the pointer is handed on. Nothing here
 * touches the legacy default stream. That is not a stylistic preference: the
 * corruption fixed in 6b6e28a7 was a legacy-default-stream copy racing a
 * worker's non-blocking read, and it only became likely when something else
 * in the process (Kokkos there) parked work on the legacy stream. AMReX
 * creates its own streams with cudaStreamCreate and does not use the legacy
 * one for kernels, but the discipline is the same either way. The Nyx side
 * calls amrex::Gpu::Device::synchronize() before the first stage of a frame,
 * so no AMReX kernel is still writing the FAB when it is read.
 */

#include "clio_nyx_insitu.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

#include <clio_cte/compressor/compressor_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_ctp/compress/compress_factory.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_runtime/clio_runtime.h>

namespace {

/**
 * Absolute error bound for error-bounded lossy compression, from
 * CLIO_NEUROPRESS_ERROR_BOUND. 0 (the default) means LOSSLESS -- what this
 * adapter did unconditionally before -- and masks NeuroPress's 16 quantize
 * actions; a positive bound makes them reachable. Same name and meaning as in
 * the sibling drivers and as gpucompress_config_t::error_bound upstream.
 *
 * Quantization then runs ON THE DEVICE here, unlike the file-replay route:
 * the chunk handed to the compressor is already device memory, so
 * compressor_runtime.cc takes its QuantizeDevice branch instead of the host
 * fallback. That is the point of the in-situ path.
 */
double ClioNpErrorBoundFromEnv() {
  const char *e = std::getenv("CLIO_NEUROPRESS_ERROR_BOUND");
  if (e == nullptr || *e == '\0') return 0.0;
  const double v = std::atof(e);
  return v > 0.0 ? v : 0.0;
}


/* FNV-1a-64, the digest every sibling example uses, so a CSV written here can
 * be replayed by bin/neuropress_field_replay --readback without translation. */
uint64_t Fnv1a(const void *p, size_t n) {
  const auto *b = static_cast<const unsigned char *>(p);
  uint64_t h = 14695981039346656037ull;
  for (size_t i = 0; i < n; ++i) {
    h ^= b[i];
    h *= 1099511628211ull;
  }
  return h;
}

/** Print a named refusal and leave. Never a return: see the header. */
[[noreturn]] void Refuse(int code, const std::string &msg) {
  std::fflush(stdout);
  std::fprintf(stderr, "\n[clio-nyx-insitu] REFUSING: %s\n", msg.c_str());
  std::fflush(stderr);
  // _exit rather than exit: this process is hosting a Clio runtime with live
  // worker threads, and running static destructors underneath them is a good
  // way to turn a clear refusal into a hang. The message is already flushed.
  _exit(code);
}

std::string EnvStr(const char *name, const char *dflt) {
  const char *e = std::getenv(name);
  return (e && *e) ? std::string(e) : std::string(dflt);
}

size_t EnvSize(const char *name, size_t dflt) {
  const char *e = std::getenv(name);
  if (!e || !*e) return dflt;
  return std::strtoull(e, nullptr, 10);
}

bool EnvBool(const char *name, bool dflt) {
  const char *e = std::getenv(name);
  if (!e || !*e) return dflt;
  return std::atoi(e) != 0;
}

/** One (box, component, chunk), as staged and as the compressor answered. */
struct BlobRecord {
  std::string name;
  size_t bytes = 0;
  uint64_t digest = 0;
  int lib = 0;
  double ratio = 0.0;
  size_t stored = 0;
  double ms = 0.0;
  // MEASURED decompression time (CUDA events around the codec call alone), or
  // <0 when nothing measured one. Needs CLIO_NEUROPRESS_EXPLORE_MEASURE_DT.
  double dt_ms = -1.0;
  bool ok = false;
  // The chunk as SUBMITTED, kept only while a bound check still needs it.
  // A digest cannot answer the lossy question -- |original - decoded| <= eb
  // needs the values, not a hash of them -- and in situ there is no file on
  // disk to re-read them from, which is the whole difference from the replay
  // route. Held for ONE frame and released in VerifyPendingBound(), so the
  // cost is a frame (48 MiB at 128^3, 384 MiB at 256^3) rather than the run.
  std::vector<char> original;
};

struct Pending {
  clio::run::Future<clio::cte::compressor::DynamicScheduleTask> fut;
  size_t record = 0;
};

/**
 * One registered device backend per chunk of a frame, allocated on the first
 * frame and REUSED by every frame after it.
 *
 * Same reasoning as the LAMMPS device path: it bounds device memory at one
 * frame rather than the whole run, and it recycles no AllocatorId mid-run.
 * Reuse is safe because frame_end() waits for every task of frame N before
 * frame N+1 stages anything into the same slots.
 *
 * One backend PER CHUNK rather than one per field addressed by offset: with
 * the runtime out of process, IpcManager::ToFullPtr resolves a registered
 * backend through its own device_ptr and ignores ShmPtr::off_, so every chunk
 * would silently come back as chunk 0 -- and each blob would still round-trip
 * bit-exact against the digest of what was submitted.
 */
struct DeviceSlot {
  ctp::ipc::AllocatorId alloc;
  char *ptr = nullptr;
  size_t bytes = 0;
};

struct Adapter {
  bool started = false;
  bool finished = false;

  // Configuration, read once from the environment in begin().
  std::string tag_name;
  size_t chunk = 4u << 20;
  std::string report_path;
  std::string raw_dir;
  bool verify = false;
  // Lossy verification: check the VALUES against the requested absolute error
  // bound instead of a digest. Mutually exclusive with `verify` in practice --
  // lossy data must fail a digest by construction, so asking for both would
  // report a failure that is correct behaviour.
  bool check_bound = false;
  double error_bound = 0.0;
  size_t bound_cursor = 0;      // records[0, bound_cursor) already checked
  size_t bound_checked = 0;     // elements compared
  size_t bound_violations = 0;  // elements outside the bound
  size_t bound_bad_blobs = 0;   // blobs that failed to read back at all
  double bound_max_err = 0.0;   // largest |original - decoded| seen

  std::unique_ptr<clio::cte::compressor::Client> compressor;
  clio::cte::core::Client *cte = nullptr;
  clio::cte::core::TagId tag_id;

  // The adapter's own stream. Never the legacy default stream.
  cudaStream_t stream = nullptr;

  // Device scratch for the valid-box extraction, grown on demand, one
  // allocation for the whole run once the largest component has been seen.
  void *scratch = nullptr;
  size_t scratch_bytes = 0;

  std::vector<DeviceSlot> slots;
  size_t slot_next = 0;

  std::vector<BlobRecord> records;
  std::vector<Pending> pending;
  std::vector<char> host_mirror;  // digest / --raw only, never on the path

  long long frames = 0;
  long long cur_step = -1;
  double stage_s = 0.0;
  std::chrono::steady_clock::time_point t_start;
  int elem_bytes = 0;  // the width every blob of this run was submitted at

  // MPI. rank/nprocs come from Nyx (ParallelDescriptor), not from an
  // environment variable, so they cannot disagree with the job the
  // simulation actually thinks it is in.
  long long rank = 0;
  long long nprocs = 1;
  int store_lock_fd = -1;  // proof this rank's store belongs to this rank
  std::string store_lock_path;

  /**
   * Which CUDA device the simulation's memory actually lives on.
   *
   * -1 until the first blob is checked, then fixed for the run. It is read
   * from cudaPointerGetAttributes on the pointer Nyx handed over -- NOT
   * assumed to be 0 -- because AllocateAndRegisterGpuBackend takes a gpu_id
   * and registering a backend on a different device from the data is a
   * cross-device copy that would either fail or silently work through
   * peer-to-peer while the log claimed one device. AMReX assigns devices
   * round-robin over the ranks on a node (AMReX_GpuDevice.cpp), so a
   * multi-GPU node reaches this with device != 0 on some ranks; a
   * single-GPU node always reads 0, which is what every measurement so far
   * was taken on.
   */
  int device = -1;

  /** Refuse unless `p` is genuinely device memory, on ONE device. */
  void RequireDevice(const void *p, const char *what) {
    cudaPointerAttributes a{};
    const cudaError_t rc = cudaPointerGetAttributes(&a, p);
    if (rc != cudaSuccess) {
      (void)cudaGetLastError();
      Refuse(CLIO_NYX_EXIT_NOT_DEVICE,
             std::string(what) + ": cudaPointerGetAttributes failed (" +
                 cudaGetErrorString(rc) +
                 "), so this pointer is not confirmed device memory. Staging "
                 "it through host memory instead would turn a GPU-resident "
                 "run into a host one with nothing in the log to show it.");
    }
    if (a.type != cudaMemoryTypeDevice) {
      const char *tn = a.type == cudaMemoryTypeManaged     ? "managed/UVM"
                       : a.type == cudaMemoryTypeHost      ? "pinned host"
                       : a.type == cudaMemoryTypeUnregistered ? "unregistered host"
                                                          : "unknown";
      Refuse(CLIO_NYX_EXIT_NOT_DEVICE,
             std::string(what) + ": memory is " + tn +
                 " (cudaPointerGetAttributes type=" +
                 std::to_string(static_cast<int>(a.type)) +
                 "), not device memory. The compressor would report GPU "
                 "residency for something that is not GPU-resident.");
    }
    if (device < 0) {
      device = a.device;
    } else if (device != a.device) {
      Refuse(CLIO_NYX_EXIT_NOT_DEVICE,
             std::string(what) + ": this blob is on CUDA device " +
                 std::to_string(a.device) + " but earlier blobs of this run "
                 "were on device " + std::to_string(device) +
                 ". The staging backends are registered on one device; "
                 "mixing them would register a backend on a device the data "
                 "is not on.");
    }
  }

  void *Scratch(size_t bytes) {
    if (bytes <= scratch_bytes) return scratch;
    if (scratch) cudaFree(scratch);
    scratch = nullptr;
    scratch_bytes = 0;
    if (cudaMalloc(&scratch, bytes) != cudaSuccess || scratch == nullptr) {
      (void)cudaGetLastError();
      Refuse(CLIO_NYX_EXIT_ALLOC,
             "could not allocate " + std::to_string(bytes) +
                 " B of device scratch for the valid-box extraction.");
    }
    scratch_bytes = bytes;
    return scratch;
  }

  void Drain() {
    for (auto &p : pending) {
      p.fut.Wait();
      auto &r = records[p.record];
      const auto &c = p.fut->context_;
      r.ok = p.fut->GetReturnCode() == 0;
      r.lib = c.compress_lib_;
      r.ratio = c.actual_compression_ratio_;
      r.ms = c.actual_compress_time_ms_;
      r.dt_ms = c.actual_decompress_time_ms_;
      // compress_lib_ == 0 is "stored raw": the codec ran, did not shrink the
      // chunk, and the ORIGINAL bytes went to the tier.
      // actual_compressed_size_ still reports the codec's output, which is
      // not what was stored.
      r.stored = (r.ok && r.lib != 0) ? c.actual_compressed_size_ : r.bytes;
    }
    pending.clear();
    // Every task of this frame has landed, so the blobs are readable and the
    // originals are still held. Check and release them here rather than at
    // end(), which is what bounds the retained bytes to one frame.
    if (check_bound) VerifyPendingBound();
  }

  /**
   * Check every record staged since the last call against the requested
   * absolute error bound, then release the originals it held.
   *
   * Called from Drain(), so it runs once per frame: the originals of frame N
   * are compared and freed before frame N+1 stages anything, which is what
   * keeps the retained bytes at one frame instead of the whole run. A run
   * that never calls frame_end still gets checked, because end() drains too.
   *
   * The comparison is elementwise on the SUBMITTED bytes, at the width the
   * run was submitted at -- not a digest. A lossy round trip is expected to
   * change the bytes; what it must not do is move any value further than
   * `error_bound` from where it started.
   */
  void VerifyPendingBound() {
    for (; bound_cursor < records.size(); ++bound_cursor) {
      auto &r = records[bound_cursor];
      if (r.original.empty()) continue;
      auto buf = CLIO_IPC->AllocateBuffer(r.bytes);
      if (buf.IsNull()) {
        std::cerr << "[clio-nyx-insitu] AllocateBuffer (bound) failed\n";
        ++bound_bad_blobs;
        r.original.clear();
        r.original.shrink_to_fit();
        continue;
      }
      std::memset(buf.ptr_, 0, r.bytes);
      auto get = compressor->AsyncDecompressExplicit(
          clio::run::PoolQuery::Local(), tag_id, r.name, 0, r.bytes, 0,
          buf.shm_.template Cast<void>(), cte->pool_id_);
      get.Wait();
      if (get->GetReturnCode() != 0) {
        std::cerr << "[clio-nyx-insitu]   READ FAILED " << r.name << "\n";
        ++bound_bad_blobs;
      } else if (elem_bytes == 4) {
        CompareBound<float>(r, buf.ptr_);
      } else if (elem_bytes == 8) {
        CompareBound<double>(r, buf.ptr_);
      } else {
        std::cerr << "[clio-nyx-insitu]   cannot bound-check " << r.name
                  << ": element width " << elem_bytes << " B\n";
        ++bound_bad_blobs;
      }
      CLIO_IPC->FreeBuffer(buf);
      r.original.clear();
      r.original.shrink_to_fit();
    }
  }

  /** Elementwise |original - decoded| <= error_bound, at width T. */
  template <typename T>
  void CompareBound(const BlobRecord &r, const char *decoded) {
    const size_t n = r.bytes / sizeof(T);
    const T *a = reinterpret_cast<const T *>(r.original.data());
    const T *b = reinterpret_cast<const T *>(decoded);
    for (size_t i = 0; i < n; ++i) {
      const double d = std::fabs(static_cast<double>(a[i]) -
                                 static_cast<double>(b[i]));
      if (d > bound_max_err) bound_max_err = d;
      if (d > error_bound) ++bound_violations;
    }
    bound_checked += n;
  }

  /** The lossy verdict, in the wording collect.py greps for. */
  bool ReportBound() {
    const bool ok = (bound_violations == 0 && bound_bad_blobs == 0);
    std::cout << "[clio-nyx-insitu] "
              << (ok ? "BOUND OK: " : "BOUND FAILED: ") << bound_checked
              << " element(s) checked against |original - decoded| <= "
              << error_bound << "; max observed error " << bound_max_err
              << "; " << bound_violations << " violation(s)";
    if (bound_bad_blobs) std::cout << "; " << bound_bad_blobs << " unreadable blob(s)";
    std::cout << std::endl;
    return ok;
  }

  bool VerifyRecords() {
    size_t bad = 0;
    for (const auto &r : records) {
      auto buf = CLIO_IPC->AllocateBuffer(r.bytes);
      if (buf.IsNull()) {
        std::cerr << "[clio-nyx-insitu] AllocateBuffer (verify) failed\n";
        return false;
      }
      std::memset(buf.ptr_, 0, r.bytes);
      auto get = compressor->AsyncDecompressExplicit(
          clio::run::PoolQuery::Local(), tag_id, r.name, 0, r.bytes, 0,
          buf.shm_.template Cast<void>(), cte->pool_id_);
      get.Wait();
      const bool ok =
          get->GetReturnCode() == 0 && Fnv1a(buf.ptr_, r.bytes) == r.digest;
      if (!ok) {
        ++bad;
        std::cerr << "[clio-nyx-insitu]   MISMATCH " << r.name
                  << " rc=" << get->GetReturnCode() << "\n";
      }
      CLIO_IPC->FreeBuffer(buf);
    }
    std::cout << "[clio-nyx-insitu] " << (bad == 0 ? "VERIFIED: " : "FAILED: ")
              << (records.size() - bad) << " of " << records.size()
              << " blobs round-tripped bit-exact through the decompressor"
              << std::endl;
    return bad == 0;
  }
};

Adapter &A() {
  static Adapter a;
  return a;
}

/** dirname(), without pulling in <filesystem> for one call. */
std::string DirOf(const std::string &path) {
  const size_t s = path.find_last_of('/');
  if (s == std::string::npos) return std::string(".");
  if (s == 0) return std::string("/");
  return path.substr(0, s);
}

/** "<stem>.rank0003<ext>" -- so N ranks do not overwrite one file. */
std::string RankQualify(const std::string &path, long long rank) {
  if (path.empty()) return path;
  char suffix[32];
  std::snprintf(suffix, sizeof(suffix), ".rank%04lld", rank);
  const size_t slash = path.find_last_of('/');
  const size_t dot = path.find_last_of('.');
  if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
    return path + suffix;
  }
  return path.substr(0, dot) + suffix + path.substr(dot);
}

/**
 * Refuse unless this rank's CLIO_SERVER_CONF directory belongs to this rank
 * alone.
 *
 * Every Clio runtime binds a TCP port and writes its bdev, tier and metadata
 * log at paths the compose file names. Point two ranks at the same compose
 * file and the second one's runtime cannot bind -- which, as the sibling
 * READMEs record, produces an EMPTY LOG rather than an error, so the failure
 * looks like the adapter silently doing nothing. An O_EXCL lock file named
 * for the rank turns that into a named refusal at the point of the mistake.
 *
 * The lock is per (store directory, rank), so a re-run of the SAME rank is
 * fine (it reopens its own lock); a different rank arriving in the same
 * directory is not.
 */
void ClaimStoreForRank(Adapter &a) {
  const std::string conf = EnvStr("CLIO_SERVER_CONF", "");
  if (conf.empty()) return;  // nothing to claim; begin() will fail anyway
  const std::string dir = DirOf(conf);
  char mine[64];
  std::snprintf(mine, sizeof(mine), "/.clio_nyx_rank%04lld.lock", a.rank);
  const std::string lock_path = dir + mine;

  // Any OTHER rank's lock in this directory means the launcher handed two
  // ranks the same store.
  for (long long r = 0; r < a.nprocs; ++r) {
    if (r == a.rank) continue;
    char other[64];
    std::snprintf(other, sizeof(other), "/.clio_nyx_rank%04lld.lock", r);
    if (::access((dir + other).c_str(), F_OK) == 0) {
      Refuse(CLIO_NYX_EXIT_TOPOLOGY,
             "rank " + std::to_string(a.rank) + " was given the store '" + dir +
                 "', which rank " + std::to_string(r) +
                 " of this job is already using. Every rank hosts its own "
                 "Clio runtime, and a runtime binds a TCP port and owns its "
                 "bdev, tier and metadata log; two ranks sharing them means "
                 "one of the two silently stores nothing. Give each rank its "
                 "own --store directory and port.");
    }
  }
  a.store_lock_fd = ::open(lock_path.c_str(), O_CREAT | O_RDWR, 0644);
  if (a.store_lock_fd >= 0) {
    char pid[64];
    const int n = std::snprintf(pid, sizeof(pid), "%d\n", (int)getpid());
    (void)!::write(a.store_lock_fd, pid, static_cast<size_t>(n));
    a.store_lock_path = lock_path;
  }
}

int BeginImpl(long long rank, long long nprocs) {
  Adapter &a = A();
  if (a.started) return 0;

  a.rank = rank;
  a.nprocs = nprocs;
  a.tag_name = EnvStr("CLIO_NYX_TAG", "nyx_insitu");
  a.chunk = EnvSize("CLIO_NYX_CHUNK", 4u << 20);
  a.report_path = EnvStr("CLIO_NYX_REPORT", "");
  a.raw_dir = EnvStr("CLIO_NYX_RAW_DIR", "");
  a.verify = EnvBool("CLIO_NYX_VERIFY", false);
  a.check_bound = EnvBool("CLIO_NYX_CHECK_BOUND", false);
  a.error_bound = ClioNpErrorBoundFromEnv();
  if (a.check_bound && a.error_bound <= 0.0) {
    // A bound check at eb=0 is a digest check wearing the wrong name, and it
    // would report "BOUND OK" for a lossless run without having tested
    // anything the digest path does not already cover. Say so and fall back.
    std::cerr << "[clio-nyx-insitu] CLIO_NYX_CHECK_BOUND set but the error "
                 "bound is 0 (lossless) -- checking digests instead.\n";
    a.check_bound = false;
    a.verify = true;
  }
  if (a.check_bound && a.verify) {
    // Lossy data cannot match a digest; asking for both would report a
    // failure that is correct behaviour.
    std::cerr << "[clio-nyx-insitu] both CLIO_NYX_VERIFY and "
                 "CLIO_NYX_CHECK_BOUND set; a lossy round trip cannot match a "
                 "digest, so the bound check wins.\n";
    a.verify = false;
  }
  a.t_start = std::chrono::steady_clock::now();
  if (a.nprocs > 1) {
    // Per-rank CSV. The blob NAMES stay unqualified on purpose -- see the
    // header -- but two ranks writing one CSV would lose half of it.
    a.report_path = RankQualify(a.report_path, a.rank);
    ClaimStoreForRank(a);
  }

  // Same call the HDF5 VOL makes lazily on first use. With CLIO_WITH_RUNTIME=1
  // the runtime comes up HERE, inside the Nyx process, composed from
  // CLIO_SERVER_CONF. That is load-bearing: a compressor task sent to a
  // runtime in another process arrives with a device pointer that means
  // nothing there unless it travels as a CUDA IPC handle.
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::cerr << "[clio-nyx-insitu] rank " << a.rank
              << ": CLIO_CTE_CLIENT_INIT failed -- is "
                 "CLIO_SERVER_CONF set and CLIO_WITH_RUNTIME=1?\n";
    return 1;
  }
  a.cte = CLIO_CTE_CLIENT;

  // The runtime MUST be in this process.
  //
  // Not a stylistic preference and not only about device pointers: the
  // compressor chimod's tasks declare their wire format as
  // SerializeStart/SerializeEnd (compressor_tasks.h), and no archive calls
  // those names -- SaveTaskArchive calls SerializeIn/SerializeOut
  // (task_archives.h). Name lookup therefore finds the inherited
  // clio::run::Task::SerializeIn, which carries routing fields only, so a
  // DynamicSchedule task sent to a runtime in another process arrives with
  // blob_name="" and size=0 and stores nothing. Measured, on host memory as
  // much as on device memory -- see the README's "Q5: MPI". In-process the
  // task object is handed over by pointer and no archive ever runs, which is
  // why this has never bitten a run before.
  //
  // Refusing here rather than staging through host memory is the whole point:
  // the alternative failure is a run that reports success and stores nothing.
  if (!CLIO_RUNTIME_MANAGER->IsRuntime()) {
    Refuse(CLIO_NYX_EXIT_TOPOLOGY,
           "rank " + std::to_string(a.rank) +
               ": the Clio runtime is NOT hosted in this process "
               "(CLIO_WITH_RUNTIME is not 1). The compressor's tasks have no "
               "cross-process wire format -- every DynamicSchedule would "
               "arrive at the runtime with blob_name=\"\" and size=0 and "
               "store nothing, while this process reported success. Host the "
               "runtime in the rank (CLIO_WITH_RUNTIME=1, one store and one "
               "port per rank).");
  }

  unsigned major = 512, minor = 0;
  const std::string pool = EnvStr("CLIO_NYX_POOL", "512.0");
  if (std::sscanf(pool.c_str(), "%u.%u", &major, &minor) < 1) {
    std::cerr << "[clio-nyx-insitu] bad CLIO_NYX_POOL '" << pool << "'\n";
    return 1;
  }
  a.compressor = std::make_unique<clio::cte::compressor::Client>(
      clio::run::PoolId(major, minor));

  auto tag = a.cte->AsyncGetOrCreateTag(a.tag_name);
  tag.Wait();
  if (tag->GetReturnCode() != 0) {
    std::cerr << "[clio-nyx-insitu] GetOrCreateTag('" << a.tag_name
              << "') failed\n";
    return 1;
  }
  a.tag_id = tag->tag_id_;

  if (cudaStreamCreateWithFlags(&a.stream, cudaStreamNonBlocking) !=
      cudaSuccess) {
    (void)cudaGetLastError();
    std::cerr << "[clio-nyx-insitu] cudaStreamCreateWithFlags failed\n";
    return 1;
  }

  std::cout << "[clio-nyx-insitu] rank " << a.rank << "/" << a.nprocs
            << " up: tag='" << a.tag_name << "' pool=" << major << "." << minor
            << " chunk="
            << (a.chunk ? std::to_string(a.chunk) : std::string("whole field"))
            << (a.verify ? " verify=on" : "") << " pid=" << getpid()
            << std::endl;
  a.started = true;
  return 0;
}

}  // namespace

extern "C" {

int clio_nyx_insitu_begin(void) { return BeginImpl(0, 1); }

int clio_nyx_insitu_begin_mpi(long long rank, long long nprocs) {
  if (nprocs < 1 || rank < 0 || rank >= nprocs) {
    Refuse(CLIO_NYX_EXIT_PRECONDITION,
           "clio_nyx_insitu_begin_mpi(rank=" + std::to_string(rank) +
               ", nprocs=" + std::to_string(nprocs) + "): not a valid rank.");
  }
  return BeginImpl(rank, nprocs);
}

int clio_nyx_insitu_frame_begin(long long step, double sim_time) {
  Adapter &a = A();
  if (!a.started) return 1;
  (void)sim_time;
  a.cur_step = step;
  a.slot_next = 0;  // frame N reuses frame N-1's slots; Drain() has waited
  ++a.frames;
  return 0;
}

int clio_nyx_insitu_stage(const char *blob_prefix, const void *dev_base,
                          long long elem_bytes, long long gx, long long gy,
                          long long gz, long long ox, long long oy,
                          long long oz, long long nx, long long ny,
                          long long nz) {
  Adapter &a = A();
  if (!a.started) return 1;
  if (blob_prefix == nullptr || dev_base == nullptr) return 1;
  if (elem_bytes != 4 && elem_bytes != 8) {
    Refuse(CLIO_NYX_EXIT_PRECONDITION,
           "element width is " + std::to_string(elem_bytes) +
               " bytes; only float32 and float64 have a NeuroPress data_type_.");
  }
  if (nx <= 0 || ny <= 0 || nz <= 0 || ox < 0 || oy < 0 || oz < 0 ||
      ox + nx > gx || oy + ny > gy || oz + nz > gz) {
    Refuse(CLIO_NYX_EXIT_PRECONDITION,
           "region does not fit inside the FAB: grown=" + std::to_string(gx) +
               "x" + std::to_string(gy) + "x" + std::to_string(gz) +
               " region origin=" + std::to_string(ox) + "," +
               std::to_string(oy) + "," + std::to_string(oz) + " size=" +
               std::to_string(nx) + "x" + std::to_string(ny) + "x" +
               std::to_string(nz));
  }
  if (a.elem_bytes == 0) {
    a.elem_bytes = static_cast<int>(elem_bytes);
  } else if (a.elem_bytes != elem_bytes) {
    Refuse(CLIO_NYX_EXIT_PRECONDITION,
           "element width changed mid-run (was " +
               std::to_string(a.elem_bytes) + ", now " +
               std::to_string(elem_bytes) + ")");
  }

  const auto t0 = std::chrono::steady_clock::now();

  // (1) The pointer Nyx handed over really is device memory. Checked here, on
  //     every blob, rather than inferred from the build's configuration:
  //     AMReX's arena is a runtime decision (amrex.the_arena_is_managed) and
  //     a managed pointer would work perfectly while making every claim of
  //     GPU residency in the log false.
  a.RequireDevice(dev_base, blob_prefix);

  const size_t field_bytes =
      static_cast<size_t>(nx) * ny * nz * static_cast<size_t>(elem_bytes);

  // (2) Extract the region into contiguous device scratch. One
  //     cudaMemcpy3DAsync on the adapter's own stream -- a strided D2D copy,
  //     no kernel, no host bounce. When the region IS the whole FAB this is
  //     still just a linear D2D copy.
  char *scratch = static_cast<char *>(a.Scratch(field_bytes));
  {
    cudaMemcpy3DParms p{};
    p.srcPtr = make_cudaPitchedPtr(const_cast<void *>(dev_base),
                                   static_cast<size_t>(gx) * elem_bytes,
                                   static_cast<size_t>(gx) * elem_bytes,
                                   static_cast<size_t>(gy));
    p.srcPos = make_cudaPos(static_cast<size_t>(ox) * elem_bytes,
                            static_cast<size_t>(oy), static_cast<size_t>(oz));
    p.dstPtr = make_cudaPitchedPtr(scratch, static_cast<size_t>(nx) * elem_bytes,
                                   static_cast<size_t>(nx) * elem_bytes,
                                   static_cast<size_t>(ny));
    p.dstPos = make_cudaPos(0, 0, 0);
    p.extent = make_cudaExtent(static_cast<size_t>(nx) * elem_bytes,
                               static_cast<size_t>(ny), static_cast<size_t>(nz));
    p.kind = cudaMemcpyDeviceToDevice;
    const cudaError_t rc = cudaMemcpy3DAsync(&p, a.stream);
    if (rc != cudaSuccess) {
      (void)cudaGetLastError();
      Refuse(CLIO_NYX_EXIT_ALLOC, std::string("cudaMemcpy3DAsync failed: ") +
                                      cudaGetErrorString(rc));
    }
    // The chunk copies below read this buffer, and the compressor reads what
    // they write from another thread and Clio's own streams. The extraction
    // has to be COMPLETE here, not merely issued.
    cudaStreamSynchronize(a.stream);
  }

  const size_t chunk = a.chunk ? a.chunk : field_bytes;
  const size_t nchunks = (field_bytes + chunk - 1) / chunk;

  clio::cte::core::Context ctx;
  // float32 = 1, float64 = 2 -- the same encoding the HDF5 VOL assigns from an
  // H5T_FLOAT's size. error_bound_ is 0 (lossless) unless
  // CLIO_NEUROPRESS_ERROR_BOUND asks otherwise.
  ctx.data_type_ = (elem_bytes == 8) ? 2 : 1;
  ctx.error_bound_ = ClioNpErrorBoundFromEnv();

  const bool need_host_bytes =
      true;  // the digest is what makes the cold read-back checkable

  for (size_t ci = 0; ci < nchunks; ++ci) {
    const size_t off = ci * chunk;
    const size_t n = std::min(chunk, field_bytes - off);

    // (3) One registered kDeviceMem backend per chunk, reused across frames.
    if (a.slot_next >= a.slots.size()) {
      DeviceSlot slot;
      slot.alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(
          /*gpu_id=*/static_cast<clio::run::u32>(a.device),
          clio::run::gpu::IpcManager::MemKind::kDeviceMem, n,
          &slot.ptr);
      if (slot.alloc.IsNull()) {
        Refuse(CLIO_NYX_EXIT_ALLOC,
               "AllocateAndRegisterGpuBackend(" + std::to_string(n) +
                   ") failed. Staging this chunk through host memory instead "
                   "would silently turn a GPU-resident run into a host one.");
      }
      slot.bytes = n;
      a.slots.push_back(slot);
    }
    DeviceSlot &slot = a.slots[a.slot_next++];
    if (slot.bytes < n) {
      Refuse(CLIO_NYX_EXIT_ALLOC,
             "staging slot is " + std::to_string(slot.bytes) +
                 " B but this chunk is " + std::to_string(n) +
                 " B; the chunk layout changed between frames, which the slot "
                 "pool cannot express. (A regrid would do this.)");
    }

    // D2D into the registered backend, on the adapter's stream, synchronised
    // before the pointer is published. cudaMemcpyAsync + an explicit sync
    // rather than ctp::GpuApi::Memcpy, which is a legacy-default-stream
    // cudaMemcpy and is NOT host-synchronous for D2D.
    const cudaError_t rc =
        cudaMemcpyAsync(slot.ptr, scratch + off, n, cudaMemcpyDeviceToDevice,
                        a.stream);
    if (rc != cudaSuccess) {
      (void)cudaGetLastError();
      Refuse(CLIO_NYX_EXIT_ALLOC,
             std::string("cudaMemcpyAsync (chunk stage) failed: ") +
                 cudaGetErrorString(rc));
    }
    cudaStreamSynchronize(a.stream);

    // (4) What Clio registered for us is device memory too. If this ever
    //     fails the compressor would receive host bytes while the run
    //     reported GPU residency.
    if (!ctp::IsDevicePointer(slot.ptr)) {
      Refuse(CLIO_NYX_EXIT_NOT_DEVICE,
             "the Clio-registered staging buffer is not device memory.");
    }

    ctp::ipc::ShmPtr<> blob_data;
    blob_data.alloc_id_ = slot.alloc;
    blob_data.off_ = reinterpret_cast<clio::run::u64>(slot.ptr);

    BlobRecord rec;
    rec.name = std::string(blob_prefix) + "/chunk_" + std::to_string(ci);
    rec.bytes = n;

    if (need_host_bytes) {
      // Instrumentation only: nothing on the compressor path reads these
      // bytes. Read from the SCRATCH (fenced above) rather than from the
      // registered buffer, so the digest cannot be taken from the destination
      // of a copy that has not landed.
      a.host_mirror.resize(n);
      cudaMemcpyAsync(a.host_mirror.data(), scratch + off, n,
                      cudaMemcpyDeviceToHost, a.stream);
      cudaStreamSynchronize(a.stream);
      rec.digest = Fnv1a(a.host_mirror.data(), n);
      // The bound check needs the VALUES back, and this host copy is the only
      // place they exist outside device memory the simulation is about to
      // overwrite. Released one frame later by VerifyPendingBound().
      if (a.check_bound) rec.original.assign(a.host_mirror.begin(),
                                             a.host_mirror.end());
      if (!a.raw_dir.empty()) {
        std::string fn = rec.name;
        for (auto &ch : fn) {
          if (ch == '/') ch = '_';
        }
        std::ofstream(a.raw_dir + "/" + fn + ".bin", std::ios::binary)
            .write(a.host_mirror.data(), static_cast<std::streamsize>(n));
      }
    }

    a.records.push_back(rec);

    Pending p;
    p.fut = a.compressor->AsyncDynamicSchedule(
        clio::run::PoolQuery::Local(), a.tag_id, rec.name, /*offset=*/0, n,
        blob_data, /*score=*/-1.0f, ctx, /*flags=*/0, a.cte->pool_id_);
    p.record = a.records.size() - 1;
    a.pending.push_back(std::move(p));
  }

  a.stage_s +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  return 0;
}

int clio_nyx_insitu_frame_end(void) {
  Adapter &a = A();
  if (!a.started) return 1;
  const auto t0 = std::chrono::steady_clock::now();
  a.Drain();
  a.stage_s +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  return 0;
}

int clio_nyx_insitu_end(void) {
  Adapter &a = A();
  if (!a.started || a.finished) return 0;
  a.finished = true;
  a.Drain();

  size_t in_total = 0, stored_total = 0, kept = 0, raw = 0, failed = 0;
  std::map<std::string, std::pair<size_t, size_t>> per_field;
  std::map<int, size_t> per_lib;
  for (const auto &r : a.records) {
    if (!r.ok) {
      ++failed;
      continue;
    }
    in_total += r.bytes;
    stored_total += r.stored;
    if (r.lib != 0) {
      ++kept;
      ++per_lib[r.lib];
    } else {
      ++raw;
    }
    auto &pf = per_field[r.name.substr(0, r.name.find('/'))];
    pf.first += r.bytes;
    pf.second += r.stored;
  }

  std::cout << std::fixed << std::setprecision(3)
            << "\n[clio-nyx-insitu] rank " << a.rank << "/" << a.nprocs
            << " stored " << a.records.size() << " blob(s) "
            << "from " << a.frames << " frame(s), " << in_total << " B in -> "
            << stored_total << " B on the tier  (stored ratio "
            << (stored_total ? double(in_total) / double(stored_total) : 0.0)
            << ")\n  compressed: " << kept << "   stored raw: " << raw
            << "   failed: " << failed << "\n";
  for (const auto &[name, io] : per_field) {
    std::cout << "  " << std::setw(10) << name << "  " << io.first << " -> "
              << io.second << "  ("
              << (io.second ? double(io.first) / io.second : 0.0) << "x)\n";
  }
  for (const auto &[lib, n] : per_lib) {
    std::cout << "  codec " << std::setw(16)
              << (lib == 0 ? std::string("raw(not-beneficial)")
                           : ctp::CompressionFactory::NameForWireId(lib))
              << " : " << n
              << " chunk(s)\n";
  }
  std::cout << "  stage+compress(wait) " << a.stage_s << " s   in-situ wall "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             a.t_start)
                   .count()
            << " s" << std::endl;

  if (!a.report_path.empty()) {
    // Exactly ../neuropress_field_replay's CSV, so `neuropress_field_replay
    // --readback` can do the cold read from a separate process without
    // knowing anything about Nyx.
    std::ofstream csv(a.report_path);
    // The first three columns and their order are ../neuropress_field_replay's
    // CSV, which is what lets `neuropress_field_replay --readback` do the cold
    // read from a separate process without knowing anything about Nyx (its
    // parser reads blob,bytes,fnv1a64 and ignores the rest). `rank` is
    // APPENDED, never inserted, for exactly that reason: it tells a reader
    // which rank owned a box without moving a column the reader depends on.
    // TWO ratios, because they answer different questions and disagreed
    // silently before this column existed:
    //   ratio        = bytes / CODEC PAYLOAD. Upstream's definition
    //                  (gpucompress_compress.cpp excludes its own header),
    //                  the NN's training label, and the number explore.csv
    //                  and selection.csv report. Do not change it: it feeds
    //                  the cost model and the SGD targets.
    //   stored_ratio = bytes / STORED. What the tier actually holds,
    //                  header included, and the basis of the aggregate
    //                  "(stored ratio N)" line printed above.
    // They differ by the 24-byte CTEC header (56 when quantized), so
    // sum(bytes)/sum(stored) never equals the mean of the `ratio` column.
    // Appended, never inserted: the first three columns are the cold
    // reader's contract.
    csv << "blob,bytes,fnv1a64,lib,codec,ratio,stored,compress_ms,"
           "decompress_ms,rc,rank,stored_ratio\n";
    for (const auto &r : a.records) {
      csv << r.name << ',' << r.bytes << ',' << std::hex << r.digest << std::dec
          << ',' << r.lib << ','
          << (r.lib == 0 ? std::string("raw(not-beneficial)")
                         : ctp::CompressionFactory::NameForWireId(r.lib))
          << ',' << r.ratio
          << ',' << r.stored << ',' << r.ms << ',' << r.dt_ms << ',' << (r.ok ? 0 : 1) << ','
          << a.rank << ','
          << (r.stored ? double(r.bytes) / double(r.stored) : 0.0) << '\n';
    }
  }

  int rc = failed ? 1 : 0;
  if (a.verify && !a.VerifyRecords()) rc = 1;
  if (a.check_bound && !a.ReportBound()) rc = 1;

  for (auto &s : a.slots) {
    if (!s.alloc.IsNull()) {
      CLIO_IPC->FreeGpuBackend(static_cast<clio::run::u32>(a.device), s.alloc);
    }
  }
  a.slots.clear();
  if (a.scratch) {
    cudaFree(a.scratch);
    a.scratch = nullptr;
    a.scratch_bytes = 0;
  }
  if (a.store_lock_fd >= 0) {
    // Removed on a CLEAN exit only. A crash leaves it behind on purpose: the
    // next run that puts a different rank in this directory should still be
    // told, and a run that puts the SAME rank back sees only its own lock.
    ::close(a.store_lock_fd);
    a.store_lock_fd = -1;
    if (!a.store_lock_path.empty()) ::unlink(a.store_lock_path.c_str());
  }
  if (a.stream) {
    cudaStreamDestroy(a.stream);
    a.stream = nullptr;
  }
  std::cout.flush();
  return rc;
}

}  // extern "C"
