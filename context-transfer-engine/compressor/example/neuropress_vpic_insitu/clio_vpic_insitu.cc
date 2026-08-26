/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file clio_vpic_insitu.cc
 * @brief VPIC -> Clio -> NeuroPress, in situ: the deck hands over its
 * device-resident field arrays and no file is written.
 *
 * The offline sibling (paper-benchmark/vpic) dumps 16 flat .f32 files per
 * frame and replays them later. This is the same data on the same GPU without
 * the file: the deck calls stage() from begin_diagnostics with the Kokkos
 * view's device pointer, the bytes go D2D into a Clio-registered backend, and
 * the compressor picks a codec per chunk exactly as it does on the replay
 * path.
 *
 * WHY THIS ADAPTER IS SIMPLER THAN THE NYX ONE. Nyx hands over a FAB carrying
 * ghost cells, so its adapter must extract the valid box with a strided
 * cudaMemcpy3DAsync before anything is contiguous. VPIC's field view is
 * LayoutLeft in a CUDA build (measured: stride0=1, stride1=n_voxels), so each
 * of the 16 variables is already a contiguous device array and stage() takes
 * a plain (pointer, count). The only copy is the one into the registered
 * staging backend, which exists because a ShmPtr carries an AllocatorId and
 * only Clio's own registration produces one.
 *
 * Everything else -- the in-process runtime requirement, per-rank stores, the
 * device-pointer checks, the digest CSV that bin/neuropress_field_replay
 * --readback can cold-read -- is deliberately identical to the Nyx adapter,
 * because the two are meant to be compared.
 */

#include "clio_vpic_insitu.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

#include <clio_cte/compressor/compressor_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_ctp/compress/compress_factory.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_runtime/clio_runtime.h>

namespace {

/* FNV-1a-64, the digest every sibling example uses, so the CSV written here
 * replays through bin/neuropress_field_replay --readback unchanged. */
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
  std::fprintf(stderr, "\n[clio-vpic-insitu] REFUSING: %s\n", msg.c_str());
  std::fflush(stderr);
  // _exit rather than exit: this process hosts a Clio runtime with live worker
  // threads, and running static destructors underneath them turns a clear
  // refusal into a hang.
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

/** One chunk, as staged and as the compressor answered. */
struct BlobRecord {
  std::string name;
  size_t bytes = 0;
  uint64_t digest = 0;
  int lib = 0;
  double ratio = 0.0;
  size_t stored = 0;
  double ms = 0.0;
  bool ok = false;
};

struct Pending {
  clio::run::Future<clio::cte::compressor::DynamicScheduleTask> fut;
  size_t record = 0;
};

/**
 * One registered device backend per chunk of a frame, allocated on the first
 * frame and reused by every frame after it.
 *
 * Bounds device memory at one frame rather than the whole run, and recycles no
 * AllocatorId mid-run. Reuse is safe because frame_end() waits for every task
 * of frame N before frame N+1 stages into the same slots. One backend PER
 * CHUNK rather than one per field addressed by offset, for the reason the Nyx
 * adapter records: a registered backend resolves through its own device_ptr.
 */
struct DeviceSlot {
  ctp::ipc::AllocatorId alloc;
  char *ptr = nullptr;
  size_t bytes = 0;
};

struct Adapter {
  bool started = false;
  bool finished = false;

  std::string tag_name;
  size_t chunk = 4u << 20;
  std::string report_path;
  std::string raw_dir;
  bool verify = false;

  std::unique_ptr<clio::cte::compressor::Client> compressor;
  clio::cte::core::Client *cte = nullptr;
  clio::cte::core::TagId tag_id;

  /** The adapter's own stream. Never the legacy default stream. */
  cudaStream_t stream = nullptr;

  std::vector<DeviceSlot> slots;
  size_t slot_next = 0;

  std::vector<BlobRecord> records;
  std::vector<Pending> pending;
  std::vector<char> host_mirror;  // digest / --raw only, never on the path

  long long frames = 0;
  long long cur_step = -1;
  double stage_s = 0.0;
  std::chrono::steady_clock::time_point t_start;
  int elem_bytes = 0;

  long long rank = 0;
  long long nprocs = 1;
  int store_lock_fd = -1;
  std::string store_lock_path;

  /**
   * Which CUDA device the simulation's memory actually lives on. -1 until the
   * first blob is checked, then fixed for the run. Read from
   * cudaPointerGetAttributes rather than assumed to be 0: Kokkos assigns a
   * device per rank on a multi-GPU node, and registering a staging backend on
   * a different device from the data would either fail or quietly work
   * through peer-to-peer while the log claimed one device.
   */
  int device = -1;

  /** Refuse unless `p` is genuinely device memory, on ONE device. */
  void RequireDevice(const void *p, const char *what) {
    cudaPointerAttributes attr{};
    const cudaError_t rc = cudaPointerGetAttributes(&attr, p);
    if (rc != cudaSuccess) {
      (void)cudaGetLastError();
      Refuse(CLIO_VPIC_EXIT_NOT_DEVICE,
             std::string(what) + ": cudaPointerGetAttributes failed (" +
                 cudaGetErrorString(rc) +
                 "), so this pointer is not confirmed device memory. Staging "
                 "it through host memory instead would turn a GPU-resident "
                 "run into a host one with nothing in the log to show it.");
    }
    if (attr.type != cudaMemoryTypeDevice) {
      const char *tn = attr.type == cudaMemoryTypeManaged ? "managed/UVM"
                       : attr.type == cudaMemoryTypeHost  ? "pinned host"
                       : attr.type == cudaMemoryTypeUnregistered
                           ? "unregistered host"
                           : "unknown";
      Refuse(CLIO_VPIC_EXIT_NOT_DEVICE,
             std::string(what) + ": memory is " + tn +
                 ", not device memory. A VPIC built for the host (or with "
                 "Kokkos in a host execution space) reaches here, and its "
                 "field view is LayoutRight and interleaved as well -- the "
                 "bytes would be wrong, not merely host-resident.");
    }
    if (device < 0) {
      device = attr.device;
    } else if (device != attr.device) {
      Refuse(CLIO_VPIC_EXIT_NOT_DEVICE,
             std::string(what) + ": this blob is on CUDA device " +
                 std::to_string(attr.device) + " but earlier blobs of this run "
                 "were on device " + std::to_string(device) + ".");
    }
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
      // compress_lib_ == 0 is "stored raw": the codec ran, did not shrink the
      // chunk, and the ORIGINAL bytes went to the tier.
      r.stored = (r.ok && r.lib != 0) ? c.actual_compressed_size_ : r.bytes;
    }
    pending.clear();
  }

  bool VerifyRecords() {
    size_t bad = 0;
    for (const auto &r : records) {
      auto buf = CLIO_IPC->AllocateBuffer(r.bytes);
      if (buf.IsNull()) {
        std::cerr << "[clio-vpic-insitu] AllocateBuffer (verify) failed\n";
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
        std::cerr << "[clio-vpic-insitu]   MISMATCH " << r.name
                  << " rc=" << get->GetReturnCode() << "\n";
      }
      CLIO_IPC->FreeBuffer(buf);
    }
    std::cout << "[clio-vpic-insitu] " << (bad == 0 ? "VERIFIED: " : "FAILED: ")
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
 * Refuse unless this rank's CLIO_SERVER_CONF directory belongs to it alone.
 *
 * A Clio runtime binds a TCP port and owns the bdev, tier and metadata log its
 * compose file names. Two ranks pointed at one store means the second one's
 * runtime cannot bind, which produces an empty log rather than an error -- the
 * failure then looks like the adapter doing nothing. An O_EXCL-style lock file
 * per rank turns that into a named refusal where the mistake was made.
 */
void ClaimStoreForRank(Adapter &a) {
  const std::string conf = EnvStr("CLIO_SERVER_CONF", "");
  if (conf.empty()) return;  // nothing to claim; begin() will fail anyway
  const std::string dir = DirOf(conf);
  char mine[64];
  std::snprintf(mine, sizeof(mine), "/.clio_vpic_rank%04lld.lock", a.rank);
  const std::string lock_path = dir + mine;

  for (long long r = 0; r < a.nprocs; ++r) {
    if (r == a.rank) continue;
    char other[64];
    std::snprintf(other, sizeof(other), "/.clio_vpic_rank%04lld.lock", r);
    if (::access((dir + other).c_str(), F_OK) == 0) {
      Refuse(CLIO_VPIC_EXIT_TOPOLOGY,
             "rank " + std::to_string(a.rank) + " was given the store '" + dir +
                 "', which rank " + std::to_string(r) +
                 " of this job is already using. Every rank hosts its own Clio "
                 "runtime, and a runtime binds a port and owns its bdev, tier "
                 "and metadata log; two ranks sharing them means one of the "
                 "two silently stores nothing. Give each rank its own store "
                 "directory and port.");
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
  a.tag_name = EnvStr("CLIO_VPIC_TAG", "vpic_insitu");
  a.chunk = EnvSize("CLIO_VPIC_CHUNK", 4u << 20);
  a.report_path = EnvStr("CLIO_VPIC_REPORT", "");
  a.raw_dir = EnvStr("CLIO_VPIC_RAW_DIR", "");
  a.verify = EnvBool("CLIO_VPIC_VERIFY", false);
  a.t_start = std::chrono::steady_clock::now();
  if (a.nprocs > 1) {
    // Per-rank CSV. Blob NAMES stay unqualified: VPIC gives every rank the
    // same field names for its own subdomain, and the stores are separate.
    a.report_path = RankQualify(a.report_path, a.rank);
    ClaimStoreForRank(a);
  }

  // With CLIO_WITH_RUNTIME=1 the runtime comes up HERE, inside the VPIC
  // process, composed from CLIO_SERVER_CONF.
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::cerr << "[clio-vpic-insitu] rank " << a.rank
              << ": CLIO_CTE_CLIENT_INIT failed -- is CLIO_SERVER_CONF set and "
                 "CLIO_WITH_RUNTIME=1?\n";
    return 1;
  }
  a.cte = CLIO_CTE_CLIENT;

  // The runtime MUST be in this process. The compressor chimod's tasks have no
  // cross-process wire format (the Nyx README's Q5 records the measurement): a
  // DynamicSchedule sent to a runtime elsewhere arrives with blob_name="" and
  // size=0 and stores nothing while this process reports success.
  if (!CLIO_RUNTIME_MANAGER->IsRuntime()) {
    Refuse(CLIO_VPIC_EXIT_TOPOLOGY,
           "rank " + std::to_string(a.rank) +
               ": the Clio runtime is NOT hosted in this process "
               "(CLIO_WITH_RUNTIME is not 1). Every DynamicSchedule would "
               "arrive at the runtime empty and store nothing while this run "
               "reported success. Host the runtime in the rank "
               "(CLIO_WITH_RUNTIME=1, one store and one port per rank).");
  }

  unsigned major = 512, minor = 0;
  const std::string pool = EnvStr("CLIO_VPIC_POOL", "512.0");
  if (std::sscanf(pool.c_str(), "%u.%u", &major, &minor) < 1) {
    std::cerr << "[clio-vpic-insitu] bad CLIO_VPIC_POOL '" << pool << "'\n";
    return 1;
  }
  a.compressor = std::make_unique<clio::cte::compressor::Client>(
      clio::run::PoolId(major, minor));

  auto tag = a.cte->AsyncGetOrCreateTag(a.tag_name);
  tag.Wait();
  if (tag->GetReturnCode() != 0) {
    std::cerr << "[clio-vpic-insitu] GetOrCreateTag('" << a.tag_name
              << "') failed\n";
    return 1;
  }
  a.tag_id = tag->tag_id_;

  if (cudaStreamCreateWithFlags(&a.stream, cudaStreamNonBlocking) !=
      cudaSuccess) {
    (void)cudaGetLastError();
    std::cerr << "[clio-vpic-insitu] cudaStreamCreateWithFlags failed\n";
    return 1;
  }

  std::cout << "[clio-vpic-insitu] rank " << a.rank << "/" << a.nprocs
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

int clio_vpic_insitu_begin(void) { return BeginImpl(0, 1); }

int clio_vpic_insitu_begin_mpi(long long rank, long long nprocs) {
  if (nprocs < 1 || rank < 0 || rank >= nprocs) {
    Refuse(CLIO_VPIC_EXIT_PRECONDITION,
           "begin_mpi(rank=" + std::to_string(rank) +
               ", nprocs=" + std::to_string(nprocs) + ") is not a valid rank.");
  }
  return BeginImpl(rank, nprocs);
}

int clio_vpic_insitu_frame_begin(long long step, double sim_time) {
  Adapter &a = A();
  (void)sim_time;
  if (!a.started) return 1;
  a.cur_step = step;
  a.slot_next = 0;
  ++a.frames;
  return 0;
}

int clio_vpic_insitu_stage(const char *blob_prefix, const void *dev_ptr,
                           long long elem_bytes, long long n_elems) {
  Adapter &a = A();
  if (!a.started) return 1;
  if (blob_prefix == nullptr || dev_ptr == nullptr) return 1;
  if (elem_bytes != 4 && elem_bytes != 8) {
    Refuse(CLIO_VPIC_EXIT_PRECONDITION,
           "element width is " + std::to_string(elem_bytes) +
               " bytes; only float32 and float64 have a NeuroPress data_type_.");
  }
  if (n_elems <= 0) {
    Refuse(CLIO_VPIC_EXIT_PRECONDITION,
           std::string(blob_prefix) + ": n_elems is " +
               std::to_string(n_elems));
  }
  if (a.elem_bytes == 0) {
    a.elem_bytes = static_cast<int>(elem_bytes);
  } else if (a.elem_bytes != elem_bytes) {
    Refuse(CLIO_VPIC_EXIT_PRECONDITION,
           "element width changed mid-run (was " +
               std::to_string(a.elem_bytes) + ", now " +
               std::to_string(elem_bytes) + ")");
  }

  const auto t0 = std::chrono::steady_clock::now();

  // (1) The pointer the deck handed over really is device memory. Checked on
  //     every blob rather than inferred from the build: a Kokkos host build
  //     reaches here with an unregistered host pointer AND a different layout.
  a.RequireDevice(dev_ptr, blob_prefix);

  const size_t total = static_cast<size_t>(n_elems) * static_cast<size_t>(elem_bytes);
  const size_t chunk = a.chunk ? a.chunk : total;

  clio::cte::core::Context ctx;
  // NeuroPress's data_type_: 1 = float32, 2 = float64. VPIC fields are float.
  ctx.data_type_ = (elem_bytes == 4) ? 1 : 2;
  ctx.error_bound_ = 0.0;  // lossless throughout, as the offline sweep is

  const bool need_host_bytes = a.verify || !a.raw_dir.empty();

  for (size_t off = 0, ci = 0; off < total; off += chunk, ++ci) {
    const size_t n = (total - off < chunk) ? (total - off) : chunk;

    // (2) One registered kDeviceMem backend per chunk, reused across frames.
    if (a.slot_next >= a.slots.size()) {
      DeviceSlot slot;
      slot.alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(
          static_cast<clio::run::u32>(a.device),
          clio::run::gpu::IpcManager::MemKind::kDeviceMem, n, &slot.ptr);
      if (slot.alloc.IsNull()) {
        Refuse(CLIO_VPIC_EXIT_ALLOC,
               "AllocateAndRegisterGpuBackend(" + std::to_string(n) +
                   ") failed. Staging this chunk through host memory instead "
                   "would silently turn a GPU-resident run into a host one.");
      }
      slot.bytes = n;
      a.slots.push_back(slot);
    }
    DeviceSlot &slot = a.slots[a.slot_next++];
    if (slot.bytes < n) {
      Refuse(CLIO_VPIC_EXIT_ALLOC,
             "staging slot is " + std::to_string(slot.bytes) +
                 " B but this chunk is " + std::to_string(n) +
                 " B; the field extent changed between frames, which the slot "
                 "pool cannot express.");
    }

    // (3) D2D into the registered backend on the adapter's stream, fenced
    //     before the pointer is published. cudaMemcpyAsync + an explicit sync
    //     rather than ctp::GpuApi::Memcpy, which is a legacy-default-stream
    //     cudaMemcpy and is NOT host-synchronous for D2D.
    const char *src = static_cast<const char *>(dev_ptr) + off;
    const cudaError_t rc = cudaMemcpyAsync(slot.ptr, src, n,
                                           cudaMemcpyDeviceToDevice, a.stream);
    if (rc != cudaSuccess) {
      (void)cudaGetLastError();
      Refuse(CLIO_VPIC_EXIT_ALLOC,
             std::string("cudaMemcpyAsync (chunk stage) failed: ") +
                 cudaGetErrorString(rc));
    }
    cudaStreamSynchronize(a.stream);

    // (4) What Clio registered for us is device memory too.
    if (!ctp::IsDevicePointer(slot.ptr)) {
      Refuse(CLIO_VPIC_EXIT_NOT_DEVICE,
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
      // bytes. Read from the SOURCE rather than from the registered buffer, so
      // the digest cannot be taken from the destination of a copy that has not
      // landed.
      a.host_mirror.resize(n);
      cudaMemcpyAsync(a.host_mirror.data(), src, n, cudaMemcpyDeviceToHost,
                      a.stream);
      cudaStreamSynchronize(a.stream);
      rec.digest = Fnv1a(a.host_mirror.data(), n);
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

int clio_vpic_insitu_frame_end(void) {
  Adapter &a = A();
  if (!a.started) return 1;
  const auto t0 = std::chrono::steady_clock::now();
  // Every task of this frame finishes before the solver may touch the fields
  // again: the staging slots are reused by the next frame, and the fields
  // themselves advance the moment this returns.
  a.Drain();
  a.stage_s +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  return 0;
}

int clio_vpic_insitu_end(void) {
  Adapter &a = A();
  if (!a.started || a.finished) return 0;
  a.finished = true;
  a.Drain();

  size_t in_total = 0, stored_total = 0, compressed = 0, raw = 0, failed = 0;
  std::map<std::string, std::pair<size_t, size_t>> per_var;  // in, stored
  std::map<int, size_t> lib_hist;
  for (const auto &r : a.records) {
    in_total += r.bytes;
    stored_total += r.stored;
    if (!r.ok) {
      ++failed;
    } else if (r.lib == 0) {
      ++raw;
    } else {
      ++compressed;
    }
    ++lib_hist[r.lib];
    // "ex/step_00025/chunk_0" -> "ex"
    const size_t s = r.name.find('/');
    const std::string var = (s == std::string::npos) ? r.name : r.name.substr(0, s);
    per_var[var].first += r.bytes;
    per_var[var].second += r.stored;
  }

  std::cout << std::fixed << std::setprecision(3)
            << "[clio-vpic-insitu] rank " << a.rank << "/" << a.nprocs
            << " stored " << a.records.size() << " blob(s) from " << a.frames
            << " frame(s), " << in_total << " B in -> " << stored_total
            << " B on the tier  (ratio "
            << (stored_total ? static_cast<double>(in_total) /
                                   static_cast<double>(stored_total)
                             : 0.0)
            << ")\n"
            << "  compressed: " << compressed << "   stored raw: " << raw
            << "   failed: " << failed << "\n";
  for (const auto &kv : per_var) {
    std::cout << "  " << std::setw(10) << kv.first << "  " << kv.second.first
              << " -> " << kv.second.second << "  ("
              << (kv.second.second ? static_cast<double>(kv.second.first) /
                                         static_cast<double>(kv.second.second)
                                   : 0.0)
              << "x)\n";
  }
  for (const auto &kv : lib_hist) {
    std::cout << "  codec " << std::setw(16)
              << ctp::CompressionFactory::NameForWireId(kv.first) << " : "
              << kv.second << " chunk(s)\n";
  }
  std::cout << "  stage+compress(wait) " << a.stage_s << " s   in-situ wall "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             a.t_start)
                   .count()
            << " s" << std::endl;

  int rc = 0;
  if (a.verify && !a.VerifyRecords()) rc = 1;

  if (!a.report_path.empty()) {
    std::ofstream csv(a.report_path);
    if (csv) {
      // Exactly the header bin/neuropress_field_replay --readback parses, so
      // the cold reader is the same Clio-only, VPIC-free binary the offline
      // sweep uses.
      csv << "blob,bytes,fnv1a64,lib,codec,ratio,stored,compress_ms,rc,rank\n";
      for (const auto &r : a.records) {
        csv << r.name << ',' << r.bytes << ',' << std::hex << r.digest
            << std::dec << ',' << r.lib << ','
            << ctp::CompressionFactory::NameForWireId(r.lib) << ',' << r.ratio
            << ',' << r.stored << ',' << r.ms << ',' << (r.ok ? 0 : 1) << ','
            << a.rank << '\n';
      }
    }
  }

  for (auto &slot : a.slots) {
    if (!slot.alloc.IsNull()) {
      CLIO_IPC->FreeGpuBackend(static_cast<clio::run::u32>(a.device),
                               slot.alloc);
    }
  }
  a.slots.clear();
  if (a.stream) {
    cudaStreamDestroy(a.stream);
    a.stream = nullptr;
  }
  if (a.store_lock_fd >= 0) {
    ::close(a.store_lock_fd);
    if (!a.store_lock_path.empty()) ::unlink(a.store_lock_path.c_str());
    a.store_lock_fd = -1;
  }
  std::cout.flush();
  return rc;
}

}  // extern "C"
