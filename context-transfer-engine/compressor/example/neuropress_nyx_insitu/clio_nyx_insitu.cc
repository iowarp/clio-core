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
#include <unistd.h>
#include <vector>

#include <clio_cte/compressor/compressor_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_ctp/compress/compress_factory.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_runtime/clio_runtime.h>

namespace {

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
  bool ok = false;
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

  /** Refuse unless `p` is genuinely device memory. */
  void RequireDevice(const void *p, const char *what) const {
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
      // compress_lib_ == 0 is "stored raw": the codec ran, did not shrink the
      // chunk, and the ORIGINAL bytes went to the tier.
      // actual_compressed_size_ still reports the codec's output, which is
      // not what was stored.
      r.stored = (r.ok && r.lib != 0) ? c.actual_compressed_size_ : r.bytes;
    }
    pending.clear();
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

}  // namespace

extern "C" {

int clio_nyx_insitu_begin(void) {
  Adapter &a = A();
  if (a.started) return 0;

  a.tag_name = EnvStr("CLIO_NYX_TAG", "nyx_insitu");
  a.chunk = EnvSize("CLIO_NYX_CHUNK", 4u << 20);
  a.report_path = EnvStr("CLIO_NYX_REPORT", "");
  a.raw_dir = EnvStr("CLIO_NYX_RAW_DIR", "");
  a.verify = EnvBool("CLIO_NYX_VERIFY", false);
  a.t_start = std::chrono::steady_clock::now();

  // Same call the HDF5 VOL makes lazily on first use. With CLIO_WITH_RUNTIME=1
  // the runtime comes up HERE, inside the Nyx process, composed from
  // CLIO_SERVER_CONF. That is load-bearing: a compressor task sent to a
  // runtime in another process arrives with a device pointer that means
  // nothing there unless it travels as a CUDA IPC handle.
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::cerr << "[clio-nyx-insitu] CLIO_CTE_CLIENT_INIT failed -- is "
                 "CLIO_SERVER_CONF set and CLIO_WITH_RUNTIME=1?\n";
    return 1;
  }
  a.cte = CLIO_CTE_CLIENT;

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

  std::cout << "[clio-nyx-insitu] up: tag='" << a.tag_name << "' pool=" << major
            << "." << minor << " chunk="
            << (a.chunk ? std::to_string(a.chunk) : std::string("whole field"))
            << (a.verify ? " verify=on" : "") << " pid=" << getpid()
            << std::endl;
  a.started = true;
  return 0;
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
  // H5T_FLOAT's size. error_bound_ stays 0: lossless.
  ctx.data_type_ = (elem_bytes == 8) ? 2 : 1;

  const bool need_host_bytes =
      true;  // the digest is what makes the cold read-back checkable

  for (size_t ci = 0; ci < nchunks; ++ci) {
    const size_t off = ci * chunk;
    const size_t n = std::min(chunk, field_bytes - off);

    // (3) One registered kDeviceMem backend per chunk, reused across frames.
    if (a.slot_next >= a.slots.size()) {
      DeviceSlot slot;
      slot.alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(
          /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem, n,
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
            << "\n[clio-nyx-insitu] stored " << a.records.size() << " blob(s) "
            << "from " << a.frames << " frame(s), " << in_total << " B in -> "
            << stored_total << " B on the tier  (ratio "
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
              << ctp::CompressionFactory::NameForWireId(lib) << " : " << n
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
    csv << "blob,bytes,fnv1a64,lib,codec,ratio,stored,compress_ms,rc\n";
    for (const auto &r : a.records) {
      csv << r.name << ',' << r.bytes << ',' << std::hex << r.digest << std::dec
          << ',' << r.lib << ','
          << ctp::CompressionFactory::NameForWireId(r.lib) << ',' << r.ratio
          << ',' << r.stored << ',' << r.ms << ',' << (r.ok ? 0 : 1) << '\n';
    }
  }

  int rc = failed ? 1 : 0;
  if (a.verify && !a.VerifyRecords()) rc = 1;

  for (auto &s : a.slots) {
    if (!s.alloc.IsNull()) CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, s.alloc);
  }
  a.slots.clear();
  if (a.scratch) {
    cudaFree(a.scratch);
    a.scratch = nullptr;
    a.scratch_bytes = 0;
  }
  if (a.stream) {
    cudaStreamDestroy(a.stream);
    a.stream = nullptr;
  }
  std::cout.flush();
  return rc;
}

}  // extern "C"
