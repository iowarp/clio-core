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

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#include <sanitizer/msan_interface.h>
#endif
#endif
#include <csignal>
#include "clio_ctp/data_structures/serialization/local_serialize.h"
#include "clio_ctp/data_structures/priv/array_vector.h"
#include "clio_ctp/thread/thread_model_manager.h"
#include "lightbeam.h"
#if CTP_IS_HOST
// EventManager (OS signaling) + Timepoint are host-only; the device pass keeps
// the busy-wait path. event_manager.h is <windows.h>-free.
#include "clio_ctp/lightbeam/event_manager.h"
#include "clio_ctp/util/timer.h"
#endif

namespace ctp::lbm {

// --- ShmTransferInfo ---
// SPSC ring buffer metadata for shared memory transport.
// The copy space is treated as a ring buffer indexed by total_written_ and
// total_read_ modulo copy_space_size_.
struct ShmTransferInfo {
  ctp::ipc::atomic<size_t> total_written_;   // Total bytes written by producer
  ctp::ipc::atomic<size_t> total_read_;      // Total bytes read by consumer
  ctp::ipc::atomic<size_t> copy_space_size_; // Ring buffer capacity (atomic for
                                         // cross-SM L2 visibility on GPU)

  CTP_CROSS_FUN ShmTransferInfo() {
    total_written_.store(0);
    total_read_.store(0);
    copy_space_size_.store(0);
  }
};

class ShmTransport
#if CTP_IS_HOST
  : public Transport
#endif
{
 public:
#if CTP_IS_HOST
  explicit ShmTransport(TransportMode mode) : Transport(mode) {
    type_ = TransportType::kShm;
  }

  ~ShmTransport() = default;

  Bulk Expose(const ctp::ipc::FullPtr<char>& ptr, size_t data_size,
              u32 flags) {
    Bulk bulk;
    bulk.data = ptr;
    bulk.size = data_size;
    bulk.flags = ctp::bitfield32_t(flags);
    return bulk;
  }

  std::string GetAddress() const { return "shm"; }

  /** Check if the server is still alive via PID probe. */
  bool IsServerAlive(const LbmContext& ctx = LbmContext()) const {
#ifndef _WIN32
    if (ctx.server_pid_ > 0) {
      if (!ctp::SystemInfo::IsProcessAlive(ctx.server_pid_)) return false;
    }
#endif
    return true;
  }

  void ClearRecvHandles(LbmMeta<>& meta) {
    for (auto& bulk : meta.recv) {
      if (bulk.data.ptr_ && !bulk.desc) {
        std::free(bulk.data.ptr_);
        bulk.data.ptr_ = nullptr;
      }
    }
  }
#endif

  /**
   * GPU-compatible static Send.
   * Serializes metadata and bulk descriptors through the SPSC ring buffer.
   * Uses the meta's allocator for temporary serialization buffers.
   *
   * @tparam MetaT Metadata type (must have allocator_type and alloc_ member)
   * @param meta Metadata with send bulk descriptors to transfer
   * @param ctx LbmContext with copy_space and shm_info_ for the ring buffer
   * @return 0 on success
   */
  template <typename MetaT>
  CTP_CROSS_FUN
  static int Send(MetaT& meta, const LbmContext& ctx) {
    using AllocT = typename MetaT::allocator_type;
    using CharVec = ctp::priv::vector<char, AllocT>;

    // 1. Serialize metadata using LocalSerialize with allocator-backed buffer
    CharVec meta_buf(meta.alloc_);
    meta_buf.reserve(ctx.shm_info_->copy_space_size_.load());
    ctp::ipc::LocalSerialize<CharVec> ar(meta_buf);
    ar(meta);
    ar.Finalize();

    // 2. Transfer serialized size then metadata
    uint32_t meta_len = static_cast<uint32_t>(meta_buf.size());
    WriteTransfer(reinterpret_cast<const char*>(&meta_len), sizeof(meta_len),
                  ctx);
    WriteTransfer(meta_buf.data(), meta_buf.size(), ctx);

#if CTP_IS_HOST
    // Wake the waiting receiver now that the leading bytes (length + metadata)
    // are in the ring, so when it wakes the data is already visible — avoids a
    // wake/data race where the receiver wakes on an empty ring and falls back
    // to a coarse re-check. It then drains any bulk frames written below as we
    // fill them (back-pressure relief). The runtime completer holds no
    // EventManager of its own; it targets the waiter's (pid, tid) via the
    // static Signal. 0 = no registered waiter (internal transfer) -> skip.
    if (ctx.signal_pid_ != 0) {
      EventManager::Signal(ctx.signal_pid_, ctx.signal_tid_);
    }
#endif

    // 3. Send each bulk with BULK_XFER or BULK_EXPOSE flag
    for (size_t i = 0; i < meta.send.size(); ++i) {
      if (meta.send[i].flags.Any(BULK_EXPOSE)) {
        // BULK_EXPOSE: Send only the ShmPtr (no data transfer)
        WriteTransfer(
            reinterpret_cast<const char*>(&meta.send[i].data.shm_),
            sizeof(meta.send[i].data.shm_), ctx);
      } else if (meta.send[i].flags.Any(BULK_XFER)) {
        // BULK_XFER: Send ShmPtr first, then data if private memory
        WriteTransfer(
            reinterpret_cast<const char*>(&meta.send[i].data.shm_),
            sizeof(meta.send[i].data.shm_), ctx);
        if (meta.send[i].data.shm_.alloc_id_.IsNull()) {
          // Private memory — also send full data bytes
          WriteTransfer(meta.send[i].data.ptr_, meta.send[i].size, ctx);
        }
      }
    }
    return 0;
  }

  /**
   * GPU-compatible static Recv.
   * Deserializes metadata and receives bulk data through the SPSC ring buffer.
   * Uses the meta's allocator for temporary deserialization buffers and
   * for allocating received private-memory bulk data.
   *
   * @tparam MetaT Metadata type (must have allocator_type and alloc_ member)
   * @param meta Metadata to populate with received data
   * @param ctx LbmContext with copy_space and shm_info_ for the ring buffer
   * @return ClientInfo with rc=0 on success
   */
  template <typename MetaT>
  CTP_CROSS_FUN
  static ClientInfo Recv(MetaT& meta, const LbmContext& ctx) {
    using AllocT = typename MetaT::allocator_type;
    using CharVec = ctp::priv::vector<char, AllocT>;
    ClientInfo info;

#if CTP_IS_HOST
    // Block (after a short busy-wait) until the sender starts the transfer,
    // instead of busy-polling an empty ring. The sender signals our
    // EventManager at the start of Send; once woken the streaming reads below
    // proceed with their existing short Yield waits (the sender only signals
    // once, so we must NOT re-block mid-transfer). Returns false only on a
    // ctx.timeout_ms deadline, surfaced as EAGAIN so the caller can react.
    if (!WaitForTransferStart(ctx)) {
      info.rc = EAGAIN;
      return info;
    }
#endif

    // 1. Receive 4-byte size prefix
    uint32_t meta_len = 0;
    ReadTransfer(reinterpret_cast<char*>(&meta_len), sizeof(meta_len), ctx);

    // 2. Receive metadata bytes into allocator-backed buffer
    CharVec meta_buf(meta_len, meta.alloc_);
    ReadTransfer(meta_buf.data(), meta_len, ctx);

    // 3. Deserialize using LocalDeserialize
    ctp::ipc::LocalDeserialize<CharVec> ar(meta_buf);
    ar(meta);

    // 4. Set up recv entries from send descriptors
    for (size_t i = 0; i < meta.send.size(); ++i) {
      Bulk recv_bulk;
      recv_bulk.size = meta.send[i].size;
      recv_bulk.flags = meta.send[i].flags;
      recv_bulk.data = ctp::ipc::FullPtr<char>::GetNull();
      meta.recv.push_back(recv_bulk);
    }

    // 5. Receive bulk data
    RecvBulksImpl(meta, ctx);

    info.rc = 0;
    return info;
  }

#if CTP_IS_HOST
  /**
   * Block until the sender begins writing the transfer, instead of busy-polling
   * an empty ring. Spins for ~10us first (covers the fast case with no syscall),
   * then sleeps on the waiter's EventManager until SendOut's start-of-transfer
   * Signal arrives. Re-checks the ring after each wake: the named auto-reset
   * event latches a signal that races the Wait, and the bounded timeout is a
   * safety net against a missed wake. Falls back to Yield when no EventManager
   * was provided (internal transfers / GPU host pass without a registered
   * waiter).
   */
  static bool WaitForTransferStart(const LbmContext& ctx) {
    auto avail = [&]() -> size_t {
      return ctx.shm_info_->total_written_.load_system() -
             ctx.shm_info_->total_read_.load_system();
    };
    if (avail() != 0) return true;
    ctp::Timepoint start;
    start.Now();
    // Phase 1: short busy-wait — covers a fast sender with no syscall.
    while (avail() == 0) {
      ctp::Timepoint now;
      now.Now();
      if (start.GetUsecFromStart(now) >= 10.0) break;
      CTP_THREAD_MODEL->Yield();
    }
    if (avail() != 0) return true;
    // Phase 2: sleep until the sender's start-of-transfer Signal, then spin
    // briefly for the data it announces. The Signal fires just BEFORE the first
    // ring write, so after a wake the bytes land within microseconds — re-check
    // with a tight spin instead of immediately re-sleeping. The bounded Wait
    // timeout re-checks the ring if the Signal is missed or no waiter
    // EventManager was registered (Yield fallback). ctx.timeout_ms (when > 0)
    // bounds the total wait so a never-produced result can't hang the caller.
    while (avail() == 0) {
      if (ctx.timeout_ms > 0) {
        ctp::Timepoint now;
        now.Now();
        if (start.GetUsecFromStart(now) >=
            static_cast<double>(ctx.timeout_ms) * 1000.0) {
          return false;  // timed out with no data
        }
      }
      if (ctx.event_manager_ != nullptr) {
        ctx.event_manager_->Wait(200);  // 200us bounded re-check
        // Woken by the sender's Signal (fired after the metadata write), so the
        // bytes are normally already visible; if the sender was descheduled
        // between Signal and the visible store, spin a little longer for them
        // before paying another (coarse) Wait.
        ctp::Timepoint woke;
        woke.Now();
        while (avail() == 0) {
          ctp::Timepoint now;
          now.Now();
          if (woke.GetUsecFromStart(now) >= 100.0) break;
          CTP_THREAD_MODEL->Yield();
        }
      } else {
        CTP_THREAD_MODEL->Yield();
      }
    }
    return true;
  }
#endif

  /**
   * Device-scope Send for GPU→GPU on same device.
   * Uses pre-allocated array_vector scratch buffer (ctx.meta_buf_).
   * Warp-parallel when called by all lanes; single-lane safe too.
   */

  /**
   * Warp-parallel strided memcpy: all 32 lanes copy 4-byte chunks.
   * Uses 4-byte (uint32_t) access for alignment safety since copy_space
   * may be offset by sizeof(uint32_t) from FutureShm base.
   * @param dst Destination buffer (device memory)
   * @param src Source buffer (device memory)
   * @param n   Number of bytes to copy
   */
  CTP_CROSS_FUN
  static void WarpMemCpy(char* dst, const char* src, size_t n) {
#if CTP_IS_GPU
    uint32_t lane = threadIdx.x & 31;
    size_t chunks4 = n / 4;
    auto* dst4 = reinterpret_cast<uint32_t*>(dst);
    auto* src4 = reinterpret_cast<const uint32_t*>(src);
    for (size_t i = lane; i < chunks4; i += 32) {
      dst4[i] = src4[i];
    }
    size_t tail_start = chunks4 * 4;
    if (lane == 0) {
      for (size_t i = tail_start; i < n; ++i) dst[i] = src[i];
    }
#else
    std::memcpy(dst, src, n);
#endif
  }

  // SendDevice and RecvDevice removed — GPU→GPU transport now serializes
  // task data directly into copy_space via wrap_vector (no intermediate buffer).
  // The archive metadata occupies copy_space[0..kDataOffset), task data follows.

 public:
  /**
   * GPU-compatible static bulk data receiver (device-scope).
   */
  template <typename MetaT>
  CTP_CROSS_FUN
  static int RecvBulksImplDevice(MetaT& meta, const LbmContext& ctx) {
    for (size_t i = 0; i < meta.recv.size(); ++i) {
      if (meta.recv[i].flags.Any(BULK_EXPOSE)) {
        ctp::ipc::ShmPtr<char> shm;
        ReadTransferDevice(reinterpret_cast<char*>(&shm), sizeof(shm), ctx);
        meta.recv[i].data.shm_ = shm;
        meta.recv[i].data.ptr_ = nullptr;
      } else if (meta.recv[i].flags.Any(BULK_XFER)) {
        ctp::ipc::ShmPtr<char> shm;
        ReadTransferDevice(reinterpret_cast<char*>(&shm), sizeof(shm), ctx);

        if (!shm.alloc_id_.IsNull()) {
          meta.recv[i].data.shm_ = shm;
          meta.recv[i].data.ptr_ = nullptr;
        } else {
          char* buf = meta.recv[i].data.ptr_;
          bool allocated = false;
          if (!buf) {
#if CTP_IS_HOST
            buf = static_cast<char*>(std::malloc(meta.recv[i].size));
#else
            auto alloc_ptr =
                meta.alloc_->template AllocateObjs<char>(meta.recv[i].size);
            buf = alloc_ptr.ptr_;
#endif
            allocated = true;
          }
          ReadTransferDevice(buf, meta.recv[i].size, ctx);
          if (allocated) {
            meta.recv[i].data.ptr_ = buf;
            meta.recv[i].data.shm_.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
            meta.recv[i].data.shm_.off_ = reinterpret_cast<size_t>(buf);
          }
        }
      }
    }
    return 0;
  }

  /**
   * GPU-compatible static bulk data receiver.
   * Allocates receive buffers using the meta's allocator for private memory.
   */
  template <typename MetaT>
  CTP_CROSS_FUN
  static int RecvBulksImpl(MetaT& meta, const LbmContext& ctx) {
    for (size_t i = 0; i < meta.recv.size(); ++i) {
      if (meta.recv[i].flags.Any(BULK_EXPOSE)) {
        // BULK_EXPOSE: Read only the ShmPtr (no data transfer)
        ctp::ipc::ShmPtr<char> shm;
        ReadTransfer(reinterpret_cast<char*>(&shm), sizeof(shm), ctx);
        meta.recv[i].data.shm_ = shm;
        meta.recv[i].data.ptr_ = nullptr;
      } else if (meta.recv[i].flags.Any(BULK_XFER)) {
        // BULK_XFER: Read ShmPtr first, then data if private memory
        ctp::ipc::ShmPtr<char> shm;
        ReadTransfer(reinterpret_cast<char*>(&shm), sizeof(shm), ctx);

        if (!shm.alloc_id_.IsNull()) {
          // Shared memory — ShmPtr passthrough, no data transfer
          meta.recv[i].data.shm_ = shm;
          meta.recv[i].data.ptr_ = nullptr;
        } else {
          // Private memory — read full data bytes
          char* buf = meta.recv[i].data.ptr_;
          bool allocated = false;
          if (!buf) {
#if CTP_IS_HOST
            buf = static_cast<char*>(std::malloc(meta.recv[i].size));
#else
            auto alloc_ptr =
                meta.alloc_->template AllocateObjs<char>(meta.recv[i].size);
            buf = alloc_ptr.ptr_;
#endif
            allocated = true;
          }

          ReadTransfer(buf, meta.recv[i].size, ctx);

          if (allocated) {
            meta.recv[i].data.ptr_ = buf;
            meta.recv[i].data.shm_.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
            meta.recv[i].data.shm_.off_ = reinterpret_cast<size_t>(buf);
          }
        }
      }
    }
    return 0;
  }

 private:
  // GPU-safe min of three values
  CTP_CROSS_FUN
  static size_t Min3(size_t a, size_t b, size_t c) {
    size_t m = (a < b) ? a : b;
    return (m < c) ? m : c;
  }

  // GPU-safe memcpy
  CTP_CROSS_FUN
  static void MemCopy(char* dst, const char* src, size_t n) {
#if CTP_IS_HOST
    std::memcpy(dst, src, n);
    // Mark destination as initialized: shared-memory src was written by another
    // process/thread without MSan tracking, so memcpy propagates "uninitialized"
    // shadow; __msan_unpoison tells MSan the bytes are now valid.
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
    __msan_unpoison(dst, n);
#endif
#endif
#else
    // Use volatile reads to bypass GPU L2 cache for cross-device visibility.
    // Copy 8 bytes at a time for performance, then handle remainder.
    // Both src and dst must be 8-byte aligned for the fast path.
    bool aligned = ((reinterpret_cast<uintptr_t>(src) % 8) == 0) &&
                   ((reinterpret_cast<uintptr_t>(dst) % 8) == 0);
    if (aligned) {
      size_t chunks8 = n / 8;
      if (chunks8 > 0) {
        const volatile unsigned long long* vsrc8 =
            reinterpret_cast<const volatile unsigned long long*>(src);
        unsigned long long* dst8 = reinterpret_cast<unsigned long long*>(dst);
        for (size_t i = 0; i < chunks8; ++i) {
          dst8[i] = vsrc8[i];
        }
      }
      size_t tail = chunks8 * 8;
      const volatile char* vsrc = src;
      for (size_t i = tail; i < n; ++i) {
        dst[i] = vsrc[i];
      }
    } else {
      // Byte-by-byte fallback for misaligned addresses (e.g., host-mapped mem)
      const volatile char* vsrc = src;
      for (size_t i = 0; i < n; ++i) {
        dst[i] = vsrc[i];
      }
    }
#endif
  }

  // SPSC ring buffer write (system-scope atomics for GPU/CPU visibility)
  CTP_CROSS_FUN
  static void WriteTransfer(const char* data, size_t size, const LbmContext& ctx) {
    size_t offset = 0;
    size_t ring_size = ctx.shm_info_->copy_space_size_.load_system();
    size_t total_written = ctx.shm_info_->total_written_.load_system();
    while (offset < size) {
      size_t total_read = ctx.shm_info_->total_read_.load_system();
      size_t space = ring_size - (total_written - total_read);
      if (space == 0) {
#if CTP_IS_HOST
        CTP_THREAD_MODEL->Yield();
#endif
        continue;
      }
      size_t write_pos = total_written % ring_size;
      size_t contig = ring_size - write_pos;
      size_t chunk = Min3(size - offset, space, contig);
      MemCopy(ctx.copy_space + write_pos, data + offset, chunk);
      offset += chunk;
      total_written += chunk;
      ctx.shm_info_->total_written_.store_system(total_written);
    }
  }

  // SPSC ring buffer write (device-scope atomics for GPU→GPU on same device)
  CTP_CROSS_FUN
  static void WriteTransferDevice(const char* data, size_t size,
                                   const LbmContext& ctx) {
    size_t offset = 0;
    size_t ring_size = ctx.shm_info_->copy_space_size_.load();
    size_t total_written = ctx.shm_info_->total_written_.load();
    while (offset < size) {
      size_t total_read = ctx.shm_info_->total_read_.load_device();
      size_t space = ring_size - (total_written - total_read);
      if (space == 0) {
#if CTP_IS_HOST
        CTP_THREAD_MODEL->Yield();
#endif
        continue;
      }
      size_t write_pos = total_written % ring_size;
      size_t contig = ring_size - write_pos;
      size_t chunk = Min3(size - offset, space, contig);
      memcpy(ctx.copy_space + write_pos, data + offset, chunk);
      // Flush copy_space data from L1 to L2 before making it visible
      // via total_written_. Without this fence, a reader on another SM
      // could see the updated total_written_ but read stale copy_space
      // data still in the writer's L1 cache.
      ctp::ipc::threadfence();
      offset += chunk;
      total_written += chunk;
      ctx.shm_info_->total_written_.store(total_written);
    }
  }

  // SPSC ring buffer read (system-scope atomics for GPU/CPU visibility)
  CTP_CROSS_FUN
  static void ReadTransfer(char* buf, size_t size, const LbmContext& ctx) {
    size_t offset = 0;
    size_t ring_size = ctx.shm_info_->copy_space_size_.load_system();
    size_t total_read = ctx.shm_info_->total_read_.load_system();
    while (offset < size) {
      size_t total_written = ctx.shm_info_->total_written_.load_system();
      size_t avail = total_written - total_read;
      if (avail == 0) {
#if CTP_IS_HOST
        CTP_THREAD_MODEL->Yield();
#endif
        continue;
      }
      size_t read_pos = total_read % ring_size;
      size_t contig = ring_size - read_pos;
      size_t chunk = Min3(size - offset, avail, contig);
      MemCopy(buf + offset, ctx.copy_space + read_pos, chunk);
      offset += chunk;
      total_read += chunk;
      ctx.shm_info_->total_read_.store_system(total_read);
    }
  }

  // SPSC ring buffer read (device-scope atomics for GPU→GPU on same device)
  CTP_CROSS_FUN
  static void ReadTransferDevice(char* buf, size_t size,
                                  const LbmContext& ctx) {
    size_t offset = 0;
    size_t ring_size = ctx.shm_info_->copy_space_size_.load();
    size_t total_read = ctx.shm_info_->total_read_.load();
    while (offset < size) {
      // Use load_device() for cross-SM L2 visibility — the writer's
      // total_written_.store() (atomicExch) lands in L2, but a volatile
      // load() on a different SM reads stale L1.
      size_t total_written = ctx.shm_info_->total_written_.load_device();
      size_t avail = total_written - total_read;
      if (avail == 0) {
#if CTP_IS_HOST
        CTP_THREAD_MODEL->Yield();
#endif
        continue;
      }
      // Fence before reading copy_space to ensure the writer's data
      // (fenced before total_written_ update) is visible in L2.
      ctp::ipc::threadfence();
      size_t read_pos = total_read % ring_size;
      size_t contig = ring_size - read_pos;
      size_t chunk = Min3(size - offset, avail, contig);
      memcpy(buf + offset, ctx.copy_space + read_pos, chunk);
      offset += chunk;
      total_read += chunk;
      ctx.shm_info_->total_read_.store(total_read);
    }
  }
};

}  // namespace ctp::lbm
