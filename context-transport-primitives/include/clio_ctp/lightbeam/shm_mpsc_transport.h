/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CTP_INCLUDE_LIGHTBEAM_SHM_MPSC_TRANSPORT_H
#define CTP_INCLUDE_LIGHTBEAM_SHM_MPSC_TRANSPORT_H

// =============================================================================
// Named multi-producer / single-consumer (MPSC) shared-memory transport.
// Issue #642.
//
// Unlike the legacy ShmTransport (where every caller supplied its own pair of
// SPSC rings inside a per-task FutureShm), this transport owns ONE named SHM
// segment created at server init. Many producers SendBytes into it
// concurrently; a single consumer RecvBytes. Data moves in <=32KB chunks
// through a ring carved out of the segment, coordinated by the
// ShmTransportHeader below. Each chunk self-describes its producing connection,
// its offset within the producer's message, and the total message size, so the
// single consumer can de-multiplex interleaved messages by connection id.
// =============================================================================

#include <cstdint>
#include <string>

#include "clio_ctp/constants/macros.h"
#include "clio_ctp/introspect/system_info.h"
#include "clio_ctp/types/atomic.h"

namespace ctp::lbm {

// --- Tunables ---------------------------------------------------------------
// Default transfer space when the caller does not specify one (128KB total,
// minus the header, is available as ring capacity).
static constexpr size_t kShmMpscDefaultSegmentSize = 128 * 1024;
// Per-chunk cap: SendBytes/RecvBytes move at most this many bytes per xfer slot.
static constexpr size_t kShmMpscChunkSize = 32 * 1024;
// Number of in-flight transfer slots the ring tracks (bounds concurrency).
static constexpr size_t kShmMpscMaxXfers = 256;

// --- Per-chunk transfer descriptor (lives in the SHM header) ----------------
// One producer fills this in, sets ready_, and the consumer drains it. conn_id_
// identifies which client connection produced the chunk (for de-mux / dead-
// connection skipping on the consumer side).
struct ShmXferHeader {
  ctp::u64 conn_id_;     // Producing connection's id
  ctp::u32 xfer_off_;    // Byte offset of this chunk within the ring (monotonic)
  ctp::u32 xfer_size_;   // Number of bytes in this chunk
  ctp::u32 rem_off_;     // Offset of this chunk within the producer's message
  ctp::u32 rem_size_;    // Total size of the producer's message
  ctp::ipc::atomic<bool> ready_;  // Producer sets after memcpy; consumer clears

  CTP_CROSS_FUN ShmXferHeader()
      : conn_id_(0), xfer_off_(0), xfer_size_(0), rem_off_(0), rem_size_(0) {
    ready_.store(false);
  }
};

// --- Segment header (placement-new'd at the start of the SHM data region) ---
// Created once by the server; clients attach and read/CAS it. The ring buffer
// is the segment bytes immediately following this header.
struct ShmTransportHeader {
  ctp::ipc::atomic<ctp::u64> head_;          // Bytes the consumer has drained
  ctp::ipc::atomic<ctp::u64> tail_;          // Bytes producers have reserved
  ctp::ipc::atomic<ctp::u64> connection_id_; // Next client connection id
  ctp::ipc::atomic<ctp::u64> xfer_id_head_;  // Next xfer slot to consume
  ctp::ipc::atomic<ctp::u64> xfer_id_tail_;  // Next xfer slot to reserve
  int pid_;                                  // Server pid (liveness probe)
  int tid_;                                  // Server tid
  size_t max_capacity_;                      // Ring capacity (segment - header)
  ShmXferHeader xfers_[kShmMpscMaxXfers];    // In-flight chunk descriptors

  CTP_CROSS_FUN ShmTransportHeader() : pid_(0), tid_(0), max_capacity_(0) {
    head_.store(0);
    tail_.store(0);
    connection_id_.store(0);
    xfer_id_head_.store(0);
    xfer_id_tail_.store(0);
    // xfers_[] are default-constructed (ready_ = false).
  }
};

}  // namespace ctp::lbm

// The transport class itself is host-only: it drives named OS shared memory and
// uses std:: containers for consumer-side reassembly. The structs above stay
// cross-platform so they can describe the layout from any context.
#if CTP_IS_HOST

#include <cerrno>
#include <cstring>
#include <unordered_map>
#include <vector>

// allocator.h must be entered before memory_backend.h/posix_shm_mmap.h: the two
// form a circular include and only the allocator-first order defines
// MemoryBackendId (== AllocatorId) before allocator.h references it.
#include "clio_ctp/memory/allocator/allocator.h"
// lightbeam.h gives the LbmMeta/Bulk/ClientInfo/BULK_* surface used by the
// high-level Send/Recv below; local_serialize.h serializes the metadata.
#include "clio_ctp/data_structures/serialization/local_serialize.h"
#include "clio_ctp/lightbeam/lightbeam.h"
#include "clio_ctp/memory/backend/posix_shm_mmap.h"
#include "clio_ctp/thread/thread_model_manager.h"
#include "clio_ctp/util/timer.h"

namespace ctp::lbm {

// DONTWAIT: RecvBytes returns -EAGAIN immediately when nothing is in flight,
// instead of blocking. (It still blocks once a transfer has been detected, to
// finish receiving that message.)
static constexpr ctp::u32 SHM_MPSC_DONTWAIT = 0x1;

// How long the consumer waits on a not-yet-ready chunk before declaring the
// producing connection dead and skipping it (microseconds).
static constexpr double kShmMpscDeadXferUs = 1'000'000.0;  // 1 second
// How long a producer waits for ring capacity / a free xfer slot before
// re-checking that the server (consumer) process is still alive (microseconds).
static constexpr double kShmMpscLivenessUs = 50'000.0;  // 50 ms

class ShmMpscTransport {
 public:
  ctp::ipc::PosixShmMmap backend_;
  ShmTransportHeader *hdr_ = nullptr;  // at backend_.data_
  char *ring_ = nullptr;               // backend_.data_ + sizeof(header)
  size_t cap_ = 0;                     // == hdr_->max_capacity_
  bool is_server_ = false;
  bool inited_ = false;
  ctp::u64 conn_id_ = 0;  // this client's connection id (server: 0)
  std::string name_;

  // Consumer-side per-connection reassembly state (single-threaded: the one
  // Recv consumer owns this map). TODO(#642): swap for mcsp_unordered_map once
  // look-ahead lets multiple drain helpers touch it.
  struct RecvConn {
    std::vector<char> buf;
    ctp::u32 total = 0;
    ctp::u32 received = 0;
  };
  std::unordered_map<ctp::u64, RecvConn> recv_conns_;

  ShmMpscTransport() = default;
  ~ShmMpscTransport() { Shutdown(); }

  ShmMpscTransport(const ShmMpscTransport &) = delete;
  ShmMpscTransport &operator=(const ShmMpscTransport &) = delete;

  /**
   * Server init: create the named segment and place the header at its start.
   * @param name      unique SHM name (e.g. "clio-<tid>")
   * @param xfer_space total transfer space (header + ring); default 128KB.
   */
  bool ServerInit(const std::string &name,
                  size_t xfer_space = kShmMpscDefaultSegmentSize) {
    name_ = name;
    // The backend enforces a >=1MB minimum and reserves a 64KB header page; the
    // ring only uses the first sizeof(header)+max_capacity bytes of the data
    // region, so request enough and ignore the slack.
    size_t backend_size = xfer_space + kBackendHeaderSizeApprox();
    if (!backend_.shm_init(
            ctp::ipc::MemoryBackendId::Get(
                static_cast<ctp::u32>(ctp::SystemInfo::GetPid()), 0),
            backend_size, name)) {
      return false;
    }
    hdr_ = new (backend_.data_) ShmTransportHeader();
    hdr_->pid_ = ctp::SystemInfo::GetPid();
    hdr_->tid_ = ctp::SystemInfo::GetTid();
    if (xfer_space <= sizeof(ShmTransportHeader)) {
      xfer_space = sizeof(ShmTransportHeader) + kShmMpscChunkSize;
    }
    cap_ = xfer_space - sizeof(ShmTransportHeader);
    // Clamp to what the data region actually holds.
    size_t avail = backend_.data_capacity_ > sizeof(ShmTransportHeader)
                       ? backend_.data_capacity_ - sizeof(ShmTransportHeader)
                       : 0;
    if (cap_ > avail) cap_ = avail;
    hdr_->max_capacity_ = cap_;
    ring_ = backend_.data_ + sizeof(ShmTransportHeader);
    is_server_ = true;
    inited_ = true;
    return true;
  }

  /** Client init: attach to the named segment and take a connection id. */
  bool ClientInit(const std::string &name) {
    name_ = name;
    if (!backend_.shm_attach(name)) {
      return false;
    }
    hdr_ = reinterpret_cast<ShmTransportHeader *>(backend_.data_);
    cap_ = hdr_->max_capacity_;
    ring_ = backend_.data_ + sizeof(ShmTransportHeader);
    conn_id_ = hdr_->connection_id_.fetch_add(1) + 1;  // 0 is reserved
    is_server_ = false;
    inited_ = true;
    return true;
  }

  void Shutdown() {
    if (!inited_) return;
    inited_ = false;
    if (is_server_) {
      backend_.shm_destroy();  // unmap + unlink the named segment
    } else {
      backend_.shm_detach();
    }
    hdr_ = nullptr;
    ring_ = nullptr;
  }

  // --- Producer ------------------------------------------------------------
  /**
   * Send `size` bytes as one logical message from this connection. Returns 0 on
   * success, -EPIPE if the server (consumer) process died mid-transfer.
   */
  int SendBytes(const char *data, size_t size) {
    ctp::u32 rem_off = 0;
    ctp::u32 rem_size = static_cast<ctp::u32>(size);
    while (rem_off < size) {
      // 1. Reserve an xfer slot; wait until it is within the in-flight window.
      ctp::u64 xfer_id = hdr_->xfer_id_tail_.fetch_add(1);
      if (!WaitSlotFree(xfer_id)) return -EPIPE;
      // 2. Chunk size.
      ctp::u32 xfer_size = static_cast<ctp::u32>(
          (size - rem_off) < kShmMpscChunkSize ? (size - rem_off)
                                               : kShmMpscChunkSize);
      // 3. Reserve ring space (monotonic).
      ctp::u64 xfer_off = hdr_->tail_.fetch_add(xfer_size);
      // 4. Wait until the consumer has drained enough that our window fits.
      if (!WaitRingCapacity(xfer_off, xfer_size)) return -EPIPE;
      // 5. Copy into the ring (handles wraparound).
      RingWrite(xfer_off, data + rem_off, xfer_size);
      // 6. Publish the descriptor, then mark ready (release).
      ShmXferHeader &slot = hdr_->xfers_[xfer_id % kShmMpscMaxXfers];
      slot.conn_id_ = conn_id_;
      slot.xfer_off_ = static_cast<ctp::u32>(xfer_off % cap_);
      slot.xfer_size_ = xfer_size;
      slot.rem_off_ = rem_off;
      slot.rem_size_ = rem_size;
      slot.ready_.store(true);
      // 7. Advance.
      rem_off += xfer_size;
    }
    return 0;
  }

  // --- Consumer ------------------------------------------------------------
  /**
   * Receive one complete message into `out` (resized to the message size), and
   * report the producing connection id via *conn_out (if non-null).
   * @return 0 on success; -EAGAIN if DONTWAIT and nothing is in flight.
   */
  int RecvBytes(std::vector<char> &out, ctp::u64 *conn_out, ctp::u32 flags = 0) {
    while (true) {
      ctp::u64 id_head = hdr_->xfer_id_head_.load();
      ctp::u64 id_tail = hdr_->xfer_id_tail_.load();
      if (id_head == id_tail && recv_conns_.empty()) {
        if (flags & SHM_MPSC_DONTWAIT) return -EAGAIN;
        CTP_THREAD_MODEL->Yield();
        continue;
      }
      if (id_head == id_tail) {
        // Nothing reserved yet but a message is partially received -> spin.
        CTP_THREAD_MODEL->Yield();
        continue;
      }
      ShmXferHeader &slot = hdr_->xfers_[id_head % kShmMpscMaxXfers];
      // Wait for the producer to publish this chunk; skip if it never arrives.
      if (!WaitChunkReady(slot)) {
        // Producer presumed dead: skip this slot. We cannot trust its size, so
        // we do NOT advance head_ (small ring leak for a dead conn) — only the
        // xfer-id cursor advances so the consumer makes progress.
        hdr_->xfer_id_head_.fetch_add(1);
        continue;
      }
      ctp::u64 conn = slot.conn_id_;
      ctp::u32 total = slot.rem_size_;
      ctp::u32 off = slot.rem_off_;
      ctp::u32 xsize = slot.xfer_size_;
      ctp::u32 xoff = slot.xfer_off_;
      // De-mux into the per-connection buffer.
      RecvConn &st = recv_conns_[conn];
      if (st.total == 0) {
        st.total = total;
        st.buf.resize(total);
        st.received = 0;
      }
      if (static_cast<size_t>(off) + xsize <= st.buf.size()) {
        RingRead(st.buf.data() + off, xoff, xsize);
      }
      st.received += xsize;
      slot.ready_.store(false);
      hdr_->head_.fetch_add(xsize);      // free ring space
      hdr_->xfer_id_head_.fetch_add(1);  // advance cursor
      if (st.received >= st.total) {
        out = std::move(st.buf);
        if (conn_out) *conn_out = conn;
        recv_conns_.erase(conn);
        return 0;
      }
    }
  }

  // --- High-level metadata + bulk API (mirrors Transport::Send/Recv) --------
  // Build a bulk descriptor referencing `ptr` (Transport::Expose parity).
  Bulk Expose(const ctp::ipc::FullPtr<char> &ptr, size_t data_size,
              ctp::u32 flags) {
    Bulk bulk;
    bulk.data = ptr;
    bulk.size = data_size;
    bulk.flags = ctp::bitfield32_t(flags);
    return bulk;
  }

  // Serialize metadata + bulk data into ONE message and SendBytes it, so a
  // producer's framing can't interleave with another's at the consumer.
  template <typename MetaT>
  int Send(MetaT &meta) {
    using AllocT = typename MetaT::allocator_type;
    using CharVec = ctp::priv::vector<char, AllocT>;
    CharVec meta_buf(meta.alloc_);
    ctp::ipc::LocalSerialize<CharVec> ar(meta_buf);
    ar(meta);
    ar.Finalize();
    std::vector<char> msg;
    uint32_t meta_len = static_cast<uint32_t>(meta_buf.size());
    AppendRaw(msg, &meta_len, sizeof(meta_len));
    AppendRaw(msg, meta_buf.data(), meta_buf.size());
    for (size_t i = 0; i < meta.send.size(); ++i) {
      if (meta.send[i].flags.Any(BULK_EXPOSE)) {
        AppendRaw(msg, &meta.send[i].data.shm_, sizeof(meta.send[i].data.shm_));
      } else if (meta.send[i].flags.Any(BULK_XFER)) {
        AppendRaw(msg, &meta.send[i].data.shm_, sizeof(meta.send[i].data.shm_));
        if (meta.send[i].data.shm_.alloc_id_.IsNull()) {
          AppendRaw(msg, meta.send[i].data.ptr_, meta.send[i].size);
        }
      }
    }
    return SendBytes(msg.data(), msg.size());
  }

  // Receive one complete message and rebuild metadata + bulks. rc=0 on success;
  // rc=EAGAIN when DONTWAIT and nothing is in flight.
  template <typename MetaT>
  ClientInfo Recv(MetaT &meta, ctp::u32 flags = 0) {
    using AllocT = typename MetaT::allocator_type;
    using CharVec = ctp::priv::vector<char, AllocT>;
    ClientInfo info;
    std::vector<char> msg;
    ctp::u64 conn = 0;
    int rc = RecvBytes(msg, &conn, flags);
    if (rc != 0) {
      info.rc = (rc < 0) ? -rc : rc;  // surface EAGAIN as a positive errno
      return info;
    }
    size_t pos = 0;
    uint32_t meta_len = 0;
    if (!ReadRaw(msg, pos, &meta_len, sizeof(meta_len))) {
      info.rc = EIO;
      return info;
    }
    CharVec meta_buf(meta_len, meta.alloc_);
    if (meta_len > 0 && !ReadRaw(msg, pos, meta_buf.data(), meta_len)) {
      info.rc = EIO;
      return info;
    }
    ctp::ipc::LocalDeserialize<CharVec> dar(meta_buf);
    dar(meta);
    for (size_t i = 0; i < meta.send.size(); ++i) {
      Bulk recv_bulk;
      recv_bulk.size = meta.send[i].size;
      recv_bulk.flags = meta.send[i].flags;
      recv_bulk.data = ctp::ipc::FullPtr<char>::GetNull();
      ctp::ipc::ShmPtr<char> shm;
      if (recv_bulk.flags.Any(BULK_EXPOSE)) {
        ReadRaw(msg, pos, &shm, sizeof(shm));
        recv_bulk.data.shm_ = shm;
        recv_bulk.data.ptr_ = nullptr;
      } else if (recv_bulk.flags.Any(BULK_XFER)) {
        ReadRaw(msg, pos, &shm, sizeof(shm));
        if (!shm.alloc_id_.IsNull()) {
          recv_bulk.data.shm_ = shm;
          recv_bulk.data.ptr_ = nullptr;
        } else {
          char *buf = static_cast<char *>(std::malloc(recv_bulk.size));
          ReadRaw(msg, pos, buf, recv_bulk.size);
          recv_bulk.data.ptr_ = buf;
          recv_bulk.data.shm_.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
          recv_bulk.data.shm_.off_ = reinterpret_cast<size_t>(buf);
        }
      }
      meta.recv.push_back(recv_bulk);
    }
    info.rc = 0;
    return info;
  }

 private:
  static void AppendRaw(std::vector<char> &v, const void *p, size_t n) {
    const char *c = static_cast<const char *>(p);
    v.insert(v.end(), c, c + n);
  }
  static bool ReadRaw(const std::vector<char> &v, size_t &pos, void *p,
                      size_t n) {
    if (pos + n > v.size()) return false;
    std::memcpy(p, v.data() + pos, n);
    pos += n;
    return true;
  }

  // The PosixShmMmap reserves a fixed 64KB header page (kBackendHeaderSize)
  // ahead of the data region; pad the request so the data region comfortably
  // holds header + ring.
  static size_t kBackendHeaderSizeApprox() { return 64 * 1024; }

  bool ServerAlive() const {
#ifndef _WIN32
    if (hdr_->pid_ > 0) return ctp::SystemInfo::IsProcessAlive(hdr_->pid_);
#endif
    return true;
  }

  // Producer: wait until this xfer slot is inside the bounded in-flight window.
  bool WaitSlotFree(ctp::u64 xfer_id) {
    ctp::Timepoint start;
    start.Now();
    while (xfer_id - hdr_->xfer_id_head_.load() >= kShmMpscMaxXfers) {
      ctp::Timepoint now;
      now.Now();
      if (start.GetUsecFromStart(now) >= kShmMpscLivenessUs) {
        if (!ServerAlive()) return false;
        start.Now();
      }
      CTP_THREAD_MODEL->Yield();
    }
    return true;
  }

  // Producer: wait until the consumer has drained enough for our ring window.
  bool WaitRingCapacity(ctp::u64 xfer_off, ctp::u32 xfer_size) {
    ctp::Timepoint start;
    start.Now();
    while (true) {
      ctp::u64 head = hdr_->head_.load();
      // rem_capacity = cap - (bytes reserved ahead of the drained point)
      ctp::u64 used = xfer_off - head;
      if (used + xfer_size <= cap_) return true;
      ctp::Timepoint now;
      now.Now();
      if (start.GetUsecFromStart(now) >= kShmMpscLivenessUs) {
        if (!ServerAlive()) return false;
        start.Now();
      }
      CTP_THREAD_MODEL->Yield();
    }
  }

  // Consumer: wait for a chunk's ready flag; false => presumed-dead, skip it.
  bool WaitChunkReady(ShmXferHeader &slot) {
    ctp::Timepoint start;
    start.Now();
    while (!slot.ready_.load()) {
      ctp::Timepoint now;
      now.Now();
      if (start.GetUsecFromStart(now) >= kShmMpscDeadXferUs) {
        return false;
      }
      CTP_THREAD_MODEL->Yield();
    }
    return true;
  }

  // Copy `size` bytes from src into the ring at monotonic offset xfer_off,
  // wrapping at cap_.
  void RingWrite(ctp::u64 xfer_off, const char *src, ctp::u32 size) {
    size_t pos = static_cast<size_t>(xfer_off % cap_);
    size_t first = cap_ - pos;
    if (first >= size) {
      std::memcpy(ring_ + pos, src, size);
    } else {
      std::memcpy(ring_ + pos, src, first);
      std::memcpy(ring_, src + first, size - first);
    }
  }

  // Copy `size` bytes out of the ring (ring offset already modulo cap_) into
  // dst, wrapping at cap_.
  void RingRead(char *dst, ctp::u32 ring_off, ctp::u32 size) {
    size_t pos = static_cast<size_t>(ring_off);
    size_t first = cap_ - pos;
    if (first >= size) {
      std::memcpy(dst, ring_ + pos, size);
    } else {
      std::memcpy(dst, ring_ + pos, first);
      std::memcpy(dst + first, ring_, size - first);
    }
  }
};

}  // namespace ctp::lbm

#endif  // CTP_IS_HOST

#endif  // CTP_INCLUDE_LIGHTBEAM_SHM_MPSC_TRANSPORT_H
