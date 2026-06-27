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
// Issue #642. WORK IN PROGRESS — header/protocol structs are final; the
// transport method bodies are being filled in incrementally (see issue).
//
// Unlike the legacy ShmTransport (where every caller supplied its own pair of
// SPSC rings inside a per-task FutureShm), this transport owns ONE named SHM
// segment created at server init. Many producers Send into it concurrently; a
// single consumer Recvs. Data moves in <=32KB chunks through a ring carved out
// of the segment, coordinated by the ShmTransportHeader below.
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
// identifies which client connection produced the chunk (for fairness / dead-
// connection skipping on the consumer side).
struct ShmXferHeader {
  ctp::u64 conn_id_;     // Producing connection's id
  ctp::u32 xfer_off_;    // Byte offset of this chunk within the ring
  ctp::u32 xfer_size_;   // Number of bytes in this chunk
  ctp::u32 rem_off_;     // Offset of this chunk within the producer's full buffer
  ctp::u32 rem_size_;    // Total remaining size of the producer's transfer
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
  ctp::ipc::atomic<ctp::u64> connection_id_; // Next client connection id (CAS'd)
  ctp::ipc::atomic<ctp::u64> xfer_id_head_;  // Next xfer slot to consume
  ctp::ipc::atomic<ctp::u64> xfer_id_tail_;  // Next xfer slot to reserve
  int pid_;                                  // Server pid (liveness probe)
  int tid_;                                  // Server tid
  size_t max_capacity_;                      // Ring capacity (segment - header)
  ShmXferHeader xfers_[kShmMpscMaxXfers];    // In-flight chunk descriptors

  CTP_CROSS_FUN ShmTransportHeader()
      : pid_(0), tid_(0), max_capacity_(0) {
    head_.store(0);
    tail_.store(0);
    connection_id_.store(0);
    xfer_id_head_.store(0);
    xfer_id_tail_.store(0);
    // xfers_[] are default-constructed (ready_ = false).
  }
};

}  // namespace ctp::lbm

#endif  // CTP_INCLUDE_LIGHTBEAM_SHM_MPSC_TRANSPORT_H
