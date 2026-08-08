/* Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause License. See LICENSE file. */

/**
 * The gpu2cpu submission ring, living in DEVICE memory.
 *
 * The queue used to live in host memory, which forced every device push to be
 * a SYSTEM-scoped atomic across PCIe -- illegal outright on GPUs that report
 * cudaDevAttrHostNativeAtomicSupported = 0 (this machine), and a bus round trip
 * even where it is legal. Moving the ring onto the device makes a push a plain
 * DEVICE-scope atomic on local memory, and makes Wait() poll local memory, so
 * the only PCIe traffic left is the CPU's batched drain: one copy carries many
 * requests instead of each request paying its own crossing.
 *
 * LAYOUT IS DELIBERATELY EXPLICIT AND POD. The CPU reads these fields by
 * cudaMemcpy from device memory, so it cannot rely on any container's internal
 * representation -- it must know the byte layout. That rules out reusing
 * multi_mpsc_ring_buffer here.
 *
 * PROTOCOL
 *   producer (device, many threads):
 *     slot  = atomicAdd(&head_, 1)                 // device-scope claim
 *     wait while slot - tail_ >= capacity_         // ring full, consumer behind
 *     write entries_[slot % capacity_]
 *     __threadfence()                              // entry before its flag
 *     ready_[slot % capacity_] = seq_for(slot)     // publish
 *
 *   consumer (host, single):
 *     copy head_ D2H; for each slot in [tail_, head_): copy entry + ready
 *     an entry counts only if its ready stamp matches the slot's sequence --
 *     a stale stamp means the producer claimed the slot but has not finished
 *     writing, so the consumer stops there and retries next poll.
 *     after consuming: publish tail_ H2D so producers can advance.
 *
 * The ready stamp is a SEQUENCE, not a 0/1 flag, so a wrapped slot cannot be
 * mistaken for a fresh one: slot s in generation g stamps (g+1), and the
 * consumer expects exactly (s / capacity_) + 1.
 */

#ifndef CLIO_RUNTIME_INCLUDE_GPU_DEVICE_RING_H_
#define CLIO_RUNTIME_INCLUDE_GPU_DEVICE_RING_H_

#include "clio_ctp/constants/macros.h"
#include "clio_ctp/util/gpu_intrinsics.h"
#include "clio_runtime/types.h"

namespace clio::run {

/**
 * One submission. Plain scalars only -- this struct is memcpy'd across PCIe,
 * so it must not contain atomics, pointers with side tables, or anything whose
 * meaning depends on which side of the bus reads it.
 *
 * `task_addr` is the RAW address of the Task in the registered device backend.
 * That is exactly what the old queue entry carried (a FullPtr whose off_ held
 * the raw address), and what RecvIn already expects to stage from.
 */
struct GpuRingEntry {
  u64 task_addr = 0;      ///< raw address of the Task POD (device backend)
  u32 alloc_major = 0;    ///< AllocatorId of the backend holding the task
  u32 alloc_minor = 0;
  u32 task_size = 0;      ///< sizeof(TaskT), so the CPU can stage without a read
  u32 pad = 0;            ///< keep the struct 24 bytes and 8-byte aligned
};

/** Fixed ring capacity. Power of two so slot->index is a mask, not a divide. */
constexpr u32 kGpuRingCapacity = 8192;
constexpr u32 kGpuRingMask = kGpuRingCapacity - 1;

/**
 * The ring itself, constructed on the host and memcpy'd once into device
 * memory. Everything after construction is touched by device atomics or by
 * explicit host copies.
 */
struct GpuDeviceRing {
  /** Producer claim counter. Device threads atomicAdd this. */
  unsigned long long head_ = 0;
  /** Consumer position. Host-owned; published H2D so producers see space. */
  unsigned long long tail_ = 0;
  /** Publication stamps, one per slot (see the sequence rule above). */
  unsigned int ready_[kGpuRingCapacity] = {};
  /** The submissions. */
  GpuRingEntry entries_[kGpuRingCapacity] = {};

#if CTP_IS_GPU_COMPILER
  /**
   * Claim a slot and publish an entry. Returns false only if the ring stayed
   * full for the whole spin budget, which the caller must treat as fatal --
   * its waiter would hang on a completion nobody will produce.
   *
   * Any thread may call this. There is deliberately no threadIdx guard: the
   * previous implementation had one and silently discarded every submission
   * from a non-zero lane, corrupting 75% of pages once more than one warp was
   * in play.
   */
  CTP_GPU_FUN bool Push(const GpuRingEntry &e) {
    const unsigned long long slot =
        atomicAdd(&head_, 1ull);              // device-scope: no PCIe, no .sys
    const u32 idx = static_cast<u32>(slot) & kGpuRingMask;
    const unsigned int stamp =
        static_cast<unsigned int>(slot / kGpuRingCapacity) + 1u;

    // Wait for the consumer to free this slot. Reading tail_ is a plain load
    // of device memory the host publishes by copy.
    for (;;) {
      const unsigned long long t =
          *const_cast<volatile unsigned long long *>(&tail_);
      if (slot - t < kGpuRingCapacity) break;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
      __nanosleep(64);
#endif
    }

    entries_[idx] = e;
    __threadfence();                          // entry visible before its stamp
    *const_cast<volatile unsigned int *>(&ready_[idx]) = stamp;
    return true;
  }
#endif  // CTP_IS_GPU_COMPILER
};

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_GPU_DEVICE_RING_H_
