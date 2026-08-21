/**
 * bam/nvme_emu.h -- an EMULATED NVMe controller backed by pinned host memory.
 *
 * Why this exists. `nvme_read_page`/`nvme_write_page` used to be stubs that
 * ignored their arguments and returned -1, and the only working backend was
 * a bare warp-cooperative copy out of pinned host memory. That bypasses the
 * protocol entirely: no submission queue, no doorbell, no completion queue,
 * none of the latency or queueing structure that a GPU page cache exists to
 * hide. Benchmarks written against it price a miss as a memcpy, which is not
 * what a miss costs on the real thing.
 *
 * What this does instead. The GPU speaks the same sequence it speaks to a
 * real drive:
 *
 *     claim an SQ slot -> write the SQE -> ring the submission doorbell
 *       -> poll the completion queue for a CQE carrying the expected phase
 *
 * and a host-side controller thread plays the role of the drive: it polls
 * the doorbell, performs the transfer, and posts the completion. The medium
 * is pinned host memory rather than flash, so this reproduces the PATH and
 * the queueing, not the media latency -- a distinction worth stating,
 * because it means an emulated miss is faster than an NVMe miss and slower
 * than the memcpy it replaces.
 *
 * The queues live in mapped pinned host memory so both sides can touch them
 * with ordinary loads and stores: the device through the pointers from
 * cudaHostGetDevicePointer, the controller thread directly.
 */
#ifndef BAM_NVME_EMU_H
#define BAM_NVME_EMU_H

#include <atomic>
#include <cstdint>
#include <thread>

#include <cuda_runtime.h>

#include "bam/types.h"

namespace bam {

/** One submission-queue entry: a page-granular read or write. */
struct EmuSqe {
  uint64_t offset;      ///< byte offset into the backing store
  uint64_t bus_addr;    ///< destination/source cache-page address (device)
  uint32_t nbytes;
  uint32_t opcode;      ///< 0 = read, 1 = write
  uint32_t cid;         ///< command id; indexes the CQ slot
  uint32_t pad;
};

/** One completion-queue entry. `phase` flips each time the ring wraps, which
 *  is how the consumer tells a fresh completion from a stale one without
 *  clearing the slot -- the same trick real NVMe uses. */
struct EmuCqe {
  uint32_t cid;
  uint32_t status;
  uint32_t phase;
  uint32_t pad;
};

/**
 * Host side of the emulated controller.
 *
 * Owns the queues, the servicing thread, and the pinned backing store. The
 * thread spins rather than sleeps: a miss is on the critical path of a GPU
 * kernel that is waiting for it, so a wakeup would dominate the latency
 * being emulated.
 */
class NvmeEmuController {
 public:
  NvmeEmuController() = default;
  ~NvmeEmuController() { stop(); }

  /**
   * @param backing      pinned host memory holding the data
   * @param backing_size bytes in `backing`
   * @param depth        queue depth (power of two)
   */
  bool start(const uint8_t *backing, size_t backing_size, uint32_t depth);
  void stop();

  /** Device-side view to hand to kernels. */
  QueuePairDevice device_state() const { return dev_; }

  /** Completions serviced so far -- the emulated drive's IOPS counter. */
  uint64_t completions() const {
    return done_.load(std::memory_order_relaxed);
  }

 private:
  void service_loop();

  const uint8_t *backing_ = nullptr;
  size_t backing_size_ = 0;
  uint32_t depth_ = 0;

  // Mapped pinned host allocations; *_d are the device-side aliases.
  EmuSqe *sq_ = nullptr;      EmuSqe *sq_d_ = nullptr;
  EmuCqe *cq_ = nullptr;      EmuCqe *cq_d_ = nullptr;
  uint32_t *sq_ready_ = nullptr; uint32_t *sq_ready_d_ = nullptr;
  uint32_t *sq_alloc_d_ = nullptr;   ///< device memory: see sq_alloc

  QueuePairDevice dev_{};
  std::thread thread_;
  std::atomic<bool> run_{false};
  std::atomic<uint64_t> done_{0};
  uint32_t head_ = 0;         ///< controller's SQ consume cursor
  /** Non-blocking stream: a default-stream copy would serialise with
   *  the very kernel waiting on this completion. */
  cudaStream_t stream_ = nullptr;
};

}  // namespace bam

#endif  // BAM_NVME_EMU_H
