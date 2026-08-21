/**
 * bam/nvme_emu.cc -- host half of the emulated NVMe controller.
 *
 * See bam/nvme_emu.h for why this exists. The short version: the device
 * side now speaks the real submission/doorbell/completion sequence, and
 * something has to play the drive. This thread does, with pinned host
 * memory as the medium.
 */
#include "bam/nvme_emu.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

namespace bam {

namespace {
/** Mapped pinned allocation plus its device-side alias. */
bool alloc_mapped(void **host, void **dev, size_t bytes) {
  if (cudaHostAlloc(host, bytes, cudaHostAllocMapped) != cudaSuccess) {
    return false;
  }
  std::memset(*host, 0, bytes);
  return cudaHostGetDevicePointer(dev, *host, 0) == cudaSuccess;
}
}  // namespace

bool NvmeEmuController::start(const uint8_t *backing, size_t backing_size,
                              uint32_t depth) {
  backing_ = backing;
  backing_size_ = backing_size;
  depth_ = depth ? depth : 1024;
  if ((depth_ & (depth_ - 1)) != 0) {
    std::fprintf(stderr, "bam::NvmeEmu: depth must be a power of two\n");
    return false;
  }

  void *h = nullptr, *d = nullptr;
  if (!alloc_mapped(&h, &d, sizeof(EmuSqe) * depth_)) return false;
  sq_ = static_cast<EmuSqe *>(h); sq_d_ = static_cast<EmuSqe *>(d);
  if (!alloc_mapped(&h, &d, sizeof(EmuCqe) * depth_)) return false;
  cq_ = static_cast<EmuCqe *>(h); cq_d_ = static_cast<EmuCqe *>(d);
  // Publication stamps: mapped host memory, one per slot.
  if (!alloc_mapped(&h, &d, sizeof(uint32_t) * depth_)) return false;
  sq_ready_ = static_cast<uint32_t *>(h);
  sq_ready_d_ = static_cast<uint32_t *>(d);
  // Slot allocator: DEVICE memory -- this GPU cannot do atomics on host
  // memory (see QueuePairDevice::sq_alloc).
  if (cudaMalloc(&sq_alloc_d_, sizeof(uint32_t)) != cudaSuccess) return false;
  if (cudaMemset(sq_alloc_d_, 0, sizeof(uint32_t)) != cudaSuccess) return false;

  dev_.sq = sq_d_;
  dev_.cq = cq_d_;
  dev_.sq_alloc = sq_alloc_d_;
  dev_.sq_ready = sq_ready_d_;
  dev_.sq_depth = depth_;
  dev_.cq_depth = depth_;
  dev_.nsid = 1;
  dev_.lba_shift = 9;

  // NON-BLOCKING STREAM, and it is not optional. A plain cudaMemcpy runs on
  // the legacy default stream, which serialises with any resident kernel --
  // and the kernel resident here is the one spinning on the completion this
  // transfer produces. That is a deadlock, and it hung the first working
  // version of this controller.
  if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) !=
      cudaSuccess) {
    std::fprintf(stderr, "bam::NvmeEmu: stream create failed\n");
    return false;
  }

  head_ = 0;
  run_.store(true, std::memory_order_release);
  thread_ = std::thread(&NvmeEmuController::service_loop, this);
  return true;
}

void NvmeEmuController::stop() {
  if (run_.exchange(false, std::memory_order_acq_rel)) {
    if (thread_.joinable()) thread_.join();
  }
  if (sq_) { cudaFreeHost(sq_); sq_ = nullptr; }
  if (cq_) { cudaFreeHost(cq_); cq_ = nullptr; }
  if (sq_ready_) { cudaFreeHost(sq_ready_); sq_ready_ = nullptr; }
  if (sq_alloc_d_) { cudaFree(sq_alloc_d_); sq_alloc_d_ = nullptr; }
  if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
}

/**
 * The drive. Polls the submission doorbell, moves the bytes, posts the
 * completion.
 *
 * The transfer is a cudaMemcpy into the GPU cache page, which is the
 * emulation's stand-in for the DMA a real controller would issue into the
 * GPU's BAR. It is issued on this thread rather than a stream so the
 * completion is only posted once the bytes have actually landed -- posting
 * early would hand the kernel a page it is allowed to read before it is
 * filled, which is the one thing a completion queue is for.
 */
void NvmeEmuController::service_loop() {
  while (run_.load(std::memory_order_acquire)) {
    // THE STAMP IS THE ARRIVAL SIGNAL, in place of a tail doorbell: the slot
    // at `head_` is due stamp (head_/depth)+1, and seeing it proves both that
    // a producer claimed the slot and that it finished writing the entry.
    const uint32_t idx = head_ & (depth_ - 1);
    const uint32_t want = (head_ / depth_) + 1u;
    const uint32_t got = __atomic_load_n(&sq_ready_[idx], __ATOMIC_ACQUIRE);
    if (got != want) {
      static const bool tr = getenv("BAM_EMU_TRACE") != nullptr;
      if (tr) {
        static uint64_t spins = 0;
        if ((spins++ % 200000000ull) == 0ull) {
          std::fprintf(stderr,
                       "[emu] idle: head=%u idx=%u want=%u got=%u done=%llu\n",
                       head_, idx, want, got,
                       (unsigned long long)done_.load());
        }
      }
      continue;
    }
    const EmuSqe sqe = sq_[idx];

    uint32_t status = 0;
    if (sqe.offset + sqe.nbytes > backing_size_) {
      status = 1;
    } else if (sqe.opcode == 0) {
      if (cudaMemcpyAsync(reinterpret_cast<void *>(sqe.bus_addr),
                          backing_ + sqe.offset, sqe.nbytes,
                          cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
          cudaStreamSynchronize(stream_) != cudaSuccess) {
        status = 2;
      }
    } else {
      if (cudaMemcpyAsync(const_cast<uint8_t *>(backing_) + sqe.offset,
                          reinterpret_cast<const void *>(sqe.bus_addr),
                          sqe.nbytes, cudaMemcpyDeviceToHost, stream_) !=
              cudaSuccess ||
          cudaStreamSynchronize(stream_) != cudaSuccess) {
        status = 2;
      }
    }

    // Completion LAST, released, so a consumer that sees the phase also sees
    // the data the phase advertises.
    const uint32_t cidx = sqe.cid & (depth_ - 1);
    cq_[cidx].cid = sqe.cid;
    cq_[cidx].status = status;
    __atomic_store_n(&cq_[cidx].phase, (sqe.cid / depth_) + 1u,
                     __ATOMIC_RELEASE);
    done_.fetch_add(1, std::memory_order_relaxed);
    ++head_;
  }
}

}  // namespace bam
