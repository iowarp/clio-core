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

#include <clio_runtime/bdev/bdev_runtime.h>
#include <clio_runtime/comutex.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_runtime/work_orchestrator.h>
#include <clio_runtime/worker.h>

#include <clio_ctp/introspect/system_info.h>
#include <clio_ctp/serialize/msgpack_wrapper.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>

#include "clio_ctp/util/timer.h"

namespace clio::run::bdev {

//===========================================================================
// WorkerIOContext Implementation
//===========================================================================

bool WorkerIOContext::Init(const std::string &file_path, chi::u32 io_depth,
                           chi::u32 worker_id) {
  if (is_initialized_) {
    return true;  // Already initialized
  }

  // Create async I/O backend via factory (io_depth passed at construction)
#if CTP_ENABLE_NIXL
  async_io_ = ctp::AsyncIoFactory::Get(io_depth, ctp::AsyncIoBackend::kNixl);
#else
  async_io_ = ctp::AsyncIoFactory::Get(io_depth);
#endif
  if (!async_io_) {
    HLOG(kError, "Worker {} failed to create async I/O backend", worker_id);
    return false;
  }

  // AsyncIO owns the file descriptors internally
  if (!async_io_->Open(file_path, O_RDWR | O_CREAT, 0644)) {
    HLOG(kError, "Worker {} failed to open file: {}", worker_id, file_path);
    async_io_.reset();
    return false;
  }

  is_initialized_ = true;
  HLOG(kDebug,
       "Worker {} I/O context initialized: event_fd={}",
       worker_id, async_io_->GetEventFd());
  return true;
}

void WorkerIOContext::Cleanup() {
  if (!is_initialized_) {
    return;
  }

  if (async_io_) {
    async_io_->Close();
    async_io_.reset();
  }

  is_initialized_ = false;
}

// Block size constants (in bytes) - 4KB, 16KB, 32KB, 64KB, 128KB, 1MB
static const size_t kBlockSizes[] = {
    4096,     // 4KB
    16384,    // 16KB
    32768,    // 32KB
    65536,    // 64KB
    131072,   // 128KB
    1048576   // 1MB
};

//===========================================================================
// Helper Functions
//===========================================================================

/**
 * Find the block type for a given I/O size (rounds to next largest)
 * @param io_size Requested I/O size
 * @param out_block_size Output parameter for the actual block size
 * @return Block type index, or -1 if larger than all cached sizes
 */
static int FindBlockTypeForSize(size_t io_size, size_t &out_block_size) {
  // Find the next block size that is larger than or equal to io_size
  for (int i = 0; i < static_cast<int>(BlockSizeCategory::kMaxCategories);
       ++i) {
    if (kBlockSizes[i] >= io_size) {
      out_block_size = kBlockSizes[i];
      return i;
    }
  }
  // If io_size is larger than all cached sizes, return -1
  out_block_size = io_size;  // Use exact size
  return -1;
}

//===========================================================================
// WorkerBlockMap Implementation
//===========================================================================

WorkerBlockMap::WorkerBlockMap() {
  // Initialize vector with 5 empty lists (one for each block size category)
  blocks_.resize(static_cast<size_t>(BlockSizeCategory::kMaxCategories));
}

bool WorkerBlockMap::AllocateBlock(int block_type, Block &block,
                                   size_t min_size) {
  if (block_type < 0 ||
      block_type >= static_cast<int>(BlockSizeCategory::kMaxCategories)) {
    return false;
  }

  auto &list = blocks_[block_type];
  if (list.empty()) {
    return false;
  }

  // For block_types where every freed block is exactly the bucket's nominal
  // size, the head element is always a fit and we exit on the first pop.
  // The largest bucket (kMaxCategories-1) is the fallthrough sink for any
  // freed block whose actual size exceeds the largest cached class — those
  // blocks can be any size >= kBlockSizes[max], so an explicit min_size
  // check is needed to avoid returning an undersized block to a caller
  // that asked for, say, 2 MiB.
  for (auto it = list.begin(); it != list.end(); ++it) {
    if (static_cast<size_t>(it->size_) >= min_size) {
      block = *it;
      list.erase(it);
      return true;
    }
  }
  return false;
}

void WorkerBlockMap::FreeBlock(Block block) {
  int block_type = static_cast<int>(block.block_type_);
  if (block_type >= 0 &&
      block_type < static_cast<int>(BlockSizeCategory::kMaxCategories)) {
    // Append to the block list
    blocks_[block_type].push_back(block);
  }
}

//===========================================================================
// GlobalBlockMap Implementation
//===========================================================================

GlobalBlockMap::GlobalBlockMap() {}

void GlobalBlockMap::Init(size_t num_workers) {
  // Pre-allocate vectors for specified number of workers
  worker_maps_.resize(num_workers);
  worker_locks_.resize(num_workers);
}

int GlobalBlockMap::FindBlockType(size_t io_size) {
  // Use the shared helper function to find block type
  size_t block_size;  // Not needed here, but required by the function signature
  return FindBlockTypeForSize(io_size, block_size);
}

bool GlobalBlockMap::AllocateBlock(int worker, size_t io_size, Block &block) {
  if (worker < 0 || static_cast<size_t>(worker) >= worker_maps_.size()) {
    return false;
  }

  size_t worker_idx = static_cast<size_t>(worker);

  // Find the next block size that is larger than this. When io_size exceeds
  // every cached class FindBlockType returns -1; mirror FreeBlocks' fallthrough
  // (which files such oversized blocks into the largest bucket via
  // FindBlockTypeForSize) so they stay reachable here too.  Without this
  // fallthrough, every freed 2 MiB block lands in bucket-max but AllocateBlock
  // never consults that bucket — so 2 MiB AllocateBlocks always falls through
  // to heap_.Allocate, growing the bdev's footprint monotonically with op
  // count.  WorkerBlockMap::AllocateBlock validates min_size for the
  // largest bucket so we don't accidentally return an undersized block.
  int block_type = FindBlockType(io_size);
  if (block_type == -1) {
    block_type = static_cast<int>(BlockSizeCategory::kMaxCategories) - 1;
  }

  // Acquire this worker's mutex using ScopedCoMutex
  {
    chi::ScopedCoMutex lock(worker_locks_[worker_idx]);
    // First attempt to allocate from this worker's map
    if (worker_maps_[worker_idx].AllocateBlock(block_type, block, io_size)) {
      return true;
    }
  }

  // If we fail, try up to 4 other workers (iterate linearly)
  size_t num_workers = worker_maps_.size();
  for (size_t i = 1; i <= 4 && i < num_workers; ++i) {
    size_t other_worker = (worker_idx + i) % num_workers;
    chi::ScopedCoMutex lock(worker_locks_[other_worker]);
    if (worker_maps_[other_worker].AllocateBlock(block_type, block, io_size)) {
      return true;
    }
  }

  return false;
}

bool GlobalBlockMap::FreeBlock(int worker, Block &block) {
  if (worker < 0 || static_cast<size_t>(worker) >= worker_maps_.size()) {
    return false;
  }

  size_t worker_idx = static_cast<size_t>(worker);

  // Free on this worker's map (with lock for thread safety)
  chi::ScopedCoMutex lock(worker_locks_[worker_idx]);
  worker_maps_[worker_idx].FreeBlock(block);
  return true;
}

//===========================================================================
// Heap Implementation
//===========================================================================

Heap::Heap() : heap_(0), total_size_(0), alignment_(4096) {}

void Heap::Init(chi::u64 total_size, chi::u32 alignment) {
  total_size_ = total_size;
  alignment_ = (alignment == 0) ? 4096 : alignment;
  heap_.store(0);
}

bool Heap::Allocate(size_t block_size, int block_type, Block &block) {
  // Align the requested block size to alignment boundary for O_DIRECT I/O
  // Formula: aligned_size = ((block_size + alignment_ - 1) / alignment_) *
  // alignment_
  chi::u32 alignment = (alignment_ == 0) ? 4096 : alignment_;

  // Align the requested size
  chi::u64 aligned_size =
      ((block_size + alignment - 1) / alignment) * alignment;
  HLOG(kDebug,
       "Allocating block: block_size = {}, alignment = {}, aligned_size = {}",
       block_size, alignment, aligned_size);

  // Atomic fetch-and-add to allocate from heap using aligned size
  chi::u64 old_heap = heap_.fetch_add(aligned_size);

  if (old_heap + aligned_size > total_size_) {
    // Out of space - rollback
    return false;
  }

  // Allocation successful - both offset and size are aligned
  block.offset_ = old_heap;
  block.size_ = aligned_size;
  block.block_type_ = static_cast<chi::u32>(block_type);
  return true;
}

chi::u64 Heap::GetRemainingSize() const {
  chi::u64 current_heap = heap_.load();
  if (current_heap >= total_size_) {
    return 0;
  }
  return total_size_ - current_heap;
}

Runtime::~Runtime() {
  // Clean up libaio (only for file-based storage)
  if (bdev_type_ == BdevType::kFile) {
    CleanupAsyncIO();
    CleanupWorkerIOContexts();
  }

  // Clean up RAM backend (vector of unique_ptr<char[]> handles itself)

  // kHbm / kPinned bdev tiers were removed — kRam/kFile handle device
  // USM source/dest pointers directly via ctp::DeviceAwareMemcpy.

  // Note: GlobalBlockMap and Heap destructors will clean up automatically
}

bool Runtime::InitializeWorkerIOContexts() {
  // Pre-allocate vector based on actual number of workers
  chi::WorkOrchestrator *work_orchestrator = CLIO_WORK_ORCHESTRATOR;
  size_t num_workers =
      work_orchestrator ? work_orchestrator->GetWorkerCount() : 16;
  worker_io_contexts_.resize(num_workers);
  // Contexts are lazily initialized when first accessed
  return true;
}

void Runtime::CleanupWorkerIOContexts() {
  for (auto &ctx : worker_io_contexts_) {
    ctx.Cleanup();
  }
  worker_io_contexts_.clear();
}

WorkerIOContext *Runtime::GetWorkerIOContext(size_t worker_id) {
  // Check bounds - vector is pre-allocated in InitializeWorkerIOContexts
  if (worker_id >= worker_io_contexts_.size()) {
    HLOG(kWarning, "Worker ID {} exceeds pre-allocated size {}", worker_id,
         worker_io_contexts_.size());
    return nullptr;
  }

  WorkerIOContext *ctx = &worker_io_contexts_[worker_id];

  // Lazy initialization: initialize on first access
  if (!ctx->is_initialized_) {
    if (!ctx->Init(file_path_, io_depth_, static_cast<chi::u32>(worker_id))) {
      HLOG(kError, "Failed to initialize I/O context for worker {}", worker_id);
      return nullptr;
    }

    // Register the eventfd with the worker's EventManager for completion notification
    int event_fd = ctx->async_io_ ? ctx->async_io_->GetEventFd() : -1;
    chi::Worker *worker = CLIO_CUR_WORKER;
    if (worker != nullptr && event_fd >= 0) {
      auto &em = worker->GetEventManager();
      if (em.AddEvent(event_fd) < 0) {
        HLOG(kWarning, "Failed to register eventfd with worker {} EventManager",
             worker_id);
      } else {
        HLOG(kDebug, "Registered eventfd {} with worker {} EventManager",
             event_fd, worker_id);
      }
    }
  }

  return ctx;
}

chi::TaskStat Runtime::GetTaskStats(const chi::Task *task) const {
  if (!task) return chi::TaskStat();
  switch (task->method_) {
    case Method::kWrite: {
      auto *wt = static_cast<const WriteTask *>(task);
      chi::TaskStat stat;
      stat.io_size_ = wt->length_;
      // wall_time = aligned pages / 500 MB/s
      size_t aligned = ((stat.io_size_ + 4095) / 4096) * 4096;
      stat.wall_time_ = static_cast<float>(aligned) / 500.0f;
      return stat;
    }
    case Method::kRead: {
      auto *rt = static_cast<const ReadTask *>(task);
      chi::TaskStat stat;
      stat.io_size_ = rt->length_;
      size_t aligned = ((stat.io_size_ + 4095) / 4096) * 4096;
      stat.wall_time_ = static_cast<float>(aligned) / 500.0f;
      return stat;
    }
    default: return chi::TaskStat();
  }
}

chi::TaskResume Runtime::Create(ctp::ipc::FullPtr<CreateTask> task,
                                chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  // Get the creation parameters
  CreateParams params = task->GetParams();

  // Get the pool name which serves as the file path for file-based operations
  std::string pool_name = task->pool_name_.str();

  HLOG(kDebug,
       "Bdev runtime received params: bdev_type={}, pool_name='{}', "
       "total_size={}, io_depth={}, alignment={}",
       static_cast<chi::u32>(params.bdev_type_), pool_name, params.total_size_,
       params.io_depth_, params.alignment_);

  // Store backend type
  bdev_type_ = params.bdev_type_;

  // Initialize storage backend based on type
  if (bdev_type_ == BdevType::kFile) {
    // Store file path for per-worker FD creation
    file_path_ = pool_name;

    // Use a temporary AsyncIO to set up the file (create/truncate)
#if CTP_ENABLE_NIXL
    auto setup_io = ctp::AsyncIoFactory::Get(io_depth_, ctp::AsyncIoBackend::kNixl);
#else
    auto setup_io = ctp::AsyncIoFactory::Get(io_depth_);
#endif
    if (!setup_io) {
      HLOG(kError, "Failed to create setup async I/O backend");
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }

    if (!setup_io->Open(pool_name, O_RDWR | O_CREAT, 0644)) {
      HLOG(kError, "Failed to open file: {}", pool_name);
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }

    // Get file size
    ssize_t current_size = setup_io->GetFileSize();
    if (current_size < 0) {
      task->return_code_ = 2;
      setup_io->Close();
      CLIO_CO_RETURN;
    }

    file_size_ = static_cast<chi::u64>(current_size);
    HLOG(kDebug, "File stat: file_size={}, params.total_size={}", file_size_,
         params.total_size_);

    if (params.total_size_ > 0 && params.total_size_ < file_size_) {
      file_size_ = params.total_size_;
    }

    // If file is empty, create it with default size (1GB)
    if (file_size_ == 0) {
      file_size_ = (params.total_size_ > 0) ? params.total_size_
                                            : (1ULL << 30);  // 1GB default
      HLOG(kDebug,
           "File is empty, setting file_size_ to {} and calling Truncate",
           file_size_);
      if (!setup_io->Truncate(static_cast<size_t>(file_size_))) {
        task->return_code_ = 3;
        HLOG(kError, "Failed to truncate file: {}", pool_name);
        setup_io->Close();
        CLIO_CO_RETURN;
      }
      HLOG(kDebug, "Truncate succeeded, file_size_={}", file_size_);
    }
    HLOG(kDebug, "Create: Final file_size_={}, initializing allocator",
         file_size_);

    // Close setup I/O — per-worker contexts will open their own
    setup_io->Close();

    // Initialize async I/O for file backend (legacy POSIX AIO)
    InitializeAsyncIO();

    // Initialize per-worker I/O contexts for parallel file access
    if (!InitializeWorkerIOContexts()) {
      HLOG(kWarning,
           "Failed to initialize per-worker I/O contexts, "
           "falling back to single FD");
    }

  } else if (bdev_type_ == BdevType::kRam) {
    // RAM-based storage initialization.
    //   capacity == 0 → default to 80% of total system DRAM (NOT
    //                    unbounded: an unbounded RAM tier lets the
    //                    allocator hand out more than physical memory and
    //                    OOM-kills the daemon on a shared compute node).
    //   capacity  > 0 → bounded; size enforced lazily on AllocateBlocks /
    //                    WriteToRam (see file_size_ in the Heap allocator).
    //
    // Either way we leave ram_pages_ empty here and grow it on the first
    // write that targets each 1 GiB slot. The prior eager allocation
    // path (new char[kRamPageSize] × N + memset) was a benchmark warm-
    // up: it forced the kernel to commit all N GiB of physical pages
    // before the timed loop. On a multi-tenant compute node — head node
    // runs jarvis + ssh fan-outs + FUSE + many IOR ranks + the daemon
    // itself — a 32 GiB upfront commit pushes physical RAM and the
    // slurm cgroup vm budget past the limit and the daemon gets killed
    // (silently: no SEGV trace, just disappears). For a 32 GiB × 4n
    // workload that's 128 GiB cluster-wide of unneeded RSS at startup.
    //
    // Even the cheaper "reserve 1 GiB per slot, touch 1 byte" variant
    // (which only commits ~128 KiB physical) costs 32 GiB of virtual
    // address space — and the slurm cgroup or RLIMIT_AS can refuse that
    // (libzmq inside the daemon then hits an unrelated allocation that
    // returns EFAULT and asserts "Bad address" in tcp.cpp). Skipping the
    // reservation entirely is the correct fix: the only producer of
    // ram_pages_ entries is WriteToRam, which already handles "page not
    // yet allocated" by allocating on the spot under ram_pages_mu_.
    ram_capacity_ = (params.total_size_ == 0) ? DefaultRamCapacityBytes()
                                               : params.total_size_;
    HLOG(kInfo,
         "RAM bdev '{}' capacity: configured={} -> using {} bytes "
         "({}% of {} total DRAM when configured as 0/0g)",
         pool_name, params.total_size_, ram_capacity_,
         static_cast<int>(kDefaultRamCapacityFraction * 100),
         ctp::SystemInfo::GetRamCapacity());
    file_size_ = ram_capacity_;  // Heap allocator's soft cap

  // BdevType::kHbm and BdevType::kPinned removed — supported tiers
  // are kFile / kRam / kNoop. PutBlob/GetBlob with HBM-resident
  // ShmPtr data buffers route through kRam (or kFile) and the bdev
  // staging path uses ctp::DeviceAwareMemcpy / IsDevicePointer.
  } else if (bdev_type_ == BdevType::kNoop) {
    // Noop backend: no storage buffer, just track allocatable size
    if (params.total_size_ == 0) {
      task->return_code_ = 4;
      co_return;
    }
    file_size_ = params.total_size_;
  }

  // Initialize common parameters
  alignment_ = params.alignment_;
  io_depth_ = params.io_depth_;

  // Initialize the data allocator
  InitializeAllocator();

  // UpdateTask is sent in PostGpuContainerCreate(), called after the GPU
  // container is registered so it arrives when the container is ready.

  // Initialize performance tracking
  start_time_ = std::chrono::high_resolution_clock::now();
  total_reads_ = 0;
  total_writes_ = 0;
  total_bytes_read_ = 0;
  total_bytes_written_ = 0;

  // Store user-provided performance characteristics
  perf_metrics_ = params.perf_metrics_;

  // Set success result
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::AllocateBlocks(ctp::ipc::FullPtr<AllocateBlocksTask> task,
                                        chi::RunContext &ctx) {
  chi::RunContext& rctx = ctx;
  CLIO_TASK_BODY_BEGIN
  HLOG(kDebug,
       "bdev::AllocateBlocks: ENTER - pool_id_=({},{}), size={}, "
       "container_id={}",
       task->pool_id_.major_, task->pool_id_.minor_, task->size_,
       container_id_);

  // Get worker ID for allocation
  int worker_id = static_cast<int>(GetWorkerID(rctx));

  chi::u64 total_size = task->size_;
  if (total_size == 0) {
    HLOG(kDebug, "bdev::AllocateBlocks: size is 0, returning empty blocks");
    task->blocks_.clear();
    task->return_code_ = 0;  // Nothing to allocate
    CLIO_CO_RETURN;
  }

  // Create local vector in private memory to build up the block list
  std::vector<Block> local_blocks;

  // Allocate the request as a SINGLE contiguous block (no kMaxBlock-chunk
  // splitting).  The old splitting path divided e.g. a 2 MiB ExtendBlob into
  // two 1 MiB Blocks, but the only consumer (CTE's AllocateFromTarget) reads
  // `allocated_blocks[0].offset_` and discards the rest — it tracks one
  // `BlobBlock(size=2 MiB)` covering the first allocator chunk plus an
  // un-tracked tail.  On overwrite, the corresponding FreeBlocks returns one
  // 2 MiB block to the largest free-list bucket, but the heap had consumed
  // two 1 MiB chunks; on the next AllocateBlocks the first 1 MiB sub-alloc
  // pops the 2 MiB block (with my min_size filter) but wastes its tail,
  // and the second 1 MiB falls through to heap_.  Net leak: 1 MiB per
  // 2 MiB overwrite.  Keeping the alloc one block end-to-end matches what
  // CTE actually stores and lets the free-list reuse cycle close cleanly.
  std::vector<size_t> io_divisions;
  io_divisions.push_back(static_cast<size_t>(total_size));

  // For each expected I/O size division, allocate a block
  for (size_t io_size : io_divisions) {
    Block block;
    bool allocated = false;

    // First attempt to allocate from the GlobalBlockMap
    if (global_block_map_.AllocateBlock(worker_id, io_size, block)) {
      allocated = true;
    } else {
      // If that fails, allocate from heap
      // Find the appropriate block type and size for this I/O size
      size_t alloc_size;
      int block_type = FindBlockTypeForSize(io_size, alloc_size);

      // If no cached size fits, use largest category
      if (block_type == -1) {
        block_type = static_cast<int>(BlockSizeCategory::kMaxCategories) - 1;
      }

      if (heap_.Allocate(alloc_size, block_type, block)) {
        allocated = true;
      }
    }

    // If allocation failed, clean up and return error
    if (!allocated) {
      // Return all allocated blocks to the GlobalBlockMap
      for (Block &allocated_block : local_blocks) {
        global_block_map_.FreeBlock(worker_id, allocated_block);
      }
      task->blocks_.clear();
      // HLOG(kError, "Out of space: {} bytes requested", total_size);
      task->return_code_ = 1;  // Out of space
      CLIO_CO_RETURN;
    }

    // Add the allocated block to the local vector
    local_blocks.push_back(block);
  }

  // Copy the local vector to the task's shared memory vector using assignment
  // operator
  // task->blocks_ = local_blocks;
  chi::u64 alloc_bytes = 0;
  for (size_t i = 0; i < local_blocks.size(); i++) {
    task->blocks_.push_back(local_blocks[i]);
    alloc_bytes += local_blocks[i].size_;
  }
  // Track LIVE allocated bytes (same per-block size FreeBlocks subtracts),
  // independent of free-list vs heap source. Drives GetStats' true
  // remaining capacity instead of heap_'s monotonic bump high-water.
  allocated_bytes_.fetch_add(alloc_bytes, std::memory_order_relaxed);

  HLOG(kDebug,
       "bdev::AllocateBlocks: SUCCESS - allocated {} blocks, "
       "task->blocks_.size()={}",
       local_blocks.size(), task->blocks_.size());

  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::FreeBlocks(ctp::ipc::FullPtr<FreeBlocksTask> task,
                                    chi::RunContext &ctx) {
  chi::RunContext& rctx = ctx;
  CLIO_TASK_BODY_BEGIN
  // Get worker ID for free operation
  int worker_id = static_cast<int>(GetWorkerID(rctx));

  // Free all blocks in the vector using GlobalBlockMap.
  //
  // Normalize block_type_ from the block's SIZE before filing it into the
  // free list. AllocateBlock picks the free list via FindBlockType(size),
  // so a freed block must be filed in that same size class to ever be
  // reused. Callers do not (and cannot) reliably track the allocator's
  // size class: BlobBlock carries only {offset,size}, so CTE's
  // FreeAllBlobBlocks passes block_type_=0. Trusting that put every freed
  // 1 MiB block in the 4 KiB list, so 1 MiB AllocateBlock never found
  // them and fell through to the monotonic heap — RAM usage grew with op
  // count regardless of the live key set until the tier cap was hit.
  // Classifying by size here makes the allocator self-consistent for
  // every caller.
  chi::u64 freed_bytes = 0;
  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    Block block_copy = task->blocks_[i];  // Make a copy since FreeBlock takes
                                          // non-const reference
    size_t cat_size = 0;
    int bt = FindBlockTypeForSize(static_cast<size_t>(block_copy.size_),
                                  cat_size);
    if (bt < 0) {
      // Larger than every cached class — mirror AllocateBlocks, which
      // uses the largest category for such sizes.
      bt = static_cast<int>(BlockSizeCategory::kMaxCategories) - 1;
    }
    block_copy.block_type_ = static_cast<chi::u32>(bt);
    freed_bytes += block_copy.size_;
    global_block_map_.FreeBlock(worker_id, block_copy);
  }
  // Reclaim live-byte accounting so GetStats' remaining recovers as
  // blocks are reused (pairs with AllocateBlocks' fetch_add). Guard the
  // subtraction so a double-free / mismatched free can't underflow the
  // unsigned counter.
  {
    chi::u64 cur = allocated_bytes_.load(std::memory_order_relaxed);
    chi::u64 dec = std::min(cur, freed_bytes);
    allocated_bytes_.fetch_sub(dec, std::memory_order_relaxed);
  }

  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Write(ctp::ipc::FullPtr<WriteTask> task,
                               chi::RunContext &ctx) {
  chi::RunContext& rctx = ctx;
  CLIO_TASK_BODY_BEGIN
  switch (bdev_type_) {
    case BdevType::kFile:
      CLIO_CO_AWAIT(WriteToFile(task, rctx));
      break;
    case BdevType::kRam:
      WriteToRam(task);
      break;
    case BdevType::kHbm:
    case BdevType::kPinned:
      // Removed tiers; reject as unsupported.
      task->return_code_ = 1;
      task->bytes_written_ = 0;
      break;
    case BdevType::kNoop:
      task->return_code_ = 0;
      task->bytes_written_ = task->length_;
      break;
    default:
      task->return_code_ = 1;
      task->bytes_written_ = 0;
      break;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Read(ctp::ipc::FullPtr<ReadTask> task,
                              chi::RunContext &ctx) {
  chi::RunContext& rctx = ctx;
  CLIO_TASK_BODY_BEGIN
  switch (bdev_type_) {
    case BdevType::kFile:
      CLIO_CO_AWAIT(ReadFromFile(task, rctx));
      break;
    case BdevType::kRam:
      ReadFromRam(task);
      break;
    case BdevType::kHbm:
    case BdevType::kPinned:
      // Removed tiers; reject as unsupported.
      task->return_code_ = 1;
      task->bytes_read_ = 0;
      break;
    case BdevType::kNoop:
      task->return_code_ = 0;
      task->bytes_read_ = task->length_;
      break;
    default:
      task->return_code_ = 1;
      task->bytes_read_ = 0;
      break;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::WriteToFile(ctp::ipc::FullPtr<WriteTask> task,
                                     chi::RunContext &ctx) {
  chi::RunContext& rctx = ctx;
  CLIO_TASK_BODY_BEGIN
  size_t worker_id = GetWorkerID(rctx);
  WorkerIOContext *io_ctx = GetWorkerIOContext(worker_id);

  auto *ipc_mgr = CLIO_IPC;
  ctp::ipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).Cast<char>();

  // libaio / POSIX-AIO can't dereference device USM, so stage through
  // a host buffer when the data lives on device. The host staging
  // buffer is sized to the largest single-block write — typical CTE
  // PutBlob is one block, but worst-case loop bound is task->length_.
  bool data_on_device = ctp::IsDevicePointer(data_ptr.ptr_);
  std::vector<char> staging;
  if (data_on_device) {
    staging.resize(task->length_);
    ctp::DeviceAwareMemcpy(staging.data(), data_ptr.ptr_, task->length_);
  }

  chi::u64 total_bytes_written = 0;
  chi::u64 data_offset = 0;

  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];

    chi::u64 remaining = task->length_ - total_bytes_written;
    if (remaining == 0) break;
    chi::u64 block_write_size = std::min(remaining, block.size_);

    void *block_data = data_on_device
                           ? static_cast<void *>(staging.data() + data_offset)
                           : static_cast<void *>(data_ptr.ptr_ + data_offset);

    if (io_ctx == nullptr || !io_ctx->is_initialized_ || !io_ctx->async_io_) {
      HLOG(kError, "WriteToFile called with invalid I/O context");
      task->return_code_ = 1;
      task->bytes_written_ = total_bytes_written;
      CLIO_CO_RETURN;
    }

    ctp::IoToken token = io_ctx->async_io_->Write(
        block_data, static_cast<size_t>(block_write_size),
        static_cast<off_t>(block.offset_));
    if (token == ctp::kInvalidIoToken) {
      HLOG(kError, "Failed to submit async write: offset={}, size={}",
           block.offset_, block_write_size);
      task->return_code_ = 2;
      task->bytes_written_ = total_bytes_written;
      CLIO_CO_RETURN;
    }

    ctp::IoResult result;
    while (!io_ctx->async_io_->IsComplete(token, result)) {
      CLIO_CO_AWAIT(chi::yield(10.0));
    }

    if (result.error_code != 0) {
      HLOG(kError, "Async write failed: error_code={}", result.error_code);
      task->return_code_ = 4;
      task->bytes_written_ = total_bytes_written;
      CLIO_CO_RETURN;
    }

    chi::u64 actual_bytes = std::min(
        static_cast<chi::u64>(result.bytes_transferred), block_write_size);
    total_bytes_written += actual_bytes;
    data_offset += actual_bytes;
  }

  task->return_code_ = 0;
  task->bytes_written_ = total_bytes_written;
  total_writes_.fetch_add(1);
  total_bytes_written_.fetch_add(task->bytes_written_);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::ReadFromFile(ctp::ipc::FullPtr<ReadTask> task,
                                      chi::RunContext &ctx) {
  chi::RunContext& rctx = ctx;
  CLIO_TASK_BODY_BEGIN
  size_t worker_id = GetWorkerID(rctx);
  WorkerIOContext *io_ctx = GetWorkerIOContext(worker_id);

  auto *ipc_mgr = CLIO_IPC;
  ctp::ipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).Cast<char>();

  // libaio / POSIX-AIO can't write into device USM. When the dest is
  // device, allocate a host staging buffer, AIO into it, then
  // DeviceAwareMemcpy to the device dest at the end.
  bool data_on_device = ctp::IsDevicePointer(data_ptr.ptr_);
  std::vector<char> staging;
  if (data_on_device) {
    staging.resize(task->length_);
  }

  chi::u64 total_bytes_read = 0;
  chi::u64 data_offset = 0;

  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];

    chi::u64 remaining = task->length_ - total_bytes_read;
    if (remaining == 0) break;
    chi::u64 block_read_size = std::min(remaining, block.size_);

    void *block_data = data_on_device
                           ? static_cast<void *>(staging.data() + data_offset)
                           : static_cast<void *>(data_ptr.ptr_ + data_offset);

    if (io_ctx == nullptr || !io_ctx->is_initialized_ || !io_ctx->async_io_) {
      HLOG(kError, "ReadFromFile called with invalid I/O context");
      task->return_code_ = 1;
      task->bytes_read_ = total_bytes_read;
      CLIO_CO_RETURN;
    }

    ctp::IoToken token = io_ctx->async_io_->Read(
        block_data, static_cast<size_t>(block_read_size),
        static_cast<off_t>(block.offset_));
    if (token == ctp::kInvalidIoToken) {
      HLOG(kError, "Failed to submit async read: offset={}, size={}",
           block.offset_, block_read_size);
      task->return_code_ = 2;
      task->bytes_read_ = total_bytes_read;
      CLIO_CO_RETURN;
    }

    ctp::IoResult result;
    while (!io_ctx->async_io_->IsComplete(token, result)) {
      CLIO_CO_AWAIT(chi::yield(10.0));
    }

    if (result.error_code != 0) {
      HLOG(kError, "Async read failed: error_code={}", result.error_code);
      task->return_code_ = 4;
      task->bytes_read_ = total_bytes_read;
      CLIO_CO_RETURN;
    }

    chi::u64 actual_bytes = std::min(
        static_cast<chi::u64>(result.bytes_transferred), block_read_size);
    total_bytes_read += actual_bytes;
    data_offset += actual_bytes;
  }

  // If we staged through a host buffer, push the freshly-read bytes
  // out to the device-USM destination. (No-op when data is on host.)
  if (data_on_device && total_bytes_read > 0) {
    ctp::DeviceAwareMemcpy(data_ptr.ptr_, staging.data(), total_bytes_read);
  }

  task->return_code_ = 0;
  task->bytes_read_ = total_bytes_read;
  total_reads_.fetch_add(1);
  total_bytes_read_.fetch_add(total_bytes_read);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Update(ctp::ipc::FullPtr<UpdateTask> task,
                                chi::RunContext &ctx) {
  // UpdateTask is meant for the GPU container only.
  // The CPU runtime receives it as a no-op.
  task->return_code_ = 0;
  (void)ctx;
  co_return;
}


chi::TaskResume Runtime::GetStats(ctp::ipc::FullPtr<GetStatsTask> task,
                                  chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  // Predict wall time from learned model using a synthetic 1 MiB R/W
  // task as the reference size for the bandwidth/latency estimate.
  ReadTask r_synthetic;
  r_synthetic.method_ = Method::kRead;
  r_synthetic.length_ = 1024 * 1024;
  WriteTask w_synthetic;
  w_synthetic.method_ = Method::kWrite;
  w_synthetic.length_ = 1024 * 1024;
  chi::TaskStat read_stat = GetTaskStats(&r_synthetic);
  chi::TaskStat write_stat = GetTaskStats(&w_synthetic);
  float read_wall_us = InferWallClockTime(Method::kRead, read_stat);
  float write_wall_us = InferWallClockTime(Method::kWrite, write_stat);
  double read_size_mb = static_cast<double>(read_stat.io_size_) / (1024.0 * 1024.0);
  double write_size_mb = static_cast<double>(write_stat.io_size_) / (1024.0 * 1024.0);
  task->metrics_.read_bandwidth_mbps_ = (read_wall_us > 0)
      ? read_size_mb / (read_wall_us * 1e-6) : perf_metrics_.read_bandwidth_mbps_;
  task->metrics_.write_bandwidth_mbps_ = (write_wall_us > 0)
      ? write_size_mb / (write_wall_us * 1e-6) : perf_metrics_.write_bandwidth_mbps_;
  task->metrics_.read_latency_us_ = read_wall_us;
  task->metrics_.write_latency_us_ = write_wall_us;
  task->metrics_.iops_ = perf_metrics_.iops_;
  // Remaining = capacity - LIVE allocated bytes. NOT heap_.GetRemainingSize()
  // (heap_ is a monotonic bump pointer never rolled back on free, so under
  // concurrent free-list misses it raced past the true live set and
  // collapsed CTE's StatTargets remaining_space_ to ~0 -> MaxBwDpe
  // rejected the only target -> ExtendBlob=2 -> PutBlob rc=12). file_size_
  // is what heap_ was Init'd with (= ram_capacity_ for kRam).
  chi::u64 live = allocated_bytes_.load(std::memory_order_relaxed);
  chi::u64 remaining = (file_size_ > live) ? (file_size_ - live) : 0;
  task->remaining_size_ = remaining;
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Destroy(ctp::ipc::FullPtr<DestroyTask> task,
                                 chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  // Worker I/O contexts (and their AsyncIO instances) are cleaned up by destructor
  // Note: GlobalBlockMap and Heap cleanup is handled by their destructors

  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::InitializeAllocator() {
  // Initialize global block map with actual number of workers
  chi::WorkOrchestrator *work_orchestrator = CLIO_WORK_ORCHESTRATOR;
  size_t num_workers =
      work_orchestrator ? work_orchestrator->GetWorkerCount() : 16;
  global_block_map_.Init(num_workers);

  // Initialize heap with total file size and alignment requirement
  heap_.Init(file_size_, alignment_);
}

size_t Runtime::GetBlockSize(int block_type) {
  if (block_type >= 0 &&
      block_type < static_cast<int>(BlockSizeCategory::kMaxCategories)) {
    return kBlockSizes[block_type];
  }
  return 0;
}

size_t Runtime::GetWorkerID(chi::RunContext &ctx) {
  // Get current worker from thread-local storage using CLIO_CUR_WORKER macro
  chi::Worker *worker = CLIO_CUR_WORKER;
  if (worker == nullptr) {
    return 0;  // Fallback to worker 0 if not in worker context
  }
  return worker->GetId();
}


chi::u64 Runtime::AlignSize(chi::u64 size) {
  if (alignment_ == 0) {
    alignment_ = 4096;  // Set to default if somehow it's 0
  }
  return ((size + alignment_ - 1) / alignment_) * alignment_;
}

void Runtime::UpdatePerformanceMetrics(bool is_write, chi::u64 bytes,
                                       double duration_us) {
  // This is a simplified implementation
  // In a real implementation, you'd maintain running averages or histograms
}

void Runtime::InitializeAsyncIO() {
  // No initialization needed for POSIX AIO fallback
}

void Runtime::CleanupAsyncIO() {
  // No cleanup needed for POSIX AIO fallback
}

char* Runtime::EnsureRamPage(size_t page_idx) {
  std::lock_guard<std::mutex> lock(ram_pages_mu_);
  if (page_idx >= ram_pages_.size()) {
    ram_pages_.resize(page_idx + 1);
  }
  if (!ram_pages_[page_idx]) {
    // Lazy alloc for pages beyond the eagerly-allocated range (or for
    // unbounded-capacity bdevs). The default-init `new char[]` reserves
    // virtual address space; physical pages fault in on first touch by
    // WriteToRam's DeviceAwareMemcpy. We do NOT memset here — that would
    // double the memory traffic (one pass to zero-fault the page, a
    // second pass to memcpy the user's data), capping Put bandwidth at
    // half of memory bandwidth. The bounded-capacity path in Create()
    // pre-allocates and pre-faults so the cost lands outside any
    // benchmark loop.
    ram_pages_[page_idx].reset(new char[kRamPageSize]);
  }
  return ram_pages_[page_idx].get();
}

char* Runtime::GetRamPage(size_t page_idx) const {
  std::lock_guard<std::mutex> lock(ram_pages_mu_);
  if (page_idx >= ram_pages_.size()) return nullptr;
  return ram_pages_[page_idx].get();
}

void Runtime::WriteToRam(ctp::ipc::FullPtr<WriteTask> task) {
  auto *ipc_mgr = CLIO_IPC;
  ctp::ipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).Cast<char>();

  chi::u64 total_bytes_written = 0;
  chi::u64 data_offset = 0;

  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];

    chi::u64 remaining = task->length_ - total_bytes_written;
    if (remaining == 0) break;
    chi::u64 block_write_size = std::min(remaining, block.size_);

    if (ram_capacity_ != std::numeric_limits<chi::u64>::max() &&
        block.offset_ + block_write_size > ram_capacity_) {
      task->return_code_ = 1;
      task->bytes_written_ = total_bytes_written;
      HLOG(kError,
           "Write to RAM beyond capacity offset: {}, length: {}, "
           "ram_capacity: {}",
           block.offset_, block_write_size, ram_capacity_);
      return;
    }

    // Walk the (offset, size) range across 1 GiB pages, allocating a page
    // on first write only. Bench-sized writes (≤ block size, typically MBs)
    // touch one page; this loop only runs >1 iteration when a block straddles
    // a 1 GiB boundary. DeviceAwareMemcpy dispatches through
    // sycl::queue::memcpy (or the CUDA equivalent) when the data ShmPtr
    // resolves to device USM, and falls back to std::memcpy otherwise.
    chi::u64 cur_off = block.offset_;
    chi::u64 left = block_write_size;
    while (left > 0) {
      size_t page_idx = static_cast<size_t>(cur_off / kRamPageSize);
      chi::u64 intra = cur_off % kRamPageSize;
      chi::u64 chunk = std::min<chi::u64>(left, kRamPageSize - intra);
      char* page = EnsureRamPage(page_idx);
      ctp::DeviceAwareMemcpy(page + intra,
                             data_ptr.ptr_ + data_offset,
                             chunk);
      cur_off += chunk;
      data_offset += chunk;
      left -= chunk;
    }

    total_bytes_written += block_write_size;
  }

  task->return_code_ = 0;
  task->bytes_written_ = total_bytes_written;

  total_writes_.fetch_add(1);
  total_bytes_written_.fetch_add(task->bytes_written_);
}

void Runtime::ReadFromRam(ctp::ipc::FullPtr<ReadTask> task) {
  auto *ipc_mgr = CLIO_IPC;
  ctp::ipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).Cast<char>();

  chi::u64 total_bytes_read = 0;
  chi::u64 data_offset = 0;

  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];

    chi::u64 remaining = task->length_ - total_bytes_read;
    if (remaining == 0) break;
    chi::u64 block_read_size = std::min(remaining, block.size_);

    if (ram_capacity_ != std::numeric_limits<chi::u64>::max() &&
        block.offset_ + block_read_size > ram_capacity_) {
      task->return_code_ = 1;
      task->bytes_read_ = total_bytes_read;
      HLOG(kError,
           "Read from RAM beyond capacity offset: {}, length: {}, "
           "ram_capacity: {}",
           block.offset_, block_read_size, ram_capacity_);
      return;
    }

    // Sparse semantics: a never-written page reads back as zeros, mirroring
    // a sparse file. DeviceAwareMemcpy handles device-USM data buffers
    // (see WriteToRam comment); for the zero-fill branch we copy from a
    // static zero scratch when the dest is device-resident, since plain
    // memset on a device USM pointer would segfault on the host.
    chi::u64 cur_off = block.offset_;
    chi::u64 left = block_read_size;
    while (left > 0) {
      size_t page_idx = static_cast<size_t>(cur_off / kRamPageSize);
      chi::u64 intra = cur_off % kRamPageSize;
      chi::u64 chunk = std::min<chi::u64>(left, kRamPageSize - intra);
      char* page = GetRamPage(page_idx);
      char *dst = data_ptr.ptr_ + data_offset;
      if (page) {
        ctp::DeviceAwareMemcpy(dst, page + intra, chunk);
      } else if (ctp::IsDevicePointer(dst)) {
        static const char kZeroScratch[4096] = {};
        chi::u64 z_left = chunk;
        chi::u64 z_off = 0;
        while (z_left > 0) {
          chi::u64 z_chunk =
              std::min<chi::u64>(z_left, sizeof(kZeroScratch));
          ctp::DeviceAwareMemcpy(dst + z_off, kZeroScratch, z_chunk);
          z_off += z_chunk;
          z_left -= z_chunk;
        }
      } else {
        memset(dst, 0, chunk);
      }
      cur_off += chunk;
      data_offset += chunk;
      left -= chunk;
    }

    total_bytes_read += block_read_size;
  }

  task->return_code_ = 0;
  task->bytes_read_ = total_bytes_read;

  total_reads_.fetch_add(1);
  total_bytes_read_.fetch_add(total_bytes_read);
}

// VIRTUAL METHOD IMPLEMENTATIONS (now in autogen/bdev_lib_exec.cc)

chi::u64 Runtime::GetWorkRemaining() const { return 0; }

chi::TaskResume Runtime::Monitor(ctp::ipc::FullPtr<MonitorTask> task,
                                 chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  (void)rctx;
  if (task->query_ == "stats") {
    // Predict wall time from learned model using a synthetic 1 MiB R/W
    // task as the reference size (matches GetStats handler above).
    ReadTask r_synthetic;
    r_synthetic.method_ = Method::kRead;
    r_synthetic.length_ = 1024 * 1024;
    WriteTask w_synthetic;
    w_synthetic.method_ = Method::kWrite;
    w_synthetic.length_ = 1024 * 1024;
    chi::TaskStat read_stat = GetTaskStats(&r_synthetic);
    chi::TaskStat write_stat = GetTaskStats(&w_synthetic);
    float read_wall_us = InferWallClockTime(Method::kRead, read_stat);
    float write_wall_us = InferWallClockTime(Method::kWrite, write_stat);
    double read_size_mb = static_cast<double>(read_stat.io_size_) / (1024.0 * 1024.0);
    double write_size_mb = static_cast<double>(write_stat.io_size_) / (1024.0 * 1024.0);
    double read_bw = (read_wall_us > 0)
        ? read_size_mb / (read_wall_us * 1e-6) : perf_metrics_.read_bandwidth_mbps_;
    double write_bw = (write_wall_us > 0)
        ? write_size_mb / (write_wall_us * 1e-6) : perf_metrics_.write_bandwidth_mbps_;

    msgpack::sbuffer sbuf;
    msgpack::packer<msgpack::sbuffer> pk(sbuf);

    pk.pack_map(13);
    pk.pack("pool_name");              pk.pack(pool_name_);
    pk.pack("bdev_type");              pk.pack(static_cast<chi::u32>(bdev_type_));
    pk.pack("total_capacity");         pk.pack(file_size_);
    pk.pack("remaining_capacity");     pk.pack(heap_.GetRemainingSize());
    pk.pack("read_bandwidth_mbps");    pk.pack(read_bw);
    pk.pack("write_bandwidth_mbps");   pk.pack(write_bw);
    pk.pack("read_latency_us");        pk.pack(static_cast<double>(read_wall_us));
    pk.pack("write_latency_us");       pk.pack(static_cast<double>(write_wall_us));
    pk.pack("iops");                   pk.pack(perf_metrics_.iops_);
    pk.pack("total_reads");            pk.pack(total_reads_.load());
    pk.pack("total_writes");           pk.pack(total_writes_.load());
    pk.pack("total_bytes_read");       pk.pack(total_bytes_read_.load());
    pk.pack("total_bytes_written");    pk.pack(total_bytes_written_.load());

    task->results_[container_id_] = std::string(sbuf.data(), sbuf.size());
  }
  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::PostGpuContainerCreate() {
  // The kHbm / kPinned tiers (which previously enqueued an UpdateTask
  // to the GPU container so it could service Write/Read directly) are
  // removed. PutBlob/GetBlob with HBM-resident data now route through
  // kRam/kFile and use the device-aware memcpy hook.
}

}  // namespace clio::run::bdev

// Define ChiMod entry points using CLIO_TASK_CC macro
CLIO_TASK_CC(clio::run::bdev::Runtime)