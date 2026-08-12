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

// Copyright 2024 IOWarp contributors
#ifndef CLIO_CTE_COMPRESSOR_COMPRESSOR_RUNTIME_H_
#define CLIO_CTE_COMPRESSOR_COMPRESSOR_RUNTIME_H_

#include <atomic>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/corwlock.h>
#include <clio_ctp/data_structures/ipc/ring_buffer.h>
#include <clio_ctp/introspect/system_info.h>
#include <memory>
#include <unordered_map>
#include <thread>
#include <vector>
#include <clio_cte/compressor/compressor_tasks.h>
#include <clio_cte/compressor/compressor_client.h>
#include <clio_cte/compressor/models/compression_features.h>
#include <clio_cte/compressor/models/qtable_predictor.h>
#include <clio_cte/compressor/models/linreg_table_predictor.h>
#include <clio_cte/compressor/models/distribution_classifier.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_interposer.h>

#ifdef CLIO_COMPRESSOR_ENABLE_DENSE_NN
#include <clio_cte/compressor/models/dense_nn_predictor.h>
#endif

namespace clio::cte::compressor {

/**
 * Compression statistics predicted by AI models
 */
struct CompressionStats {
  int compress_lib_;           // Compression library ID
  int compress_preset_;        // Compression preset (0=balanced, 1=best, 2=default, 3=fast)
  double compression_ratio_;   // Predicted compression ratio
  double compress_time_ms_;    // Predicted compression time in milliseconds
  double decompress_time_ms_;  // Predicted decompression time in milliseconds
  double psnr_db_;             // Predicted PSNR for lossy compression (0 for lossless)

  CompressionStats()
      : compress_lib_(0), compress_preset_(2), compression_ratio_(1.0), compress_time_ms_(0.0),
        decompress_time_ms_(0.0), psnr_db_(0.0) {}

  CompressionStats(int lib, int preset, double ratio, double comp_time, double decomp_time, double psnr)
      : compress_lib_(lib), compress_preset_(preset), compression_ratio_(ratio), compress_time_ms_(comp_time),
        decompress_time_ms_(decomp_time), psnr_db_(psnr) {}
};

/**
 * CTE Compressor Runtime Container
 * Implements compression scheduling and execution
 */
class Runtime : public clio::cte::core::CoreInterposer {
public:
  using CreateParams = CompressorConfig; // Required for CLIO_TASK_CC (defined in compressor_tasks.h)

  Runtime() = default;
  /**
   * Stops the codec drainer thread.
   *
   * std::thread's destructor calls std::terminate if the thread is still
   * joinable, and DestroyCodecContext -- the only place that joined it -- is
   * reachable only from Create's failure path, never at shutdown. So a run
   * that had used batched decompression aborted on teardown with "terminate
   * called without an active exception". Only the thread is stopped here; the
   * CUDA teardown stays in DestroyCodecContext, where the context is known to
   * still be valid.
   */
  ~Runtime() override;


  /**
   * Per-task cost estimate for the scheduler (see Container::GetTaskStats).
   *
   * compute_ is the feature Container::InferCpuTime multiplies its learned
   * per-method coefficient by; leaving it 0 — as every chimod but MOD_NAME and
   * admin did — collapses that model to one constant per method with no
   * dependence on request size. wall_time_ seeds InferWallClockTime at the
   * ~500 MB/s house convention. Both coefficients are then learned from real
   * completions, so these only need the right order of magnitude.
   */
  clio::run::TaskStat GetTaskStats(const clio::run::Task *task) const override {
    clio::run::TaskStat stat;
    if (task == nullptr) {
      return stat;
    }
    switch (task->method_) {
      case Method::kCompress: {
        const auto *t = static_cast<const CompressTask *>(task);
        stat.io_size_ = t->size_;
        // Compression is genuinely CPU-bound: ~300 MB/s, i.e. ~300 bytes per
        // microsecond of CPU. That is 30x more CPU per byte than a plain copy,
        // which is exactly the distinction the cost model needs to see.
        stat.compute_ = static_cast<size_t>(t->size_ / 300.0f) + 5;
        stat.wall_time_ = static_cast<float>(t->size_) / 300.0f;
        return stat;
      }
      case Method::kDecompress: {
        const auto *t = static_cast<const DecompressTask *>(task);
        stat.io_size_ = t->size_;
        // Decompression runs ~3x faster than compression.
        stat.compute_ = static_cast<size_t>(t->size_ / 900.0f) + 5;
        stat.wall_time_ = static_cast<float>(t->size_) / 900.0f;
        return stat;
      }
      default:
        return clio::cte::core::CoreInterposer::GetTaskStats(task);
    }
  }


private:
  // Client for this ChiMod
  Client client_;

  // Core client for target monitoring
  /**
   * Compress src[0..size) per ctx into a fresh SHM buffer laid out as
   * CompressionHeader + codec bytes. Returns true and sets stored/
   * stored_size on success (and ORs the transform bits + stats into ctx);
   * returns false when compression failed or was not beneficial — the
   * caller must then store the RAW bytes with ctx.compress_lib_ cleared.
   */
  bool CompressIntoShm(clio::cte::core::Context &ctx, const char *src,
                       clio::run::u64 size, ctp::ipc::FullPtr<char> *stored,
                       clio::run::u64 *stored_size);

  /**
   * Decompress a stored CompressionHeader+codec buffer into dst (capacity
   * dst_cap). Sets *out_size to the original size. Returns 0, or a nonzero
   * rc (3 = decompressor creation failed, 5 = decompress failed,
   * 6 = header invalid / dst too small).
   */
  int DecompressStored(const char *stored, clio::run::u64 stored_size,
                       char *dst, clio::run::u64 dst_cap,
                       clio::run::u64 *out_size);

  std::unique_ptr<clio::cte::core::Client> core_client_;

  /**
   * Create the container (Method::kCreate)
   * Initializes predictors and loads AI models
   */
  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask> &task);

  /**
   * Destroy the container (Method::kDestroy)
   * Cleanup resources and predictors
   */
  clio::run::TaskResume Destroy(clio::run::shared_ptr<DestroyTask> &task);

  /**
   * Monitor container state (Method::kMonitor)
   * Polls core for target information and serializes results with msgpack
   */
  clio::run::TaskResume Monitor(clio::run::shared_ptr<MonitorTask> &task);

  /**
   * Dynamic compression scheduling (Method::kDynamicSchedule)
   * Analyzes data and determines optimal compression strategy
   */
  clio::run::TaskResume DynamicSchedule(clio::run::shared_ptr<DynamicScheduleTask> &task);

  /**
   * Compress data (Method::kCompress)
   * Executes compression with specified library and parameters
   */
  clio::run::TaskResume Compress(clio::run::shared_ptr<CompressTask> &task);

  /**
   * Decompress data (Method::kDecompress)
   * Executes decompression with specified library and parameters
   */
  clio::run::TaskResume Decompress(clio::run::shared_ptr<DecompressTask> &task);

  /**
   * Sample this node's CPU utilization and aggregated worker load
   * (Method::kPollNodeLoad). Writes results into task->sample_.
   */
  clio::run::TaskResume PollNodeLoad(clio::run::shared_ptr<PollNodeLoadTask> &task);

  /**
   * Periodic task that iterates the tracked consumer list and dispatches
   * PollNodeLoad to each consumer node (Method::kPollConsumers).
   */
  clio::run::TaskResume PollConsumers(clio::run::shared_ptr<PollConsumersTask> &task);

  // ---- Interposed core data verbs (issue #886 interposition) ----
  // The compressor speaks the CTE core's task interface: a default put with
  // Context::compress_lib_ set is compressed transparently before landing on
  // the next pool; reads of transformed blobs are decompressed — including
  // PARTIAL and VECTORED reads (fetch stored, decompress once, slice into
  // each requested region), which the raw core cannot serve for compressed
  // data. GetBlobSize reports the LOGICAL (original) size. MultiPutBlob is
  // forwarded verbatim (batch records carry no per-record Context, so
  // compression is not requestable on that path).
  clio::run::TaskResume PutBlob(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task);
  clio::run::TaskResume GetBlob(
      clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task);
  clio::run::TaskResume GetBlobSize(
      clio::run::shared_ptr<clio::cte::core::GetBlobSizeTask> &task);
  clio::run::TaskResume MultiPutBlob(
      clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> &task);

  /**
   * POD variants, used by DEVICE producers such as gpu_vector.
   *
   * Without these the interposer's default case forwards a Pod task straight
   * to the next pool, so pages written from a kernel are stored UNCOMPRESSED
   * while everything reports success -- compression silently does nothing for
   * the one producer that most needs it.
   *
   * Their blob_data_ usually points at DEVICE memory, so every access goes
   * through ctp::DeviceAwareMemcpy rather than a plain memcpy.
   */
  clio::run::TaskResume CompressPodPutBlob(
      clio::run::shared_ptr<clio::cte::core::PodPutBlobTask> &task);
  clio::run::TaskResume DecompressPodGetBlob(
      clio::run::shared_ptr<clio::cte::core::PodGetBlobTask> &task);

  /** Batched POD paging (kPodMultiPutBlob / kPodMultiGetBlob). Each record is
   *  fanned into a scalar Pod task and run through the scalar handler above,
   *  so compression semantics are identical per page and no codec logic is
   *  duplicated. The win the batch is actually after is on the SUBMISSION
   *  side -- one device->host queue entry per batch instead of per page. */
  clio::run::TaskResume CompressPodMultiPutBlob(
      clio::run::shared_ptr<clio::cte::core::PodMultiPutBlobTask> &task);
  clio::run::TaskResume DecompressPodMultiGetBlob(
      clio::run::shared_ptr<clio::cte::core::PodMultiGetBlobTask> &task);

  /**
   * Schedule a task by resolving Dynamic pool queries.
   */
  clio::run::PoolQuery ScheduleTask(const clio::run::shared_ptr<clio::run::Task> &task) override;

  // Autogen-provided methods
  void Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
            clio::run::u32 container_id = 0) override;
  clio::run::TaskResume Run(clio::run::u32 method,
                      clio::run::shared_ptr<clio::run::Task> task_ptr) override;
  clio::run::u64 GetWorkRemaining() const override;
  void SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive& archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;

  // Container virtual method implementations (defined in autogen/compressor_lib_exec.cc)
  void LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> AllocLoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive) override;
  clio::run::shared_ptr<clio::run::Task> NewCopyTask(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task_ptr,
                                        bool deep) override;
  clio::run::shared_ptr<clio::run::Task> NewTask(clio::run::u32 method) override;
  void AggregateOut(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task,
                 const clio::run::shared_ptr<clio::run::Task>& replica_task) override;
  void LocalLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive &archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> LocalAllocLoadTask(clio::run::u32 method,
                                               clio::run::DefaultLoadArchive &archive) override;
  void AggregateIn(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &agg_task,
                   const clio::run::shared_ptr<clio::run::Task> &member_task) override;
  void LocalSaveTask(clio::run::u32 method, clio::run::DefaultSaveArchive &archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;

private:
  // AI model predictors
  std::unique_ptr<QTablePredictor> qtable_predictor_;
  std::unique_ptr<LinRegTablePredictor> linreg_predictor_;
  // Note: DistributionClassifier is a template - use DistributionClassifierFactory for type-erased access

#ifdef CLIO_COMPRESSOR_ENABLE_DENSE_NN
  std::unique_ptr<DenseNNPredictor> nn_predictor_;
#endif

  // Compression telemetry ring buffer for performance monitoring
  using CompressionTelemetryLog = ctp::ipc::ring_buffer<CompressionTelemetry, CLIO_TASK_ALLOC_T>;
  ctp::ipc::ShmPtr<CompressionTelemetryLog> compression_telemetry_log_;
  std::atomic<std::uint64_t> compression_logical_time_;
  /** Stream every GPU codec runs on, created once in Create() and handed to
   *  CompressionFactory. Owned here, so it outlives any compressor the factory
   *  produced. Null on a build or host with no GPU, which the codecs read as
   *  "use the default stream". */
  void *gpu_stream_ = nullptr;

  /**
   * Device buffers for the COMPRESSED side of a GPU codec operation: one
   * allocation, carved into equal slabs, one slab per in-flight operation.
   *
   * This is not a staging copy of the payload. The payload never needs one:
   * on a read the destination is already the caller's device page and nvcomp
   * decompresses straight into it; on a write the source is already the
   * device page. What needs a buffer is the compressed bitstream -- it has to
   * be read out of (or written into) a CTE blob, and on a write its size is
   * not known until the codec finishes, so it cannot go directly into a
   * right-sized blob buffer.
   *
   * Preallocated because the alternative is nvcomp allocating it per call.
   * nvcomp allocates NOTHING when handed device pointers on both sides; it
   * falls back to cudaMalloc only because a host buffer forces it to. That
   * fallback does not merely cost time, it hangs: cudaMalloc and especially
   * cudaFree synchronize the device, and a gpu_vector page fault runs while
   * its block spins waiting for the very operation being set up -- so the
   * allocation waits on a kernel that is waiting on the allocation. (Measured:
   * 128 MB through nvcomp made no progress in 200 s.) Stream creation had the
   * identical failure mode, which is why the stream above is created once too.
   *
   * A request larger than a slab, or one arriving with every slab out, is
   * refused. Callers must fall back to the host path or store raw -- never
   * allocate.
   */
  char *gpu_scratch_base_ = nullptr;
  size_t gpu_scratch_slab_ = 0;
  std::vector<char *> gpu_scratch_free_;
  std::mutex gpu_scratch_mu_;

  /** @return a device buffer of at least `bytes`, or nullptr. Never allocates. */
  char *AcquireGpuScratch(size_t bytes);
  /** Return a buffer from AcquireGpuScratch; null is ignored. */
  void ReleaseGpuScratch(char *ptr);
  /** @return true if `wire_id` names a codec that runs on the GPU. */
  static bool IsGpuCodec(int wire_id);

  /**
   * A DEDICATED CUDA context for GPU codecs, and the device buffers they use.
   *
   * A GPU codec cannot run in the context that raised the page fault. An
   * indefinitely-resident kernel blocks every kernel launched after it in the
   * same context -- measured with a standalone reproducer: a 1-block marker
   * kernel launched behind a spinner never executed at all, while the same
   * marker behind a FINITE long kernel overlapped at full speed. The gpu_vector
   * consumer spins until its fault completes, so nvcomp's kernels never ran and
   * the fault never returned.
   *
   * Contexts, unlike kernels within a context, ARE time-sliced by the driver.
   * A codec kernel in a second context runs while the consumer spins (measured:
   * 100 ms), and cuMemcpyPeer bridges the result back into the faulting
   * context's page -- a copy-engine operation, and copy engines were never the
   * thing being blocked (which is why the raw path always worked).
   *
   * Created ONCE here, at module creation. cuCtxCreate and cuCtxDestroy both
   * synchronize, so doing either while a kernel spins is its own deadlock.
   */
  void *codec_ctx_ = nullptr;    // CUcontext, owned
  void *primary_ctx_ = nullptr;  // CUcontext of the faulting side, not owned
  /**
   * A pool of independent codec slots, NOT one shared buffer.
   *
   * Each slot is a device staging buffer for the compressed bytes plus its own
   * stream, so operations proceed concurrently. That is the whole point: a
   * codec operation waits on a driver context switch (~7.6 ms), and switches
   * are per SLICE, not per operation. Serializing behind one mutex made every
   * fault pay its own switch, which is why prefetching more pages changed
   * nothing at all -- the extra fetches simply queued. With independent slots
   * the concurrent faults land in the same slice and share its cost.
   *
   * Only the compressed side needs a buffer. The decompressed side is the
   * caller's page, written directly.
   */
public:
  /** Chunk table of one stored blob (public so the parser can fill it). */
  struct BlobChunksPub {
    unsigned long long orig = 0;
    unsigned long long chunk_raw = 0;
    std::vector<unsigned long long> rel;
    std::vector<unsigned long long> csz;
  };

  /** One page's worth of work for the batched decompressor. */
  struct DecompItem {
    const void *src_device = nullptr;  // stored blob, device memory
    size_t stored_size = 0;
    void *dst_device = nullptr;        // page, device memory
    size_t dst_bytes = 0;
  };

private:

  /**
   * The compression module's ONE CUDA stream.
   *
   * Created on first use inside the codec context and reused for every
   * operation. There is no reason for a stream per compression: a stream is
   * an ordering domain, and the batched decompressor already expresses all
   * the parallelism by handing nvcomp every chunk at once. cudaStreamCreate
   * is also not free -- creating one per fault deadlocked against a resident
   * kernel before the fault path learned to yield.
   */
  void *ModuleStream();
  void *module_stream_ = nullptr;

  /**
   * The module's stream POOL, created once at first use.
   *
   * MEASURED: nvcomp's batched decompress costs ~151us per launch on one
   * stream, and that cost is kernel DURATION, not host overhead -- it
   * overlaps almost perfectly across streams (150.6 / 77.0 / 39.7 / 20.9 us
   * effective at 1/2/4/8 streams). Serializing every decode on a single
   * stream was the whole reason the compressed fault path lost to raw. A
   * fixed pool created once is still "one set of streams for the module",
   * not a stream per compression.
   */
  static constexpr size_t kModuleStreams = 8;
  void *module_streams_[kModuleStreams] = {};
  std::atomic<unsigned> module_stream_rr_{0};
  void *ModuleStreamRR();

  // ---- persistent state for the batched decoder (drain thread ONLY) ----
  //
  // The first implementation paid, PER DRAIN: one cudaStreamSynchronize per
  // item to read its header, seven cudaMalloc/cudaFree pairs for the
  // descriptor arrays and temp, and uploads from pageable std::vector memory
  // (which cudaMemcpyAsync stages synchronously). ~260 drains made that
  // thousands of hidden synchronization points before nvcomp ever ran --
  // the decompression itself was never the cost.
  /** Pinned staging for batched header reads (cache misses only). */
  void *bd_hdrpin_ = nullptr;
  size_t bd_hdrpin_cap_ = 0;
  /** Parsed chunk table per stored blob. A blob's table never changes, so a
   *  page refaulted N times parses its header ONCE, not N times. */
  struct CachedChunks {
    size_t stored_size = 0;
    int wire = 0;        // codec from the blob's CTEC header
    BlobChunksPub bc;
  };
  std::unordered_map<const void *, CachedChunks> bd_cache_;
  /** Bumped on every put; the drain clears bd_cache_ when it moves, because a
   *  rewritten blob's chunk table is stale the moment the put lands. */
  std::atomic<unsigned long long> tier_write_gen_{0};
  unsigned long long bd_cache_gen_ = ~0ull;

  struct CodecSlot {
    void *buf = nullptr;   // compressed bytes, CUdeviceptr in codec_ctx_
    void *obuf = nullptr;  // decompressed bytes, plain ctx2 device memory
    void *stream = nullptr;  // CUstream, in codec_ctx_
  };
  std::vector<CodecSlot> codec_slots_;
  std::vector<size_t> codec_free_;  // indices of slots not in use
  size_t codec_buf_bytes_ = 0;
  std::mutex codec_mu_;  // guards codec_free_ ONLY, never a whole operation

  /** Take a slot, or SIZE_MAX if all are busy (caller falls back). */
  size_t AcquireCodecSlot();
  void ReleaseCodecSlot(size_t idx);

  /**
   * BATCHED GPU decompression -- INCOMPLETE, off unless
   * CLIO_COMPRESS_GPU_BATCH=1.
   *
   * Entering the codec context costs a fixed ~2.33 ms driver time slice,
   * unaffected by spin backoff or resident block count, so one page per entry
   * caps this path near 110 MB/s. Batching is the only lever, and the pages
   * are there to batch: each CUDA block faults independently.
   *
   * State of the work, so the next person does not repeat it:
   *  - A dedicated thread owns the batch. Three earlier versions made a
   *    WAITING FIBER the leader and each deadlocked differently; a fiber that
   *    is itself blocking a GPU block cannot also be responsible for other
   *    fibers' progress.
   *  - The codec must NOT write the caller's managed page from this thread.
   *    Doing so hangs inside Decompress: the page is mapped by the faulting
   *    context too, and the migration that write needs cannot happen while
   *    that context's kernel is resident. Decompress into plain ctx2 memory
   *    and cuMemcpyDtoDAsync into the page -- copies were never blocked.
   *  - With that, items complete and batching demonstrably groups requests
   *    (three in one context entry, observed). It then STALLS after a few
   *    batches. That is the open bug.
   *
   * Two properties keep every failure benign and must stay: requests are
   * shared_ptr, so a waiter that gives up cannot leave the drainer writing
   * into a dead stack frame; and waiting is bounded, so a wedged drainer
   * degrades to the host path instead of hanging.
   */
  struct PendingDecomp {
    /**
     * The request owns its compressed bytes.
     *
     * Pointing at the waiter's SHM buffer was a use-after-free waiting to
     * happen: on timeout the waiter marks itself abandoned and proceeds down
     * the host path, which consumes and FREES that buffer, while the drainer
     * may still be reading it. shared_ptr keeps this struct alive; it says
     * nothing about the buffer the struct points at. Copying costs one memcpy
     * of the COMPRESSED bytes, which are small by construction -- that is the
     * whole point of having compressed them.
     */
    std::vector<char> stored_bytes;
    /**
     * The stored image in DEVICE memory, when the requester already fetched
     * it there (the gpu_vector fault path does). Preferred over stored_bytes:
     * the batched decompressor takes device pointers, so this needs no upload
     * and the compressed payload never touches the host at all.
     */
    const void *src_device = nullptr;
    /**
     * True only when src_device is STABLE storage (the blob's home on a
     * device tier). Scratch buffers are reused across pages -- same pointer,
     * different blob every fault, and compressed sizes cluster tightly enough
     * that (pointer, size) collides -- so caching a scratch blob's chunk
     * table serves ANOTHER page's table on the next fault. Measured: every
     * failure in a mixed batch carried the same stale csz from a previous
     * occupant of its scratch slot. Only stable sources may be cached.
     */
    bool src_stable = false;
    size_t stored_size = 0;
    void *dst = nullptr;
    size_t dst_bytes = 0;
    std::atomic<bool> done{false};
    std::atomic<bool> abandoned{false};
    bool ok = false;
  };

  /**
   * Per-REQUEST async decompress: the worker coroutine that owns the fault
   * launches its own decode on the module stream and yield-polls the event.
   *
   * This exists because the drain-thread design, however its batching was
   * tuned, added a cross-thread hop to every fault: worker -> queue -> drain
   * -> GPU -> retire poll -> publish -> worker's own poll. Measured 580us per
   * fault against raw's ~65us, with batches averaging 1.7 because faults
   * arrive staggered, not together. The raw path is fast precisely because
   * the worker enqueues its own transfer and polls its own flag; this gives
   * the codec path the same shape. Concurrent workers' kernels queue
   * back-to-back on the ONE module stream -- that queue is the pipeline.
   *
   * A slot holds the descriptor arenas for up to kSlotMaxChunks chunks
   * (pinned + device, preallocated once). The shared temp is safe because a
   * single stream serializes the kernels that use it.
   */
  struct DecompSlot {
    void *pin = nullptr;
    void *dev = nullptr;
    void *ev = nullptr;      // cudaEvent_t
    void *ev0 = nullptr;     // timing event at launch, for diagnostics
    /** Per-slot nvcomp temp: launches on DIFFERENT streams run concurrently
     *  and cannot share scratch the way the serialized design could. */
    void *temp = nullptr;
    size_t temp_cap = 0;
    size_t nch = 0;
    std::vector<size_t> item_first;
    std::vector<size_t> item_n;
    bool busy = false;
  };
  static constexpr size_t kDecompSlots = 64;
  static constexpr size_t kSlotMaxChunks = 64;
  DecompSlot dslots_[kDecompSlots];
  std::mutex dslot_mu_;   // slot claim/free + temp high-water only
  void *dtemp_ = nullptr;
  size_t dtemp_cap_ = 0;
  size_t dtemp_bytes_ = 0;
  size_t dtemp_hw_nch_ = 0;
  size_t dtemp_hw_unc_ = 0;

  /**
   * Launch the decode of one or more pages sharing a slot.
   * @param items  each: {src_device, stored_size, header snapshot, dst,
   *               dst_bytes}; every chunk of every item rides one launch.
   * @return slot index, or -1 (no slot free / nothing parseable -- caller
   *         falls back).
   */
  struct OneDecomp {
    const void *src = nullptr;
    size_t stored = 0;
    const char *hdr = nullptr;
    size_t hdr_len = 0;
    void *dst = nullptr;
    size_t dst_bytes = 0;
  };
  int LaunchDecompOne(const OneDecomp *items, size_t n);

  /**
   * COMBINING front door for the fault path, built around the measured
   * ~151us FIXED cost of one nvcompBatched*DecompressAsync launch (flat from
   * 1 to 64 chunks -- 150.6us/chunk at batch=1, 2.4us/chunk at batch=64).
   * Nothing else about this path's cost matters; only chunks per launch.
   *
   * A faulting coroutine pushes its request and then: if a launch is in
   * flight, it simply yield-polls its own flag -- its request rides the NEXT
   * launch, which whoever gets there first will issue with EVERYTHING queued
   * by then. The 151us the GPU spends on a launch IS the accumulation window
   * for the next one; no linger, no drain thread, no cross-thread handoff.
   */
  struct CombineReq {
    OneDecomp od;
    std::vector<char> hdr_copy;   // keeps od.hdr alive across the wait
    std::atomic<int> state{0};    // 0 pending, 1 ok, 2 failed
  };
  std::mutex comb_mu_;            // guards comb_q_ only
  std::vector<std::shared_ptr<CombineReq>> comb_q_;
  std::atomic<bool> comb_launching_{false};

  /** Push `req`, then either launch (taking everything queued) or yield until
   *  someone else's launch serves it. Returns with req->state settled. */
  clio::run::TaskResume CombinedDecompWait(std::shared_ptr<CombineReq> req);
  /** @return -1 still running, else a bitmask-free result: 1 all ok, 0 any
   *  failed. Frees the slot when it returns >= 0. Item i's own result is in
   *  ok_out[i] when provided. */
  int DecompPoll(int slot, char *ok_out, size_t n);

  /**
   * One in-flight decompress batch. NOTHING in the compressor blocks on it:
   * the launch records an event and returns; RetireBatches() polls the event
   * (cudaEventQuery, never a synchronize) and publishes results when it has
   * fired. The drain thread's sleep between polls is the yield.
   *
   * Each segment owns its arenas so several batches can be in flight without
   * sharing buffers: pinned host (async copies from pageable memory silently
   * synchronize), a device mirror, and a grow-only nvcomp temp with the
   * high-water marks that let TempSize be skipped when a batch fits a shape
   * already computed.
   */
  struct BatchSeg {
    void *pin = nullptr;
    size_t pin_cap = 0;
    void *dev = nullptr;
    size_t dev_cap = 0;
    void *temp = nullptr;
    size_t temp_cap = 0;
    size_t temp_bytes = 0;   // requirement at the high-water shape
    size_t hw_nch = 0;       // high-water chunk count for temp reuse
    size_t hw_maxunc = 0;
    void *ev = nullptr;      // cudaEvent_t, created on first use
    bool busy = false;
    std::vector<std::shared_ptr<PendingDecomp>> owners;
    std::vector<size_t> item_first;
    std::vector<size_t> item_n;
    size_t nch = 0;
  };
  static constexpr size_t kBatchSegs = 4;
  BatchSeg bd_segs_[kBatchSegs];

  /**
   * Launch one batch covering `batch`'s device-resident items, WITHOUT
   * waiting for it. Items it cannot serve are published done/!ok immediately
   * so their waiters take the fallback path. Returns false if no segment was
   * free (caller retires and retries).
   */
  bool LaunchDecompBatch(std::vector<std::shared_ptr<PendingDecomp>> &batch);

  /** Poll every busy segment; publish + free the finished ones. Never blocks. */
  size_t RetireBatches();
  std::mutex batch_mu_;
  std::vector<std::shared_ptr<PendingDecomp>> batch_;
  std::thread batch_thread_;
  std::atomic<bool> batch_stop_{false};
  bool batch_enabled_ = false;

  void BatchDrainLoop();
  void RunDecompBatch(std::vector<std::shared_ptr<PendingDecomp>> &batch);

  /** Build codec_ctx_ + its buffers. Best effort; false leaves GPU codecs off. */
  bool InitCodecContext();
  /** Tear the codec context down. Must not run while a kernel is resident. */
  void DestroyCodecContext();
  /** @return true if the GPU codec path is usable. */
  bool HasCodecContext() const { return codec_ctx_ != nullptr; }

  /**
   * Decompress a stored blob into `dst_device` using a GPU codec, entirely on
   * the device. Returns false if the GPU path is unavailable or fails, in
   * which case the caller must use the host path.
   */
  bool GpuDecompressToDevice(const char *stored_host, size_t stored_size,
                             void *dst_device, size_t dst_bytes);
  /**
   * Decompress a stored blob already resident in DEVICE memory into device
   * memory, with no copy of the payload in either direction.
   *
   * Takes the two header fields it needs rather than the header struct, which
   * this header cannot see. The caller validates the header (it has to bring
   * those 32 bytes back from the device anyway).
   *
   * @param wire_id  codec from the blob's header
   * @param payload  compressed byte count, excluding the header
   */
  bool GpuDecompressFromDevice(const char *stored_device, size_t stored_size,
                               int wire_id, size_t payload, void *dst_device,
                               size_t dst_bytes);
  /**
   * Compress `src_device` with a GPU codec. On success fills `out_host` (which
   * must hold at least the codec's bound) and sets *out_size.
   */
  bool GpuCompressFromDevice(int wire_id, const void *src_device, size_t size,
                             char *out_host, size_t out_cap, size_t *out_size);
  /**
   * Compress `src_device` with a GPU codec, leaving the result in DEVICE
   * memory (`out_device`). Device pointer in, device pointer out: nothing is
   * staged through the host, so a blob bound for the kHBM tier never leaves
   * the GPU.
   */
  bool GpuCompressToDevice(int wire_id, const void *src_device, size_t size,
                           char *out_device, size_t out_cap, size_t *out_size);
#if CTP_ENABLE_GPU
  /** ShmPtr addressing a pointer inside the registered device scratch. */
  ctp::ipc::ShmPtr<void> ScratchShmPtr(char *p) const;
#endif
  /** @return the CPU codec of the same family as a GPU codec, or 0 if none.
   *  Used on the device page-fault path, where a GPU codec cannot run. */
  /** Eligible for nvcomp's BATCHED decompress API (nvcomp family only).
   *  Distinct from IsGpuCodec, which means "must run on the device". */
  static bool IsNvcompBatchedCodec(int wire_id);

  static int CpuEquivalentCodec(int gpu_wire_id);

  // Configuration
  CompressorConfig config_;

  // Target state cache for compression/tiering decisions
  std::unordered_map<std::string, TargetState> target_states_;
  std::mutex target_states_mutex_;

  // Maximum number of distinct consumer nodes tracked PER TAG. Per-tag
  // (rather than per-container) tracking lets the compressor route
  // future Compress traffic for tag T toward the most-recent reader of
  // T rather than the union of all consumers seen across all tags. The
  // bound applies per tag so a tag with more readers than this caps at
  // kMaxConsumersPerTag and silently drops the rest (kDebug-logged).
  static constexpr std::size_t kMaxConsumersPerTag = 32;

  // Per-tag consumer node-id sets (small unsorted vector per tag, size
  // capped at kMaxConsumersPerTag). Map entry is created on first
  // Decompress for the tag. Guarded by tag_consumers_lock_.
  //
  // Skipped entirely when CompressorConfig::tracking_enabled_ is false —
  // ScheduleTask then routes Compress via DirectHash(tag_id) and the
  // periodic PollConsumers becomes a no-op.
  std::unordered_map<clio::cte::core::TagId, std::vector<clio::run::u32>>
      tag_consumers_;
  clio::run::CoRwLock tag_consumers_lock_;

  // Previous CPU times sample, used by PollNodeLoad to compute CPU%.
  ctp::CpuTimes prev_cpu_times_;
  std::mutex cpu_times_mutex_;

  /**
   * Append node_id to tag_consumers_[tag_id] if not already present and
   * under the kMaxConsumersPerTag cap. No-op when
   * CompressorConfig::tracking_enabled_ is false. Acquires
   * tag_consumers_lock_ as a writer when an insert is needed.
   * @param tag_id Tag that the inbound Decompress is reading.
   * @param node_id Originating node ID of the Decompress request.
   */
  void RegisterConsumer(const clio::cte::core::TagId &tag_id, clio::run::u32 node_id);

  /**
   * Pick the best consumer node for placing future Compress traffic for
   * `tag_id`. Returns:
   *   - When tracking_enabled_=false OR no consumers known: invalid
   *     (caller should fall back to DirectHash).
   *   - When tracking_enabled_=true AND consumers known: the most
   *     recently registered consumer (back of the per-tag vector).
   * Acquires tag_consumers_lock_ as a reader.
   * @param tag_id Tag the Compress task is operating on.
   * @param node_id_out Out param: receives the chosen node ID on success.
   * @return true if a consumer was found, false otherwise.
   */
  bool PickConsumerForTag(const clio::cte::core::TagId &tag_id,
                          clio::run::u32 &node_id_out);

  /**
   * Estimate compression statistics using AI models
   * @param chunk Pointer to data chunk
   * @param chunk_size Size of chunk in bytes
   * @param context Compression context with parameters
   * @return Vector of compression statistics for candidate libraries
   */
  std::vector<CompressionStats> EstCompressionStats(
      const void* chunk, clio::run::u64 chunk_size, const Context& context);

  /**
   * Estimate workflow compression time for a specific tier
   * @param chunk_size Size of chunk in bytes
   * @param tier_bw Tier bandwidth in bytes/second
   * @param stats Compression statistics for library
   * @param context Compression context
   * @return Estimated time in milliseconds
   */
  double EstWorkflowCompressTime(
      clio::run::u64 chunk_size, double tier_bw, const CompressionStats& stats,
      const Context& context);

  /**
   * Find best compression for ratio optimization
   * @param chunk Pointer to data chunk
   * @param chunk_size Size of chunk
   * @param container_id Container ID for placement
   * @param stats Vector of compression statistics
   * @param context Compression context
   * @return Tuple of (tier_id, compress_lib, compress_preset, estimated_time, tier_score)
   */
  std::tuple<int, int, int, double, float> BestCompressRatio(
      const void* chunk, clio::run::u64 chunk_size, int container_id,
      const std::vector<CompressionStats>& stats, const Context& context);

  /**
   * Find best compression for time optimization
   * @param chunk Pointer to data chunk
   * @param chunk_size Size of chunk
   * @param container_id Container ID for placement
   * @param stats Vector of compression statistics
   * @param context Compression context
   * @return Tuple of (tier_id, compress_lib, compress_preset, estimated_time, tier_score)
   */
  std::tuple<int, int, int, double, float> BestCompressTime(
      const void* chunk, clio::run::u64 chunk_size, int container_id,
      const std::vector<CompressionStats>& stats, const Context& context);

  /**
   * Choose best compression based on context objective
   * @param context Compression context
   * @param chunk Pointer to data chunk
   * @param chunk_size Size of chunk
   * @param container_id Container ID for placement
   * @param stats Vector of compression statistics
   * @return Tuple of (tier_id, compress_lib, compress_preset, estimated_time, tier_score)
   */
  std::tuple<int, int, int, double, float> BestCompressForNode(
      const Context& context, const void* chunk, clio::run::u64 chunk_size,
      int container_id, const std::vector<CompressionStats>& stats);

  /**
   * Log compression telemetry for performance monitoring
   * @param telemetry Compression telemetry entry
   */
  void LogCompressionTelemetry(const CompressionTelemetry& telemetry);
};

} // namespace clio::cte::compressor

#endif // CLIO_CTE_COMPRESSOR_COMPRESSOR_RUNTIME_H_
