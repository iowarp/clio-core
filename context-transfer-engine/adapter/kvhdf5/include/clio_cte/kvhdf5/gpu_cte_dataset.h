#pragma once

// Host control plane for a GPU-resident dataset on the new iowarp producer-only
// CTE model. Lifts the proven mechanics of the integration reference
// (test_cte_devmem_putget): preallocate registered kDeviceMem backends for the
// per-chunk PodPutBlob/PodGetBlob tasks (self-contained, embedded fut_) and the blob data
// buffers, stamp the task prototypes onto the device once, build a device-side
// ChunkDesc array, and hand out a small by-value GpuDatasetHandle the user's
// kernel submits.
//
// Generalized to N chunks (Phase 2): the single-chunk case is just N==1. Each
// chunk gets its own task slot pair and its own distinct region of the data
// backend (concurrent multi-chunk Put/Get can't alias one buffer). The handle
// carries a pointer to the device ChunkDesc array + the count, so its size is
// independent of N.
//
// The surface is Handle(), not a host-callable WriteChunk/ReadChunk chunk store:
// the host only preallocates and stamps task slots, and the device submits its own
// I/O from inside the compute kernel. An earlier host-orchestrated CPU control
// plane (File/Dataset over a synchronous BlobBackend concept) could not express
// that inversion and was deleted rather than carried forward.
// It owns three device allocations, so it is move-only and frees them in the dtor.

#include "defines.h"
#include "chunking.h"
#include "dataset_meta.h"  // Layout, DatasetMeta
#include "gpu_dataset_handle.h"
#include "tag_resolve.h"  // ResolveTagId (path->TagId)
#include "meta_write.h"   // WriteDatasetMeta (publish Layout as the __meta blob)

#include <clio_runtime/ipc_manager.h>
#include <clio_runtime/types.h>
#include <clio_runtime/gpu/gpu_info.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/future.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include "tag_path.h"  // CanonicalTag (path->tag string)

#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Host-only control plane: guarded out of the nvcc device pass (kernels need
// only GpuDatasetHandle, included above). Mirrors how the reference guards its
// host bring-up class.
#if !CTP_IS_DEVICE_PASS

namespace kvhdf5 {

class GpuCteDataset {
    clio::run::IpcManager* ipc_ = nullptr;
    uint32_t gpu_id_ = 0;

    ctp::ipc::AllocatorId task_alloc_{};  // N task slots + co-located futures
    ctp::ipc::AllocatorId data_alloc_{};  // N distinct blob data regions
    ctp::ipc::AllocatorId desc_alloc_{};  // device ChunkDesc array
    byte_t* task_base_ = nullptr;
    byte_t* data_base_ = nullptr;
    ChunkDesc* desc_base_ = nullptr;      // device array (count_ entries)

    std::vector<ChunkDesc> host_descs_;   // host mirror (for host accessors + H2D)
    uint32_t count_ = 0;
    uint32_t pool_ = 0;                    // resident data buffers M (M==count_ if unpooled)
    uint32_t grid_ = 0;                    // producer grid G (blocks); G divides M
    uint64_t data_bytes_ = 0;              // registered DATA backend size (device payload)
    // Set once any I/O-status accessor is called. Purely for the destructor's
    // "you never checked" warning; mutable so the accessors stay const.
    mutable bool io_checked_ = false;
    // Device's out-of-range chunk-coordinate report: points into the task backend's
    // tail pad (kPinnedHost, so this is a plain host load). Holds first offending
    // coordinate + 1, or 0 for "never happened". Owned by task_alloc_, so there is
    // nothing extra to free.
    uint32_t* oob_ = nullptr;

    GpuDatasetHandle handle_{};

    // Task is self-contained now (embedded fut_, no co-located gpu::FutureShm),
    // so each slot is just the POD task itself.
    static constexpr uint32_t kPutSlot = sizeof(cte::PodPutBlobTask);
    static constexpr uint32_t kGetSlot = sizeof(cte::PodGetBlobTask);
    static constexpr uint32_t kSlotPair = kPutSlot + kGetSlot;

    // One chunk's blob name (NUL-terminated C-string) + raw byte count.
    struct ChunkSpec {
        const char* name;
        uint64_t bytes;
    };

public:
    // Memory placement of the blob DATA backend. kDeviceMem (default) keeps the
    // payload in HBM and the bdev server D2H-copies it out. kPinnedHost lands the
    // payload straight in host memory so the server needs no D2H DMA — this lets
    // its disk writes overlap the producer's compute (the in-process server D2H
    // does not overlap compute; see the data-backend alloc in Init). The task and
    // descriptor backends are unaffected by this choice.
    using MemKind = clio::run::gpu::IpcManager::MemKind;

    // Single chunk: `name` is a NUL-terminated chunk-blob name (<= kMaxBlobNameLen),
    // `bytes` the chunk's raw byte count. `data_kind` places the blob data backend
    // (default kDeviceMem). Throws on any iowarp failure.
    GpuCteDataset(clio::run::IpcManager* ipc, clio::run::IpcManagerGpuInfo info,
                  uint32_t gpu_id, cte::TagId tag, const char* name,
                  uint64_t bytes, MemKind data_kind = MemKind::kDeviceMem)
        : ipc_(ipc), gpu_id_(gpu_id) {
        ChunkSpec spec{name, bytes};
        Init(info, tag, {&spec, 1}, /*pool_size=*/0, data_kind);
    }

    // Multi-chunk: derive one chunk per layout chunk-coordinate. The blob name is
    // the chunk coord text (chunking::ChunkCoordToName); the size is the (uniform)
    // chunk byte count. Requires each dim divisible by its chunk dim (equal-size
    // chunks; edge-chunk handling is deferred). Throws on any iowarp failure.
    //
    // pool_size (Phase 3 "P"): bound the resident data buffers to M = pool_size,
    // M <= N. Chunk c reuses data buffer c % M; its task still has a distinct blob
    // name, so N chunks stream through M buffers. pool_size == 0 (default) keeps
    // one buffer per chunk (M == N). Pooling requires uniform chunk size (the
    // Layout ctor already produces that).
    //
    // grid_size (G) is the producer kernel's BLOCK COUNT, and it is part of the
    // correctness contract, not a tuning hint — see the GridSize()/Depth() note
    // below for the exclusivity argument. G must divide M. grid_size == 0
    // (default) means G == M, i.e. depth D == 1: the historical "grid == M,
    // one buffer per block, reuse serialized by Write() = Send().Wait()" contract.
    //
    // NOTE (judgment call): grid_size is appended AFTER data_kind rather than
    // paired next to pool_size, because three existing callers pass data_kind
    // POSITIONALLY as the 7th argument (occupancy_bench.cu:705,754 and
    // gray_scott_threeway_bench.cu:986). Inserting a parameter before it would
    // silently reinterpret their MemKind as a grid size.
    GpuCteDataset(clio::run::IpcManager* ipc, clio::run::IpcManagerGpuInfo info,
                  uint32_t gpu_id, cte::TagId tag, const Layout& layout,
                  uint32_t pool_size = 0, MemKind data_kind = MemKind::kDeviceMem,
                  uint32_t grid_size = 0)
        : ipc_(ipc), gpu_id_(gpu_id) {
        if (!layout.Valid())
            throw std::runtime_error("GpuCteDataset: invalid layout");
        const size_t rank = layout.dims.size();
        for (size_t i = 0; i < rank; ++i)
            if (layout.dims[i] % layout.chunk_dims[i] != 0)
                throw std::runtime_error(
                    "GpuCteDataset: non-divisible dims (edge chunks unsupported)");

        uint64_t chunk_elems = 1;
        for (uint64_t cd : layout.chunk_dims) chunk_elems *= cd;
        const uint64_t chunk_bytes = chunk_elems * layout.elem_size;
        const uint64_t cc = layout.ChunkCount();

        // Own the chunk-coord names so ChunkSpec's const char* stay valid through
        // Init (vector<string> must not reallocate after we take .c_str()).
        std::vector<std::string> names;
        names.reserve(cc);
        for (uint64_t idx = 0; idx < cc; ++idx) {
            uint64_t coord[MAX_DIMS] = {};
            chunking::ChunkIndexToCoord(idx, layout.Dims(), layout.ChunkDims(),
                                        {coord, rank});
            char buf[chunking::kMaxBlobNameLen + 1];
            auto nm = chunking::ChunkCoordToName({coord, rank}, buf);
            if (nm.empty())
                throw std::runtime_error("GpuCteDataset: chunk name too long");
            names.emplace_back(nm.data(), nm.size());
        }
        std::vector<ChunkSpec> specs;
        specs.reserve(cc);
        for (const auto& n : names) specs.push_back({n.c_str(), chunk_bytes});

        Init(info, tag, {specs.data(), specs.size()}, pool_size, data_kind,
             grid_size);
    }

    // Primary path->GPU surface: the producer needs only a dataset `path` (-> tag)
    // and a `layout` (chunk geometry) — no host metadata struct. `path` is
    // canonicalized into the CTE tag (the path->tag scheme) and resolved to a TagId
    // via get-or-create on `cte_client`; the blob key stays the chunk coordinate,
    // so chunks address as (path-tag, chunk-coord). Throws if the path has no valid
    // segment or on any iowarp failure. pool_size / grid_size forward to the
    // Phase 3 pool + pipeline-depth contract (see the multi-chunk ctor).
    //
    // SELF-DESCRIBING: this also publishes `layout` into the tag as the reserved
    // __meta blob (meta_blob.h), once, on the host, off the timed path. That is
    // what makes an exported .h5 possible -- the chunks alone do not encode dims,
    // chunk geometry, or element type, so without __meta the dataset is not
    // reconstructible from the store.
    //
    // The publish is BEST-EFFORT and happens AFTER construction: a failed __meta
    // write costs only exportability, not the producer's data, so it is reported
    // and not thrown (matching this class's REPORT-don't-throw I/O philosophy --
    // see the dtor). And publishing only once the backends are allocated means a
    // construction failure never strands an orphan __meta describing chunks that
    // will never be written. Chunks are not written until the caller runs a
    // kernel, so publishing here vs. before construction is equivalent to any
    // reader.
    //
    // This is why FromPath (and FromDataset) is the blessed surface: the raw
    // (TagId, Layout) constructor below takes no cte::Client and therefore
    // CANNOT publish __meta. A dataset built through it stores chunks that no
    // exporter can interpret. Prefer FromPath unless you specifically want the
    // bare, unexportable form.
    static GpuCteDataset FromPath(clio::run::IpcManager* ipc,
                                  clio::run::IpcManagerGpuInfo info, uint32_t gpu_id,
                                  cte::Client* cte_client, std::string_view path,
                                  const Layout& layout, uint32_t pool_size = 0,
                                  MemKind data_kind = MemKind::kDeviceMem,
                                  uint32_t grid_size = 0) {
        std::string tag_name = tagpath::CanonicalTag(path);
        if (tag_name.empty())
            throw std::runtime_error("GpuCteDataset::FromPath: empty/invalid tag path");
        cte::TagId tag = ResolveTagId(cte_client, tag_name);
        GpuCteDataset ds(ipc, info, gpu_id, tag, layout, pool_size, data_kind,
                         grid_size);
        // Best-effort by design: WriteDatasetMeta logs its own error and the
        // dataset stays fully usable for chunk writes even if __meta is lost
        // (only .h5 exportability is), so the producer is not aborted here.
        (void)WriteDatasetMeta(cte_client, tag, layout);
        return ds;
    }

    // Convenience over FromPath for callers that already hold a DatasetMeta (the
    // host directory model). The GPU path itself depends only on path + layout.
    static GpuCteDataset FromDataset(clio::run::IpcManager* ipc,
                                     clio::run::IpcManagerGpuInfo info, uint32_t gpu_id,
                                     cte::Client* cte_client,
                                     const DatasetMeta& meta,
                                     uint32_t pool_size = 0,
                                     MemKind data_kind = MemKind::kDeviceMem,
                                     uint32_t grid_size = 0) {
        return FromPath(ipc, info, gpu_id, cte_client, meta.path, meta.layout,
                        pool_size, data_kind, grid_size);
    }

    // Loud, non-fatal safety net. A failed PutBlob means the data is GONE, and
    // before this the loss was completely silent (see the I/O status accessors
    // above). We deliberately do NOT throw: a destructor that throws terminates,
    // and a lost write does not corrupt the caller's computation — only its
    // output. So the library REPORTS and the caller DECIDES (ThrowIfIoFailed()
    // is there for callers who consider a lost write fatal). What must never
    // happen again is losing the data and saying nothing.
    ~GpuCteDataset() {
        WarnIfIoFailureUnchecked();
        Free();
    }

    GpuCteDataset(const GpuCteDataset&) = delete;
    GpuCteDataset& operator=(const GpuCteDataset&) = delete;

    GpuCteDataset(GpuCteDataset&& o) noexcept { MoveFrom(o); }
    GpuCteDataset& operator=(GpuCteDataset&& o) noexcept {
        if (this != &o) { Free(); MoveFrom(o); }
        return *this;
    }

    [[nodiscard]] GpuDatasetHandle Handle() const { return handle_; }
    [[nodiscard]] uint32_t ChunkCount() const { return count_; }
    // Resident data-buffer count M (== ChunkCount() when unpooled).
    [[nodiscard]] uint32_t PoolSize() const { return pool_; }
    // Bytes of DEVICE payload this dataset holds resident: the registered data
    // backend size (M * chunk_bytes when pooled, else the sum of chunk sizes).
    // This is the figure that must stay bounded when buffer groups are reused
    // across snapshots — measure THIS, not a dataset count.
    [[nodiscard]] uint64_t DeviceDataBytes() const { return data_bytes_; }

    // ---- The producer-launch contract (was "grid == M"; now generalized) ----
    // The producer kernel MUST launch grid == GridSize(), and G == GridSize()
    // MUST divide M == PoolSize(). Init() enforces the divisibility; the grid is
    // the caller's to get right (GpuDatasetHandle::WritePipelined strides by the
    // handle's own G and idles blocks >= G, so an OVERSIZED launch is harmless;
    // an undersized one leaves chunks unwritten, which any read-back catches).
    //
    // Why it is a correctness contract and not a tuning hint. The host maps chunk
    // c's data buffer to c % M. A block b walks c = b, b+G, b+2G, ..., so for
    // c = b + i*G we get c % M == b + (i mod D)*G where D == M/G. Block b
    // therefore touches exactly the buffer set {b, b+G, ..., b+(D-1)*G} — and
    // distinct b are distinct residues mod G, so those sets are DISJOINT across
    // blocks. Buffer exclusivity is free; no change to the c % M mapping.
    //
    // Given exclusivity, the reuse distance is exactly M (chunk c's buffer is next
    // claimed by chunk c + D*G == c + M), so the whole pipeline invariant is:
    // drain chunk c-M before filling chunk c. D == 1 collapses to fully
    // synchronous; D == N/G (M == N) collapses to fire-all. One code path.
    [[nodiscard]] uint32_t GridSize() const { return grid_; }
    /** Pipeline depth D = M / G: in-flight Puts per block (>= 1). 0 if moved-from. */
    [[nodiscard]] uint32_t Depth() const { return grid_ ? pool_ / grid_ : 0; }

    // Indexed device-buffer / size accessors (host side, for zero/verify).
    [[nodiscard]] byte_t* DeviceData(uint32_t c) const { return host_descs_[c].data; }
    [[nodiscard]] uint64_t ChunkBytes(uint32_t c) const { return host_descs_[c].size; }

    // Single-chunk convenience: chunk 0.
    [[nodiscard]] byte_t* DeviceData() const { return host_descs_[0].data; }
    [[nodiscard]] uint64_t Bytes() const { return host_descs_[0].size; }

    // ---- I/O completion status -------------------------------------------
    // A PutBlob whose bytes do not fit in the registered bdev target FAILS in the
    // runtime, which records it correctly (core_runtime.cc:1155 sets
    // task->return_code_ = 10 + alloc_result). The device-side submit path then
    // discards that code: GpuDatasetHandle::SubmitWait only spins on the
    // completion flag, and Write()/Read() return void — so a producer kernel
    // cannot signal its own I/O failure, and the blob is lost SILENTLY. That is
    // how the "~16 backend ceiling" was misdiagnosed; see
    // test/e2e/backend_ceiling_test.cu and traces/05-backend-ceiling.md.
    //
    // These accessors surface the code the runtime already wrote. The task
    // backend is kPinnedHost (mandatory — see Init), so its slots are directly
    // CPU-readable and no device-side ABI change is needed. Call these AFTER
    // synchronizing the submitting kernel; a nonzero code means the data did not
    // land (10 + 3 == 13 is "bdev out of capacity").
    [[nodiscard]] int PutStatus(uint32_t c) const {
        return reinterpret_cast<const cte::PodPutBlobTask*>(
            task_base_ + static_cast<uint64_t>(c) * kSlotPair)->GetReturnCode();
    }
    [[nodiscard]] int GetStatus(uint32_t c) const {
        return reinterpret_cast<const cte::PodGetBlobTask*>(
            task_base_ + static_cast<uint64_t>(c) * kSlotPair + kPutSlot)->GetReturnCode();
    }

    /**
     * Index of the first chunk whose Put failed, or -1 if every chunk landed.
     * Calling this (or IoFailed()) marks the dataset's I/O status as INSPECTED,
     * which suppresses the destructor's "never checked" warning below.
     */
    [[nodiscard]] int FirstFailedPut() const {
        io_checked_ = true;
        for (uint32_t c = 0; c < count_; ++c)
            if (PutStatus(c) != 0) return static_cast<int>(c);
        return -1;
    }

    /**
     * Chunk coordinate the kernel passed to a device accessor that was NOT in
     * [0, ChunkCount()), or -1 if every coordinate was in range. Only the FIRST
     * such coordinate is kept (the device CASes from 0), because the first one is
     * the one that localizes the bug.
     *
     * This is a REJECTED ACCESS, not a failed write: the guard turned it into a
     * no-op before it could touch the descriptor array, so nothing was read, nothing
     * was submitted, and no memory was corrupted. It is surfaced through the I/O
     * status API rather than a channel of its own because the observable
     * consequence is the same as a failed Put — the caller believes a chunk was
     * written and it was not.
     *
     * Reading the word is a plain host load (the task backend is kPinnedHost), so
     * this is safe to call even while a kernel is resident. Marks as checked.
     */
    [[nodiscard]] int OutOfRangeChunk() const {
        io_checked_ = true;
        if (oob_ == nullptr) return -1;
        const uint32_t v = *oob_;
        return v == 0 ? -1 : static_cast<int>(v - 1);
    }

    /**
     * True if ANY chunk's Put failed (its data did not land) OR the kernel submitted
     * an out-of-range chunk coordinate. Marks as checked.
     */
    [[nodiscard]] bool IoFailed() const {
        return FirstFailedPut() >= 0 || OutOfRangeChunk() >= 0;
    }

    /**
     * Throwing form, for callers whose output is worthless if a write was lost —
     * the Gray-Scott benchmark above all, since a dropped write there would mean
     * reporting I/O throughput for bytes that never reached storage.
     *
     * The library itself never throws on your behalf (a failed Put does not
     * corrupt the computation, only the output — that is the caller's call to
     * make, not ours). This is the opt-in way to say "a lost write is fatal".
     */
    void ThrowIfIoFailed(const char* what = "GpuCteDataset") const {
        // Check the coordinate guard FIRST: an out-of-range access means the kernel
        // is indexing wrong, which is a more fundamental defect than a Put that was
        // correctly addressed and merely didn't fit.
        int oob = OutOfRangeChunk();
        if (oob >= 0)
            throw std::runtime_error(
                std::string(what) + ": kernel submitted OUT-OF-RANGE chunk "
                "coordinate " + std::to_string(oob) + " (dataset has " +
                std::to_string(count_) + " chunks, valid range [0, " +
                std::to_string(count_) + ")). The access was REJECTED — no memory "
                "was read or written for it, and any data that coordinate was "
                "meant to carry did NOT land. Fix the kernel's chunk indexing.");
        int c = FirstFailedPut();
        if (c < 0) return;
        throw std::runtime_error(
            std::string(what) + ": PutBlob FAILED on chunk " + std::to_string(c) +
            " (rc=" + std::to_string(PutStatus(static_cast<uint32_t>(c))) +
            "; 13 == bdev out of capacity). The data did NOT land. Size the bdev "
            "to at least the total bytes written.");
    }

    // ---- Host key re-arm: reuse this dataset for another snapshot ------------
    //
    // Re-target every chunk's Put/Get task slot to a NEW destination tag, reusing
    // the SAME task slots + data buffers + geometry. This is the host "key re-arm"
    // that lets a fixed set of buffer groups serve many snapshots with device
    // memory CONSTANT in the snapshot count (DESIGN §3.3): snapshot s and s+2 share
    // one dataset/buffer group but persist under distinct tags.
    //
    // Only tag_id_ changes. blob_name_ (the chunk coordinate), blob_data_ (the
    // device buffer), size_ and all geometry are IDENTICAL across snapshots, so a
    // re-arm is a handful of POD writes with nothing to re-allocate and nothing to
    // re-copy to the device — the device ChunkDesc array still points at the same
    // task slots + buffers (only the slot CONTENTS change), so Handle() stays valid.
    //
    // Completion state resets automatically and needs no help here: the device Send
    // clears fut_.is_complete_ on every submit, and the runtime resets the task's
    // run_ctx_ on reuse (the producer-only reuse fix in ipc_gpu2cpu.cc — without it
    // reuse deadlocks at the 2nd fire). We only clear return_code_ so a fresh
    // ThrowIfIoFailed()/PutStatus() reflects THIS snapshot's writes, and reset the
    // "never checked" guard so the destructor still nags if the caller stops
    // checking.
    //
    // ORDERING (caller's responsibility): the prior snapshot's Puts on these slots
    // MUST have drained (WriteWait / WriteDrainAll, i.e. the buffer-reuse barrier
    // the pool already enforces) BEFORE re-arming — otherwise the host would rewrite
    // tag_id_ while the CPU worker is still reading it for the in-flight Put. The
    // DEVICE never reads tag_id_ (only the worker does, at execute time), so no
    // device fence is needed; the drain barrier supplies the host-visible ordering.
    void Rearm(cte::TagId new_tag) {
        for (uint32_t c = 0; c < count_; ++c) {
            byte_t* put_slot = task_base_ + static_cast<uint64_t>(c) * kSlotPair;
            byte_t* get_slot = put_slot + kPutSlot;
            auto* put = reinterpret_cast<cte::PodPutBlobTask*>(put_slot);
            auto* get = reinterpret_cast<cte::PodGetBlobTask*>(get_slot);
            put->tag_id_ = new_tag;
            get->tag_id_ = new_tag;
            put->return_code_.store(0);
            get->return_code_.store(0);
        }
        // NOT cleared: the out-of-range report. A return_code_ is a per-snapshot I/O
        // outcome and belongs to the writes being re-armed; an out-of-range
        // coordinate is a kernel indexing DEFECT, and the geometry it violated
        // (count_) is identical across every snapshot this dataset serves. Keeping
        // it sticky for the dataset's lifetime means one bad snapshot cannot be
        // papered over by the next re-arm.
        io_checked_ = false;
    }

private:
    void Init(clio::run::IpcManagerGpuInfo info, cte::TagId tag,
              cstd::span<const ChunkSpec> specs, uint32_t pool_size = 0,
              MemKind data_kind = MemKind::kDeviceMem, uint32_t grid_size = 0) {
        count_ = static_cast<uint32_t>(specs.size());
        if (count_ == 0)
            throw std::runtime_error("GpuCteDataset: zero chunks");

        // Bounded data-buffer pool (Phase 3 "P"): M resident buffers, chunk c uses
        // buffer c % M. pool_size 0 means one buffer per chunk (M == N).
        //
        // These throw rather than clamp ON PURPOSE. The failure mode of a bad
        // (M, G) pair is a buffer being refilled while its Put is still draining,
        // i.e. SILENT DATA CORRUPTION of an already-submitted chunk. That must
        // die at construction, loudly, not degrade quietly at runtime.
        if (pool_size > count_)
            throw std::runtime_error(
                "GpuCteDataset: pool_size (M=" + std::to_string(pool_size) +
                ") exceeds chunk count (N=" + std::to_string(count_) + ")");
        pool_ = (pool_size == 0) ? count_ : pool_size;

        // Producer grid G. Default 0 => G == M (depth D == 1), the historical
        // "grid == M" contract. G must divide M, which (with the blockIdx-strided
        // walk) is exactly what makes each block's buffer set disjoint from every
        // other block's. See the GridSize() comment for the derivation.
        grid_ = (grid_size == 0) ? pool_ : grid_size;
        if (grid_ == 0)
            throw std::runtime_error("GpuCteDataset: grid_size must be >= 1");
        if (grid_ > pool_)
            throw std::runtime_error(
                "GpuCteDataset: grid_size (G=" + std::to_string(grid_) +
                ") exceeds pool_size (M=" + std::to_string(pool_) +
                "); every block needs at least one buffer");
        if (pool_ % grid_ != 0)
            throw std::runtime_error(
                "GpuCteDataset: grid_size (G=" + std::to_string(grid_) +
                ") does not divide pool_size (M=" + std::to_string(pool_) +
                "); buffer exclusivity across blocks requires G | M");

        // Queue-depth bound. Pooling bounds TOTAL outstanding submissions to <= M
        // (each of G blocks has at most D == M/G Puts in flight). The gpu2cpu ring
        // is num_lanes=1, num_prio=2, depth=gpu_queue_depth (default 16;
        // config_manager.h:402, gpu_info.h:59, gpu2cpu_init_hip.cc:98), so its lane
        // holds depth-1 == 15 entries.
        //
        // M is NOT bounded by that depth. The lane is a
        // multi_mpsc_host_consumer_ring_buffer == MPSC | FIXED_SIZE |
        // WAIT_FOR_SPACE | HOST_CONSUMER (multi_ring_buffer.h:273-277), and its
        // Emplace (ring_buffer.h:530) takes the WaitForSpace branch: a producer
        // that finds the ring full SPINS on a system-scope volatile re-read of
        // head_ until the CPU worker pops a slot. Push therefore BLOCKS — it never
        // drops and never returns false. (ipc_gpu2cpu_impl.h:114 even flags that
        // Push's return value is dropped, and states it is safe only while this
        // lane stays WAIT_FOR_SPACE.) The CPU worker drains continuously and its
        // progress does not depend on the GPU, so a parked producer always makes
        // progress: no deadlock. This is why the existing async arm fires 48 chunks
        // (M == N == 48) through a depth-16 ring and works — in-flight count is not
        // ring occupancy.
        //
        // So M > usable is CORRECT but self-throttling: blocks stall inside Push.
        // Warn, do not throw. Read the depth from `info` rather than assuming the
        // 16 default — gpu_queue_depth is configurable (gpu.queue_depth in the
        // server YAML), and a warning quoting a stale constant is worse than none.
        const uint32_t lane_usable =
            info.gpu_queue_depth ? info.gpu_queue_depth - 1 : 0;
        if (pool_ > lane_usable)
            HLOG(kInfo,
                 "GpuCteDataset: pool M={} exceeds the gpu2cpu lane's usable "
                 "capacity (~{} entries at gpu_queue_depth={}). This is SAFE — the "
                 "lane is WAIT_FOR_SPACE, so a full ring back-pressures the "
                 "submitting block (spin until the CPU worker pops), it never drops "
                 "a task — but submissions beyond that will serialize on the ring.",
                 pool_, lane_usable, info.gpu_queue_depth);

        const bool pooling = pool_ < count_;

        // Validate names + size the data backend. Non-pooled: prefix-offset every
        // chunk into one backend (sizes may differ). Pooled: M uniform-size slots.
        uint64_t total_data = 0;
        uint64_t slot_bytes = specs[0].bytes;
        for (const auto& s : specs) {
            if (std::strlen(s.name) > chunking::kMaxBlobNameLen)
                throw std::runtime_error("GpuCteDataset: blob name too long");
            total_data += s.bytes;
            if (pooling && s.bytes != slot_bytes)
                throw std::runtime_error(
                    "GpuCteDataset: buffer pool requires uniform chunk size");
        }
        const uint64_t data_bytes =
            pooling ? static_cast<uint64_t>(pool_) * slot_bytes : total_data;
        data_bytes_ = data_bytes;

        // (a) Task backend: count_ * (put-slot + get-slot), + pad.
        // MUST be kPinnedHost, matching the gpu_vector adapter reference.
        // The kernel reaches the task via UVA (FullPtr addressing + the
        // embedded fut_.is_complete_ poll both work on a pinned-host slot).
        // Critically, this is what makes the async (many-in-flight) path
        // correct: the server's gpu2cpu RecvIn treats a kDeviceMem task as
        // device-resident and D2H-copies it into a SHARED thread_local
        // scratch, then enqueues that scratch pointer non-owning for deferred
        // execution — so with >1 put in flight, concurrent tasks alias and
        // clobber the same scratch, and some never signal completion (drain
        // hangs). A pinned-host task makes IsDevicePointer() false, so RecvIn
        // skips the scratch entirely and each task keeps its own distinct
        // slot. (The old note here — "kPinnedHost didn't help, race is
        // upstream" — was against the pre-9266bd19 co-located-FutureShm model,
        // which no longer exists.)
        // The tail pad also houses the device's out-of-range report word (see
        // GpuDatasetHandle::oob_): it needs to be device-writable AND host-readable
        // with no device operation, which is exactly what this kPinnedHost backend
        // already is. 128 bytes of pad rather than 64 so the word can sit on a
        // 64-byte boundary past the last slot without running off the end —
        // kSlotPair is a sum of two task sizeofs and is not guaranteed to leave the
        // slot array 64-aligned.
        char* task_base = nullptr;
        const uint64_t task_bytes =
            static_cast<uint64_t>(count_) * kSlotPair + 128;
        task_alloc_ = ipc_->AllocateAndRegisterGpuBackend(
            gpu_id_, MemKind::kPinnedHost, task_bytes, &task_base);
        task_base_ = reinterpret_cast<byte_t*>(task_base);
        if (task_alloc_.IsNull() || task_base_ == nullptr)
            throw std::runtime_error("GpuCteDataset: task backend alloc failed");

        // Out-of-range report word, 64-aligned inside the tail pad. Zero it here:
        // the backend's contents are not guaranteed clean, and a stale nonzero word
        // would report a violation that never happened.
        const uint64_t oob_off =
            ((static_cast<uint64_t>(count_) * kSlotPair + 63) / 64) * 64;
        oob_ = reinterpret_cast<uint32_t*>(task_base_ + oob_off);
        *oob_ = 0;

        // (b) Data backend: one buffer partitioned into M distinct regions
        // (M == N when not pooling; M == pool_ < N when pooling, chunks share).
        // Placement is `data_kind` (default kDeviceMem = on-GPU, the bdev server
        // D2H-copies it out; kPinnedHost lands the payload in host memory so the
        // server needs no D2H — see the MemKind doc on the public ctors). The
        // per-chunk concurrency fix lives in the TASK MemKind above, not here.
        char* data_base = nullptr;
        data_alloc_ = ipc_->AllocateAndRegisterGpuBackend(
            gpu_id_, data_kind, data_bytes, &data_base);
        data_base_ = reinterpret_cast<byte_t*>(data_base);
        if (data_alloc_.IsNull() || data_base_ == nullptr) {
            ipc_->FreeGpuBackend(gpu_id_, task_alloc_);
            throw std::runtime_error("GpuCteDataset: data backend alloc failed");
        }

        // (c) Descriptor array backend: count_ ChunkDescs the kernel reads.
        char* desc_base = nullptr;
        const uint64_t desc_bytes =
            static_cast<uint64_t>(count_) * sizeof(ChunkDesc);
        desc_alloc_ = ipc_->AllocateAndRegisterGpuBackend(
            gpu_id_, MemKind::kDeviceMem, desc_bytes, &desc_base);
        desc_base_ = reinterpret_cast<ChunkDesc*>(desc_base);
        if (desc_alloc_.IsNull() || desc_base_ == nullptr) {
            ipc_->FreeGpuBackend(gpu_id_, data_alloc_);
            ipc_->FreeGpuBackend(gpu_id_, task_alloc_);
            throw std::runtime_error("GpuCteDataset: desc backend alloc failed");
        }

        // Stamp each chunk's task pair and build the host ChunkDesc mirror.
        // Pooled: chunk c's data points at buffer c % pool_ (shared). Non-pooled:
        // each chunk gets its own prefix-offset region.
        host_descs_.resize(count_);
        uint64_t data_off = 0;
        for (uint32_t c = 0; c < count_; ++c) {
            byte_t* put_slot = task_base_ + static_cast<uint64_t>(c) * kSlotPair;
            byte_t* get_slot = put_slot + kPutSlot;
            byte_t* chunk_data =
                pooling ? data_base_ + static_cast<uint64_t>(c % pool_) * slot_bytes
                        : data_base_ + data_off;
            StampChunk(tag, specs[c].name, specs[c].bytes, put_slot, get_slot,
                       chunk_data);
            host_descs_[c] = {MakeFullPtr<cte::PodPutBlobTask>(put_slot),
                              MakeFullPtr<cte::PodGetBlobTask>(get_slot),
                              chunk_data, specs[c].bytes};
            if (!pooling) data_off += specs[c].bytes;
        }

        // Copy the descriptor array to the device once.
        ctp::GpuApi::Memcpy(desc_base_, host_descs_.data(),
                            count_ * sizeof(ChunkDesc));

        handle_ = {info, desc_base_, count_, pool_, grid_, oob_};
    }

    // Placement-new this chunk's Pod Put/Get prototypes on the host and copy
    // them into its registered device task slots. Each task is self-contained
    // (its completion record lives in the embedded fut_, no co-located
    // FutureShm), so the slot holds only the POD task. shm.off_ carries the raw
    // device data pointer with a null alloc_id (the kernel reads it as an
    // absolute address).
    void StampChunk(cte::TagId tag, const char* name, uint64_t bytes,
                    byte_t* put_slot, byte_t* get_slot, byte_t* chunk_data) {
        ctp::ipc::ShmPtr<> blob_shm;
        blob_shm.alloc_id_.SetNull();
        blob_shm.off_ = reinterpret_cast<clio::run::u64>(chunk_data);

        alignas(64) byte_t put_proto[kPutSlot];
        std::memset(put_proto, 0, sizeof(put_proto));
        auto* put = new (put_proto) cte::PodPutBlobTask(
            clio::run::CreateTaskId(), cte::kCtePoolId,
            clio::run::PoolQuery::ToLocalCpu(), tag, name,
            /*offset=*/clio::run::u64(0), bytes, blob_shm,
            /*score=*/-1.0f, cte::Context(), /*flags=*/clio::run::u32(0));
        put->fut_.task_size_ = sizeof(cte::PodPutBlobTask);
        ctp::GpuApi::Memcpy(put_slot, put_proto, sizeof(put_proto));

        alignas(64) byte_t get_proto[kGetSlot];
        std::memset(get_proto, 0, sizeof(get_proto));
        auto* get = new (get_proto) cte::PodGetBlobTask(
            clio::run::CreateTaskId(), cte::kCtePoolId,
            clio::run::PoolQuery::ToLocalCpu(), tag, name,
            /*offset=*/clio::run::u64(0), bytes, /*flags=*/clio::run::u32(0),
            blob_shm);
        get->fut_.task_size_ = sizeof(cte::PodGetBlobTask);
        ctp::GpuApi::Memcpy(get_slot, get_proto, sizeof(get_proto));
    }

    template<typename TaskT>
    static ctp::ipc::FullPtr<TaskT> MakeFullPtr(byte_t* device_addr) {
        ctp::ipc::FullPtr<TaskT> fp;
        fp.shm_.alloc_id_.SetNull();
        fp.shm_.off_ = reinterpret_cast<clio::run::u64>(device_addr);
        fp.ptr_ = reinterpret_cast<TaskT*>(device_addr);
        return fp;
    }

    /**
     * Destructor safety net: shout if a Put failed and NOBODY EVER LOOKED.
     *
     * Guarded on task_base_/count_ so a moved-from shell (MoveFrom nulls both) is
     * silent, and on io_checked_ so a caller who already inspected the status --
     * and has therefore handled it however they chose -- is not nagged.
     */
    void WarnIfIoFailureUnchecked() const {
        if (io_checked_ || task_base_ == nullptr || count_ == 0) return;
        if (oob_ != nullptr && *oob_ != 0) {
            HLOG(kError,
                 "GpuCteDataset: kernel submitted OUT-OF-RANGE chunk coordinate {} "
                 "(dataset has {} chunks) and its status was NEVER CHECKED. The "
                 "access was rejected, so nothing was corrupted -- but whatever that "
                 "coordinate was meant to write DID NOT LAND. Call "
                 "ThrowIfIoFailed()/IoFailed()/OutOfRangeChunk() after submitting, "
                 "and fix the kernel's chunk indexing.",
                 *oob_ - 1, count_);
            return;
        }
        for (uint32_t c = 0; c < count_; ++c) {
            int rc = PutStatus(c);
            if (rc == 0) continue;
            HLOG(kError,
                 "GpuCteDataset: chunk {} PutBlob FAILED (rc={}, 13 == bdev out "
                 "of capacity) and its status was NEVER CHECKED -- THE DATA WAS "
                 "SILENTLY LOST. Call ThrowIfIoFailed()/IoFailed() after "
                 "submitting, and size the bdev to at least the total bytes "
                 "written.", c, rc);
            return;  // one line per dataset is enough; don't spam per chunk
        }
    }

    void Free() {
        if (!desc_alloc_.IsNull()) ipc_->FreeGpuBackend(gpu_id_, desc_alloc_);
        if (!data_alloc_.IsNull()) ipc_->FreeGpuBackend(gpu_id_, data_alloc_);
        if (!task_alloc_.IsNull()) ipc_->FreeGpuBackend(gpu_id_, task_alloc_);
    }

    void MoveFrom(GpuCteDataset& o) {
        ipc_ = o.ipc_;
        gpu_id_ = o.gpu_id_;
        task_alloc_ = o.task_alloc_;
        data_alloc_ = o.data_alloc_;
        desc_alloc_ = o.desc_alloc_;
        task_base_ = o.task_base_;
        data_base_ = o.data_base_;
        desc_base_ = o.desc_base_;
        host_descs_ = std::move(o.host_descs_);
        count_ = o.count_;
        pool_ = o.pool_;
        grid_ = o.grid_;
        data_bytes_ = o.data_bytes_;
        oob_ = o.oob_;
        handle_ = o.handle_;
        io_checked_ = o.io_checked_;
        o.io_checked_ = true;  // moved-from shell owns nothing; never warn for it
        o.task_alloc_.SetNull();
        o.data_alloc_.SetNull();
        o.desc_alloc_.SetNull();
        o.task_base_ = nullptr;
        o.data_base_ = nullptr;
        o.desc_base_ = nullptr;
        o.oob_ = nullptr;
        o.count_ = 0;
        o.pool_ = 0;
        o.grid_ = 0;
    }
};

}  // namespace kvhdf5

#endif  // !CTP_IS_DEVICE_PASS
