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
// This is NOT a BlobBackend — its surface is Handle(), not WriteChunk/ReadChunk.
// It owns three device allocations, so it is move-only and frees them in the dtor.

#include "defines.h"
#include "chunking.h"
#include "cpu_dataset.h"  // Layout, DatasetMeta
#include "gpu_dataset_handle.h"
#include "tag_resolve.h"  // ResolveTagId (path->TagId)

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
    // M < N. Chunk c reuses data buffer c % M; its task still has a distinct blob
    // name, so N chunks stream through M buffers. pool_size == 0 (default) keeps
    // one buffer per chunk (M == N). Pooling requires uniform chunk size (the
    // Layout ctor already produces that). Buffer-exclusivity: the producer kernel
    // MUST launch grid == M so each pooled buffer is owned by a single block and
    // reuse is serialized by that block's Write() = Send().Wait().
    GpuCteDataset(clio::run::IpcManager* ipc, clio::run::IpcManagerGpuInfo info,
                  uint32_t gpu_id, cte::TagId tag, const Layout& layout,
                  uint32_t pool_size = 0, MemKind data_kind = MemKind::kDeviceMem)
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

        Init(info, tag, {specs.data(), specs.size()}, pool_size, data_kind);
    }

    // Primary path->GPU surface: the producer needs only a dataset `path` (-> tag)
    // and a `layout` (chunk geometry) — no host metadata struct. `path` is
    // canonicalized into the CTE tag (the path->tag scheme) and resolved to a TagId
    // via get-or-create on `cte_client`; the blob key stays the chunk coordinate,
    // so chunks address as (path-tag, chunk-coord). Throws if the path has no valid
    // segment or on any iowarp failure. pool_size forwards to the Phase 3 pool.
    static GpuCteDataset FromPath(clio::run::IpcManager* ipc,
                                  clio::run::IpcManagerGpuInfo info, uint32_t gpu_id,
                                  cte::Client* cte_client, std::string_view path,
                                  const Layout& layout, uint32_t pool_size = 0,
                                  MemKind data_kind = MemKind::kDeviceMem) {
        std::string tag_name = tagpath::CanonicalTag(path);
        if (tag_name.empty())
            throw std::runtime_error("GpuCteDataset::FromPath: empty/invalid tag path");
        cte::TagId tag = ResolveTagId(cte_client, tag_name);
        return GpuCteDataset(ipc, info, gpu_id, tag, layout, pool_size, data_kind);
    }

    // Convenience over FromPath for callers that already hold a DatasetMeta (the
    // host directory model). The GPU path itself depends only on path + layout.
    static GpuCteDataset FromDataset(clio::run::IpcManager* ipc,
                                     clio::run::IpcManagerGpuInfo info, uint32_t gpu_id,
                                     cte::Client* cte_client,
                                     const DatasetMeta& meta,
                                     uint32_t pool_size = 0,
                                     MemKind data_kind = MemKind::kDeviceMem) {
        return FromPath(ipc, info, gpu_id, cte_client, meta.path, meta.layout,
                        pool_size, data_kind);
    }

    ~GpuCteDataset() { Free(); }

    GpuCteDataset(const GpuCteDataset&) = delete;
    GpuCteDataset& operator=(const GpuCteDataset&) = delete;

    GpuCteDataset(GpuCteDataset&& o) noexcept { MoveFrom(o); }
    GpuCteDataset& operator=(GpuCteDataset&& o) noexcept {
        if (this != &o) { Free(); MoveFrom(o); }
        return *this;
    }

    [[nodiscard]] GpuDatasetHandle Handle() const { return handle_; }
    [[nodiscard]] uint32_t ChunkCount() const { return count_; }
    // Resident data-buffer count M (== ChunkCount() when unpooled). The producer
    // kernel must launch grid == PoolSize() so each buffer is owned by one block.
    [[nodiscard]] uint32_t PoolSize() const { return pool_; }

    // Indexed device-buffer / size accessors (host side, for zero/verify).
    [[nodiscard]] byte_t* DeviceData(uint32_t c) const { return host_descs_[c].data; }
    [[nodiscard]] uint64_t ChunkBytes(uint32_t c) const { return host_descs_[c].size; }

    // Single-chunk convenience: chunk 0.
    [[nodiscard]] byte_t* DeviceData() const { return host_descs_[0].data; }
    [[nodiscard]] uint64_t Bytes() const { return host_descs_[0].size; }

private:
    void Init(clio::run::IpcManagerGpuInfo info, cte::TagId tag,
              cstd::span<const ChunkSpec> specs, uint32_t pool_size = 0,
              MemKind data_kind = MemKind::kDeviceMem) {
        count_ = static_cast<uint32_t>(specs.size());
        if (count_ == 0)
            throw std::runtime_error("GpuCteDataset: zero chunks");

        // Bounded data-buffer pool (Phase 3 "P"): M resident buffers, chunk c uses
        // buffer c % M. pool_size 0 or >= count_ means one buffer per chunk (M==N).
        pool_ = (pool_size == 0 || pool_size >= count_) ? count_ : pool_size;
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
        char* task_base = nullptr;
        const uint64_t task_bytes =
            static_cast<uint64_t>(count_) * kSlotPair + 64;
        task_alloc_ = ipc_->AllocateAndRegisterGpuBackend(
            gpu_id_, MemKind::kPinnedHost, task_bytes, &task_base);
        task_base_ = reinterpret_cast<byte_t*>(task_base);
        if (task_alloc_.IsNull() || task_base_ == nullptr)
            throw std::runtime_error("GpuCteDataset: task backend alloc failed");

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

        handle_ = {info, desc_base_, count_};
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
        handle_ = o.handle_;
        o.task_alloc_.SetNull();
        o.data_alloc_.SetNull();
        o.desc_alloc_.SetNull();
        o.task_base_ = nullptr;
        o.data_base_ = nullptr;
        o.desc_base_ = nullptr;
        o.count_ = 0;
        o.pool_ = 0;
    }
};

}  // namespace kvhdf5

#endif  // !CTP_IS_DEVICE_PASS
