#pragma once

// Device-facing handle for the iowarp GPU producer path.
//
// A GpuDatasetHandle is a small, by-value POD built on the host by GpuCteDataset
// and passed BY VALUE into the user's compute kernel. It describes a dataset of
// N chunks: the per-chunk Put/Get tasks + data buffers live in a device-resident
// ChunkDesc array (chunks_), and the handle carries only a pointer to it + the
// count. This keeps the kernel-arg size constant regardless of N (inlining N
// FullPtr pairs by value would hit the CUDA kernel-parameter ceiling).
//
// Inside the kernel the user fills Data(c) for a chunk and calls Write(c) (or
// Read(c)) to submit that chunk's pre-built task — the I/O is fused into the
// user's launch instead of being orchestrated from the host. A block can own one
// chunk (chunk_id = blockIdx) or grid-stride over a range of chunks; both index
// the same chunks_ array, so one-block-per-chunk is just the gridDim >= N case of
// the grid-stride form. The no-arg Data()/Write()/... overloads target chunk 0
// (the single-chunk specialization).
//
// Contract: the kernel MUST run CLIO_GPU_INIT(handle.info_, nullptr) at
// block scope before calling Write()/Read(). That macro declares a *kernel-
// local* g_ipc_manager_ptr and does the block-wide ClientInitGpu + __syncthreads
// — so Write()/Read() can't see it and instead re-fetch the per-block IpcManager
// via GetBlockIpcManager() (the same accessor the macro uses).

#include "defines.h"

#include <clio_runtime/types.h>
#include <clio_runtime/gpu/future.h>
#include <clio_runtime/gpu/gpu_info.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_cte/core/core_tasks.h>

#include <cstdint>

// Whether an out-of-range chunk coordinate additionally TRAPS the kernel (on top of
// the always-on host-visible report). Defaults to on in debug, off in release, so a
// developer hits it immediately but a shipped kernel degrades to the reported no-op.
//
// A trap destroys the CUDA context, which is unrecoverable in a shared-server,
// single-process test binary — so that binary defines this to 0 (see the e2e
// CMakeLists) to exercise the graceful path in-process. Define it uniformly for a
// whole binary (a target compile definition), never per-TU: InRange/ReportOutOfRange
// are inline __device__ members, so mismatched values across TUs would be an ODR
// violation.
#ifndef KVHDF5_GPU_OOR_TRAP
#  ifdef NDEBUG
#    define KVHDF5_GPU_OOR_TRAP 0
#  else
#    define KVHDF5_GPU_OOR_TRAP 1
#  endif
#endif

namespace kvhdf5 {

namespace cte = clio::cte::core;

// One element of the device-resident per-chunk array. Built on the host
// (GpuCteDataset stamps the prototypes and fills these), copied to the device
// once, and read by the kernel. The FullPtrs address that chunk's pre-built
// PutBlob/GetBlob task slots; data/size address its distinct device buffer.
struct ChunkDesc {
    ctp::ipc::FullPtr<cte::PodPutBlobTask> put_fp;
    ctp::ipc::FullPtr<cte::PodGetBlobTask> get_fp;
    byte_t* data = nullptr;     // this chunk's registered device blob buffer
    uint64_t size = 0;
    // Submit-probe record claimed by this chunk's in-flight *Async, read back by
    // its matching *Wait. Needed because the fire and the drain are separate
    // calls (the async path fires every chunk before draining any), so the slot
    // cannot live in a per-block scratch. kProbeNoSlot = probe off.
    uint32_t probe_slot = clio::run::gpu::kProbeNoSlot;
};

struct GpuDatasetHandle {
    clio::run::IpcManagerGpuInfo info_;
    ChunkDesc* chunks_ = nullptr;   // device array, count_ entries
    uint32_t count_ = 0;            // N chunks
    // The bounded-pool contract, carried by value so WritePipelined can enforce it
    // without the caller re-deriving it. Two uint32s; the handle must stay small
    // (it is a kernel parameter). pool_ == count_ and grid_ == pool_ for an
    // unpooled dataset, which is exactly depth D == 1 over N buffers.
    uint32_t pool_ = 0;             // M resident data buffers (host maps c -> c % M)
    uint32_t grid_ = 0;             // G producer blocks; G divides M; D = M / G
    // ---- Out-of-range chunk-coordinate report --------------------------------
    //
    // Sticky, one word per dataset, holding (first offending c) + 1 so that 0 means
    // "no violation" without burning a sentinel coordinate. Points INTO the task
    // backend's tail pad (GpuCteDataset::Init), which is kPinnedHost — deliberately:
    //
    //  * device-writable through UVA, so the kernel needs no extra allocation and
    //    no cudaMemcpy to report;
    //  * host-readable with ZERO device operations, which is the binding constraint.
    //    The persistent-kernel arm keeps one cooperative grid resident for the whole
    //    run and completes its writes with no device-side op; a report channel that
    //    needed a D2H (or any device call) on the completion path could not schedule
    //    under that resident grid and would deadlock it. A mapped pinned word cannot.
    //
    // Null only for a default-constructed handle, in which case reporting is skipped
    // and the guard still degrades to a safe no-op.
    uint32_t* oob_ = nullptr;

#if CTP_IS_GPU_COMPILER
    __device__ uint32_t Count() const { return count_; }
    __device__ uint32_t PoolSize() const { return pool_; }
    __device__ uint32_t GridSize() const { return grid_; }
    __device__ uint32_t Depth() const { return pool_ / grid_; }
    __device__ byte_t* Data(uint32_t c) const {
        if (!InRange(c)) return nullptr;
        return chunks_[c].data;
    }
    __device__ uint64_t Size(uint32_t c) const {
        if (!InRange(c)) return 0;
        return chunks_[c].size;
    }

    // Submit chunk c's pre-built Put/Get task and wait. Thread-0 of the block
    // enqueues; all other threads no-op (iowarp's threadIdx==0 producer guard).
    // Probing defaults to true so callers that don't specify it keep the exact
    // prior codegen; the bench opts into Write<false>/WriteAsync<false> to shed
    // the SendIn submit-probe registers on its hot non-probing runs.
    template<bool Probing = true>
    __device__ void Write(uint32_t c) const {
        if (!InRange(c)) return;
        Submit<Probing>(c, chunks_[c].put_fp);
    }
    __device__ void Read(uint32_t c) const {
        if (!InRange(c)) return;
        Submit(c, chunks_[c].get_fp);
    }

    // Async split of Write/Read (the "Phase 2" overlap path): *Async fires the
    // chunk's pre-built task WITHOUT waiting; *Wait drains it later. Firing many
    // chunks before draining lets the server's CPU-side IO of earlier chunks
    // overlap the GPU-side compute of later ones. Requires a distinct buffer per
    // in-flight chunk (i.e. an unpooled M==N dataset), so a chunk's fill can't
    // clobber another chunk's buffer while its put is still draining. Thread-0
    // only; pair every *Async(c) with exactly one *Wait(c) before the kernel
    // exits (the host Synchronize waits the kernel, not the server's puts).
    template<bool Probing = true>
    __device__ void WriteAsync(uint32_t c) const {
        if (!InRange(c)) return;
        SubmitAsync<Probing>(c, chunks_[c].put_fp);
    }
    __device__ void ReadAsync(uint32_t c) const {
        if (!InRange(c)) return;
        SubmitAsync(c, chunks_[c].get_fp);
    }
    __device__ void WriteWait(uint32_t c) const {
        if (!InRange(c)) return;
        SubmitWait(c, chunks_[c].put_fp);
    }
    __device__ void ReadWait(uint32_t c) const {
        if (!InRange(c)) return;
        SubmitWait(c, chunks_[c].get_fp);
    }

    // ---- Bounded-pool double buffering: fill + fire, one fused kernel --------
    //
    // Streams ALL N chunks this block owns through the dataset's M resident data
    // buffers, keeping at most D = M/G Puts in flight per block. This is the ONLY
    // supported way to produce into a POOLED dataset (M < N), and it encodes the
    // discipline so a caller cannot get it wrong.
    //
    // Why fill and fire MUST be fused. The old bench shape — one kernel fills all
    // N buffers, a second kernel fires all N Puts — is CORRECT ONLY at M == N.
    // With M < N the fill kernel writes chunk c and chunk c+M into the SAME
    // buffer before either is submitted, so the first chunk's payload is gone
    // before anyone reads it. There is no fix for that outside a fused loop.
    //
    // Exclusivity (free, no host-mapping change). The host maps chunk c's buffer
    // to c % M. Block b walks c = b, b+G, b+2G, ...; for c = b + i*G,
    // c % M == b + (i mod D)*G. So block b touches exactly {b, b+G, ...,
    // b+(D-1)*G} — stride-G — and distinct b are distinct residues mod G, so no
    // two blocks ever share a buffer.
    //
    // The invariant. Given exclusivity, chunk c's buffer is next claimed by chunk
    // c + D*G == c + M. The reuse distance is exactly M, so the whole pipeline
    // reduces to: DRAIN c-M BEFORE FILLING c. Endpoints collapse for free —
    //   D == 1   (M == G): wait(c-G) == the previous chunk this block did =>
    //                      fully synchronous, one buffer per block.
    //   D == N/G (M == N): c-M < 0 always => the in-loop wait never fires and
    //                      everything drains in the tail => today's fire-all.
    // One code path, no special-casing.
    //
    // `fill(c, data, size)` is invoked BLOCK-WIDE (every thread) to fill chunk c's
    // buffer. It must be uniform across the block (no divergent __syncthreads()).
    //
    // THE LOAD-BEARING LINE is the __syncthreads() immediately after the wait.
    // WriteWait is THREAD-0-ONLY (iowarp's producer guard). Without that barrier a
    // non-zero thread races straight into `fill` and clobbers the buffer while
    // chunk c-M's Put is still draining on the CPU side => SILENT DATA CORRUPTION
    // of an already-submitted chunk. Every other sync here is ordinary hygiene;
    // that one is the entire safety property.
    //
    // Grid: launch with gridDim.x == GridSize(). We stride by the handle's own G
    // (not gridDim.x) so an OVERSIZED launch is harmless — surplus blocks own no
    // chunks and return. An UNDERSIZED launch leaves chunks unwritten, which any
    // read-back catches loudly; it cannot corrupt.
    // TailDrain: whether this kernel also drains the <= D Puts still in flight when
    // its chunk walk ends. It is NOT what enforces the buffer-reuse invariant —
    // the in-loop WriteWait(c-M) is — so deferring it does not weaken correctness,
    // and it costs REAL overlap:
    //
    //   With TailDrain=true the kernel cannot RETIRE until this dataset's last D
    //   Puts have landed. Anything queued behind it on the stream (the next
    //   snapshot's compute) is therefore held behind THIS snapshot's I/O, and the
    //   pipeline degenerates to compute + io per snapshot instead of
    //   max(compute, io). At M == N the tail condition (c + M >= N) is true for
    //   EVERY chunk, so it collapses all the way to "fire all, then drain all,
    //   in-kernel" — i.e. it performs like the synchronous arm, which is exactly
    //   what we measured (pooled ~= sync ~2.1 s, vs async ~1.7 s).
    //
    // TailDrain=false leaves those <= D Puts in flight past kernel exit; the CALLER
    // MUST drain them (a WriteWait over every chunk — idempotent for the ones the
    // loop already drained, since the wait just polls an already-set completion
    // flag) BEFORE it reads the blobs back, destroys the dataset, or REFILLS ANY OF
    // ITS BUFFERS. Safe when each producer launch owns its own dataset (the
    // one-dataset-per-snapshot shape): nothing refills those buffers in between, so
    // the drain can be deferred to a single kernel at the end — which is precisely
    // how the fire-all path already overlaps I/O with later compute.
    //
    // Re-launching this kernel on the SAME dataset with TailDrain=false and no
    // intervening drain is a buffer-reuse race: chunk c would be refilled while the
    // previous launch's Put on c may still be draining. Default is true (safe).
    //
    // *** TailDrain=false REQUIRES a PRE-WARMED drain kernel. This is a CORRECTNESS
    // requirement, not a perf tuning knob — get it wrong and you DEADLOCK. ***
    //
    // CUDA 12 defaults to CUDA_MODULE_LOADING=LAZY: a kernel's device code is loaded
    // on its FIRST launch, and that load is a device-wide SYNCHRONIZING driver
    // operation. With TailDrain=false the producer kernel is still RESIDENT and
    // spinning in WriteWait when the host issues the drain kernel. If that drain
    // launch is the drain kernel's first, its lazy module load blocks the device —
    // including the CUDA calls the CPU-side CTE server needs to COMPLETE the very
    // Puts the producer is spinning on. The producer waits on the server, the server
    // waits on the module load, the module load waits on the producer to retire.
    // Nothing moves. (Same mechanism as the cold-launch device serialization written
    // up in context-runtime/test/unit/gpu/GPU_SUBMIT_SERIALIZES_DEVICE.md, but
    // escalated from "serializes" to "hangs", because here the two kernels are
    // mutually dependent rather than merely queued.)
    //
    // Fix: force the module load WITHOUT launching, before the producer goes
    // resident — cudaFuncGetAttributes on the drain kernel (and on this one), or
    // CUDA_MODULE_LOADING=EAGER in the environment. A cudaDeviceSynchronize between
    // the producer and the drain also "fixes" it, but that sync is exactly the
    // overlap the deferral exists to buy, so it defeats the purpose.
    // TailDrain=true has no such hazard: it never leaves a producer resident across
    // another kernel's launch.
    template<bool Probing = true, bool TailDrain = true, typename FillFn>
    __device__ void WritePipelined(FillFn fill) const {
        const uint32_t g = grid_, m = pool_, n = count_, b = blockIdx.x;
        if (b >= g) return;  // surplus blocks own no chunks
        for (uint32_t c = b; c < n; c += g) {
            // Reclaim this chunk's buffer from the chunk that used it M ago. THIS is
            // the invariant; it stays put regardless of TailDrain.
            //
            // c and m are uint32_t, so `c - m` would WRAP (not go negative) on
            // underflow and index chunks_ astronomically out of range. The `c >= m`
            // test is what makes that unreachable: it is evaluated on the same two
            // values in the same expression, so the subtraction is only ever taken
            // when it cannot wrap, and the result obeys c - m < c < n == count_ —
            // in range by construction. WriteWait's own guard is therefore
            // belt-and-braces here, and it costs nothing: it is thread-0-only and
            // contains no barrier, so a guarded skip can never desynchronize the
            // __syncthreads() below.
            if (c >= m) WriteWait(c - m);   // thread-0 only
            __syncthreads();                // <-- ALL threads wait for that drain
            fill(c, chunks_[c].data, chunks_[c].size);
            // Each thread publishes its own stores system-wide; the CPU-side Put
            // reads this buffer, and SendIn's fence only covers thread 0.
            __threadfence_system();
            __syncthreads();
            WriteAsync<Probing>(c);         // thread-0 only; do NOT wait here
        }
        // Tail: the chunks this block fired whose buffers were never re-claimed
        // (c + M >= N) are still in flight. At most D of them.
        if (TailDrain) {
            for (uint32_t c = b; c < n; c += g)
                if (c + m >= n) WriteWait(c);
            __syncthreads();
        }
    }

    // Deferred companion to WritePipelined<..., TailDrain=false>: drain every chunk
    // this block owns. Idempotent for chunks the pipelined loop already drained (the
    // wait polls a completion flag that is already set), so it is safe to run over
    // the whole range rather than tracking which <= D are outstanding.
    __device__ void WriteDrainAll() const {
        const uint32_t g = grid_, n = count_, b = blockIdx.x;
        if (b >= g) return;
        for (uint32_t c = b; c < n; c += g) WriteWait(c);
        __syncthreads();
    }

    // ---- Device self-addressing: in-kernel per-snapshot key stamp -----------
    //
    // Re-target chunk c's Put/Get to a new destination `tag` from INSIDE the
    // kernel, before firing — the tag-table addressing model (DESIGN §3.3) that
    // lets a persistent kernel serve many snapshots out of a fixed set of reused
    // buffer groups: the host pre-builds a device-visible TagId[num_snaps] table
    // off the hot path, and the kernel stamps chunk c's tag = table[snapshot]
    // just before WriteAsync(c). The blob name (chunk coordinate) is unchanged,
    // so snapshot s lands as its OWN dataset (tag) under the same coordinate key.
    //
    // The runtime reads task->tag_id_ at store time (core_runtime.cc:1085), so a
    // device-set tag routes EXACTLY like a host-set one (same field, same path;
    // gpu_vector already sets task fields in-kernel this way). Thread-0 only
    // (iowarp's producer guard) — the write is published system-wide by the
    // __threadfence_system that WriteAsync issues before the task is enqueued, so
    // the CPU worker reads the fresh tag. Call SetTag(c, ...) then WriteAsync(c)
    // from the SAME thread-0 with no barrier between; do NOT let a non-zero thread
    // fire before the stamp lands.
    __device__ void SetTag(uint32_t c, cte::TagId tag) const {
        if (clio::run::gpu::IpcManager::GetGpuThreadId() != 0) return;
        if (!InRange(c)) return;   // inside the thread-0 guard: one report, not 256
        chunks_[c].put_fp.ptr_->tag_id_ = tag;
        chunks_[c].get_fp.ptr_->tag_id_ = tag;
    }

    // Single-chunk convenience: target chunk 0. Routed through the indexed forms so
    // they inherit the same bounds guard — chunk 0 is in range for every dataset
    // Init() will build (it rejects zero chunks), so this is identical codegen for a
    // live handle; it only catches a default-constructed / moved-from one, where
    // chunks_ is null and the old form dereferenced it.
    __device__ byte_t* Data() const { return Data(0); }
    __device__ uint64_t Size() const { return Size(0); }
    __device__ void Write() const { Write(0); }
    __device__ void Read() const { Read(0); }
    __device__ void WriteAsync() const { WriteAsync(0); }
    __device__ void WriteWait() const { WriteWait(0); }
    __device__ void ReadAsync() const { ReadAsync(0); }
    __device__ void ReadWait() const { ReadWait(0); }

private:
    // ---- Chunk-coordinate bounds guard --------------------------------------
    //
    // Every device accessor funnels its coordinate through here BEFORE it can
    // subscript chunks_. Out of range used to read past the end of the device
    // ChunkDesc array and then dereference whatever pointer/FullPtr it found —
    // undefined behaviour that could silently corrupt unrelated device memory.
    // Now it is a no-op that the host can see.
    //
    // Cost, on the in-range path: ONE compare of c against count_, a field the
    // handle already carries by value in the kernel parameter bank. No load, no
    // memory traffic, no divergence that was not already there. The report side
    // is entirely off the fall-through path.
    //
    // Uniformity: c is block-uniform at every call site in this header, so the
    // branch is warp- and block-uniform where the caller is whole-block (Data,
    // Size) and sits under the existing thread-0 producer guard where one exists
    // (SetTag). CRITICALLY, no guarded early-return here skips a __syncthreads()
    // or a grid.sync(): the submit/drain paths this guards contain no barrier at
    // all, and WritePipelined's barriers sit outside the guarded calls. A resident
    // cooperative grid therefore cannot desynchronize on a rejected coordinate.
    __device__ bool InRange(uint32_t c) const {
        if (c < count_) return true;
        ReportOutOfRange(c);
        return false;
    }

    // Record the FIRST offending coordinate and keep it: atomicCAS from 0 means a
    // later, less interesting violation cannot overwrite the one that started it.
    // Stored as c+1 so 0 stays a clean "never happened" for the host.
    //
    // *oob_ is a mapped pinned-host word, so the store lands system-visible with no
    // device operation and no D2H — see the oob_ member comment for why that
    // matters to the persistent arm. _system scope matches how the gpu2cpu ring
    // already does its cross-PCIe atomics on pinned memory.
    //
    // Debug builds additionally trap, so a bad coordinate fails loudly in
    // development instead of being discovered later in the host status. Release
    // builds never trap: the guard degrades to the silent-but-reported no-op.
    //
    // __noinline__ so the cold report body (a system-scope atomic, and the trap
    // in debug) is NOT copied into every accessor's call site — it keeps InRange's
    // fall-through to a single compare and holds the per-kernel register delta at
    // the hot path to ~zero, which §eval publishes.
    __device__ __noinline__ void ReportOutOfRange(uint32_t c) const {
        if (oob_) atomicCAS_system(oob_, 0u, c + 1u);
#if KVHDF5_GPU_OOR_TRAP
        __trap();
#endif
    }

    template<bool Probing = true, typename TaskT>
    __device__ void Submit(uint32_t c, const ctp::ipc::FullPtr<TaskT>& fp) const {
        SubmitAsync<Probing>(c, fp);
        SubmitWait(c, fp);
    }

    // Fire fp's task; thread-0 enqueues, others no-op. Discards the returned
    // future — SubmitWait rebuilds it from fp's slot (below), so nothing needs
    // to be retained across the fire/drain gap.
    //
    // `c` is carried only for the submit probe: it claims this submit's record
    // and parks the slot both on the block-shared IpcManager (so SendIn can stamp
    // the fence/push hops without a signature change) and on the ChunkDesc (so
    // this chunk's later *Wait can find the same record). Every probe branch is
    // dead when the probe is off.
    template<bool Probing = true, typename TaskT>
    __device__ void SubmitAsync(uint32_t c, const ctp::ipc::FullPtr<TaskT>& fp) const {
        auto* ipc = clio::run::gpu::IpcManager::GetBlockIpcManager();
        if (clio::run::gpu::IpcManager::GetGpuThreadId() != 0) return;
        if (ipc->gpu_info_.probe_.On()) {
            uint32_t slot = clio::run::gpu::ProbeClaim(ipc->gpu_info_.probe_);
            ipc->probe_slot_ = slot;
            chunks_[c].probe_slot = slot;
        }
        (void)ipc->Send<Probing>(fp);
    }

    // Drain fp's task: thread-0 polls task->fut_.is_complete_. The task is now
    // self-contained (its completion record lives in the POD's embedded fut_,
    // no co-located gpu::FutureShm), so the wait is reconstructed statelessly
    // from fp alone — a fresh gpu::Future over the same task slot reads the same
    // completion flag the CPU worker flips. No stored future is needed.
    template<typename TaskT>
    __device__ void SubmitWait(uint32_t c, const ctp::ipc::FullPtr<TaskT>& fp) const {
        if (clio::run::gpu::IpcManager::GetGpuThreadId() != 0) return;
        auto* ipc = clio::run::gpu::IpcManager::GetBlockIpcManager();
        clio::run::gpu::SubmitProbeRec* prec =
            clio::run::gpu::ProbeRec(ipc->gpu_info_.probe_, chunks_[c].probe_slot);
        // Held in registers across the spin; committed after it. Both clocks: the
        // wall clock to join against the CPU's SendOut, the cycle counter to give
        // this record its own cycles->ns ratio (same block, so clock64 is coherent
        // with the c_* stamps SendIn took above).
        unsigned long long w_begin_ns = 0, w_begin_cy = 0;
        if (prec) {
            w_begin_ns = clio::run::gpu::ProbeNowNs();
            w_begin_cy = clio::run::gpu::ProbeCycles();
        }
        clio::run::gpu::Future<TaskT> fut(fp, sizeof(TaskT));
        fut.Wait();
        if (prec) {
            const unsigned long long w_end_cy = clio::run::gpu::ProbeCycles();
            const unsigned long long w_end_ns = clio::run::gpu::ProbeNowNs();
            prec->d_wait_begin = w_begin_ns;
            prec->d_wait_end = w_end_ns;
            prec->c_wait_begin = w_begin_cy;
            prec->c_wait_end = w_end_cy;
        }
    }
#endif  // CTP_IS_GPU_COMPILER
};

// No is_trivially_copyable static_assert here on purpose: iowarp's
// FullPtr<T> declares user-provided copy/move ctors, so neither trait holds — yet
// the reference passes FullPtr and IpcManagerGpuInfo BY VALUE straight into a
// __global__ kernel and it works. ChunkDesc bundles those same proven types and
// is read from a device array (never copy-constructed on device); the handle
// itself stays small (info_ + pointer + count), so the guarantee is the runtime
// round-trip, not a trait.

}  // namespace kvhdf5
