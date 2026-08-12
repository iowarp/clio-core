#pragma once

// Runtime side of the __meta scheme (see meta_blob.h): publish a dataset's
// Layout into its own tag as the reserved kMetaBlobName blob, so the dataset
// becomes self-describing and an exporter can reconstruct its shape and type
// without any out-of-band information.
//
// Host-only and iowarp-facing, so -- exactly like tag_resolve.h -- it is kept
// out of the shared headers (layout.h/chunking.h/meta_blob.h) and guarded out of
// the device pass. That keeps the no-iowarp unit build of the pure layout/meta
// code clean.

#include <clio_ctp/constants/macros.h>  // CTP_IS_DEVICE_PASS

#if !CTP_IS_DEVICE_PASS

#include "layout.h"
#include "meta_blob.h"

#include <clio_runtime/ipc_manager.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace kvhdf5 {

namespace cte = clio::cte::core;

namespace detail {
// RAII for a runtime shm buffer: frees on every scope exit, including an
// exception thrown between AllocateBuffer and the manual free (e.g. from
// AsyncPutBlob/Wait). Move-only so it can't double-free.
struct ShmBufGuard {
    clio::run::IpcManager* ipc;
    ctp::ipc::FullPtr<char> buf;
    ShmBufGuard(clio::run::IpcManager* i, ctp::ipc::FullPtr<char> b)
        : ipc(i), buf(b) {}
    ShmBufGuard(const ShmBufGuard&) = delete;
    ShmBufGuard& operator=(const ShmBufGuard&) = delete;
    ~ShmBufGuard() { if (ipc != nullptr && !buf.IsNull()) ipc->FreeBuffer(buf); }
};
}  // namespace detail

/**
 * Write `layout`'s description into tag `tag` as the reserved __meta blob.
 * Returns true iff the blob landed.
 *
 * Idempotent: re-publishing the same dataset overwrites the blob with identical
 * bytes, so re-opening an existing dataset is harmless.
 *
 * NOT ON THE HOT PATH, and must stay that way: this is a ~150-byte host PutBlob
 * performed ONCE per dataset at construction. It is deliberately not per-step --
 * the gray-scott benchmark measures write throughput, and a per-step metadata
 * Put would land inside the timed region and tax every arm.
 *
 * FAILURE MODEL. A null client or an invalid/non-representable layout is a
 * programmer error at the call site and THROWS. A runtime failure to actually
 * store the blob (shm exhaustion, PutBlob rc != 0) is REPORTED and returns
 * false: losing __meta costs only .h5 exportability, never the producer's chunk
 * data, so it must not abort a producer that may not even care about export
 * (the benchmark above all). This matches GpuCteDataset's own I/O philosophy --
 * report the loss, let the caller decide -- rather than throwing on an
 * output-only failure. [[nodiscard]] so a caller that DOES care about export
 * can't silently ignore it; FromPath opts out explicitly.
 */
[[nodiscard]] inline bool WriteDatasetMeta(cte::Client* client, cte::TagId tag,
                                           const Layout& layout) {
    if (client == nullptr)
        throw std::runtime_error("WriteDatasetMeta: null CTE client");
    if (!layout.Valid())
        throw std::runtime_error("WriteDatasetMeta: invalid layout");

    const MetaBlob meta = layout.ToMetaBlob();
    // Catches the edge-chunk case (and anything else MetaBlob rejects) at WRITE
    // time -- a bug in the caller's layout, so it throws.
    if (!meta.Valid())
        throw std::runtime_error(
            "WriteDatasetMeta: layout is not representable as a __meta blob "
            "(non-divisible dims / unknown dtype?)");

    auto* ipc = CLIO_CPU_IPC;
    ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(sizeof(MetaBlob));
    if (buf.IsNull()) {
        HLOG(kError,
             "WriteDatasetMeta: shm allocation failed; __meta not written. This "
             "dataset will NOT be exportable to .h5. Chunk data is unaffected.");
        return false;
    }
    detail::ShmBufGuard guard(ipc, buf);
    EncodeMeta(meta, buf.ptr_);

    ctp::ipc::ShmPtr<> shm_ptr(buf.shm_);
    auto task = client->AsyncPutBlob(tag, kMetaBlobName, /*offset=*/0,
                                     sizeof(MetaBlob), shm_ptr);
    task.Wait();
    const int rc = task->GetReturnCode();
    if (rc != 0) {
        HLOG(kError,
             "WriteDatasetMeta: PutBlob(__meta) FAILED (rc={}, 13 == bdev out of "
             "capacity); __meta not stored. This dataset will NOT be exportable "
             "to .h5 (the exporter needs it). Chunk data is unaffected.", rc);
        return false;
    }
    return true;
}

}  // namespace kvhdf5

#endif  // !CTP_IS_DEVICE_PASS
