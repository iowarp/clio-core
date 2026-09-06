/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

#ifndef CLIO_BDEV_S3_TRANSPORT_H_
#define CLIO_BDEV_S3_TRANSPORT_H_

#include <clio_runtime/bdev/transports/bdev_transport.h>
#include <clio_runtime/bdev/transports/block_allocator.h>

#include <cstddef>
#include <memory>
#include <vector>

#ifdef CLIO_ENABLE_AMAZON_DRIVE
#include <clio_runtime/bdev/transports/s3_rest.h>
#endif

namespace clio::run::bdev {

/**
 * Amazon S3 block-device transport. Each block is stored as an S3 object (key
 * `[prefix/]block_<offset>`); a sparse (never-written) block reads back as
 * zeros via the 404 -> zero-fill convention. Mirrors GcsBdevTransport, with
 * SigV4 request signing in place of GCS's bearer token.
 *
 * Deliberately holds no AWS SDK object: linking aws-cpp-sdk-core into a process
 * that calls CLIO_INIT corrupts runtime init, and unlike the CAE assimilator
 * this code cannot fork+exec its way out at per-block granularity. See
 * s3_rest.h.
 */
class S3BdevTransport : public BdevTransport {
 public:
  S3BdevTransport() = default;
  ~S3BdevTransport() override { Destroy(); }

  bool Init(const CreateParams& params, const std::string& pool_name,
            Runtime* runtime) override;
  void Destroy() override;

  bool AllocateBlocks(size_t size, int worker_id, std::vector<Block>& blocks) override;
  void FreeBlocks(int worker_id, const std::vector<Block>& blocks) override;

  clio::run::TaskResume WriteBlocks(ctp::ipc::FullPtr<WriteTask> task) override;
  clio::run::TaskResume ReadBlocks(ctp::ipc::FullPtr<ReadTask> task) override;

  clio::run::u64 GetCapacity() const override { return allocator_.GetCapacity(); }
  clio::run::u64 GetRemainingSize() const override { return allocator_.GetRemainingSize(); }

 private:
  StandardBlockAllocator allocator_;
  clio::run::u64 s3_capacity_{0};

#ifdef CLIO_ENABLE_AMAZON_DRIVE
  std::unique_ptr<s3::S3RestClient> client_;

  // One reusable (keep-alive) HTTP connection per worker, indexed by worker id.
  // Lock-free like FsBdevTransport::io_contexts_: S3 ops run synchronously in a
  // worker's task body, so a worker only ever touches its own slot. Sized with
  // ElasticHeadroom so ids handed to workers spawned after Init still land in
  // range. Default-constructed slots are lazy (null session -> connect on first
  // use), so no eager init loop is needed.
  std::vector<s3::S3Connection> conns_;

  /** The connection slot for `worker_id`, or nullptr if out of range. */
  s3::S3Connection* GetWorkerConnection(size_t worker_id);
#endif
};

} // namespace clio::run::bdev

#endif // CLIO_BDEV_S3_TRANSPORT_H_
