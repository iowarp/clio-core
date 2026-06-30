/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

#ifndef CLIO_BDEV_GCS_TRANSPORT_H_
#define CLIO_BDEV_GCS_TRANSPORT_H_

#include <clio_runtime/bdev/transports/bdev_transport.h>
#include <clio_runtime/bdev/transports/block_allocator.h>

#include <memory>

#ifdef CLIO_ENABLE_GOOGLE_CLOUD
#include <clio_runtime/bdev/transports/gcs_rest.h>
#endif

namespace clio::run::bdev {

/**
 * Google Cloud Storage block-device transport. Each block is stored as a GCS
 * object (key `[prefix/]block_<offset>`); a sparse (never-written) block reads
 * back as zeros via the 404 -> zero-fill convention. Mirrors S3BdevTransport but
 * uses a Poco-based REST client and bearer-token auth (GCS_ACCESS_TOKEN).
 */
class GcsBdevTransport : public BdevTransport {
 public:
  GcsBdevTransport() = default;
  ~GcsBdevTransport() override { Destroy(); }

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
  clio::run::u64 gcs_capacity_{0};

#ifdef CLIO_ENABLE_GOOGLE_CLOUD
  std::unique_ptr<gcs::GcsRestClient> client_;
#endif
};

} // namespace clio::run::bdev

#endif // CLIO_BDEV_GCS_TRANSPORT_H_
