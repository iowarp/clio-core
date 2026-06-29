/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

#ifndef CLIO_BDEV_S3_TRANSPORT_H_
#define CLIO_BDEV_S3_TRANSPORT_H_

#include <clio_runtime/bdev/transports/bdev_transport.h>
#include <clio_runtime/bdev/transports/block_allocator.h>

#ifdef CLIO_ENABLE_AMAZON_DRIVE
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#endif

namespace clio::run::bdev {

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
  Aws::SDKOptions options_;
  std::unique_ptr<Aws::S3::S3Client> s3_client_;
  Aws::String bucket_name_;
  static std::atomic<int> init_count_;
#endif
};

} // namespace clio::run::bdev

#endif // CLIO_BDEV_S3_TRANSPORT_H_
