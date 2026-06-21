/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

#include <clio_runtime/bdev/transports/s3_bdev_transport.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/worker.h>
#include <clio_runtime/work_orchestrator.h>

#ifdef CLIO_ENABLE_AMAZON_DRIVE
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/DeleteObjectRequest.h>
#include <aws/core/utils/memory/stl/AWSStreamFwd.h>
#include <aws/core/utils/stream/PreallocatedStreamBuf.h>
#include <aws/core/utils/StringUtils.h>
#include <iostream>
#endif

namespace clio::run::bdev {

#ifdef CLIO_ENABLE_AMAZON_DRIVE
std::atomic<int> S3BdevTransport::init_count_{0};

class PreallocatedStreamBuf : public std::streambuf {
 public:
  PreallocatedStreamBuf(char* base, size_t length) {
    setp(base, base + length);
    setg(base, base, base + length);
  }
};
#endif

bool S3BdevTransport::Init(const CreateParams& params,
                           const std::string& pool_name, Runtime* runtime) {
#ifdef CLIO_ENABLE_AMAZON_DRIVE
  if (init_count_.fetch_add(1) == 0) {
    Aws::InitAPI(options_);
  }

  // The bdev pool_name acts as the bucket name
  bucket_name_ = Aws::String(pool_name.c_str());

  Aws::Client::ClientConfiguration clientConfig;
  const char* region_env = std::getenv("AWS_DEFAULT_REGION");
  if (region_env) {
    clientConfig.region = region_env;
  }
  s3_client_ = std::make_unique<Aws::S3::S3Client>(clientConfig);

  s3_capacity_ = (params.total_size_ == 0) ? (1ULL << 40) : params.total_size_; // Default to 1TB

  clio::run::WorkOrchestrator *work_orchestrator = CLIO_WORK_ORCHESTRATOR;
  size_t num_workers = work_orchestrator ? work_orchestrator->GetWorkerCount() : 16;
  allocator_.Init(num_workers, s3_capacity_, params.alignment_);

  return true;
#else
  HLOG(kError, "CLIO_ENABLE_AMAZON_DRIVE is not defined. Cannot use S3 bdev.");
  return false;
#endif
}

void S3BdevTransport::Destroy() {
#ifdef CLIO_ENABLE_AMAZON_DRIVE
  s3_client_.reset();
  if (init_count_.fetch_sub(1) == 1) {
    Aws::ShutdownAPI(options_);
  }
#endif
}

bool S3BdevTransport::AllocateBlocks(size_t size, int worker_id, std::vector<Block>& blocks) {
  return allocator_.AllocateBlocks(size, worker_id, blocks);
}

void S3BdevTransport::FreeBlocks(int worker_id, const std::vector<Block>& blocks) {
#ifdef CLIO_ENABLE_AMAZON_DRIVE
  for (const auto& block : blocks) {
    Aws::S3::Model::DeleteObjectRequest request;
    request.SetBucket(bucket_name_);
    request.SetKey("block_" + std::to_string(block.offset_));
    s3_client_->DeleteObject(request); // Best effort, ignore errors
  }
#endif
  allocator_.FreeBlocks(worker_id, blocks);
}

clio::run::TaskResume S3BdevTransport::WriteBlocks(ctp::ipc::FullPtr<WriteTask> task, clio::run::RunContext &ctx) {
  clio::run::RunContext& rctx = ctx;
  CLIO_TASK_BODY_BEGIN

#ifdef CLIO_ENABLE_AMAZON_DRIVE
  auto *ipc_mgr = CLIO_IPC;
  ctp::ipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).Cast<char>();

  bool data_on_device = ctp::IsDevicePointer(data_ptr.ptr_);
  std::vector<char> staging;
  if (data_on_device) {
    staging.resize(task->length_);
    ctp::DeviceAwareMemcpy(staging.data(), data_ptr.ptr_, task->length_);
  }

  clio::run::u64 total_bytes_written = 0;
  clio::run::u64 data_offset = 0;

  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];
    clio::run::u64 remaining = task->length_ - total_bytes_written;
    if (remaining == 0) break;
    clio::run::u64 block_write_size = std::min(remaining, block.size_);

    char *block_data = data_on_device
                           ? staging.data() + data_offset
                           : data_ptr.ptr_ + data_offset;

    Aws::S3::Model::PutObjectRequest request;
    request.SetBucket(bucket_name_);
    request.SetKey("block_" + std::to_string(block.offset_));

    // Create a memory stream from the buffer
    auto stream_buf = std::make_shared<PreallocatedStreamBuf>(block_data, static_cast<size_t>(block_write_size));
    auto input_data = std::make_shared<std::iostream>(stream_buf.get());
    request.SetBody(input_data);

    auto outcome = s3_client_->PutObject(request);
    if (!outcome.IsSuccess()) {
      HLOG(kError, "PutObject failed: {}", outcome.GetError().GetMessage().c_str());
      task->return_code_ = 2;
      task->bytes_written_ = total_bytes_written;
      CLIO_CO_RETURN;
    }

    total_bytes_written += block_write_size;
    data_offset += block_write_size;
  }

  task->return_code_ = 0;
  task->bytes_written_ = total_bytes_written;
#else
  task->return_code_ = 1;
  task->bytes_written_ = 0;
#endif

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume S3BdevTransport::ReadBlocks(ctp::ipc::FullPtr<ReadTask> task, clio::run::RunContext &ctx) {
  clio::run::RunContext& rctx = ctx;
  CLIO_TASK_BODY_BEGIN

#ifdef CLIO_ENABLE_AMAZON_DRIVE
  auto *ipc_mgr = CLIO_IPC;
  ctp::ipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).Cast<char>();

  bool data_on_device = ctp::IsDevicePointer(data_ptr.ptr_);
  std::vector<char> staging;
  if (data_on_device) {
    staging.resize(task->length_);
  }

  clio::run::u64 total_bytes_read = 0;
  clio::run::u64 data_offset = 0;

  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];
    clio::run::u64 remaining = task->length_ - total_bytes_read;
    if (remaining == 0) break;
    clio::run::u64 block_read_size = std::min(remaining, block.size_);

    char *block_data = data_on_device
                           ? staging.data() + data_offset
                           : data_ptr.ptr_ + data_offset;

    Aws::S3::Model::GetObjectRequest request;
    request.SetBucket(bucket_name_);
    request.SetKey("block_" + std::to_string(block.offset_));

    auto outcome = s3_client_->GetObject(request);
    if (!outcome.IsSuccess()) {
      HLOG(kError, "GetObject failed: {}", outcome.GetError().GetMessage().c_str());
      task->return_code_ = 2;
      task->bytes_read_ = total_bytes_read;
      CLIO_CO_RETURN;
    }

    auto& body = outcome.GetResult().GetBody();
    body.read(block_data, block_read_size);

    total_bytes_read += block_read_size;
    data_offset += block_read_size;
  }

  if (data_on_device && total_bytes_read > 0) {
    ctp::DeviceAwareMemcpy(data_ptr.ptr_, staging.data(), total_bytes_read);
  }

  task->return_code_ = 0;
  task->bytes_read_ = total_bytes_read;
#else
  task->return_code_ = 1;
  task->bytes_read_ = 0;
#endif

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

} // namespace clio::run::bdev
