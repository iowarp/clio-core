/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

#include <clio_runtime/bdev/transports/gcs_bdev_transport.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/worker.h>
#include <clio_runtime/work_orchestrator.h>

#ifdef CLIO_ENABLE_GOOGLE_CLOUD
#include <cstring>
#include <string>
#include <vector>
#endif

namespace clio::run::bdev {

#ifdef CLIO_ENABLE_GOOGLE_CLOUD
namespace {
// Parse a bdev pool name into (bucket, prefix). Accepts a bare bucket
// ("clio-bdev"), an optional gcs:// scheme ("gcs://clio-bdev/pool_0"), and an
// optional key prefix after the first slash. Trailing slashes are stripped.
void ParsePoolName(const std::string &pool_name, std::string &bucket,
                   std::string &prefix) {
  std::string s = pool_name;
  const std::string scheme = "gcs://";
  if (s.rfind(scheme, 0) == 0) {
    s = s.substr(scheme.size());
  }
  size_t slash = s.find('/');
  if (slash == std::string::npos) {
    bucket = s;
    prefix.clear();
  } else {
    bucket = s.substr(0, slash);
    prefix = s.substr(slash + 1);
  }
  while (!prefix.empty() && prefix.back() == '/') {
    prefix.pop_back();
  }
}
}  // namespace
#endif

bool GcsBdevTransport::Init(const CreateParams& params,
                            const std::string& pool_name, Runtime* runtime) {
#ifdef CLIO_ENABLE_GOOGLE_CLOUD
  std::string bucket, prefix;
  ParsePoolName(pool_name, bucket, prefix);
  if (bucket.empty()) {
    HLOG(kError, "GCS bdev: could not parse a bucket from pool_name '{}'",
         pool_name.c_str());
    return false;
  }

  gcs::GcsConfig config = gcs::GcsRestClient::ConfigFromEnv(bucket, prefix);
  if (config.token.empty()) {
    HLOG(kError,
         "GCS bdev: GCS_ACCESS_TOKEN is not set. Export it from "
         "`gcloud auth print-access-token` before creating a gcs pool.");
    return false;
  }
  client_ = std::make_unique<gcs::GcsRestClient>(config);

  gcs::GcsResult ensured = client_->EnsureBucket();
  if (!ensured.error.empty()) {
    HLOG(kError, "GCS bdev: {}", ensured.error.c_str());
    client_.reset();
    return false;
  }

  gcs_capacity_ = (params.total_size_ == 0) ? (1ULL << 40) : params.total_size_; // Default 1TB

  clio::run::WorkOrchestrator *work_orchestrator = CLIO_WORK_ORCHESTRATOR;
  size_t num_workers = work_orchestrator ? work_orchestrator->GetWorkerCount() : 16;
  allocator_.Init(num_workers, gcs_capacity_, params.alignment_);

  HLOG(kInfo,
       "GCS bdev ready: bucket='{}' prefix='{}' endpoint='{}' capacity={}",
       config.bucket.c_str(), config.prefix.c_str(), config.endpoint.c_str(),
       gcs_capacity_);
  return true;
#else
  HLOG(kError, "CLIO_ENABLE_GOOGLE_CLOUD is not defined. Cannot use GCS bdev.");
  return false;
#endif
}

void GcsBdevTransport::Destroy() {
#ifdef CLIO_ENABLE_GOOGLE_CLOUD
  client_.reset();
#endif
}

bool GcsBdevTransport::AllocateBlocks(size_t size, int worker_id, std::vector<Block>& blocks) {
  return allocator_.AllocateBlocks(size, worker_id, blocks);
}

void GcsBdevTransport::FreeBlocks(int worker_id, const std::vector<Block>& blocks) {
#ifdef CLIO_ENABLE_GOOGLE_CLOUD
  if (client_) {
    for (const auto& block : blocks) {
      client_->DeleteObject(client_->KeyForOffset(block.offset_)); // best effort
    }
  }
#endif
  allocator_.FreeBlocks(worker_id, blocks);
}

clio::run::TaskResume GcsBdevTransport::WriteBlocks(ctp::ipc::FullPtr<WriteTask> task, clio::run::RunContext &ctx) {
  clio::run::RunContext& rctx = ctx;
  CLIO_TASK_BODY_BEGIN

#ifdef CLIO_ENABLE_GOOGLE_CLOUD
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

    gcs::GcsResult res = client_->PutObject(
        client_->KeyForOffset(block.offset_), block_data,
        static_cast<size_t>(block_write_size));
    if (!res.error.empty()) {
      HLOG(kError, "GCS PutObject failed: {}", res.error.c_str());
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

clio::run::TaskResume GcsBdevTransport::ReadBlocks(ctp::ipc::FullPtr<ReadTask> task, clio::run::RunContext &ctx) {
  clio::run::RunContext& rctx = ctx;
  CLIO_TASK_BODY_BEGIN

#ifdef CLIO_ENABLE_GOOGLE_CLOUD
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

    size_t got = 0;
    gcs::GcsResult res = client_->GetObject(
        client_->KeyForOffset(block.offset_), block_data,
        static_cast<size_t>(block_read_size), &got);

    if (res.not_found) {
      // Sparse block: object was never written -> read back as zeros.
      std::memset(block_data, 0, static_cast<size_t>(block_read_size));
    } else if (!res.error.empty()) {
      HLOG(kError, "GCS GetObject failed: {}", res.error.c_str());
      task->return_code_ = 2;
      task->bytes_read_ = total_bytes_read;
      CLIO_CO_RETURN;
    } else if (got < block_read_size) {
      // Short object: zero-fill the unwritten tail.
      std::memset(block_data + got, 0,
                  static_cast<size_t>(block_read_size) - got);
    }

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
