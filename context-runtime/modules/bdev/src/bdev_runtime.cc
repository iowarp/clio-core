/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

#include <clio_runtime/bdev/bdev_runtime.h>
#include <clio_runtime/comutex.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_runtime/work_orchestrator.h>
#include <clio_runtime/worker.h>
#include <clio_ctp/introspect/system_info.h>
#include <clio_ctp/serialize/msgpack_wrapper.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>

#include "clio_ctp/util/timer.h"

namespace clio::run::bdev {

chi::TaskStat Runtime::GetTaskStats(const chi::Task *task) const {
  if (!task) return chi::TaskStat();
  switch (task->method_) {
    case Method::kWrite: {
      auto *wt = static_cast<const WriteTask *>(task);
      chi::TaskStat stat;
      stat.io_size_ = wt->length_;
      size_t aligned = ((stat.io_size_ + 4095) / 4096) * 4096;
      stat.wall_time_ = static_cast<float>(aligned) / 500.0f;
      return stat;
    }
    case Method::kRead: {
      auto *rt = static_cast<const ReadTask *>(task);
      chi::TaskStat stat;
      stat.io_size_ = rt->length_;
      size_t aligned = ((stat.io_size_ + 4095) / 4096) * 4096;
      stat.wall_time_ = static_cast<float>(aligned) / 500.0f;
      return stat;
    }
    default: return chi::TaskStat();
  }
}

size_t Runtime::GetWorkerID(chi::RunContext &ctx) {
  chi::Worker *worker = CLIO_CUR_WORKER;
  if (worker == nullptr) {
    return 0;
  }
  return worker->GetId();
}

chi::TaskResume Runtime::Create(ctp::ipc::FullPtr<CreateTask> task, chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  
  CreateParams params = task->GetParams();
  bdev_type_ = params.bdev_type_;
  
  transport_ = BdevTransportFactory::Create(bdev_type_);
  if (!transport_) {
    if (bdev_type_ == BdevType::kNoop) {
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }
    HLOG(kError, "Failed to create bdev transport for type {}", static_cast<int>(bdev_type_));
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }

  // pool_name doubles as the file path (kFile) / S3 bucket (kS3); it lives on
  // the create task, not in CreateParams.
  if (!transport_->Init(params, task->pool_name_.str(), this)) {
    HLOG(kError, "Failed to initialize bdev transport");
    task->return_code_ = 2;
    CLIO_CO_RETURN;
  }

  perf_metrics_ = params.perf_metrics_;
  start_time_ = std::chrono::high_resolution_clock::now();
  total_reads_ = 0;
  total_writes_ = 0;
  total_bytes_read_ = 0;
  total_bytes_written_ = 0;

  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::AllocateBlocks(ctp::ipc::FullPtr<AllocateBlocksTask> task, chi::RunContext &ctx) {
  chi::RunContext& rctx = ctx;
  CLIO_TASK_BODY_BEGIN

  if (bdev_type_ == BdevType::kNoop) {
    task->return_code_ = 0;
    CLIO_CO_RETURN;
  }

  int worker_id = static_cast<int>(GetWorkerID(rctx));
  std::vector<Block> local_blocks;

  if (!transport_->AllocateBlocks(task->size_, worker_id, local_blocks)) {
    task->blocks_.clear();
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }

  for (const auto& block : local_blocks) {
    task->blocks_.push_back(block);
  }

  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::FreeBlocks(ctp::ipc::FullPtr<FreeBlocksTask> task, chi::RunContext &ctx) {
  chi::RunContext& rctx = ctx;
  CLIO_TASK_BODY_BEGIN

  if (bdev_type_ == BdevType::kNoop) {
    task->return_code_ = 0;
    CLIO_CO_RETURN;
  }

  int worker_id = static_cast<int>(GetWorkerID(rctx));
  std::vector<Block> local_blocks;
  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    local_blocks.push_back(task->blocks_[i]);
  }

  transport_->FreeBlocks(worker_id, local_blocks);

  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Write(ctp::ipc::FullPtr<WriteTask> task, chi::RunContext &ctx) {
  chi::RunContext& rctx = ctx;
  CLIO_TASK_BODY_BEGIN

  if (bdev_type_ == BdevType::kNoop) {
    task->return_code_ = 0;
    task->bytes_written_ = task->length_;
    CLIO_CO_RETURN;
  }

  if (transport_) {
    CLIO_CO_AWAIT(transport_->WriteBlocks(task, rctx));
    total_writes_.fetch_add(1);
    total_bytes_written_.fetch_add(task->bytes_written_);
  } else {
    task->return_code_ = 1;
    task->bytes_written_ = 0;
  }

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Read(ctp::ipc::FullPtr<ReadTask> task, chi::RunContext &ctx) {
  chi::RunContext& rctx = ctx;
  CLIO_TASK_BODY_BEGIN

  if (bdev_type_ == BdevType::kNoop) {
    task->return_code_ = 0;
    task->bytes_read_ = task->length_;
    CLIO_CO_RETURN;
  }

  if (transport_) {
    CLIO_CO_AWAIT(transport_->ReadBlocks(task, rctx));
    total_reads_.fetch_add(1);
    total_bytes_read_.fetch_add(task->bytes_read_);
  } else {
    task->return_code_ = 1;
    task->bytes_read_ = 0;
  }

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Update(ctp::ipc::FullPtr<UpdateTask> task, chi::RunContext &ctx) {
  task->return_code_ = 0;
  (void)ctx;
  co_return;
}

chi::TaskResume Runtime::GetStats(ctp::ipc::FullPtr<GetStatsTask> task, chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  ReadTask r_synthetic;
  r_synthetic.method_ = Method::kRead;
  r_synthetic.length_ = 1024 * 1024;
  WriteTask w_synthetic;
  w_synthetic.method_ = Method::kWrite;
  w_synthetic.length_ = 1024 * 1024;
  chi::TaskStat read_stat = GetTaskStats(&r_synthetic);
  chi::TaskStat write_stat = GetTaskStats(&w_synthetic);
  float read_wall_us = InferWallClockTime(Method::kRead, read_stat);
  float write_wall_us = InferWallClockTime(Method::kWrite, write_stat);
  double read_size_mb = static_cast<double>(read_stat.io_size_) / (1024.0 * 1024.0);
  double write_size_mb = static_cast<double>(write_stat.io_size_) / (1024.0 * 1024.0);
  task->metrics_.read_bandwidth_mbps_ = (read_wall_us > 0)
      ? read_size_mb / (read_wall_us * 1e-6) : perf_metrics_.read_bandwidth_mbps_;
  task->metrics_.write_bandwidth_mbps_ = (write_wall_us > 0)
      ? write_size_mb / (write_wall_us * 1e-6) : perf_metrics_.write_bandwidth_mbps_;
  task->metrics_.read_latency_us_ = read_wall_us;
  task->metrics_.write_latency_us_ = write_wall_us;
  task->metrics_.iops_ = perf_metrics_.iops_;

  if (transport_) {
    task->remaining_size_ = transport_->GetRemainingSize();
  } else {
    task->remaining_size_ = 0;
  }

  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Destroy(ctp::ipc::FullPtr<DestroyTask> task, chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  if (transport_) {
    transport_->Destroy();
    transport_.reset();
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::u64 Runtime::GetWorkRemaining() const { return 0; }

chi::TaskResume Runtime::Monitor(ctp::ipc::FullPtr<MonitorTask> task, chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  (void)rctx;
  if (task->query_ == "stats") {
    ReadTask r_synthetic;
    r_synthetic.method_ = Method::kRead;
    r_synthetic.length_ = 1024 * 1024;
    WriteTask w_synthetic;
    w_synthetic.method_ = Method::kWrite;
    w_synthetic.length_ = 1024 * 1024;
    chi::TaskStat read_stat = GetTaskStats(&r_synthetic);
    chi::TaskStat write_stat = GetTaskStats(&w_synthetic);
    float read_wall_us = InferWallClockTime(Method::kRead, read_stat);
    float write_wall_us = InferWallClockTime(Method::kWrite, write_stat);
    double read_size_mb = static_cast<double>(read_stat.io_size_) / (1024.0 * 1024.0);
    double write_size_mb = static_cast<double>(write_stat.io_size_) / (1024.0 * 1024.0);
    double read_bw = (read_wall_us > 0)
        ? read_size_mb / (read_wall_us * 1e-6) : perf_metrics_.read_bandwidth_mbps_;
    double write_bw = (write_wall_us > 0)
        ? write_size_mb / (write_wall_us * 1e-6) : perf_metrics_.write_bandwidth_mbps_;

    msgpack::sbuffer sbuf;
    msgpack::packer<msgpack::sbuffer> pk(sbuf);

    pk.pack_map(13);
    pk.pack("pool_name");              pk.pack(pool_name_);
    pk.pack("bdev_type");              pk.pack(static_cast<chi::u32>(bdev_type_));
    pk.pack("total_capacity");         pk.pack(transport_ ? transport_->GetCapacity() : 0);
    pk.pack("remaining_capacity");     pk.pack(transport_ ? transport_->GetRemainingSize() : 0);
    pk.pack("read_bandwidth_mbps");    pk.pack(read_bw);
    pk.pack("write_bandwidth_mbps");   pk.pack(write_bw);
    pk.pack("read_latency_us");        pk.pack(static_cast<double>(read_wall_us));
    pk.pack("write_latency_us");       pk.pack(static_cast<double>(write_wall_us));
    pk.pack("iops");                   pk.pack(perf_metrics_.iops_);
    pk.pack("total_reads");            pk.pack(total_reads_.load());
    pk.pack("total_writes");           pk.pack(total_writes_.load());
    pk.pack("total_bytes_read");       pk.pack(total_bytes_read_.load());
    pk.pack("total_bytes_written");    pk.pack(total_bytes_written_.load());

    task->results_[container_id_] = std::string(sbuf.data(), sbuf.size());
  }
  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::PostGpuContainerCreate() {}

}  // namespace clio::run::bdev

// Define ChiMod entry points using CLIO_TASK_CC macro
CLIO_TASK_CC(clio::run::bdev::Runtime)