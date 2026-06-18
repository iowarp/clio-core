/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

#ifndef BDEV_RUNTIME_H_
#define BDEV_RUNTIME_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/comutex.h>
#include "bdev_client.h"
#include "bdev_tasks.h"
#include <clio_runtime/bdev/transports/bdev_transport.h>
#include <vector>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>

namespace clio::run::bdev {

/**
 * Runtime container for bdev operations
 */
class Runtime : public chi::Container {
 public:
  // Required typedef for CLIO_TASK_CC macro
  using CreateParams = clio::run::bdev::CreateParams;
  
  Runtime() : bdev_type_(BdevType::kFile),
              total_reads_(0), total_writes_(0),
              total_bytes_read_(0), total_bytes_written_(0) {
    start_time_ = std::chrono::high_resolution_clock::now();
  }
  ~Runtime() override = default;

  /**
   * Get live task statistics for this task instance.
   */
  chi::TaskStat GetTaskStats(const chi::Task *task) const override;

  chi::TaskResume Create(ctp::ipc::FullPtr<CreateTask> task, chi::RunContext& ctx);
  chi::TaskResume AllocateBlocks(ctp::ipc::FullPtr<AllocateBlocksTask> task, chi::RunContext& ctx);
  chi::TaskResume FreeBlocks(ctp::ipc::FullPtr<FreeBlocksTask> task, chi::RunContext& ctx);
  chi::TaskResume Write(ctp::ipc::FullPtr<WriteTask> task, chi::RunContext& ctx);
  chi::TaskResume Read(ctp::ipc::FullPtr<ReadTask> task, chi::RunContext& ctx);
  chi::TaskResume GetStats(ctp::ipc::FullPtr<GetStatsTask> task, chi::RunContext& ctx);
  chi::TaskResume Update(ctp::ipc::FullPtr<UpdateTask> task, chi::RunContext& ctx);
  chi::TaskResume Monitor(ctp::ipc::FullPtr<MonitorTask> task, chi::RunContext &rctx);
  chi::TaskResume Destroy(ctp::ipc::FullPtr<DestroyTask> task, chi::RunContext& ctx);

  void Init(const chi::PoolId &pool_id, const std::string &pool_name,
            chi::u32 container_id = 0) override;
  void PostGpuContainerCreate() override;
  chi::TaskResume Run(chi::u32 method, ctp::ipc::FullPtr<chi::Task> task_ptr,
                      chi::RunContext& rctx) override;
  chi::u64 GetWorkRemaining() const override;

  void SaveTask(chi::u32 method, chi::SaveTaskArchive& archive,
                ctp::ipc::FullPtr<chi::Task> task_ptr) override;
  void LoadTask(chi::u32 method, chi::LoadTaskArchive& archive,
                ctp::ipc::FullPtr<chi::Task> task_ptr) override;
  ctp::ipc::FullPtr<chi::Task> AllocLoadTask(chi::u32 method, chi::LoadTaskArchive& archive) override;

  void LocalLoadTask(chi::u32 method, chi::DefaultLoadArchive& archive,
                     ctp::ipc::FullPtr<chi::Task> task_ptr) override;
  ctp::ipc::FullPtr<chi::Task> LocalAllocLoadTask(chi::u32 method, chi::DefaultLoadArchive& archive) override;
  void LocalSaveTask(chi::u32 method, chi::DefaultSaveArchive& archive,
                     ctp::ipc::FullPtr<chi::Task> task_ptr) override;

  ctp::ipc::FullPtr<chi::Task> NewCopyTask(chi::u32 method, ctp::ipc::FullPtr<chi::Task> orig_task_ptr,
                                        bool deep) override;
  ctp::ipc::FullPtr<chi::Task> NewTask(chi::u32 method) override;
  void Aggregate(chi::u32 method, ctp::ipc::FullPtr<chi::Task> orig_task,
                 const ctp::ipc::FullPtr<chi::Task>& replica_task) override;
  void DelTask(chi::u32 method, ctp::ipc::FullPtr<chi::Task> task_ptr) override;

 private:
  Client client_;
  BdevType bdev_type_;                            

  std::unique_ptr<BdevTransport> transport_;

  std::atomic<chi::u64> total_reads_;
  std::atomic<chi::u64> total_writes_;
  std::atomic<chi::u64> total_bytes_read_;
  std::atomic<chi::u64> total_bytes_written_;
  std::chrono::high_resolution_clock::time_point start_time_;
  
  PerfMetrics perf_metrics_;

  size_t GetWorkerID(chi::RunContext& ctx);

  void UpdatePerformanceMetrics(bool is_write, chi::u64 bytes,
                                double duration_us);
};

} // namespace clio::run::bdev

#endif // BDEV_RUNTIME_H_