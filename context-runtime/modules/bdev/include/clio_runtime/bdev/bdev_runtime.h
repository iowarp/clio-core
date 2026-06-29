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
class Runtime : public clio::run::Container {
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
  clio::run::TaskStat GetTaskStats(const clio::run::Task *task) const override;

  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask> &task);
  clio::run::TaskResume AllocateBlocks(clio::run::shared_ptr<AllocateBlocksTask> &task);
  clio::run::TaskResume FreeBlocks(clio::run::shared_ptr<FreeBlocksTask> &task);
  clio::run::TaskResume Write(clio::run::shared_ptr<WriteTask> &task);
  clio::run::TaskResume Read(clio::run::shared_ptr<ReadTask> &task);
  clio::run::TaskResume GetStats(clio::run::shared_ptr<GetStatsTask> &task);
  clio::run::TaskResume Update(clio::run::shared_ptr<UpdateTask> &task);
  clio::run::TaskResume Monitor(clio::run::shared_ptr<MonitorTask> &task);
  clio::run::TaskResume Destroy(clio::run::shared_ptr<DestroyTask> &task);

  void Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
            clio::run::u32 container_id = 0) override;
  void PostGpuContainerCreate() override;
  clio::run::TaskResume Run(clio::run::u32 method,
                      clio::run::shared_ptr<clio::run::Task> task_ptr) override;
  clio::run::u64 GetWorkRemaining() const override;

  void SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive& archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  void LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive& archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> AllocLoadTask(clio::run::u32 method, clio::run::LoadTaskArchive& archive) override;

  void LocalLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive& archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> LocalAllocLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive& archive) override;
  void LocalSaveTask(clio::run::u32 method, clio::run::DefaultSaveArchive& archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;

  clio::run::shared_ptr<clio::run::Task> NewCopyTask(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task_ptr,
                                        bool deep) override;
  clio::run::shared_ptr<clio::run::Task> NewTask(clio::run::u32 method) override;
  void AggregateOut(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task,
                 const clio::run::shared_ptr<clio::run::Task>& replica_task) override;

 private:
  Client client_;
  BdevType bdev_type_;                            

  std::unique_ptr<BdevTransport> transport_;

  std::atomic<clio::run::u64> total_reads_;
  std::atomic<clio::run::u64> total_writes_;
  std::atomic<clio::run::u64> total_bytes_read_;
  std::atomic<clio::run::u64> total_bytes_written_;
  std::chrono::high_resolution_clock::time_point start_time_;
  
  PerfMetrics perf_metrics_;

  size_t GetWorkerID(clio::run::RunContext& rctx);

  void UpdatePerformanceMetrics(bool is_write, clio::run::u64 bytes,
                                double duration_us);
};

} // namespace clio::run::bdev

#endif // BDEV_RUNTIME_H_