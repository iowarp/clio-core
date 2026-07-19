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
#include <string>
#include <thread>

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
  ~Runtime() override;

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
  clio::run::TaskResume SetLifespan(clio::run::shared_ptr<SetLifespanTask> &task);
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

 public:
  // ---------------------------------------------------------------------
  // Perf-stats persistence (issue #747).
  //
  // The bdev's real performance knowledge lives in two places: the
  // PerfMetrics snapshot (bandwidth/latency/iops) and the runtime's learned
  // per-method wall-clock model coefficients (method_model_wall_[kRead/
  // kWrite]) that GetStats derives its bandwidth numbers from. Both are
  // saved to a small per-bdev stats file so a restarted session starts from
  // the previous session's estimates instead of re-learning from the 1.0
  // model seed. Files live under $CLIO_BDEV_STATS_DIR (default
  // <home>/.clio/bdev_perf), keyed by the sanitized pool name.
  //
  // The file helpers are static so they can be unit-tested without a
  // running container.
  // ---------------------------------------------------------------------

  /** Resolve the stats-file path for a bdev pool name. */
  static std::string MakePerfStatsPath(const std::string &pool_name);

  /**
   * Write a perf-stats file (small text format, tmp+rename).
   * @param path Destination file
   * @param metrics Perf metrics snapshot
   * @param model_wall_read Learned wall-model coefficient for kRead
   * @param model_wall_write Learned wall-model coefficient for kWrite
   * @return true on success
   */
  static bool SavePerfStatsFile(const std::string &path,
                                const PerfMetrics &metrics,
                                float model_wall_read, float model_wall_write);

  /**
   * Read a perf-stats file previously written by SavePerfStatsFile.
   * @param path Source file
   * @param metrics Output metrics (only overwritten on success)
   * @param model_wall_read Output kRead wall coefficient
   * @param model_wall_write Output kWrite wall coefficient
   * @return true if the file existed and parsed
   */
  static bool LoadPerfStatsFile(const std::string &path, PerfMetrics &metrics,
                                float &model_wall_read,
                                float &model_wall_write);

 private:
  /** Load persisted stats into perf_metrics_ + the wall model (Create). */
  void LoadPerfStats();

  /** Persist current stats, throttled to one write per ~10s (GetStats). */
  void SavePerfStats(bool force);

  Client client_;
  BdevType bdev_type_;

  std::unique_ptr<BdevTransport> transport_;

  std::atomic<clio::run::u64> total_reads_;
  std::atomic<clio::run::u64> total_writes_;
  std::atomic<clio::run::u64> total_bytes_read_;
  std::atomic<clio::run::u64> total_bytes_written_;
  std::chrono::high_resolution_clock::time_point start_time_;

  PerfMetrics perf_metrics_;

  // Predictive device health: estimated days until drive failure.
  // Populated by a background thread that periodically queries the
  // local prediction server (server.py). Default 999999 = healthy.
  std::atomic<clio::run::u32> predicted_ttl_days_{999999};

  // Path to the device file this bdev owns (set in Init, used for SMART reads)
  std::string device_path_;

  // Background health-poll thread and its stop flag.
  std::thread health_poll_thread_;
  std::atomic<bool> health_poll_stop_{false};

  // How often (seconds) to re-poll the prediction server.
  static constexpr unsigned kHealthPollIntervalSec = 300;  // every 5 minutes

  // URL of the local prediction server.  Override via CLIO_PRED_SERVER env var.
  static constexpr const char *kDefaultPredServerUrl =
      "http://127.0.0.1:8000/predict/auto";

  /**
   * Reads SMART attributes from the OS for device_path_ and posts them to the
   * local prediction server.  Runs on health_poll_thread_.
   */
  void PollPredictionServer();

  /**
   * Stop the background health poll thread and destroy the bdev transport.
   *
   * This is shared by Destroy() and the destructor so shutdown is safe even if
   * the container is torn down without the explicit destroy task.
   */
  void StopHealthPolling();

  // Perf-stats persistence state (issue #747)
  std::string perf_stats_path_;
  std::chrono::steady_clock::time_point last_perf_save_{};
  static constexpr double kPerfSaveThrottleSec = 10.0;

  size_t GetWorkerID(clio::run::RunContext& rctx);

  void UpdatePerformanceMetrics(bool is_write, clio::run::u64 bytes,
                                double duration_us);
};

} // namespace clio::run::bdev

#endif // BDEV_RUNTIME_H_
