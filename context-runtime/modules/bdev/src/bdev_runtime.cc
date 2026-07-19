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
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <thread>

#include "clio_ctp/util/timer.h"

namespace clio::run::bdev {

Runtime::~Runtime() { StopHealthPolling(); }

// ---------------------------------------------------------------------------
// Perf-stats persistence (issue #747)
// ---------------------------------------------------------------------------

namespace {
constexpr const char *kPerfStatsHeader = "clio_bdev_perf_v1";
}  // namespace

std::string Runtime::MakePerfStatsPath(const std::string &pool_name) {
  // Master off-switch: CLIO_BDEV_PERSIST_STATS=0 disables persistence
  // entirely (e.g. benchmarks that must start from a cold model).
  if (ctp::SystemInfo::Getenv("CLIO_BDEV_PERSIST_STATS") == "0") {
    return std::string();
  }
  std::string dir = ctp::SystemInfo::Getenv("CLIO_BDEV_STATS_DIR");
  if (dir.empty()) {
    std::string home = ctp::SystemInfo::GetHomeDir();
    if (home.empty()) {
      return std::string();  // nowhere to persist
    }
    dir = home + "/.clio/bdev_perf";
  }
  // Sanitize the pool name (it may be a filesystem path or "ram::name")
  // into a flat file name.
  std::string name = pool_name;
  for (char &c : name) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '.' && c != '-') {
      c = '_';
    }
  }
  return dir + "/" + name + ".perf";
}

bool Runtime::SavePerfStatsFile(const std::string &path,
                                const PerfMetrics &metrics,
                                float model_wall_read,
                                float model_wall_write) {
  if (path.empty()) return false;
  try {
    namespace fs = std::filesystem;
    fs::create_directories(fs::path(path).parent_path());
    // Write to a tmp file then rename so a crash mid-write never leaves a
    // truncated stats file for the next session to trip over.
    const std::string tmp = path + ".tmp";
    {
      std::ofstream ofs(tmp, std::ios::trunc);
      if (!ofs.is_open()) return false;
      ofs << kPerfStatsHeader << "\n"
          << "read_bandwidth_mbps " << metrics.read_bandwidth_mbps_ << "\n"
          << "write_bandwidth_mbps " << metrics.write_bandwidth_mbps_ << "\n"
          << "read_latency_us " << metrics.read_latency_us_ << "\n"
          << "write_latency_us " << metrics.write_latency_us_ << "\n"
          << "iops " << metrics.iops_ << "\n"
          << "model_wall_read " << model_wall_read << "\n"
          << "model_wall_write " << model_wall_write << "\n";
      if (!ofs.good()) return false;
    }
    std::error_code ec;
    fs::remove(path, ec);  // Windows rename does not overwrite
    fs::rename(tmp, path, ec);
    return !ec;
  } catch (const std::exception &) {
    return false;
  }
}

bool Runtime::LoadPerfStatsFile(const std::string &path, PerfMetrics &metrics,
                                float &model_wall_read,
                                float &model_wall_write) {
  if (path.empty()) return false;
  std::ifstream ifs(path);
  if (!ifs.is_open()) return false;
  std::string header;
  if (!std::getline(ifs, header) || header != kPerfStatsHeader) {
    return false;
  }
  PerfMetrics m;
  float wall_read = 0.0f;
  float wall_write = 0.0f;
  std::string key;
  double value = 0.0;
  while (ifs >> key >> value) {
    if (key == "read_bandwidth_mbps") m.read_bandwidth_mbps_ = value;
    else if (key == "write_bandwidth_mbps") m.write_bandwidth_mbps_ = value;
    else if (key == "read_latency_us") m.read_latency_us_ = value;
    else if (key == "write_latency_us") m.write_latency_us_ = value;
    else if (key == "iops") m.iops_ = value;
    else if (key == "model_wall_read") wall_read = static_cast<float>(value);
    else if (key == "model_wall_write") wall_write = static_cast<float>(value);
    // Unknown keys are ignored (forward compatibility).
  }
  metrics = m;
  model_wall_read = wall_read;
  model_wall_write = wall_write;
  return true;
}

void Runtime::LoadPerfStats() {
  perf_stats_path_ = MakePerfStatsPath(pool_name_);
  PerfMetrics m;
  float wall_read = 0.0f;
  float wall_write = 0.0f;
  if (!LoadPerfStatsFile(perf_stats_path_, m, wall_read, wall_write)) {
    return;  // first session for this bdev — keep configured defaults
  }
  perf_metrics_ = m;
  // Seed the learned wall-clock model so InferWallClockTime (which GetStats
  // derives its bandwidth from) starts warm instead of at the 1.0 seed.
  if (wall_read > 0.0f && Method::kRead < method_model_wall_.size()) {
    method_model_wall_[Method::kRead] = wall_read;
  }
  if (wall_write > 0.0f && Method::kWrite < method_model_wall_.size()) {
    method_model_wall_[Method::kWrite] = wall_write;
  }
  HLOG(kInfo,
       "bdev '{}': restored perf stats from {} (read {} MB/s, write {} MB/s)",
       pool_name_, perf_stats_path_, perf_metrics_.read_bandwidth_mbps_,
       perf_metrics_.write_bandwidth_mbps_);
}

void Runtime::SavePerfStats(bool force) {
  if (perf_stats_path_.empty()) return;
  auto now = std::chrono::steady_clock::now();
  if (!force &&
      std::chrono::duration<double>(now - last_perf_save_).count() <
          kPerfSaveThrottleSec) {
    return;
  }
  last_perf_save_ = now;
  float wall_read = (Method::kRead < method_model_wall_.size())
                        ? method_model_wall_[Method::kRead] : 0.0f;
  float wall_write = (Method::kWrite < method_model_wall_.size())
                         ? method_model_wall_[Method::kWrite] : 0.0f;
  SavePerfStatsFile(perf_stats_path_, perf_metrics_, wall_read, wall_write);
}

clio::run::TaskStat Runtime::GetTaskStats(const clio::run::Task *task) const {
  if (!task) return clio::run::TaskStat();
  switch (task->method_) {
    case Method::kWrite: {
      auto *wt = static_cast<const WriteTask *>(task);
      clio::run::TaskStat stat;
      stat.io_size_ = wt->length_;
      size_t aligned = ((stat.io_size_ + 4095) / 4096) * 4096;
      stat.wall_time_ = static_cast<float>(aligned) / 500.0f;
      return stat;
    }
    case Method::kRead: {
      auto *rt = static_cast<const ReadTask *>(task);
      clio::run::TaskStat stat;
      stat.io_size_ = rt->length_;
      size_t aligned = ((stat.io_size_ + 4095) / 4096) * 4096;
      stat.wall_time_ = static_cast<float>(aligned) / 500.0f;
      return stat;
    }
    default: return clio::run::TaskStat();
  }
}

size_t Runtime::GetWorkerID(clio::run::RunContext &rctx) {
  clio::run::Worker *worker = CLIO_CUR_WORKER;
  if (worker == nullptr) {
    return 0;
  }
  return worker->GetId();
}

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
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

  // Capture the device path so the poll thread can look up SMART attributes.
  device_path_ = task->pool_name_.str();

  // Start background health-poll thread.  It will periodically query the
  // local prediction server and update predicted_ttl_days_.
  health_poll_stop_.store(false, std::memory_order_relaxed);
  health_poll_thread_ = std::thread(&Runtime::PollPredictionServer, this);

  // Restore persisted performance stats from a previous session (issue
  // #747) so the latency-bandwidth model does not have to be re-estimated
  // on every startup. Overrides the configured defaults when a stats file
  // exists.
  LoadPerfStats();

  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::AllocateBlocks(clio::run::shared_ptr<AllocateBlocksTask> &task) {
  CLIO_TASK_BODY_BEGIN

  if (bdev_type_ == BdevType::kNoop) {
    task->return_code_ = 0;
    CLIO_CO_RETURN;
  }

  clio::run::Worker *worker = CLIO_CUR_WORKER;
  int worker_id = (worker != nullptr) ? static_cast<int>(worker->GetId()) : 0;
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

clio::run::TaskResume Runtime::FreeBlocks(clio::run::shared_ptr<FreeBlocksTask> &task) {
  CLIO_TASK_BODY_BEGIN

  if (bdev_type_ == BdevType::kNoop) {
    task->return_code_ = 0;
    CLIO_CO_RETURN;
  }

  clio::run::Worker *worker = CLIO_CUR_WORKER;
  int worker_id = (worker != nullptr) ? static_cast<int>(worker->GetId()) : 0;
  std::vector<Block> local_blocks;
  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    local_blocks.push_back(task->blocks_[i]);
  }

  transport_->FreeBlocks(worker_id, local_blocks);

  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Write(clio::run::shared_ptr<WriteTask> &task) {
  CLIO_TASK_BODY_BEGIN

  if (bdev_type_ == BdevType::kNoop) {
    task->return_code_ = 0;
    task->bytes_written_ = task->length_;
    CLIO_CO_RETURN;
  }

  if (transport_) {
    CLIO_CO_AWAIT(transport_->WriteBlocks(ctp::ipc::FullPtr<WriteTask>(task.get())));
    total_writes_.fetch_add(1);
    total_bytes_written_.fetch_add(task->bytes_written_);
  } else {
    task->return_code_ = 1;
    task->bytes_written_ = 0;
  }

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Read(clio::run::shared_ptr<ReadTask> &task) {
  CLIO_TASK_BODY_BEGIN

  if (bdev_type_ == BdevType::kNoop) {
    task->return_code_ = 0;
    task->bytes_read_ = task->length_;
    CLIO_CO_RETURN;
  }

  if (transport_) {
    CLIO_CO_AWAIT(transport_->ReadBlocks(ctp::ipc::FullPtr<ReadTask>(task.get())));
    total_reads_.fetch_add(1);
    total_bytes_read_.fetch_add(task->bytes_read_);
  } else {
    task->return_code_ = 1;
    task->bytes_read_ = 0;
  }

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Update(clio::run::shared_ptr<UpdateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetStats(clio::run::shared_ptr<GetStatsTask> &task) {
  CLIO_TASK_BODY_BEGIN
  ReadTask r_synthetic;
  r_synthetic.method_ = Method::kRead;
  r_synthetic.length_ = 1024 * 1024;
  WriteTask w_synthetic;
  w_synthetic.method_ = Method::kWrite;
  w_synthetic.length_ = 1024 * 1024;
  clio::run::TaskStat read_stat = GetTaskStats(&r_synthetic);
  clio::run::TaskStat write_stat = GetTaskStats(&w_synthetic);
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

  // Expose the latest ML-predicted TTL so the CTE can make
  // TTL-aware allocation decisions without any external daemon.
  task->predicted_ttl_days_ = predicted_ttl_days_.load(std::memory_order_relaxed);

  // Cache the derived metrics and persist them (throttled) so the next
  // session starts from this one's estimates (issue #747). Merge field-wise:
  // a zero derived value (model not warmed up yet) must not clobber a
  // configured/restored stat.
  if (task->metrics_.read_bandwidth_mbps_ > 0.0) {
    perf_metrics_.read_bandwidth_mbps_ = task->metrics_.read_bandwidth_mbps_;
  }
  if (task->metrics_.write_bandwidth_mbps_ > 0.0) {
    perf_metrics_.write_bandwidth_mbps_ = task->metrics_.write_bandwidth_mbps_;
  }
  if (task->metrics_.read_latency_us_ > 0.0) {
    perf_metrics_.read_latency_us_ = task->metrics_.read_latency_us_;
  }
  if (task->metrics_.write_latency_us_ > 0.0) {
    perf_metrics_.write_latency_us_ = task->metrics_.write_latency_us_;
  }
  SavePerfStats(/*force=*/false);

  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

/**
 * Update the in-memory predicted lifespan that bdev reports to callers.
 *
 * @param task SetLifespan task carrying the new remaining-life estimate.
 * @return Task resume state for the CLIO scheduler.
 */
clio::run::TaskResume Runtime::SetLifespan(
    clio::run::shared_ptr<SetLifespanTask> &task) {
  CLIO_TASK_BODY_BEGIN
  predicted_ttl_days_.store(task->lifespan_days_, std::memory_order_relaxed);
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Destroy(clio::run::shared_ptr<DestroyTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Final perf-stats save (unthrottled) so the next session inherits
  // everything learned in this one (issue #747).
  SavePerfStats(/*force=*/true);
  StopHealthPolling();
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::u64 Runtime::GetWorkRemaining() const { return 0; }

clio::run::TaskResume Runtime::Monitor(clio::run::shared_ptr<MonitorTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (task->query_ == "stats") {
    ReadTask r_synthetic;
    r_synthetic.method_ = Method::kRead;
    r_synthetic.length_ = 1024 * 1024;
    WriteTask w_synthetic;
    w_synthetic.method_ = Method::kWrite;
    w_synthetic.length_ = 1024 * 1024;
    clio::run::TaskStat read_stat = GetTaskStats(&r_synthetic);
    clio::run::TaskStat write_stat = GetTaskStats(&w_synthetic);
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

    pk.pack_map(16);
    pk.pack("pool_name");              pk.pack(pool_name_);
    pk.pack("bdev_type");              pk.pack(static_cast<clio::run::u32>(bdev_type_));
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

    std::string health_json = ctp::SystemInfo::GetDeviceHealthStats(pool_name_);
    std::string drive_type = ctp::SystemInfo::DeriveDriveType(pool_name_);
    std::string prediction_json = ctp::SystemInfo::PredictDriveFailure(drive_type, health_json, pool_name_);

    pk.pack("device_health");          pk.pack(health_json);
    pk.pack("failure_prediction");     pk.pack(prediction_json);
    pk.pack("predicted_ttl_days");     pk.pack(predicted_ttl_days_.load(std::memory_order_relaxed));

    task->results_[container_id_] = std::string(sbuf.data(), sbuf.size());
  }
  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::PostGpuContainerCreate() {}

void Runtime::StopHealthPolling() {
  health_poll_stop_.store(true, std::memory_order_relaxed);
  if (health_poll_thread_.joinable()) {
    health_poll_thread_.join();
  }
  if (transport_) {
    transport_->Destroy();
    transport_.reset();
  }
}

// ---------------------------------------------------------------------------
// PollPredictionServer
//
// Background thread body.  Every kHealthPollIntervalSec seconds it:
//   1. Reads SMART attributes for this device via SystemInfo.
//   2. Posts them to the local prediction server (server.py).
//   3. Parses "days_remaining" from the JSON response.
//   4. Stores the result in predicted_ttl_days_ so GetStats() can expose it
//      to the CTE on the next poll without any external daemon involved.
// ---------------------------------------------------------------------------
void Runtime::PollPredictionServer() {
  // Derive the drive type string ("hdd" or "ssd") from the pool / device name.
  const std::string drive_type = ctp::SystemInfo::DeriveDriveType(pool_name_);

  while (!health_poll_stop_.load(std::memory_order_relaxed)) {
    try {
      // Step 1: collect SMART attributes.
      const std::string health_json =
          ctp::SystemInfo::GetDeviceHealthStats(pool_name_);

      // Step 2 & 3: post to prediction server and get back a JSON response
      // that contains "days_remaining": <int> | null.
      const std::string pred_json =
          ctp::SystemInfo::PredictDriveFailure(drive_type, health_json, pool_name_);

      // Parse days_remaining out of the JSON string.  We use a lightweight
      // hand-rolled search to avoid pulling in a full JSON library here.
      //  • "days_remaining": 42   → 42
      //  • "days_remaining": null → keep current value (healthy)
      auto extract_days = [](const std::string &json) -> clio::run::u32 {
        const std::string key = "\"days_remaining\"";
        auto pos = json.find(key);
        if (pos == std::string::npos) return 999999;
        pos += key.size();
        // Skip whitespace and colon
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':')) ++pos;
        // "null" means the server says the drive is healthy
        if (json.size() >= pos + 4 && json.substr(pos, 4) == "null") return 999999;
        // Parse integer
        if (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos]))) {
          clio::run::u32 days = 0;
          while (pos < json.size() &&
                 std::isdigit(static_cast<unsigned char>(json[pos]))) {
            days = days * 10 + static_cast<clio::run::u32>(json[pos] - '0');
            ++pos;
          }
          return days;
        }
        return 999999;  // Unparseable → treat as healthy
      };

      clio::run::u32 new_ttl = extract_days(pred_json);
      predicted_ttl_days_.store(new_ttl, std::memory_order_relaxed);

    } catch (...) {
      // If the prediction server is unreachable, silently retain the last
      // known TTL rather than falsely marking the drive as dead.
    }

    // Sleep in 1-second increments so we can wake promptly on stop.
    for (unsigned s = 0;
         s < kHealthPollIntervalSec &&
         !health_poll_stop_.load(std::memory_order_relaxed);
         ++s) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
}

}  // namespace clio::run::bdev

// Define ChiMod entry points using CLIO_TASK_CC macro
CLIO_TASK_CC(clio::run::bdev::Runtime)
