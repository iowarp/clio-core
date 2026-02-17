/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * CTE Score/Demotion Benchmark
 *
 * This benchmark measures the impact of blob reorganization (demotion) using
 * SetScore (AsyncReorganizeBlob) on Put performance.
 *
 * Order of operations per step:
 *   1. Start async reorganization of blobs from previous step (demotion)
 *   2. Busy wait (simulate compute, reorganization runs in background)
 *   3. Wait for reorganization to complete
 *   4. Put new blobs
 *
 * Usage:
 *   mpirun -n <nprocs> wrp_cte_score_bench <data_per_rank_step> <busy_wait_sec>
 * <num_steps> <demotion_pct>
 *
 * Parameters:
 *   data_per_rank_step: Amount of data per rank per step (e.g., 100m, 1g)
 *   busy_wait_sec: Time to busy wait during each step (seconds, e.g., 0.5)
 *   num_steps: Number of steps to run
 *   demotion_pct: Percentage of blobs to demote each step (0-100)
 *
 * Configuration:
 *   Use cte_score_bench_config.yaml with 8GB RAM (score 1.0) and 16GB NVMe
 * (score 0.0)
 */

#include <chimaera/chimaera.h>
#include <mpi.h>
#include <wrp_cte/core/core_client.h>
#include <hermes_shm/util/logging.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace std::chrono;

namespace {

// Blob size for this benchmark (1MB per blob)
constexpr chi::u64 kBlobSize = 1024 * 1024;

// Scores for tiered storage (higher score = higher priority/faster tier)
constexpr float kFastTierScore = 1.0f;  // RAM (fast tier)
constexpr float kSlowTierScore = 0.0f;  // NVMe (slow tier)

/**
 * Parse size string with k/K, m/M, g/G suffixes
 */
chi::u64 ParseSize(const std::string &size_str) {
  double size = 0.0;
  chi::u64 multiplier = 1;

  std::string num_str;
  char suffix = 0;

  for (char c : size_str) {
    if (std::isdigit(c) || c == '.') {
      num_str += c;
    } else if (c == 'k' || c == 'K' || c == 'm' || c == 'M' || c == 'g' ||
               c == 'G') {
      suffix = std::tolower(c);
      break;
    }
  }

  if (num_str.empty()) {
    return 0;
  }

  size = std::stod(num_str);

  switch (suffix) {
    case 'k':
      multiplier = 1024;
      break;
    case 'm':
      multiplier = 1024 * 1024;
      break;
    case 'g':
      multiplier = 1024ULL * 1024 * 1024;
      break;
    default:
      multiplier = 1;
      break;
  }

  return static_cast<chi::u64>(size * multiplier);
}

/**
 * Convert bytes to human-readable string
 */
std::string FormatSize(chi::u64 bytes) {
  if (bytes >= 1024ULL * 1024 * 1024) {
    return std::to_string(bytes / (1024ULL * 1024 * 1024)) + " GB";
  } else if (bytes >= 1024 * 1024) {
    return std::to_string(bytes / (1024 * 1024)) + " MB";
  } else if (bytes >= 1024) {
    return std::to_string(bytes / 1024) + " KB";
  } else {
    return std::to_string(bytes) + " B";
  }
}

/**
 * Calculate bandwidth in MB/s
 */
double CalcBandwidthMBs(chi::u64 total_bytes, double milliseconds) {
  if (milliseconds <= 0.0) return 0.0;
  double seconds = milliseconds / 1000.0;
  double megabytes = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
  return megabytes / seconds;
}

/**
 * Generate predictable blob name for a given rank, step, and blob index
 */
std::string GetBlobName(int rank, int step, int blob_idx) {
  return "blob_r" + std::to_string(rank) + "_s" + std::to_string(step) + "_b" +
         std::to_string(blob_idx);
}

/**
 * Generate predictable tag name for a given rank and step
 */
std::string GetTagName(int rank, int step) {
  return "tag_r" + std::to_string(rank) + "_s" + std::to_string(step);
}

/**
 * Busy wait for specified duration
 */
void BusyWait(double seconds) {
  if (seconds <= 0.0) return;

  auto start = high_resolution_clock::now();
  auto target_duration = duration<double>(seconds);

  while (true) {
    auto now = high_resolution_clock::now();
    if (now - start >= target_duration) {
      break;
    }
    // Busy loop - simulate compute work
    volatile int x = 0;
    for (int i = 0; i < 1000; ++i) {
      x += i;
    }
    (void)x;
  }
}

}  // namespace

/**
 * Benchmark result for a single demotion case
 */
struct BenchmarkResult {
  int demotion_pct;
  double put_time_ms;
  double demotion_time_ms;
  double total_time_ms;
  chi::u64 total_bytes;
  double put_bw_mbs;
  double total_bw_mbs;
};

/**
 * Main benchmark class
 */
class CTEScoreBenchmark {
 public:
  CTEScoreBenchmark(int rank, int num_procs, chi::u64 data_per_rank_step,
                    double busy_wait_sec, int num_steps, int demotion_pct)
      : rank_(rank),
        num_procs_(num_procs),
        data_per_rank_step_(data_per_rank_step),
        busy_wait_sec_(busy_wait_sec),
        num_steps_(num_steps),
        demotion_pct_(demotion_pct) {
    // Calculate number of blobs per step
    blobs_per_step_ = (data_per_rank_step_ + kBlobSize - 1) / kBlobSize;
    if (blobs_per_step_ == 0) {
      blobs_per_step_ = 1;
    }
  }

  /**
   * Run the benchmark
   */
  void Run() {
    if (rank_ == 0) {
      PrintBenchmarkInfo();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    BenchmarkResult result = RunDemotionCase(demotion_pct_);

    if (rank_ == 0) {
      PrintResult(result);
    }

    // Cleanup blobs after benchmark
    CleanupBlobs();
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank_ == 0) {
      HLOG(kInfo, "Benchmark complete.");
    }
  }

 private:
  void PrintBenchmarkInfo() {
    HLOG(kInfo, "=== CTE Score/Demotion Benchmark ===");
    HLOG(kInfo, "MPI processes: {}", num_procs_);
    HLOG(kInfo, "Data per rank per step: {}", FormatSize(data_per_rank_step_));
    HLOG(kInfo, "Blob size: {}", FormatSize(kBlobSize));
    HLOG(kInfo, "Blobs per step per rank: {}", blobs_per_step_);
    HLOG(kInfo, "Busy wait per step: {} seconds", busy_wait_sec_);
    HLOG(kInfo, "Number of steps: {}", num_steps_);
    HLOG(kInfo, "Demotion percentage: {}%", demotion_pct_);
    HLOG(kInfo, "Total data per rank: {}",
         FormatSize(data_per_rank_step_ * num_steps_));
    HLOG(kInfo, "Total data (all ranks): {}",
         FormatSize(data_per_rank_step_ * num_steps_ * num_procs_));
    HLOG(kInfo, "Fast tier score: {} (RAM)", kFastTierScore);
    HLOG(kInfo, "Slow tier score: {} (NVMe)", kSlowTierScore);
    HLOG(kInfo, "=====================================");
  }

  void PrintResult(const BenchmarkResult &r) {
    HLOG(kInfo, "");
    HLOG(kInfo, "========== BENCHMARK RESULTS ==========");
    HLOG(kInfo, "Demotion: {}%", r.demotion_pct);
    HLOG(kInfo, "Put time: {} ms", r.put_time_ms);
    HLOG(kInfo, "Demotion wait time: {} ms", r.demotion_time_ms);
    HLOG(kInfo, "Total time: {} ms", r.total_time_ms);
    HLOG(kInfo, "Put bandwidth: {} MB/s", r.put_bw_mbs);
    HLOG(kInfo, "Total bandwidth: {} MB/s", r.total_bw_mbs);
    HLOG(kInfo, "=========================================");
  }

  /**
   * Run a single demotion case
   *
   * Order of operations per step:
   * 1. Start async reorganization of blobs from previous step (demotion)
   * 2. Busy wait (simulate compute, reorganization runs in background)
   * 3. Wait for reorganization to complete
   * 4. Put new blobs
   */
  BenchmarkResult RunDemotionCase(int demotion_pct) {
    BenchmarkResult result;
    result.demotion_pct = demotion_pct;
    result.total_bytes = data_per_rank_step_ * num_steps_;

    double total_put_time_ms = 0.0;
    double total_demotion_time_ms = 0.0;

    // Allocate shared memory buffer for Put operations
    auto shm_buffer = CHI_IPC->AllocateBuffer(kBlobSize);
    std::memset(shm_buffer.ptr_, rank_ & 0xFF, kBlobSize);
    hipc::ShmPtr<> shm_ptr = shm_buffer.shm_.template Cast<void>();

    auto *cte_client = WRP_CTE_CLIENT;

    for (int step = 0; step < num_steps_; ++step) {
      // Step 1: Start async reorganization of blobs from PREVIOUS step
      std::vector<chi::Future<wrp_cte::core::ReorganizeBlobTask>> reorg_tasks;

      if (step > 0 && demotion_pct > 0) {
        int prev_step = step - 1;
        std::string prev_tag_name = GetTagName(rank_, prev_step);
        wrp_cte::core::Tag prev_tag(prev_tag_name);
        wrp_cte::core::TagId prev_tag_id = prev_tag.GetTagId();

        int blobs_to_demote =
            (blobs_per_step_ * demotion_pct + 99) / 100;  // Round up
        reorg_tasks.reserve(blobs_to_demote);

        // Demote first N blobs from previous step (predictable selection)
        for (int b = 0; b < blobs_to_demote; ++b) {
          std::string blob_name = GetBlobName(rank_, prev_step, b);
          auto task = cte_client->AsyncReorganizeBlob(prev_tag_id, blob_name,
                                                      kSlowTierScore);
          reorg_tasks.push_back(task);
        }
      }

      // Step 2: Busy wait (simulate compute, reorganization runs in background)
      BusyWait(busy_wait_sec_);

      // Step 3: Wait for reorganization to complete (measure only this wait)
      auto demote_wait_start = high_resolution_clock::now();
      for (auto &task : reorg_tasks) {
        task.Wait();
      }
      auto demote_wait_end = high_resolution_clock::now();

      if (step > 0 && !reorg_tasks.empty()) {
        total_demotion_time_ms +=
            duration_cast<microseconds>(demote_wait_end - demote_wait_start).count() /
            1000.0;
      }

      // Step 4: Put new blobs to fast tier (RAM)
      auto put_start = high_resolution_clock::now();

      std::string tag_name = GetTagName(rank_, step);
      wrp_cte::core::Tag tag(tag_name);

      std::vector<chi::Future<wrp_cte::core::PutBlobTask>> put_tasks;
      put_tasks.reserve(blobs_per_step_);

      for (chi::u64 b = 0; b < blobs_per_step_; ++b) {
        std::string blob_name = GetBlobName(rank_, step, b);
        auto task =
            tag.AsyncPutBlob(blob_name, shm_ptr, kBlobSize, 0, kFastTierScore);
        put_tasks.push_back(task);
      }

      // Wait for all Puts to complete
      for (auto &task : put_tasks) {
        task.Wait();
      }

      auto put_end = high_resolution_clock::now();
      total_put_time_ms +=
          duration_cast<microseconds>(put_end - put_start).count() / 1000.0;

      // Synchronize across ranks at end of step
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // Free shared memory buffer
    CHI_IPC->FreeBuffer(shm_buffer);

    // Gather timing results from all ranks
    double global_max_put_time_ms = 0.0;
    double global_max_demotion_time_ms = 0.0;

    MPI_Reduce(&total_put_time_ms, &global_max_put_time_ms, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_demotion_time_ms, &global_max_demotion_time_ms, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    result.put_time_ms = global_max_put_time_ms;
    result.demotion_time_ms = global_max_demotion_time_ms;
    result.total_time_ms = global_max_put_time_ms + global_max_demotion_time_ms;
    result.put_bw_mbs =
        CalcBandwidthMBs(result.total_bytes * num_procs_, result.put_time_ms);
    result.total_bw_mbs =
        CalcBandwidthMBs(result.total_bytes * num_procs_, result.total_time_ms);

    return result;
  }

  /**
   * Clean up all blobs created in this benchmark
   * Explicitly deletes each blob first, then deletes the tags
   */
  void CleanupBlobs() {
    auto *cte_client = WRP_CTE_CLIENT;

    if (rank_ == 0) {
      HLOG(kInfo, "  Cleaning up blobs...");
    }

    // First, delete all blobs explicitly
    for (int step = 0; step < num_steps_; ++step) {
      std::string tag_name = GetTagName(rank_, step);

      // Create tag object to get tag ID
      wrp_cte::core::Tag tag(tag_name);
      wrp_cte::core::TagId tag_id = tag.GetTagId();

      // Delete each blob in this tag
      std::vector<chi::Future<wrp_cte::core::DelBlobTask>> del_blob_tasks;
      del_blob_tasks.reserve(blobs_per_step_);

      for (chi::u64 b = 0; b < blobs_per_step_; ++b) {
        std::string blob_name = GetBlobName(rank_, step, b);
        auto task = cte_client->AsyncDelBlob(tag_id, blob_name);
        del_blob_tasks.push_back(task);
      }

      // Wait for all blob deletions to complete
      for (auto &task : del_blob_tasks) {
        task.Wait();
      }
    }

    // Then delete the tags
    for (int step = 0; step < num_steps_; ++step) {
      std::string tag_name = GetTagName(rank_, step);
      auto task = cte_client->AsyncDelTag(tag_name);
      task.Wait();
    }

    if (rank_ == 0) {
      HLOG(kInfo, "  Cleanup complete.");
    }
  }

  int rank_;
  int num_procs_;
  chi::u64 data_per_rank_step_;
  double busy_wait_sec_;
  int num_steps_;
  int demotion_pct_;
  chi::u64 blobs_per_step_;
};

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, num_procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  // Check arguments
  if (argc != 5) {
    if (rank == 0) {
      HLOG(kError, "Usage: {} <data_per_rank_step> <busy_wait_sec> <num_steps> <demotion_pct>",
           argv[0]);
      HLOG(kError,
           "  data_per_rank_step: Amount of data per rank per step (e.g., 100m, 1g)");
      HLOG(kError, "  busy_wait_sec: Busy wait time per step in seconds (e.g., 0.5)");
      HLOG(kError, "  num_steps: Number of steps to run");
      HLOG(kError, "  demotion_pct: Percentage of blobs to demote (0-100)");
      HLOG(kError, "");
      HLOG(kError, "Example:");
      HLOG(kError, "  mpirun -n 4 wrp_cte_score_bench 100m 0.5 10 50");
      HLOG(kError, "");
      HLOG(kError, "Environment variables:");
      HLOG(kError, "  WRP_RUNTIME_CONF: Path to chimaera configuration file");
    }
    MPI_Finalize();
    return 1;
  }

  chi::u64 data_per_rank_step = ParseSize(argv[1]);
  double busy_wait_sec = std::stod(argv[2]);
  int num_steps = std::atoi(argv[3]);
  int demotion_pct = std::atoi(argv[4]);

  // Validate parameters
  if (data_per_rank_step == 0 || num_steps <= 0 || demotion_pct < 0 || demotion_pct > 100) {
    if (rank == 0) {
      HLOG(kError, "Invalid parameters");
      HLOG(kError, "  data_per_rank_step must be > 0");
      HLOG(kError, "  num_steps must be > 0");
      HLOG(kError, "  demotion_pct must be 0-100");
    }
    MPI_Finalize();
    return 1;
  }

  // Initialize Chimaera runtime
  if (rank == 0) {
    HLOG(kInfo, "Initializing Chimaera runtime...");
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true)) {
    if (rank == 0) {
      HLOG(kError, "Failed to initialize Chimaera runtime");
    }
    MPI_Finalize();
    return 1;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Initialize CTE client
  if (!wrp_cte::core::WRP_CTE_CLIENT_INIT()) {
    if (rank == 0) {
      HLOG(kError, "Failed to initialize CTE client");
    }
    MPI_Finalize();
    return 1;
  }

  if (rank == 0) {
    HLOG(kInfo, "Runtime and client initialized successfully");
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Small delay to ensure initialization is complete
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // Run benchmark
  CTEScoreBenchmark benchmark(rank, num_procs, data_per_rank_step,
                              busy_wait_sec, num_steps, demotion_pct);
  benchmark.Run();

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return 0;
}
