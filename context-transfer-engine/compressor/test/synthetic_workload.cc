/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**
 * Synthetic Workload Generator with MPI
 *
 * This program generates synthetic workloads with configurable data patterns
 * and calls the compressor chimod for dynamic scheduling. It simulates
 * scientific simulation I/O patterns with controllable compute/I/O phases.
 *
 * Usage:
 *   mpirun -n <nprocs> ./synthetic_workload_exec [options]
 *
 * Options:
 *   --io-size <size>        I/O size per rank (e.g., "1MB", "128KB") [default: 1MB]
 *   --transfer-size <size>  Transfer chunk size (e.g., "64KB") [default: 64KB]
 *   --compute-time <ms>     Compute time per iteration in ms [default: 100]
 *   --iterations <n>        Number of iterations [default: 10]
 *   --pattern <spec>        Data pattern specification (see below) [default: grayscott:100]
 *   --compress <option>     Compression: none, dynamic, zstd, lz4, etc. [default: dynamic]
 *   --output <path>         Output file path [default: synthetic_output.bp]
 *
 * Pattern specification:
 *   Format: <pattern1>:<percent1>,<pattern2>:<percent2>,...
 *   Patterns: uniform, gaussian, constant, gradient, sinusoidal, repeating, grayscott
 *   Example: grayscott:70,gaussian:20,uniform:10
 *
 * Environment variables:
 *   WRP_CTE_COMPRESS_TRACE: Set to "on" to enable compression tracing
 */

#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <random>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <map>
#include <getopt.h>
#include <ctime>

#include <chimaera/chimaera.h>
#include <hermes_shm/util/config_parse.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/compressor/compressor_client.h>

#include "synthetic_data_generator.h"

// Use the pattern types from the header
using PatternType = wrp_cte::PatternType;
using PatternSpec = wrp_cte::PatternSpec;
using DataGenerator = wrp_cte::SyntheticDataGenerator;

// Configuration structure
struct WorkloadConfig {
  size_t io_size_per_rank;      // I/O size per rank in bytes
  size_t transfer_size;         // Transfer chunk size in bytes
  int compute_time_ms;          // Compute time per iteration
  int iterations;               // Number of iterations
  std::vector<PatternSpec> patterns;  // Data pattern specifications
  std::string compress_option;  // Compression option
  std::string output_path;      // Output file path
  bool trace_enabled;           // Enable compression tracing

  WorkloadConfig()
      : io_size_per_rank(1024 * 1024),  // 1MB
        transfer_size(64 * 1024),        // 64KB
        compute_time_ms(100),
        iterations(10),
        compress_option("dynamic"),
        output_path("synthetic_output.bp"),
        trace_enabled(false) {
    // Default pattern: 100% grayscott
    patterns.push_back({PatternType::kGrayscott, 1.0});
  }
};

// Get compression library ID from name
int GetCompressLibId(const std::string& name) {
  static const std::map<std::string, int> lib_map = {
      {"none", 0},
      {"dynamic", -1},  // Special: dynamic scheduling
      {"zstd", 1},
      {"lz4", 2},
      {"brotli", 3},
      {"bzip2", 4},
      {"blosc2", 5},
      {"fpzip", 6},
      {"lzma", 7},
      {"snappy", 8},
      {"sz3", 9},
      {"zfp", 10},
      {"zlib", 11}
  };

  auto it = lib_map.find(name);
  return (it != lib_map.end()) ? it->second : 0;
}

void PrintUsage(const char* prog) {
  std::cerr << "Usage: mpirun -n <nprocs> " << prog << " [options]\n"
            << "\nOptions:\n"
            << "  --io-size <size>        I/O size per rank (e.g., \"1MB\") [default: 1MB]\n"
            << "  --transfer-size <size>  Transfer chunk size (e.g., \"64KB\") [default: 64KB]\n"
            << "  --compute-time <ms>     Compute time per iteration [default: 100]\n"
            << "  --iterations <n>        Number of iterations [default: 10]\n"
            << "  --pattern <spec>        Pattern: <name>:<pct>,... [default: grayscott:100]\n"
            << "  --compress <option>     none, dynamic, zstd, lz4, etc. [default: dynamic]\n"
            << "  --output <path>         Output file path [default: synthetic_output.bp]\n"
            << "  --trace                 Enable compression tracing\n"
            << "  --help                  Show this help message\n"
            << "\nPatterns: uniform, gaussian, constant, gradient, sinusoidal,\n"
            << "          repeating, grayscott, bimodal, exponential\n"
            << "\nExample:\n"
            << "  mpirun -n 4 " << prog << " --io-size 4MB --compute-time 200 "
            << "--pattern grayscott:70,gaussian:30 --compress dynamic\n";
}

WorkloadConfig ParseArgs(int argc, char** argv) {
  WorkloadConfig config;

  static struct option long_options[] = {
      {"io-size", required_argument, 0, 'i'},
      {"transfer-size", required_argument, 0, 't'},
      {"compute-time", required_argument, 0, 'c'},
      {"iterations", required_argument, 0, 'n'},
      {"pattern", required_argument, 0, 'p'},
      {"compress", required_argument, 0, 'x'},
      {"output", required_argument, 0, 'o'},
      {"trace", no_argument, 0, 'T'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}
  };

  int opt;
  int option_index = 0;

  while ((opt = getopt_long(argc, argv, "i:t:c:n:p:x:o:Th", long_options, &option_index)) != -1) {
    switch (opt) {
      case 'i':
        config.io_size_per_rank = hshm::ConfigParse::ParseSize(optarg);
        break;
      case 't':
        config.transfer_size = hshm::ConfigParse::ParseSize(optarg);
        break;
      case 'c':
        config.compute_time_ms = std::stoi(optarg);
        break;
      case 'n':
        config.iterations = std::stoi(optarg);
        break;
      case 'p':
        config.patterns = DataGenerator::ParsePatternSpec(optarg);
        break;
      case 'x':
        config.compress_option = optarg;
        break;
      case 'o':
        config.output_path = optarg;
        break;
      case 'T':
        config.trace_enabled = true;
        break;
      case 'h':
        PrintUsage(argv[0]);
        MPI_Finalize();
        exit(0);
      default:
        break;
    }
  }

  return config;
}

// Simulate compute work using CPU time (not wall clock time)
// This ensures the compute phase runs for the specified CPU time
// regardless of interference from other processes or OS scheduling
void SimulateCompute(int duration_ms) {
  if (duration_ms <= 0) return;

  // Convert target duration to CPU clock ticks
  clock_t target_ticks = (duration_ms * CLOCKS_PER_SEC) / 1000;
  clock_t start_ticks = clock();
  clock_t end_ticks = start_ticks + target_ticks;

  // Do actual computation to consume CPU time
  volatile double result = 0.0;
  while (clock() < end_ticks) {
    for (int i = 0; i < 1000; i++) {
      result += std::sin(static_cast<double>(i) * 0.001);
    }
  }
}

int main(int argc, char** argv) {
  // Initialize MPI
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // Parse command-line arguments
  WorkloadConfig config = ParseArgs(argc, argv);

  // Print configuration on rank 0
  if (rank == 0) {
    std::cout << "=== Synthetic Workload Generator ===" << std::endl;
    std::cout << "MPI ranks: " << nprocs << std::endl;
    std::cout << "I/O size per rank: " << config.io_size_per_rank << " bytes" << std::endl;
    std::cout << "Transfer size: " << config.transfer_size << " bytes" << std::endl;
    std::cout << "Compute time: " << config.compute_time_ms << " ms" << std::endl;
    std::cout << "Iterations: " << config.iterations << std::endl;
    std::cout << "Compression: " << config.compress_option << std::endl;
    std::cout << "Output: " << config.output_path << std::endl;
    std::cout << "Trace: " << (config.trace_enabled ? "enabled" : "disabled") << std::endl;
    std::cout << "Patterns: ";
    for (const auto& p : config.patterns) {
      std::cout << static_cast<int>(p.type) << ":" << (p.percentage * 100) << "% ";
    }
    std::cout << std::endl;
    std::cout << "======================================" << std::endl;
  }

  // Initialize CTE client (assumes Chimaera runtime is already running)
  if (!wrp_cte::core::WRP_CTE_CLIENT_INIT("", chi::PoolQuery::Local())) {
    if (rank == 0) {
      std::cerr << "Error: Failed to initialize CTE client. Make sure chimaera runtime is started." << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  // Get the global CTE client
  (void)wrp_cte::core::g_cte_client;  // Client is accessed via Tag class

  // Create compressor client if compression is enabled
  std::unique_ptr<wrp_cte::compressor::Client> compressor_client;

  if (config.compress_option != "none") {
    compressor_client = std::make_unique<wrp_cte::compressor::Client>();
    auto create_task = compressor_client->AsyncCreate(
        chi::PoolQuery::Local(),
        "wrp_cte_compressor",
        chi::PoolId(513, 0));
    create_task.Wait();
    if (create_task->GetReturnCode() == 0) {
      if (rank == 0) {
        std::cout << "Compressor client initialized" << std::endl;
      }
    } else if (rank == 0) {
      std::cerr << "Warning: Failed to create compressor pool, using no compression" << std::endl;
    }
  }

  // Allocate data buffer
  size_t num_elements = config.io_size_per_rank / sizeof(float);
  std::vector<float> data_buffer(num_elements);

  // Create tag for this workload
  std::string tag_name = "synthetic_workload_" + std::to_string(rank);
  wrp_cte::core::Tag tag(tag_name);

  // Timing statistics
  std::vector<double> compute_times;
  std::vector<double> io_times;
  std::vector<double> compress_times;
  std::vector<double> total_times;

  // Get compression library ID
  int compress_lib = GetCompressLibId(config.compress_option);
  bool use_dynamic = (compress_lib == -1);

  // Pending async operations from previous iteration
  std::vector<chi::Future<wrp_cte::core::PutBlobTask>> pending_futures;
  std::vector<hipc::FullPtr<char>> pending_buffers;  // Keep SHM buffers alive

  // Start end-to-end wall clock timer
  MPI_Barrier(MPI_COMM_WORLD);
  auto e2e_start = std::chrono::steady_clock::now();

  // Main iteration loop
  for (int iter = 0; iter < config.iterations; iter++) {
    auto iter_start = std::chrono::steady_clock::now();

    // Wait for pending operations from previous iteration before starting compute
    // This ensures all I/O is complete before we start the next simulation step
    if (!pending_futures.empty()) {
      for (auto& future : pending_futures) {
        future.Wait();
      }
      pending_futures.clear();
      // Release SHM buffers now that operations are complete
      for (auto& buf : pending_buffers) {
        CHI_IPC->FreeBuffer(buf);
      }
      pending_buffers.clear();
    }

    // Phase 1: Compute (uses CPU time, not wall clock)
    auto compute_start = std::chrono::steady_clock::now();
    SimulateCompute(config.compute_time_ms);
    auto compute_end = std::chrono::steady_clock::now();

    // Phase 2: Generate data
    DataGenerator::GenerateMixedData(data_buffer.data(), num_elements,
                                      config.patterns, rank, iter);

    // Phase 3: I/O with compression (async)
    auto io_start = std::chrono::steady_clock::now();

    // Create context for compression
    wrp_cte::core::Context context;
    context.dynamic_compress_ = use_dynamic ? 2 : (compress_lib > 0 ? 1 : 0);
    context.compress_lib_ = use_dynamic ? 0 : compress_lib;
    context.compress_preset_ = 2;  // Balanced
    context.data_type_ = 2;  // Float
    context.trace_ = config.trace_enabled;

    // Transfer data in chunks using async operations
    size_t bytes_written = 0;

    while (bytes_written < config.io_size_per_rank) {
      size_t chunk_size = std::min(config.transfer_size,
                                   config.io_size_per_rank - bytes_written);

      // Get blob name for this chunk
      std::string blob_name = "iter" + std::to_string(iter) +
                              "_chunk" + std::to_string(bytes_written / config.transfer_size);

      // Allocate shared memory for async operation
      auto shm_buffer = CHI_IPC->AllocateBuffer(chunk_size);

      // Copy data to shared memory
      const char* chunk_ptr = reinterpret_cast<const char*>(
          data_buffer.data() + bytes_written / sizeof(float));
      std::memcpy(shm_buffer.ptr_, chunk_ptr, chunk_size);

      // Convert ShmPtr<char> to ShmPtr<void> for async put
      hipc::ShmPtr<> shm_ptr(shm_buffer.shm_);

      // Async put blob with compression context
      auto future = tag.AsyncPutBlob(blob_name, shm_ptr, chunk_size,
                                      0, 0.5f, context);
      pending_futures.push_back(std::move(future));
      pending_buffers.push_back(shm_buffer);

      bytes_written += chunk_size;
    }

    auto io_end = std::chrono::steady_clock::now();
    auto iter_end = std::chrono::steady_clock::now();

    // Record timing (I/O time is just the submission time, actual I/O happens async)
    double compute_ms = std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
    double io_ms = std::chrono::duration<double, std::milli>(io_end - io_start).count();
    double total_ms = std::chrono::duration<double, std::milli>(iter_end - iter_start).count();

    compute_times.push_back(compute_ms);
    io_times.push_back(io_ms);
    compress_times.push_back(0.0);  // Compress time tracked separately
    total_times.push_back(total_ms);

    // Progress report
    if (rank == 0 && (iter + 1) % std::max(1, config.iterations / 10) == 0) {
      std::cout << "Iteration " << (iter + 1) << "/" << config.iterations
                << " - Compute: " << std::fixed << std::setprecision(1) << compute_ms
                << "ms, I/O submit: " << io_ms << "ms, Total: " << total_ms << "ms" << std::endl;
    }
  }

  // Wait for any remaining pending operations
  for (auto& future : pending_futures) {
    future.Wait();
  }
  for (auto& buf : pending_buffers) {
    CHI_IPC->FreeBuffer(buf);
  }
  pending_futures.clear();
  pending_buffers.clear();

  // End-to-end wall clock time
  MPI_Barrier(MPI_COMM_WORLD);
  auto e2e_end = std::chrono::steady_clock::now();
  double e2e_time_ms = std::chrono::duration<double, std::milli>(e2e_end - e2e_start).count();

  // Calculate local averages
  double avg_compute = 0, avg_io = 0, avg_total = 0;
  for (int i = 0; i < config.iterations; i++) {
    avg_compute += compute_times[i];
    avg_io += io_times[i];
    avg_total += total_times[i];
  }
  avg_compute /= config.iterations;
  avg_io /= config.iterations;
  avg_total /= config.iterations;

  // Reduce to get global averages and max end-to-end time
  double global_avg_compute, global_avg_io, global_avg_total;
  double global_max_e2e_time;
  MPI_Reduce(&avg_compute, &global_avg_compute, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&avg_io, &global_avg_io, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&avg_total, &global_avg_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&e2e_time_ms, &global_max_e2e_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  global_avg_compute /= nprocs;
  global_avg_io /= nprocs;
  global_avg_total /= nprocs;

  // Print final statistics
  if (rank == 0) {
    std::cout << "\n=== Final Statistics ===" << std::endl;
    std::cout << "Average compute time (CPU): " << std::fixed << std::setprecision(2)
              << global_avg_compute << " ms" << std::endl;
    std::cout << "Average I/O submit time: " << global_avg_io << " ms" << std::endl;
    std::cout << "Average iteration time: " << global_avg_total << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << "End-to-end wall clock time: " << global_max_e2e_time << " ms ("
              << (global_max_e2e_time / 1000.0) << " s)" << std::endl;

    double total_data_mb = (config.io_size_per_rank * nprocs * config.iterations) / (1024.0 * 1024.0);
    double e2e_time_s = global_max_e2e_time / 1000.0;
    double throughput_mb_s = total_data_mb / e2e_time_s;

    std::cout << "Total data: " << total_data_mb << " MB" << std::endl;
    std::cout << "End-to-end throughput: " << throughput_mb_s << " MB/s" << std::endl;
    std::cout << "=========================" << std::endl;
  }

  MPI_Finalize();
  return 0;
}
