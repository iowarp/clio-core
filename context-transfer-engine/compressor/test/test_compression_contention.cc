/**
 * Compression Contention Benchmark
 *
 * Simulates a producer-consumer workflow to measure how compression
 * performance is affected by CPU contention from concurrent workloads.
 *
 * Test setup:
 *   - Producer: 4 simulation threads generating data
 *   - Consumer: 2 compression threads processing data
 *   - Queue: Ring buffer between producer and consumer
 *
 * Metrics collected:
 *   - Simulation slowdown due to compression contention
 *   - Compression time with and without contention
 *   - CPU interference between workflows
 */

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <semaphore>
#include <mutex>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <memory>
#include <queue>
#include <condition_variable>
#include <sys/resource.h>
#include <sys/times.h>

#include <hermes_shm/compress/compress_factory.h>

using namespace std::chrono;

// ============================================================================
// Configuration
// ============================================================================

constexpr size_t CHUNK_SIZE_KB = 4096;  // 4 MB chunks
constexpr size_t CHUNK_SIZE = CHUNK_SIZE_KB * 1024;
constexpr int NUM_SIM_THREADS = 4;
constexpr int NUM_COMPRESS_THREADS = 2;
constexpr int QUEUE_DEPTH = 32;
constexpr int BUSY_TIME_SECONDS = 4;
constexpr int NUM_OUTPUTS = 16;

// ============================================================================
// Thread-safe Queue
// ============================================================================

template<typename T>
class ThreadSafeQueue {
 public:
  explicit ThreadSafeQueue(size_t max_size) : max_size_(max_size) {}

  bool Push(T item) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.size() >= max_size_) {
      full_events_++;
      return false;
    }
    queue_.push(std::move(item));
    cv_.notify_one();
    return true;
  }

  bool Pop(T& item, int timeout_ms = 100) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (cv_.wait_for(lock, milliseconds(timeout_ms),
                     [this] { return !queue_.empty(); })) {
      item = std::move(queue_.front());
      queue_.pop();
      return true;
    }
    empty_poll_events_++;
    return false;
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  size_t FullEvents() const { return full_events_; }
  size_t EmptyPollEvents() const { return empty_poll_events_; }

 private:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  size_t max_size_;
  std::atomic<size_t> full_events_{0};
  std::atomic<size_t> empty_poll_events_{0};
};

// ============================================================================
// Data Chunk
// ============================================================================

struct DataChunk {
  std::vector<char> data;
  int chunk_id;

  DataChunk() : chunk_id(-1) {}
  DataChunk(size_t size, int id) : data(size), chunk_id(id) {}
};

// ============================================================================
// Timing Utilities
// ============================================================================

struct CpuTime {
  double user_sec;
  double system_sec;

  double Total() const { return user_sec + system_sec; }
};

CpuTime GetCpuTime() {
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  return {
    static_cast<double>(usage.ru_utime.tv_sec) +
        static_cast<double>(usage.ru_utime.tv_usec) / 1e6,
    static_cast<double>(usage.ru_stime.tv_sec) +
        static_cast<double>(usage.ru_stime.tv_usec) / 1e6
  };
}

// ============================================================================
// Simulation Thread (Producer)
// ============================================================================

class SimulationThread {
 public:
  SimulationThread(int thread_id, int busy_time_sec)
      : thread_id_(thread_id), busy_time_sec_(busy_time_sec) {}

  void Run() {
    auto start = high_resolution_clock::now();
    auto end_time = start + seconds(busy_time_sec_);

    std::mt19937 rng(thread_id_);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // CPU-intensive simulation work
    double result = 0.0;
    while (high_resolution_clock::now() < end_time) {
      for (int i = 0; i < 100000; ++i) {
        result += std::sin(dist(rng)) * std::cos(dist(rng));
      }
    }

    // Prevent optimization
    volatile double sink = result;
    (void)sink;
  }

 private:
  int thread_id_;
  int busy_time_sec_;
};

// ============================================================================
// Compression Thread (Consumer)
// ============================================================================

class CompressionThread {
 public:
  CompressionThread(hshm::Compressor* compressor,
                   ThreadSafeQueue<std::shared_ptr<DataChunk>>* queue,
                   std::atomic<bool>* done,
                   std::atomic<size_t>* total_output_size)
      : compressor_(compressor),
        queue_(queue),
        done_(done),
        total_output_size_(total_output_size) {}

  void Run() {
    std::vector<char> output_buf(CHUNK_SIZE * 2);  // Pre-allocate output buffer
    while (!done_->load() || queue_->Size() > 0) {
      std::shared_ptr<DataChunk> chunk;
      if (queue_->Pop(chunk, 100)) {
        // Compress the chunk
        size_t output_size = output_buf.size();
        compressor_->Compress(output_buf.data(), output_size,
                             chunk->data.data(), chunk->data.size());
        total_output_size_->fetch_add(output_size);
      }
    }
  }

 private:
  hshm::Compressor* compressor_;
  ThreadSafeQueue<std::shared_ptr<DataChunk>>* queue_;
  std::atomic<bool>* done_;
  std::atomic<size_t>* total_output_size_;
};

// ============================================================================
// Benchmark Results
// ============================================================================

struct BenchmarkResult {
  std::string library_name;
  std::string preset;
  int sim_threads;
  int compress_threads;
  int queue_depth;
  size_t output_size_kb;

  // Timing
  double busy_time_sec;
  double target_cpu_time_sec;
  double wall_clock_sec;

  // Simulation metrics
  double sim_baseline_sec;
  double sim_slowdown_sec;
  double sim_slowdown_pct;
  double sim_cpu_sec;
  double sim_wall_sec;
  double sim_cpu_interference_sec;
  double sim_cpu_interference_pct;

  // Compression metrics
  double compress_cpu_sec;
  double compress_wall_sec;
  double compress_baseline_sec;
  double compress_cpu_interference_sec;
  double compress_cpu_interference_pct;
  double compress_contention_sec;
  double compress_contention_pct;

  // Output metrics
  int total_outputs;
  double input_mb;
  double output_mb;
  double compression_ratio;

  // Queue metrics
  size_t queue_full_events;
  size_t empty_poll_events;
};

// ============================================================================
// Run Benchmark
// ============================================================================

BenchmarkResult RunBenchmark(const std::string& lib_name,
                             const std::string& preset) {
  BenchmarkResult result;
  result.library_name = lib_name;
  result.preset = preset;
  result.sim_threads = NUM_SIM_THREADS;
  result.compress_threads = NUM_COMPRESS_THREADS;
  result.queue_depth = QUEUE_DEPTH;
  result.output_size_kb = CHUNK_SIZE_KB;
  result.busy_time_sec = BUSY_TIME_SECONDS;
  result.target_cpu_time_sec = BUSY_TIME_SECONDS * NUM_SIM_THREADS;
  result.total_outputs = NUM_OUTPUTS;
  result.input_mb = (CHUNK_SIZE * NUM_OUTPUTS) / (1024.0 * 1024.0);

  // Create compressor using factory
  std::unique_ptr<hshm::Compressor> compressor;
  if (lib_name != "NONE") {
    compressor = hshm::CompressionFactory::GetPreset(lib_name, hshm::CompressionPreset::BEST);
  }

  // Create queue and control flags
  ThreadSafeQueue<std::shared_ptr<DataChunk>> queue(QUEUE_DEPTH);
  std::atomic<bool> done{false};
  std::atomic<size_t> total_output_size{0};

  // Generate test data
  std::vector<std::shared_ptr<DataChunk>> chunks;
  std::mt19937 rng(42);
  for (int i = 0; i < NUM_OUTPUTS; ++i) {
    auto chunk = std::make_shared<DataChunk>(CHUNK_SIZE, i);
    // Fill with compressible random data
    for (size_t j = 0; j < CHUNK_SIZE; ++j) {
      chunk->data[j] = static_cast<char>(rng() % 64 + 32);  // Printable ASCII
    }
    chunks.push_back(chunk);
  }

  // Baseline: Measure simulation time without contention
  auto sim_baseline_start = high_resolution_clock::now();
  {
    std::vector<std::thread> sim_threads;
    for (int i = 0; i < NUM_SIM_THREADS; ++i) {
      sim_threads.emplace_back([i]() {
        SimulationThread sim(i, BUSY_TIME_SECONDS);
        sim.Run();
      });
    }
    for (auto& t : sim_threads) t.join();
  }
  auto sim_baseline_end = high_resolution_clock::now();
  result.sim_baseline_sec = duration<double>(sim_baseline_end - sim_baseline_start).count();

  // Baseline: Measure compression time without contention
  if (compressor) {
    std::vector<char> output_buf(CHUNK_SIZE * 2);
    auto compress_baseline_start = high_resolution_clock::now();
    for (auto& chunk : chunks) {
      size_t output_size = output_buf.size();
      compressor->Compress(output_buf.data(), output_size,
                          chunk->data.data(), chunk->data.size());
    }
    auto compress_baseline_end = high_resolution_clock::now();
    result.compress_baseline_sec = duration<double>(compress_baseline_end - compress_baseline_start).count();
  } else {
    result.compress_baseline_sec = 0;
  }

  // Main test: Run with contention
  done.store(false);
  total_output_size.store(0);

  auto cpu_start = GetCpuTime();
  auto wall_start = high_resolution_clock::now();

  // Start compression threads
  std::vector<std::thread> compress_threads;
  if (compressor) {
    for (int i = 0; i < NUM_COMPRESS_THREADS; ++i) {
      compress_threads.emplace_back([&]() {
        CompressionThread ct(compressor.get(), &queue, &done, &total_output_size);
        ct.Run();
      });
    }
  }

  // Start simulation threads and push data to queue
  std::vector<std::thread> sim_threads;
  for (int i = 0; i < NUM_SIM_THREADS; ++i) {
    sim_threads.emplace_back([i]() {
      SimulationThread sim(i, BUSY_TIME_SECONDS);
      sim.Run();
    });
  }

  // Push chunks to queue during simulation
  for (auto& chunk : chunks) {
    while (!queue.Push(chunk)) {
      std::this_thread::sleep_for(milliseconds(10));
    }
  }

  // Wait for simulation to complete
  for (auto& t : sim_threads) t.join();

  // Signal compression threads to finish
  done.store(true);
  for (auto& t : compress_threads) t.join();

  auto wall_end = high_resolution_clock::now();
  auto cpu_end = GetCpuTime();

  // Calculate metrics
  result.wall_clock_sec = duration<double>(wall_end - wall_start).count();
  result.sim_cpu_sec = cpu_end.Total() - cpu_start.Total();
  result.sim_wall_sec = result.wall_clock_sec * NUM_SIM_THREADS;

  result.sim_slowdown_sec = result.wall_clock_sec - result.sim_baseline_sec;
  result.sim_slowdown_pct = (result.sim_slowdown_sec / result.sim_baseline_sec) * 100.0;

  result.sim_cpu_interference_sec = result.sim_cpu_sec - result.target_cpu_time_sec;
  result.sim_cpu_interference_pct = (result.sim_cpu_interference_sec / result.target_cpu_time_sec) * 100.0;

  result.compress_cpu_sec = cpu_end.Total() - cpu_start.Total();
  result.compress_wall_sec = result.wall_clock_sec;

  if (result.compress_baseline_sec > 0) {
    result.compress_cpu_interference_sec = result.compress_cpu_sec - result.compress_baseline_sec;
    result.compress_cpu_interference_pct =
        (result.compress_cpu_interference_sec / result.compress_baseline_sec) * 100.0;
    result.compress_contention_sec = result.compress_wall_sec - result.compress_baseline_sec;
    result.compress_contention_pct =
        (result.compress_contention_sec / result.compress_baseline_sec) * 100.0;
  } else {
    result.compress_cpu_interference_sec = 0;
    result.compress_cpu_interference_pct = 0;
    result.compress_contention_sec = 0;
    result.compress_contention_pct = 0;
  }

  result.output_mb = total_output_size.load() / (1024.0 * 1024.0);
  result.compression_ratio = result.input_mb / std::max(result.output_mb, 0.001);

  result.queue_full_events = queue.FullEvents();
  result.empty_poll_events = queue.EmptyPollEvents();

  return result;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
  std::cout << "============================================================\n";
  std::cout << "COMPRESSION CONTENTION BENCHMARK\n";
  std::cout << "============================================================\n";
  std::cout << "Configuration:\n";
  std::cout << "  Simulation threads: " << NUM_SIM_THREADS << "\n";
  std::cout << "  Compression threads: " << NUM_COMPRESS_THREADS << "\n";
  std::cout << "  Chunk size: " << CHUNK_SIZE_KB << " KB\n";
  std::cout << "  Queue depth: " << QUEUE_DEPTH << "\n";
  std::cout << "  Busy time: " << BUSY_TIME_SECONDS << " seconds\n";
  std::cout << "  Outputs: " << NUM_OUTPUTS << "\n";
  std::cout << "============================================================\n\n";

  // Define libraries to test
  std::vector<std::string> libraries = {
    "NONE",
    "ZSTD",
    "LZ4",
    "ZLIB",
    "SNAPPY",
    "BROTLI",
    "LZMA",
    "BZIP2",
    "Blosc2",
#ifdef HSHM_ENABLE_LIBPRESSIO
    "ZFP",
    "SZ3",
    "FPZIP",
#endif
  };

  std::vector<BenchmarkResult> results;

  for (const auto& name : libraries) {
    std::cout << "Testing " << name << "... " << std::flush;
    auto result = RunBenchmark(name, "best");
    results.push_back(result);
    std::cout << "done (wall=" << std::fixed << std::setprecision(2)
              << result.wall_clock_sec << "s, slowdown="
              << result.sim_slowdown_pct << "%)\n";
  }

  // Write results to CSV
  std::string output_file = "results/paper/compression_contention_results.csv";
  std::ofstream csv(output_file);
  csv << "library_name,preset,sim_threads,compress_threads,queue_depth,"
      << "output_size_kb,busy_time_sec,target_cpu_time_sec,wall_clock_sec,"
      << "sim_baseline_sec,sim_slowdown_sec,sim_slowdown_pct,"
      << "sim_cpu_sec,sim_wall_sec,sim_cpu_interference_sec,sim_cpu_interference_pct,"
      << "compress_cpu_sec,compress_wall_sec,compress_baseline_sec,"
      << "compress_cpu_interference_sec,compress_cpu_interference_pct,"
      << "compress_contention_sec,compress_contention_pct,"
      << "total_outputs,input_mb,output_mb,compression_ratio,"
      << "queue_full_events,empty_poll_events\n";

  for (const auto& r : results) {
    csv << r.library_name << "," << r.preset << ","
        << r.sim_threads << "," << r.compress_threads << ","
        << r.queue_depth << "," << r.output_size_kb << ","
        << r.busy_time_sec << "," << r.target_cpu_time_sec << ","
        << r.wall_clock_sec << "," << r.sim_baseline_sec << ","
        << r.sim_slowdown_sec << "," << r.sim_slowdown_pct << ","
        << r.sim_cpu_sec << "," << r.sim_wall_sec << ","
        << r.sim_cpu_interference_sec << "," << r.sim_cpu_interference_pct << ","
        << r.compress_cpu_sec << "," << r.compress_wall_sec << ","
        << r.compress_baseline_sec << "," << r.compress_cpu_interference_sec << ","
        << r.compress_cpu_interference_pct << "," << r.compress_contention_sec << ","
        << r.compress_contention_pct << "," << r.total_outputs << ","
        << r.input_mb << "," << r.output_mb << "," << r.compression_ratio << ","
        << r.queue_full_events << "," << r.empty_poll_events << "\n";
  }

  csv.close();
  std::cout << "\nResults saved to: " << output_file << "\n";

  // Print summary
  std::cout << "\n============================================================\n";
  std::cout << "SUMMARY\n";
  std::cout << "============================================================\n";
  std::cout << std::setw(12) << "Library"
            << std::setw(15) << "Wall Time"
            << std::setw(15) << "Slowdown %"
            << std::setw(15) << "Ratio\n";
  std::cout << std::string(57, '-') << "\n";

  for (const auto& r : results) {
    std::cout << std::setw(12) << r.library_name
              << std::setw(15) << std::fixed << std::setprecision(2) << r.wall_clock_sec
              << std::setw(15) << r.sim_slowdown_pct
              << std::setw(15) << r.compression_ratio << "\n";
  }

  return 0;
}
