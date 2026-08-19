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
 * Compressor ChiMod Functional Tests
 *
 * Tests the actual functionality of the compressor chimod tasks:
 * - CompressTask: Compression with various libraries
 * - DecompressTask: Decompression and data integrity
 * - DynamicScheduleTask: Intelligent compression selection
 * - Round-trip: Compress + Decompress data verification
 *
 * NOTE: The compressor API has been changed to integrate with the core module.
 * CompressTask now has the same inputs as PutBlobTask and calls PutBlob internally.
 * DecompressTask now has the same inputs as GetBlobTask and calls GetBlob internally.
 * These tests require a fully initialized CTE environment with core pool.
 */

#include "simple_test.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <vector>
#include <random>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/compressor/compressor_client.h>
#include <clio_cte/compressor/compressor_tasks.h>
#include <clio_cte/compressor/compressor_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#if CTP_ENABLE_NVCOMP
#include <cuda_runtime.h>
#include <clio_ctp/util/gpu_api.h>
#endif

using namespace clio::cte::compressor;

namespace {

// Compression library IDs
namespace CompLib {
  constexpr int NONE = 0;
  constexpr int BROTLI = 0;
  constexpr int BZIP2 = 1;
  constexpr int BLOSC2 = 2;
  constexpr int FPZIP = 3;
  constexpr int LZ4 = 4;
  constexpr int LZMA = 5;
  constexpr int SNAPPY = 6;
  constexpr int SZ3 = 7;
  constexpr int ZFP = 8;
  constexpr int ZLIB = 9;
  constexpr int ZSTD = 10;
  // GPU compressors (require nvcomp build). These 8 wire ids are exactly
  // NeuroPress's GPUCOMPRESS_ALGO_AUTO action space (gpucompress.h) --
  // LZ4/SNAPPY/DEFLATE/GDEFLATE/ZSTD/ANS/CASCADED/BITCOMP -- registered in
  // CompressionFactory's frozen wire-id namespace.
  constexpr int NVCOMP_LZ4 = 11;
  constexpr int NVCOMP_SNAPPY = 12;
  constexpr int NVCOMP_ZSTD = 13;
  constexpr int NVCOMP_GDEFLATE = 14;
  constexpr int NVCOMP_DEFLATE = 15;
  constexpr int NVCOMP_ANS = 16;
  constexpr int NVCOMP_CASCADED = 21;
  constexpr int NVCOMP_BITCOMP = 22;
  // Explicit-selection-only algorithms (not part of NeuroPress's NN action
  // space, but selectable through Clio's static compress_lib_ path like
  // everything else). CUSZ/CUSZP are LOSSY (error-bounded); NDZIP is
  // LOSSLESS. All three require float-aligned data (checked in their
  // Compress() -- input_size % sizeof(float) must be 0).
  constexpr int CUSZ = 18;
  constexpr int NDZIP = 19;
  constexpr int CUSZP = 20;
}

/**
 * Generate test data with specified pattern
 */
std::vector<char> GenerateTestData(size_t size, const std::string& pattern) {
  std::vector<char> data(size);

  if (pattern == "zeros") {
    // All zeros - highly compressible
    std::fill(data.begin(), data.end(), 0);
  } else if (pattern == "ones") {
    // All ones - highly compressible
    std::fill(data.begin(), data.end(), 1);
  } else if (pattern == "repeating") {
    // Repeating pattern - moderately compressible
    const char pattern_bytes[] = {0x01, 0x02, 0x03, 0x04};
    for (size_t i = 0; i < size; ++i) {
      data[i] = pattern_bytes[i % 4];
    }
  } else if (pattern == "random") {
    // Random data - poorly compressible
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<> dis(0, 255);
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<char>(dis(gen));
    }
  } else if (pattern == "text") {
    // Text-like data - moderately compressible
    const char* text = "The quick brown fox jumps over the lazy dog. ";
    size_t text_len = strlen(text);
    for (size_t i = 0; i < size; ++i) {
      data[i] = text[i % text_len];
    }
  }

  return data;
}

/**
 * Generate smooth scientific-data-like float array (a sum of a few sine
 * waves, like a sampled sensor/simulation signal). SZ-family compressors
 * (cuSZ/cuSZp/ndzip) are designed for this kind of data, not arbitrary
 * bytes reinterpreted as float -- that risks NaN/Inf bit patterns that
 * are meaningless to compress and undefined to error-bound against.
 */
std::vector<float> GenerateFloatTestData(size_t num_floats) {
  std::vector<float> data(num_floats);
  for (size_t i = 0; i < num_floats; ++i) {
    double x = static_cast<double>(i) * 0.001;
    data[i] = static_cast<float>(50.0 * std::sin(x) + 10.0 * std::sin(x * 17.0) +
                                 5.0 * std::sin(x * 101.0));
  }
  return data;
}

/**
 * Initialize CLIO Runtime runtime for compressor tests
 */
void InitializeClio() {
  // Initialize CLIO Runtime runtime in client mode with runtime
  bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
  if (!success) {
    throw std::runtime_error("Failed to initialize Clio runtime");
  }
}

/**
 * Cleanup CLIO Runtime runtime
 */
void CleanupClio() {
  // Client finalize handled by CLIO_CLIENT destructor
}

/**
 * Create and return pool ID for core chimod
 */
clio::run::PoolId CreateCorePool() {
  clio::run::PoolId core_pool_id = clio::run::PoolId(1, 1);
  clio::cte::core::Client core_client;

  clio::cte::core::CreateParams core_params;
  auto create_task = core_client.AsyncCreate(
      clio::run::PoolQuery::Local(),
      "test_core_pool",
      core_pool_id,
      core_params);
  create_task.Wait();

  // Without a registered storage target, ExtendBlob's DPE placement step
  // sees an empty target list and every real PutBlob fails with "No storage
  // devices configured" (core_runtime.cc's ExtendBlob, error_code=1 ->
  // return_code_=11) -- regardless of which compressor was used or whether
  // compression itself succeeded. RAM-backed, sized generously (16 GiB) so
  // functional tests can safely exercise multi-GB payloads (e.g. the 1 GiB
  // GPU compressor sweep) even in the worst case where several SECTIONs in
  // a row fall back to storing the raw, uncompressed blob.
  core_client.Init(core_pool_id);
  auto reg_task = core_client.AsyncRegisterTarget(
      "test_compressor_ram_target", clio::run::bdev::BdevType::kRam,
      static_cast<clio::run::u64>(16) * 1024 * 1024 * 1024,
      clio::run::PoolQuery::Local(), clio::run::PoolId(700, 0));
  reg_task.Wait();
  if (reg_task->GetReturnCode() != 0) {
    throw std::runtime_error("CreateCorePool: failed to register storage target");
  }

  return core_pool_id;
}

/**
 * Create and return pool ID for compressor chimod
 */
clio::run::PoolId CreateCompressorPool() {
  clio::run::PoolId compressor_pool_id = clio::run::PoolId(2, 1);
  Client compressor_client;

  auto create_task = compressor_client.AsyncCreateCompressor(
      clio::run::PoolQuery::Local(),
      "test_compressor_pool",
      compressor_pool_id);
  create_task.Wait();

  return compressor_pool_id;
}

/**
 * Create and return pool ID for a compressor chimod configured with a
 * specific CompressorConfig (e.g. to point it at a predictor model on
 * disk), rather than the all-defaults pool CreateCompressorPool() makes.
 */
clio::run::PoolId CreateCompressorPoolWithConfig(
    const clio::run::PoolId &compressor_pool_id,
    const std::string &pool_name, const CompressorConfig &config) {
  Client compressor_client;

  auto create_task = compressor_client.AsyncCreateCompressor(
      clio::run::PoolQuery::Local(), pool_name, compressor_pool_id, config);
  create_task.Wait();

  return compressor_pool_id;
}

/**
 * Test fixture for CTE integration tests
 */
struct CTETestFixture {
  clio::run::PoolId core_pool_id_;
  clio::run::PoolId compressor_pool_id_;
  clio::cte::core::Client core_client_;
  Client compressor_client_;
  clio::cte::core::TagId tag_id_;

  CTETestFixture() {
    InitializeClio();
    core_pool_id_ = CreateCorePool();
    compressor_pool_id_ = CreateCompressorPool();
    core_client_.Init(core_pool_id_);
    compressor_client_.Init(compressor_pool_id_);

    // Create a test tag
    auto tag_task = core_client_.AsyncGetOrCreateTag("test_tag");
    tag_task.Wait();
    tag_id_ = tag_task->tag_id_;
  }

  ~CTETestFixture() {
    CleanupClio();
  }

  /**
   * Allocate shared memory and copy data to it
   */
  ctp::ipc::FullPtr<char> AllocateAndCopyData(const std::vector<char>& data) {
    auto shm_buffer = CLIO_IPC->AllocateBuffer(data.size());
    if (!shm_buffer.IsNull()) {
      std::memcpy(shm_buffer.ptr_, data.data(), data.size());
    }
    return shm_buffer;
  }

  /**
   * Read data from shared memory
   */
  std::vector<char> ReadFromSharedMemory(ctp::ipc::FullPtr<char>& buffer, size_t size) {
    std::vector<char> data(size);
    if (!buffer.IsNull()) {
      std::memcpy(data.data(), buffer.ptr_, size);
    }
    return data;
  }
};

}  // namespace

/**
 * Test Case 1: Basic Compress and Store via PutBlob
 * Tests that CompressTask properly compresses data and stores via core PutBlob
 */
TEST_CASE("Basic Compress and Store", "[compressor][functional][basic]") {
  CTETestFixture fixture;

  auto test_data = GenerateTestData(16 * 1024, "text");

  // Allocate shared memory for input data
  auto shm_buffer = fixture.AllocateAndCopyData(test_data);
  REQUIRE(!shm_buffer.IsNull());

  ctp::ipc::ShmPtr<> blob_data = shm_buffer.shm_.template Cast<void>();

  Context context;
  context.compress_lib_ = CompLib::LZ4;
  context.compress_preset_ = 2;

  // Call AsyncCompress which compresses and stores via PutBlob
  auto task = fixture.compressor_client_.AsyncCompress(
      clio::run::PoolQuery::Local(),
      fixture.tag_id_,
      "test_blob_compress",
      0,  // offset
      test_data.size(),
      blob_data,
      0.5f,  // score
      context,
      0,  // flags
      fixture.core_pool_id_);
  task.Wait();

  REQUIRE(task->return_code_ == 0);
  INFO("Compression completed successfully");

  // Cleanup
  CLIO_IPC->FreeBuffer(shm_buffer);
}

/**
 * Test Case 2: Decompress and Retrieve via GetBlob
 * Tests that DecompressTask properly retrieves and decompresses data
 */
TEST_CASE("Decompress and Retrieve", "[compressor][functional][basic]") {
  CTETestFixture fixture;

  auto original_data = GenerateTestData(16 * 1024, "text");

  // First, compress and store the data
  auto put_buffer = fixture.AllocateAndCopyData(original_data);
  REQUIRE(!put_buffer.IsNull());

  ctp::ipc::ShmPtr<> put_blob_data = put_buffer.shm_.template Cast<void>();

  Context context;
  context.compress_lib_ = CompLib::LZ4;
  context.compress_preset_ = 2;

  auto compress_task = fixture.compressor_client_.AsyncCompress(
      clio::run::PoolQuery::Local(),
      fixture.tag_id_,
      "test_blob_roundtrip",
      0,
      original_data.size(),
      put_blob_data,
      0.5f,
      context,
      0,
      fixture.core_pool_id_);
  compress_task.Wait();
  REQUIRE(compress_task->return_code_ == 0);

  CLIO_IPC->FreeBuffer(put_buffer);

  // Now retrieve and decompress
  auto get_buffer = CLIO_IPC->AllocateBuffer(original_data.size());
  REQUIRE(!get_buffer.IsNull());

  ctp::ipc::ShmPtr<> get_blob_data = get_buffer.shm_.template Cast<void>();

  auto decompress_task = fixture.compressor_client_.AsyncDecompressExplicit(
      clio::run::PoolQuery::Local(),
      fixture.tag_id_,
      "test_blob_roundtrip",
      0,
      original_data.size(),
      0,  // flags
      get_blob_data,
      fixture.core_pool_id_);
  decompress_task.Wait();

  REQUIRE(decompress_task->return_code_ == 0);
  REQUIRE(decompress_task->output_size_ == original_data.size());

  // Verify data integrity
  auto retrieved_data = fixture.ReadFromSharedMemory(get_buffer, original_data.size());
  REQUIRE(std::memcmp(original_data.data(), retrieved_data.data(), original_data.size()) == 0);

  INFO("Round-trip compression/decompression verified");
  CLIO_IPC->FreeBuffer(get_buffer);
}

/**
 * Test Case 3: Dynamic Schedule
 * Tests that DynamicScheduleTask selects optimal compression
 */
TEST_CASE("Dynamic Schedule Compression", "[compressor][functional][dynamic]") {
  CTETestFixture fixture;

  auto test_data = GenerateTestData(64 * 1024, "text");

  auto shm_buffer = fixture.AllocateAndCopyData(test_data);
  REQUIRE(!shm_buffer.IsNull());

  ctp::ipc::ShmPtr<> blob_data = shm_buffer.shm_.template Cast<void>();

  Context context;
  context.dynamic_compress_ = 0;  // Enable dynamic compression selection
  context.max_performance_ = false;  // Optimize for ratio

  auto task = fixture.compressor_client_.AsyncDynamicSchedule(
      clio::run::PoolQuery::Local(),
      fixture.tag_id_,
      "test_blob_dynamic",
      0,
      test_data.size(),
      blob_data,
      0.5f,
      context,
      0,
      fixture.core_pool_id_);
  task.Wait();

  REQUIRE(task->return_code_ == 0);
  INFO("DynamicSchedule selected compression library: " << task->context_.compress_lib_);
  INFO("Tier score: " << task->tier_score_);

  CLIO_IPC->FreeBuffer(shm_buffer);
}

/**
 * Test Case 3b (issue #693): with a NeuroPress model configured,
 * DynamicSchedule must actually consult clio_ctp::compress::model's wider
 * candidate set (11 CPU compressors x 3 presets via NeuroPressCandidateStats)
 * instead of silently continuing to use the old 4-library hardcoded
 * candidate_lib_configs list in EstCompressionStats(). The old hardcoded
 * list can only ever pick wire ids {1 bzip2, 4 lz4, 9 zlib, 10 zstd} (or 0
 * as the "nothing beat ratio 1.0" fallback); this test drives DynamicSchedule
 * across several distinct data patterns/sizes and requires at least one
 * selection to land on a wire id that ONLY the wider action space can reach
 * (blosc2/fpzip/lzma/snappy/sz3/zfp), proving the NeuroPress path is what
 * actually ran, not just that the old path happens to still work.
 */
TEST_CASE("DynamicSchedule - NeuroPress reaches the wider action space",
          "[compressor][functional][dynamic][neuropress][693]") {
#ifndef CLIO_CTP_NEUROPRESS_WEIGHTS_DIR
#error "CLIO_CTP_NEUROPRESS_WEIGHTS_DIR must be set by CMake"
#endif

  CTETestFixture fixture;

  // Point a second, separately-configured compressor pool at the same
  // trained NeuroPress weights the other tests already validate,
  // instead of the fixture's default (unconfigured) compressor pool.
  CompressorConfig config;
  config.neuropress_model_path_ = CLIO_CTP_NEUROPRESS_WEIGHTS_DIR;
  clio::run::PoolId neuropress_pool_id =
      CreateCompressorPoolWithConfig(clio::run::PoolId(2, 42),
                                     "test_compressor_pool_neuropress",
                                     config);
  Client neuropress_client;
  neuropress_client.Init(neuropress_pool_id);

  struct Trial { std::string pattern; size_t size; };
  std::vector<Trial> trials = {
      {"zeros", 4 * 1024},      {"zeros", 1024 * 1024},
      {"ones", 64 * 1024},      {"repeating", 64 * 1024},
      {"repeating", 1024 * 1024}, {"text", 64 * 1024},
      {"text", 1024 * 1024},    {"random", 64 * 1024},
  };

  // Wire ids only reachable through the wider action space
  // NeuroPressCandidateStats draws from (11 CPU compressors + NeuroPress's
  // actual trained 8-algorithm nvcomp GPU action space -- cuSZ/nDzip/cuSZp/
  // zfp-sycl are deliberately excluded from dynamic selection: the network
  // was never trained on them, see neuropress_bridge.cc) -- the old
  // hardcoded candidate_lib_configs list in EstCompressionStats()
  // only ever emits {1 bzip2, 4 lz4, 9 zlib, 10 zstd} (wire id 0 excluded:
  // it's BestCompressRatio's "nothing beat ratio 1.0" fallback sentinel as
  // well as brotli's real wire id, so seeing it alone wouldn't prove
  // anything). Anything else -- including a GPU pick, which is an even
  // stronger signal that NeuroPress's own ranking (not the old heuristic)
  // actually ran -- proves the wider space was reached.
  const std::set<int> kOldHardcodedOrAmbiguous = {0, 1, 4, 9, 10};

  std::set<int> observed_libs;
  for (const auto &trial : trials) {
    auto test_data = GenerateTestData(trial.size, trial.pattern);
    auto shm_buffer = fixture.AllocateAndCopyData(test_data);
    REQUIRE(!shm_buffer.IsNull());
    ctp::ipc::ShmPtr<> blob_data = shm_buffer.shm_.template Cast<void>();

    Context context;
    context.dynamic_compress_ = 0;  // Dynamic mode
    context.max_performance_ = false;  // Optimize for ratio

    auto task = neuropress_client.AsyncDynamicSchedule(
        clio::run::PoolQuery::Local(), fixture.tag_id_,
        "test_blob_neuropress_" + trial.pattern, 0, test_data.size(),
        blob_data, 0.5f, context, 0, fixture.core_pool_id_);
    task.Wait();

    REQUIRE(task->return_code_ == 0);
    observed_libs.insert(task->context_.compress_lib_);

    CLIO_IPC->FreeBuffer(shm_buffer);
  }

  bool reached_wider_action_space = false;
  for (int lib : observed_libs) {
    if (kOldHardcodedOrAmbiguous.count(lib) == 0) {
      reached_wider_action_space = true;
      break;
    }
  }
  REQUIRE(reached_wider_action_space);
}

/**
 * Exploration: the K-way search, its winner-adoption path, and the training
 * samples it produces.
 *
 * Exploration is off by default in BOTH trees, which is exactly why nothing
 * exercised it -- recorded as coverage gap G3 during the NeuroPress parity
 * investigation. It is not a side path: when an explored alternative beats
 * the primary on cost, Clio RE-STORES the blob with that alternative's bytes
 * (the adoption block in Runtime::Compress), so a bug there silently replaces
 * stored data with something produced by a different codec and different
 * preprocessing. It also feeds Train() with real-outcome samples.
 *
 * Gating, from CompressorConfig: exploration needs online learning on (that
 * block is what computes error_pct) AND exploration_enabled_, and then fires
 * when error_pct > exploration_threshold_. The threshold is dropped to 0 so
 * it fires on every chunk rather than only when the predictor happens to be
 * badly wrong.
 *
 * The assertion is round-trip integrity per pattern: whatever exploration
 * adopts, decompressing the stored blob must reproduce the original bytes.
 * That covers the case the adoption path gets wrong most plausibly --
 * swapping the payload while leaving the header describing the primary's
 * codec, preset, shuffle or quantization state.
 */
TEST_CASE("Exploration - adopted winners still round-trip",
          "[compressor][functional][dynamic][neuropress][exploration][693]") {
#ifndef CLIO_CTP_NEUROPRESS_WEIGHTS_DIR
#error "CLIO_CTP_NEUROPRESS_WEIGHTS_DIR must be set by CMake"
#endif

  CTETestFixture fixture;

  CompressorConfig config;
  config.neuropress_model_path_ = CLIO_CTP_NEUROPRESS_WEIGHTS_DIR;
  config.neuropress_online_learning_enabled_ = true;   // computes error_pct
  config.neuropress_exploration_enabled_ = true;       // the path under test
  config.neuropress_exploration_threshold_ = 0.0f;     // fire every chunk
  config.neuropress_exploration_k_ = 3;                // NeuroPress's default

  clio::run::PoolId explore_pool_id =
      CreateCompressorPoolWithConfig(clio::run::PoolId(2, 43),
                                     "test_compressor_pool_explore", config);
  Client explore_client;
  explore_client.Init(explore_pool_id);

  struct Trial { std::string pattern; size_t size; };
  const std::vector<Trial> trials = {
      {"zeros", 64 * 1024},     {"repeating", 64 * 1024},
      {"text", 256 * 1024},     {"random", 64 * 1024},
      {"repeating", 1024 * 1024},
  };

  for (const auto &trial : trials) {
    const auto original = GenerateTestData(trial.size, trial.pattern);
    auto put_buffer = fixture.AllocateAndCopyData(original);
    REQUIRE(!put_buffer.IsNull());
    ctp::ipc::ShmPtr<> put_blob_data = put_buffer.shm_.template Cast<void>();

    const std::string blob_name = "explore_blob_" + trial.pattern + "_" +
                                  std::to_string(trial.size);

    Context context;
    context.dynamic_compress_ = 0;     // dynamic mode -> NeuroPress ranks
    context.max_performance_ = false;  // optimize for ratio

    auto task = explore_client.AsyncDynamicSchedule(
        clio::run::PoolQuery::Local(), fixture.tag_id_, blob_name, 0,
        original.size(), put_blob_data, 0.5f, context, 0,
        fixture.core_pool_id_);
    task.Wait();
    REQUIRE(task->return_code_ == 0);
    CLIO_IPC->FreeBuffer(put_buffer);

    // THE ACTUAL CHECK: read the stored blob back. If exploration adopted an
    // alternative, these are that alternative's bytes under a header it also
    // had to rewrite -- codec id, preset, shuffle element size and any
    // quantization parameters. Getting any of those wrong yields either a
    // decode failure or a buffer of the right LENGTH holding wrong data,
    // which is why the comparison is byte-for-byte and not just on size.
    auto get_buffer = CLIO_IPC->AllocateBuffer(original.size());
    REQUIRE(!get_buffer.IsNull());
    ctp::ipc::ShmPtr<> get_blob_data = get_buffer.shm_.template Cast<void>();

    auto decompress_task = explore_client.AsyncDecompressExplicit(
        clio::run::PoolQuery::Local(), fixture.tag_id_, blob_name, 0,
        original.size(), 0, get_blob_data, fixture.core_pool_id_);
    decompress_task.Wait();

    REQUIRE(decompress_task->return_code_ == 0);
    REQUIRE(decompress_task->output_size_ == original.size());

    std::vector<char> retrieved(original.size());
    std::memcpy(retrieved.data(), get_buffer.ptr_, original.size());
    REQUIRE(std::memcmp(original.data(), retrieved.data(),
                        original.size()) == 0);

    CLIO_IPC->FreeBuffer(get_buffer);
  }
}

/**
 * Selection must not depend on WHERE the chunk lives.
 *
 * Recorded as coverage gap G2. Profiling during the NeuroPress parity
 * investigation showed the default functional path runs host-resident: only
 * InferKernel launches, so statistics come from DataStatisticsFactory on the
 * host rather than the device stats kernels, and the ranking consumes those
 * host numbers. The device path computes the same three features with
 * StatsPass1Kernel/StatsPass2DevKernel/EntropyFromHistKernel instead, and
 * feeds InferKernelDeviceStats.
 *
 * Two different implementations of the features, therefore two chances to
 * diverge -- and nothing asserted they agree END TO END. The parity suite
 * compares the feature values themselves, but it hands both sides the SAME
 * bytes; it cannot catch a selection that changes because the chunk happened
 * to arrive in device memory.
 *
 * This drives identical bytes through DynamicSchedule twice, once from host
 * SHM and once from a registered device backend, and requires the same
 * algorithm out of both. It also round-trips each, since a device-resident
 * chunk takes a different preprocessing and storage path.
 */
TEST_CASE("DynamicSchedule - selection is the same for host- and "
          "device-resident chunks",
          "[compressor][functional][dynamic][neuropress][residency][693]") {
#ifndef CLIO_CTP_NEUROPRESS_WEIGHTS_DIR
#error "CLIO_CTP_NEUROPRESS_WEIGHTS_DIR must be set by CMake"
#endif
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    return;  // no GPU: the device half of the comparison cannot run
  }

  CTETestFixture fixture;

  CompressorConfig config;
  config.neuropress_model_path_ = CLIO_CTP_NEUROPRESS_WEIGHTS_DIR;
  clio::run::PoolId pool_id = CreateCompressorPoolWithConfig(
      clio::run::PoolId(2, 44), "test_compressor_pool_residency", config);
  Client client;
  client.Init(pool_id);

  // Patterns spanning the entropy range the selector actually discriminates
  // on -- a single operating point could agree by luck.
  const std::vector<std::string> patterns = {"zeros", "repeating", "text",
                                             "random"};
  constexpr size_t kSize = 256 * 1024;

  for (const auto &pattern : patterns) {
    const auto data = GenerateTestData(kSize, pattern);

    // ---- Host-resident: the chunk lives in CPU shared memory. ----
    auto host_buf = fixture.AllocateAndCopyData(data);
    REQUIRE(!host_buf.IsNull());
    ctp::ipc::ShmPtr<> host_blob = host_buf.shm_.template Cast<void>();

    Context host_ctx;
    host_ctx.dynamic_compress_ = 0;
    host_ctx.max_performance_ = false;
    auto host_task = client.AsyncDynamicSchedule(
        clio::run::PoolQuery::Local(), fixture.tag_id_,
        "residency_host_" + pattern, 0, data.size(), host_blob, 0.5f,
        host_ctx, 0, fixture.core_pool_id_);
    host_task.Wait();
    REQUIRE(host_task->return_code_ == 0);
    const int host_lib = host_task->context_.compress_lib_;
    const int host_preset = host_task->context_.compress_preset_;
    CLIO_IPC->FreeBuffer(host_buf);

    // ---- Device-resident: register a real device backend so the ShmPtr
    //      resolves to device memory instead of being staged to the host.
    char *registered = nullptr;
    ctp::ipc::AllocatorId alloc_id = CLIO_IPC->AllocateAndRegisterGpuBackend(
        /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
        kSize, &registered);
    if (alloc_id.IsNull() || registered == nullptr) {
      continue;  // device backend unavailable here; host half already checked
    }
    REQUIRE(cudaMemcpy(registered, data.data(), kSize,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(ctp::IsDevicePointer(registered));

    ctp::ipc::ShmPtr<> dev_blob;
    dev_blob.alloc_id_ = alloc_id;
    dev_blob.off_ = reinterpret_cast<clio::run::u64>(registered);

    Context dev_ctx;
    dev_ctx.dynamic_compress_ = 0;
    dev_ctx.max_performance_ = false;
    auto dev_task = client.AsyncDynamicSchedule(
        clio::run::PoolQuery::Local(), fixture.tag_id_,
        "residency_dev_" + pattern, 0, data.size(), dev_blob, 0.5f, dev_ctx,
        0, fixture.core_pool_id_);
    dev_task.Wait();
    REQUIRE(dev_task->return_code_ == 0);
    const int dev_lib = dev_task->context_.compress_lib_;
    const int dev_preset = dev_task->context_.compress_preset_;

    // THE ACTUAL CHECK: same bytes in, same selection out, regardless of
    // which side of the PCIe bus they arrived on.
    REQUIRE(dev_lib == host_lib);
    REQUIRE(dev_preset == host_preset);

    // And the device-resident blob must still round-trip: it took a
    // different preprocessing and storage path to get here.
    auto get_buf = CLIO_IPC->AllocateBuffer(data.size());
    REQUIRE(!get_buf.IsNull());
    ctp::ipc::ShmPtr<> get_blob = get_buf.shm_.template Cast<void>();
    auto dec = client.AsyncDecompressExplicit(
        clio::run::PoolQuery::Local(), fixture.tag_id_,
        "residency_dev_" + pattern, 0, data.size(), 0, get_blob,
        fixture.core_pool_id_);
    dec.Wait();
    REQUIRE(dec->return_code_ == 0);
    REQUIRE(dec->output_size_ == data.size());
    REQUIRE(std::memcmp(data.data(), get_buf.ptr_, data.size()) == 0);

    CLIO_IPC->FreeBuffer(get_buf);
    CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, alloc_id);
  }
}

/**
 * Test Case 4: Multiple Compression Libraries
 * Tests compression with various libraries
 */
TEST_CASE("Multiple Compression Libraries", "[compressor][functional][libraries]") {
  CTETestFixture fixture;

  auto test_data = GenerateTestData(16 * 1024, "text");

  std::vector<std::pair<int, std::string>> libraries = {
    {CompLib::LZ4, "LZ4"},
    {CompLib::ZSTD, "ZSTD"},
    {CompLib::ZLIB, "ZLIB"}
  };

  for (const auto& [lib_id, lib_name] : libraries) {
    SECTION(lib_name) {
      auto shm_buffer = fixture.AllocateAndCopyData(test_data);
      REQUIRE(!shm_buffer.IsNull());

      ctp::ipc::ShmPtr<> blob_data = shm_buffer.shm_.template Cast<void>();

      Context context;
      context.compress_lib_ = lib_id;
      context.compress_preset_ = 2;

      std::string blob_name = "test_blob_" + lib_name;
      auto task = fixture.compressor_client_.AsyncCompress(
          clio::run::PoolQuery::Local(),
          fixture.tag_id_,
          blob_name,
          0,
          test_data.size(),
          blob_data,
          0.5f,
          context,
          0,
          fixture.core_pool_id_);
      task.Wait();

      REQUIRE(task->return_code_ == 0);
      INFO(lib_name << " compression completed successfully");

      CLIO_IPC->FreeBuffer(shm_buffer);
    }
  }
}

/**
 * Test Case 5: No Compression (Passthrough)
 * Tests that data with compress_lib_ = 0 is stored without compression
 */
TEST_CASE("No Compression Passthrough", "[compressor][functional][passthrough]") {
  CTETestFixture fixture;

  auto test_data = GenerateTestData(8 * 1024, "random");

  auto shm_buffer = fixture.AllocateAndCopyData(test_data);
  REQUIRE(!shm_buffer.IsNull());

  ctp::ipc::ShmPtr<> blob_data = shm_buffer.shm_.template Cast<void>();

  Context context;
  context.compress_lib_ = 0;  // No compression

  auto task = fixture.compressor_client_.AsyncCompress(
      clio::run::PoolQuery::Local(),
      fixture.tag_id_,
      "test_blob_passthrough",
      0,
      test_data.size(),
      blob_data,
      0.5f,
      context,
      0,
      fixture.core_pool_id_);
  task.Wait();

  REQUIRE(task->return_code_ == 0);
  INFO("Passthrough (no compression) completed successfully");

  CLIO_IPC->FreeBuffer(shm_buffer);
}

/**
 * Test Case 6: Error Handling - Invalid Parameters
 */
TEST_CASE("Error Handling - Invalid Parameters", "[compressor][functional][error]") {
  CTETestFixture fixture;

  auto test_data = GenerateTestData(1024, "text");

  SECTION("Null blob data") {
    Context context;
    context.compress_lib_ = CompLib::LZ4;

    auto task = fixture.compressor_client_.AsyncCompress(
        clio::run::PoolQuery::Local(),
        fixture.tag_id_,
        "test_blob_null",
        0,
        test_data.size(),
        ctp::ipc::ShmPtr<>::GetNull(),  // Null data
        0.5f,
        context,
        0,
        fixture.core_pool_id_);
    task.Wait();

    // Should fail gracefully
    REQUIRE(task->return_code_ != 0);
    INFO("Correctly handled null blob data");
  }

  SECTION("Zero size") {
    auto shm_buffer = fixture.AllocateAndCopyData(test_data);
    REQUIRE(!shm_buffer.IsNull());

    ctp::ipc::ShmPtr<> blob_data = shm_buffer.shm_.template Cast<void>();

    Context context;
    context.compress_lib_ = CompLib::LZ4;

    auto task = fixture.compressor_client_.AsyncCompress(
        clio::run::PoolQuery::Local(),
        fixture.tag_id_,
        "test_blob_zero_size",
        0,
        0,  // Zero size
        blob_data,
        0.5f,
        context,
        0,
        fixture.core_pool_id_);
    task.Wait();

    // Should fail gracefully
    REQUIRE(task->return_code_ != 0);
    INFO("Correctly handled zero size");

    CLIO_IPC->FreeBuffer(shm_buffer);
  }
}

#if CTP_ENABLE_NVCOMP
/**
 * Test Case 7: GPU compression via nvcomp (nvcomp-lz4, lib id 11)
 * Validates the compressor chimod can select and round-trip the GPU compressor
 * end-to-end (Compress -> PutBlob -> GetBlob -> Decompress). Returns early
 * (passes without a skip marker) when no GPU device is present.
 */
TEST_CASE("NvComp GPU Round-trip", "[compressor][functional][nvcomp][gpu]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    INFO("No CUDA device available; skipping nvcomp functional test");
    return;
  }

  CTETestFixture fixture;

  // Every wire id in NeuroPress's GPUCOMPRESS_ALGO_AUTO action space
  // (gpucompress.h) -- statically selecting each one and round-tripping
  // through Clio's actual Compress/Decompress tasks proves the whole chain
  // (CompressionFactory -> NvComp wrapper -> CTE storage -> decompress)
  // correctly passes data for every algorithm NeuroPress can pick, before
  // trusting the NN to pick among them dynamically.
  std::vector<std::pair<int, std::string>> algorithms = {
      {CompLib::NVCOMP_LZ4, "nvcomp-lz4"},
      {CompLib::NVCOMP_SNAPPY, "nvcomp-snappy"},
      {CompLib::NVCOMP_DEFLATE, "nvcomp-deflate"},
      {CompLib::NVCOMP_GDEFLATE, "nvcomp-gdeflate"},
      {CompLib::NVCOMP_ZSTD, "nvcomp-zstd"},
      {CompLib::NVCOMP_ANS, "nvcomp-ans"},
      {CompLib::NVCOMP_CASCADED, "nvcomp-cascaded"},
      {CompLib::NVCOMP_BITCOMP, "nvcomp-bitcomp"},
  };

  for (const auto& [lib_id, lib_name] : algorithms) {
    SECTION(lib_name) {
      // 64 MiB: large enough to be representative of real GPU chunk-based
      // compression (nvcomp's per-chunk overhead/behavior on a 64 KiB
      // buffer is not representative of the multi-MB writes Clio actually
      // handles). The *stored* (compressed) blob is far smaller than this,
      // so it comfortably fits the 256 MiB test RAM target even across all
      // 8 SECTION iterations.
      auto original_data = GenerateTestData(64 * 1024 * 1024, "text");

      // Compress + store via the chimod using the GPU compressor.
      auto put_buffer = fixture.AllocateAndCopyData(original_data);
      REQUIRE(!put_buffer.IsNull());
      ctp::ipc::ShmPtr<> put_blob_data = put_buffer.shm_.template Cast<void>();

      Context context;
      context.compress_lib_ = lib_id;
      context.compress_preset_ = 2;

      std::string blob_name = "test_blob_" + lib_name;
      auto compress_task = fixture.compressor_client_.AsyncCompress(
          clio::run::PoolQuery::Local(), fixture.tag_id_, blob_name, 0,
          original_data.size(), put_blob_data, 0.5f, context, 0,
          fixture.core_pool_id_);
      compress_task.Wait();
      REQUIRE(compress_task->return_code_ == 0);
      CLIO_IPC->FreeBuffer(put_buffer);

      // Retrieve + decompress via the chimod and verify integrity.
      auto get_buffer = CLIO_IPC->AllocateBuffer(original_data.size());
      REQUIRE(!get_buffer.IsNull());
      ctp::ipc::ShmPtr<> get_blob_data = get_buffer.shm_.template Cast<void>();

      auto decompress_task = fixture.compressor_client_.AsyncDecompressExplicit(
          clio::run::PoolQuery::Local(), fixture.tag_id_, blob_name, 0,
          original_data.size(), 0, get_blob_data, fixture.core_pool_id_);
      decompress_task.Wait();
      REQUIRE(decompress_task->return_code_ == 0);
      REQUIRE(decompress_task->output_size_ == original_data.size());

      auto retrieved_data =
          fixture.ReadFromSharedMemory(get_buffer, original_data.size());
      REQUIRE(std::memcmp(original_data.data(), retrieved_data.data(),
                          original_data.size()) == 0);
      INFO(lib_name << " GPU round-trip verified");
      CLIO_IPC->FreeBuffer(get_buffer);
    }
  }
}
#endif  // CTP_ENABLE_NVCOMP

#if CTP_ENABLE_NVCOMP || CTP_ENABLE_CUSZ || CTP_ENABLE_NDZIP || CTP_ENABLE_CUSZP
/**
 * Static selection of every GPU compressor Clio can build against, run at
 * 1 GiB: NeuroPress's 8-algorithm nvcomp action space (lossless, byte data)
 * plus the three explicit-selection-only float compressors NeuroPress also
 * defines (GPUCOMPRESS_ALGO_CUSZ/NDZIP/CUSZP) -- cuSZ and cuSZp are LOSSY
 * (error-bounded), ndzip is LOSSLESS. Same static-selection proof as the
 * 64 MiB "NvComp GPU Round-trip" test above, scaled up and covering the
 * complete algorithm set instead of just the NN-selectable subset.
 */
TEST_CASE("GPU Compressor Round-trip - 1GiB dataset",
          "[compressor][functional][gpu][1gb]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    INFO("No CUDA device available; skipping 1GiB GPU compressor sweep");
    return;
  }

  CTETestFixture fixture;
  constexpr size_t kOneGiB = static_cast<size_t>(1) * 1024 * 1024 * 1024;

#if CTP_ENABLE_NVCOMP
  {
    std::vector<std::pair<int, std::string>> lossless_byte_algos = {
        {CompLib::NVCOMP_LZ4, "nvcomp-lz4"},
        {CompLib::NVCOMP_SNAPPY, "nvcomp-snappy"},
        {CompLib::NVCOMP_DEFLATE, "nvcomp-deflate"},
        {CompLib::NVCOMP_GDEFLATE, "nvcomp-gdeflate"},
        {CompLib::NVCOMP_ZSTD, "nvcomp-zstd"},
        {CompLib::NVCOMP_ANS, "nvcomp-ans"},
        {CompLib::NVCOMP_CASCADED, "nvcomp-cascaded"},
        {CompLib::NVCOMP_BITCOMP, "nvcomp-bitcomp"},
    };
    auto original_data = GenerateTestData(kOneGiB, "text");

    for (const auto& [lib_id, lib_name] : lossless_byte_algos) {
      SECTION(lib_name) {
        auto put_buffer = fixture.AllocateAndCopyData(original_data);
        REQUIRE(!put_buffer.IsNull());
        ctp::ipc::ShmPtr<> put_blob_data = put_buffer.shm_.template Cast<void>();

        Context context;
        context.compress_lib_ = lib_id;
        context.compress_preset_ = 2;

        std::string blob_name = "test_blob_1gb_" + lib_name;
        auto compress_task = fixture.compressor_client_.AsyncCompress(
            clio::run::PoolQuery::Local(), fixture.tag_id_, blob_name, 0,
            original_data.size(), put_blob_data, 0.5f, context, 0,
            fixture.core_pool_id_);
        compress_task.Wait();
        REQUIRE(compress_task->return_code_ == 0);
        CLIO_IPC->FreeBuffer(put_buffer);

        auto get_buffer = CLIO_IPC->AllocateBuffer(original_data.size());
        REQUIRE(!get_buffer.IsNull());
        ctp::ipc::ShmPtr<> get_blob_data = get_buffer.shm_.template Cast<void>();

        auto decompress_task = fixture.compressor_client_.AsyncDecompressExplicit(
            clio::run::PoolQuery::Local(), fixture.tag_id_, blob_name, 0,
            original_data.size(), 0, get_blob_data, fixture.core_pool_id_);
        decompress_task.Wait();
        REQUIRE(decompress_task->return_code_ == 0);
        REQUIRE(decompress_task->output_size_ == original_data.size());

        auto retrieved_data =
            fixture.ReadFromSharedMemory(get_buffer, original_data.size());
        REQUIRE(std::memcmp(original_data.data(), retrieved_data.data(),
                            original_data.size()) == 0);
        INFO(lib_name << " 1GiB lossless round-trip verified");
        CLIO_IPC->FreeBuffer(get_buffer);
      }
    }
  }
#endif  // CTP_ENABLE_NVCOMP

#if CTP_ENABLE_CUSZ || CTP_ENABLE_NDZIP || CTP_ENABLE_CUSZP
  {
    // cuSZ/ndzip/cuSZp all require float-aligned input (Compress() rejects
    // input_size % sizeof(float) != 0), and cuSZ/cuSZp are LOSSY -- exact
    // memcmp is the wrong check for them. lossless=true (ndzip) verifies
    // byte-exact; lossless=false (cuSZ/cuSZp, BALANCED preset -> eb=1e-3)
    // verifies the decompressed signal stays close to the original (a
    // generous tolerance -- proving Clio's pipeline passes float data
    // through correctly, not re-verifying the SZ libraries' own internal
    // error-bound guarantees, which is those projects' own concern).
    struct FloatAlgo { int lib_id; std::string name; bool lossless; };
    std::vector<FloatAlgo> float_algos = {
#if CTP_ENABLE_NDZIP
        {CompLib::NDZIP, "ndzip", true},
#endif
#if CTP_ENABLE_CUSZ
        {CompLib::CUSZ, "cusz", false},
#endif
#if CTP_ENABLE_CUSZP
        {CompLib::CUSZP, "cuszp", false},
#endif
    };

    auto original_floats = GenerateFloatTestData(kOneGiB / sizeof(float));
    const size_t data_bytes = original_floats.size() * sizeof(float);
    std::vector<char> original_bytes(data_bytes);
    std::memcpy(original_bytes.data(), original_floats.data(), data_bytes);

    for (const auto& algo : float_algos) {
      SECTION(algo.name) {
        auto put_buffer = fixture.AllocateAndCopyData(original_bytes);
        REQUIRE(!put_buffer.IsNull());
        ctp::ipc::ShmPtr<> put_blob_data = put_buffer.shm_.template Cast<void>();

        Context context;
        context.compress_lib_ = algo.lib_id;
        context.compress_preset_ = 2;  // BALANCED

        std::string blob_name = "test_blob_1gb_" + algo.name;
        auto compress_task = fixture.compressor_client_.AsyncCompress(
            clio::run::PoolQuery::Local(), fixture.tag_id_, blob_name, 0,
            data_bytes, put_blob_data, 0.5f, context, 0,
            fixture.core_pool_id_);
        compress_task.Wait();
        REQUIRE(compress_task->return_code_ == 0);
        CLIO_IPC->FreeBuffer(put_buffer);

        auto get_buffer = CLIO_IPC->AllocateBuffer(data_bytes);
        REQUIRE(!get_buffer.IsNull());
        ctp::ipc::ShmPtr<> get_blob_data = get_buffer.shm_.template Cast<void>();

        auto decompress_task = fixture.compressor_client_.AsyncDecompressExplicit(
            clio::run::PoolQuery::Local(), fixture.tag_id_, blob_name, 0,
            data_bytes, 0, get_blob_data, fixture.core_pool_id_);
        decompress_task.Wait();
        REQUIRE(decompress_task->return_code_ == 0);
        REQUIRE(decompress_task->output_size_ == data_bytes);

        auto retrieved_bytes = fixture.ReadFromSharedMemory(get_buffer, data_bytes);
        const float* retrieved_floats =
            reinterpret_cast<const float*>(retrieved_bytes.data());

        if (algo.lossless) {
          REQUIRE(std::memcmp(original_bytes.data(), retrieved_bytes.data(),
                              data_bytes) == 0);
        } else {
          double max_abs_error = 0.0;
          bool saw_nan_or_inf = false;
          for (size_t i = 0; i < original_floats.size(); ++i) {
            float v = retrieved_floats[i];
            if (!std::isfinite(v)) { saw_nan_or_inf = true; break; }
            double err = std::fabs(static_cast<double>(v) -
                                   static_cast<double>(original_floats[i]));
            if (err > max_abs_error) max_abs_error = err;
          }
          REQUIRE_FALSE(saw_nan_or_inf);
          // BALANCED preset eb=1e-3 (cuSZ relative, cuSZp absolute) against
          // a signal with amplitude ~65 -- 1.0 is a generous ~15-1000x
          // margin over the nominal bound, so this catches real breakage
          // (garbage/zeroed output) without being brittle to the exact
          // error-bound semantics each library implements internally.
          REQUIRE(max_abs_error < 1.0);
          INFO(algo.name << " max abs error: " << max_abs_error);
        }
        INFO(algo.name << " 1GiB round-trip verified");
        CLIO_IPC->FreeBuffer(get_buffer);
      }
    }
  }
#endif  // CTP_ENABLE_CUSZ || CTP_ENABLE_NDZIP || CTP_ENABLE_CUSZP
}
#endif  // CTP_ENABLE_NVCOMP || CTP_ENABLE_CUSZ || CTP_ENABLE_NDZIP || CTP_ENABLE_CUSZP

#if CTP_ENABLE_NVCOMP
/**
 * Static compression (explicit context.compress_lib_, AsyncCompress -- no
 * NeuroPress/dynamic analysis) with the SOURCE data living in a real
 * cudaMalloc'd device allocation, not host memory. Every other static test
 * above (including the 1GiB sweep) stages its input via
 * fixture.AllocateAndCopyData(), which is plain host SHM -- none of them
 * prove Runtime::Compress() is safe when the bytes only exist on the GPU.
 *
 * The compressor task API only accepts a host-resident ShmPtr
 * (Runtime::Compress() dereferences task->blob_data_ directly on the host
 * via CLIO_IPC->ToFullPtr), so there is no "hand it a device pointer
 * directly" form -- any caller (the HDF5 VOL connector included, see
 * clio_dataset_write() in adapter/hdf5_vol/clio_vol.cc) must stage GPU data
 * into host SHM with a device-aware copy first. This test proves that
 * staging pattern plus the static compression path together produce a
 * correct round-trip when the source truly is device memory.
 */
TEST_CASE("Static Compress - real GPU device pointer as input",
          "[compressor][functional][gpu][static][693]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    INFO("No CUDA device available; skipping GPU device-pointer static "
         "compression test");
    return;
  }

  CTETestFixture fixture;

  // Compressible, non-random content (repeating pattern) so a real codec
  // does something meaningful with it.
  constexpr size_t kSize = 256 * 1024;
  std::vector<char> original_data(kSize);
  for (size_t i = 0; i < kSize; ++i) {
    original_data[i] = static_cast<char>((i / 64) % 16);
  }

  char *d_src = nullptr;
  REQUIRE(cudaMalloc(&d_src, kSize) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_src, original_data.data(), kSize,
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(ctp::IsDevicePointer(d_src));

  auto put_buffer = CLIO_IPC->AllocateBuffer(kSize);
  REQUIRE(!put_buffer.IsNull());
  ctp::DeviceAwareMemcpy(put_buffer.ptr_, d_src, kSize);
  cudaFree(d_src);
  ctp::ipc::ShmPtr<> put_blob_data = put_buffer.shm_.template Cast<void>();

  Context context;
  context.compress_lib_ = CompLib::LZ4;  // explicit/static, not analyzed
  context.compress_preset_ = 2;

  std::string blob_name = "test_blob_gpu_static";
  auto compress_task = fixture.compressor_client_.AsyncCompress(
      clio::run::PoolQuery::Local(), fixture.tag_id_, blob_name, 0,
      original_data.size(), put_blob_data, 0.5f, context, 0,
      fixture.core_pool_id_);
  compress_task.Wait();
  REQUIRE(compress_task->return_code_ == 0);
  CLIO_IPC->FreeBuffer(put_buffer);

  auto get_buffer = CLIO_IPC->AllocateBuffer(original_data.size());
  REQUIRE(!get_buffer.IsNull());
  ctp::ipc::ShmPtr<> get_blob_data = get_buffer.shm_.template Cast<void>();

  auto decompress_task = fixture.compressor_client_.AsyncDecompressExplicit(
      clio::run::PoolQuery::Local(), fixture.tag_id_, blob_name, 0,
      original_data.size(), 0, get_blob_data, fixture.core_pool_id_);
  decompress_task.Wait();
  REQUIRE(decompress_task->return_code_ == 0);
  REQUIRE(decompress_task->output_size_ == original_data.size());

  // Verify by copying the decompressed result back into a fresh device
  // allocation and comparing there -- proving both ends of a real device
  // round-trip (GPU in, GPU out), not just that the host-side copy matches.
  char *d_dst = nullptr;
  REQUIRE(cudaMalloc(&d_dst, original_data.size()) == cudaSuccess);
  ctp::DeviceAwareMemcpy(d_dst, get_buffer.ptr_, original_data.size());
  std::vector<char> rbuf(original_data.size());
  REQUIRE(cudaMemcpy(rbuf.data(), d_dst, original_data.size(),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  cudaFree(d_dst);

  REQUIRE(std::memcmp(original_data.data(), rbuf.data(),
                      original_data.size()) == 0);
  CLIO_IPC->FreeBuffer(get_buffer);
}

TEST_CASE("Compress keeps a device-resident input's compressed output "
          "on-device (issue #693)",
          "[compressor][functional][gpu][static][693]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    INFO("No CUDA device available; skipping");
    return;
  }

  CTETestFixture fixture;

  // Compressible, non-random content -- unlike test_cte_devmem_putget's
  // "Static Compress" test above, blob_data_ here is passed straight as a
  // device ShmPtr (no DeviceAwareMemcpy-into-host staging first), so
  // Compress() sees a genuinely device-resident input and takes the
  // output_on_device path.
  constexpr size_t kSize = 256 * 1024;
  std::vector<char> original_data(kSize);
  for (size_t i = 0; i < kSize; ++i) {
    original_data[i] = static_cast<char>((i / 64) % 16);
  }

  auto *ipc = CLIO_IPC;
  char *device_base = nullptr;
  auto alloc_id = ipc->AllocateAndRegisterGpuBackend(
      /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem, kSize,
      &device_base);
  REQUIRE(!alloc_id.IsNull());
  REQUIRE(device_base != nullptr);
  ctp::GpuApi::Memcpy(device_base, original_data.data(), kSize);

  ctp::ipc::ShmPtr<> blob_data;
  blob_data.alloc_id_ = alloc_id;
  blob_data.off_ = reinterpret_cast<clio::run::u64>(device_base);

  Context context;
  context.compress_lib_ = CompLib::NVCOMP_LZ4;  // explicit/static, GPU-native
  context.compress_preset_ = 2;

  std::string blob_name = "test_blob_gpu_output_device";
  auto compress_task = fixture.compressor_client_.AsyncCompress(
      clio::run::PoolQuery::Local(), fixture.tag_id_, blob_name, 0,
      original_data.size(), blob_data, 0.5f, context, 0,
      fixture.core_pool_id_);
  compress_task.Wait();
  REQUIRE(compress_task->return_code_ == 0);
  ipc->FreeGpuBackend(/*gpu_id=*/0, alloc_id);

  // ---- Decompress and verify the round trip. ----
  auto get_buffer = CLIO_IPC->AllocateBuffer(original_data.size());
  REQUIRE(!get_buffer.IsNull());
  ctp::ipc::ShmPtr<> get_blob_data = get_buffer.shm_.template Cast<void>();

  auto decompress_task = fixture.compressor_client_.AsyncDecompressExplicit(
      clio::run::PoolQuery::Local(), fixture.tag_id_, blob_name, 0,
      original_data.size(), 0, get_blob_data, fixture.core_pool_id_);
  decompress_task.Wait();
  REQUIRE(decompress_task->return_code_ == 0);
  REQUIRE(decompress_task->output_size_ == original_data.size());

  REQUIRE(std::memcmp(original_data.data(), get_buffer.ptr_,
                      original_data.size()) == 0);
  CLIO_IPC->FreeBuffer(get_buffer);
}
#endif  // CTP_ENABLE_NVCOMP

// Main function using simple_test.h framework
SIMPLE_TEST_MAIN()
