/**
 * Exploration must not leave the blob sized for the candidate it rejected.
 *
 * DynamicSchedule compresses the model's primary pick and STORES it, then
 * explores alternatives and stores a winner over the top. PutBlob overwrites
 * bytes but does not shorten a blob that was already longer, so adopting a
 * winner smaller than the primary left the difference allocated and
 * unreadable: the blob kept the PRIMARY's length while holding the WINNER's
 * payload. Measured at 4-9x the needed space, and every ratio derived from
 * the tier was wrong by that factor.
 *
 * The invariant asserted here is the one that catches it regardless of which
 * codec wins: what the core pool physically holds must equal what the chosen
 * codec actually produced. context_.actual_compressed_size_ is set to the
 * adopted result's total (header + payload), so the two must agree exactly.
 */
#include "simple_test.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/core/content_transfer_engine.h>
#include <clio_cte/compressor/compressor_client.h>
#include <clio_ctp/compress/compress_factory.h>

#if CTP_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace {

bool HaveGpu() {
#if CTP_ENABLE_CUDA
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
#else
  return false;
#endif
}

/* Exploration ON, threshold 0 so every chunk is explored rather than only the
 * ones the model is unsure about, and online learning ON because the
 * exploration block sits inside that guard (upstream nests it the same way). */
std::string WriteComposeConfig(const std::string &path) {
  std::ofstream f(path);
  f << "networking:\n  port: 9413\n"
    << "runtime:\n  num_threads: 4\n  queue_depth: 1024\n"
    << "compose:\n"
    << "  - mod_name: clio_bdev\n    pool_name: \"ram::chi_default_bdev\"\n"
    << "    pool_query: local\n    pool_id: \"301.0\"\n"
    << "    bdev_type: ram\n    capacity: \"256MB\"\n"
    << "  - mod_name: clio_cte_compressor\n    pool_name: cte_compressor\n"
    << "    pool_query: local\n    pool_id: \"512.0\"\n"
    << "    next_pool_id: \"513.0\"\n"
    << "    neuropress_model_path: \"" << CLIO_CTP_NEUROPRESS_WEIGHTS_DIR << "\"\n"
    << "    neuropress_online_learning_enabled: true\n"
    << "    neuropress_exploration_enabled: true\n"
    << "    neuropress_exploration_k: 31\n"
    << "    neuropress_exploration_threshold: 0.0\n"
    << "  - mod_name: clio_cte_core\n    pool_name: cte_core\n"
    << "    pool_query: local\n    pool_id: \"513.0\"\n"
    << "    storage:\n      - path: \"ram::cte_ram_tier1\"\n"
    << "        bdev_type: \"ram\"\n        capacity_limit: \"128MB\"\n"
    << "        score: 1.0\n"
    << "    dpe:\n      dpe_type: \"max_bw\"\n";
  f.close();
  return path;
}

}  // namespace

TEST_CASE("Exploration stores exactly what the adopted codec needs",
          "[compressor][neuropress][explore][693]") {
  if (!HaveGpu()) {
    INFO("No CUDA device -- skipping");
    return;
  }
  const std::string cfg = WriteComposeConfig("/tmp/np_explore_single_write.yaml");
  SIMPLE_TEST_SETENV("CLIO_SERVER_CONF", cfg.c_str());
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(std::chrono::seconds(1));
  // Compose created every pool; bind to the entrypoint (the compressor).
  CLIO_CTE_CLIENT->Init(clio::cte::core::kCtePoolId);

  auto *cte_client = CLIO_CTE_CLIENT;
  auto tag_task = cte_client->AsyncGetOrCreateTag("np_explore_tag");
  tag_task.Wait();
  auto tag_id = tag_task->tag_id_;

  // Smooth float32, the shape the model was trained on and the shape where
  // the codecs differ enough for exploration to have something to adopt.
  const size_t kElems = 256 * 1024;
  const size_t kBytes = kElems * sizeof(float);
  auto buf = CLIO_IPC->AllocateBuffer(kBytes);
  REQUIRE(!buf.IsNull());
  auto *fp = reinterpret_cast<float *>(buf.ptr_);
  for (size_t i = 0; i < kElems; ++i) {
    fp[i] = std::sin(static_cast<double>(i) * 0.001) * 100.0;
  }
  ctp::ipc::ShmPtr<> blob_data = buf.shm_.template Cast<void>();

  clio::cte::core::Context ctx;
  ctx.dynamic_compress_ = 2;  // DYNAMIC: let the model choose
  ctx.data_type_ = 1;         // float32

  clio::cte::compressor::Client compressor(clio::cte::core::kCtePoolId);
  auto sched = compressor.AsyncDynamicSchedule(
      clio::run::PoolQuery::Local(), tag_id, "np_explore_blob",
      /*offset=*/0, kBytes, blob_data, -1.0f, ctx, 0,
      clio::run::PoolId(513, 0));
  sched.Wait();
  REQUIRE(sched->GetReturnCode() == 0);

  const clio::run::u64 chosen_size = sched->context_.actual_compressed_size_;
  INFO("chosen codec " +
       ctp::CompressionFactory::NameForWireId(sched->context_.compress_lib_) +
       ", produced " + std::to_string(chosen_size) + " B");
  REQUIRE(chosen_size > 0);
  REQUIRE(chosen_size < kBytes);

  // Ask the CORE pool what it is PHYSICALLY holding. The compressor's own
  // GetBlobSize is overridden to report the LOGICAL size and cannot answer.
  clio::cte::core::Client core(clio::run::PoolId(513, 0));
  auto sz = core.AsyncGetBlobSize(tag_id, "np_explore_blob");
  sz.Wait();
  REQUIRE(sz->GetReturnCode() == 0);
  INFO("core holds " + std::to_string(sz->size_) + " B, chosen codec needs " +
       std::to_string(chosen_size) + " B");

  // THE INVARIANT. Larger than the chosen result means a rejected candidate's
  // footprint is still allocated.
  REQUIRE(sz->size_ == chosen_size);

  // And the bytes must still come back: a blob sized correctly but truncated
  // into its own payload would satisfy the check above and be useless.
  auto get_buf = CLIO_IPC->AllocateBuffer(kBytes);
  REQUIRE(!get_buf.IsNull());
  std::memset(get_buf.ptr_, 0, kBytes);
  auto get_task = cte_client->AsyncGetBlob(
      tag_id, "np_explore_blob", 0, kBytes, 0,
      get_buf.shm_.template Cast<void>());
  get_task.Wait();
  REQUIRE(get_task->GetReturnCode() == 0);
  auto *got = reinterpret_cast<float *>(get_buf.ptr_);
  size_t bad = 0;
  for (size_t i = 0; i < kElems; ++i) {
    if (std::memcmp(&got[i], &fp[i], sizeof(float)) != 0) ++bad;
  }
  INFO(std::to_string(bad) + " mismatching floats after round trip");
  REQUIRE(bad == 0);
}

SIMPLE_TEST_MAIN()
