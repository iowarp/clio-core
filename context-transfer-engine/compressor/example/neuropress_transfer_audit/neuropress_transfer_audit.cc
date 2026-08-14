/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_transfer_audit.cc
 * @brief Large-chunk (default 2 x 256 MiB) NeuroPress transfer audit for nsys.
 *
 * WHY THIS EXISTS SEPARATELY FROM neuropress_data_path_trace
 * ----------------------------------------------------------
 * The sibling tracer answers "does the payload stay on the device", and it
 * answers it with its own CUPTI record. This one answers a different question
 * -- "which transfers are REDUNDANT" -- and it answers it with Nsight Systems,
 * because that question needs a profiler's view: every copy, its stream, its
 * duration, and the call stack region it happened under, including copies made
 * by libraries this process never calls directly (nvcomp, the CUDA runtime).
 *
 * The mechanism that makes nsys usable here is NVTX. Each phase pushes an NVTX
 * range, so `nsys stats` can attribute every cudaMemcpy to the phase that
 * caused it instead of producing one undifferentiated list. Without that the
 * profile shows 40-odd copies and no way to tell setup from steady state.
 *
 * WHY 256 MiB
 * -----------
 * At 4-16 MiB the per-chunk control traffic (~1 KiB) is invisible next to the
 * payload and every ratio looks fine. Scaling the chunk 16x while the control
 * traffic stays FIXED is what separates transfers that scale with the data
 * (real, unavoidable) from transfers that do not (fixed overhead, and the only
 * candidates for elimination). Two chunks, so per-chunk cost can be separated
 * from one-time setup by differencing.
 *
 * CONFIGURATION
 * -------------
 * Inference-only: exploration and online learning are both OFF, which is the
 * shipped default on both sides (g_exploration_enabled{false},
 * g_online_learning_enabled{false}). Stated explicitly here anyway, because an
 * audit whose configuration is implicit is not reproducible.
 *
 * Ratio cost model: CLIO_NEUROPRESS_COST_W_CT=0 / _W_DT=0 leaves
 * cost = w_io * size / (ratio * bw), i.e. selection on compression ratio
 * alone. Set by this program unless the caller already set them, so the run is
 * self-describing rather than depending on the shell that launched it.
 */

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "clio_cte/compressor/compressor_client.h"
#include "clio_cte/compressor/compressor_tasks.h"
#include "clio_cte/core/content_transfer_engine.h"
#include "clio_ctp/compress/compress_factory.h"
#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/ipc_manager.h"

extern "C" bool NpAuditGenerateChunk(float *device_buf, size_t num_elements,
                                     unsigned int seed);
extern "C" bool NpAuditCountMismatches(const float *a, const float *b,
                                       size_t num_elements,
                                       unsigned long long *out);

namespace {

/** RAII NVTX range. This is what makes the nsys profile attributable. */
class NvtxPhase {
 public:
  explicit NvtxPhase(const char *name) {
    nvtxRangePushA(name);
    std::cout << "[phase] BEGIN " << name << std::endl;
    name_ = name;
  }
  ~NvtxPhase() {
    nvtxRangePop();
    std::cout << "[phase] END   " << name_ << std::endl;
  }

 private:
  std::string name_;
};

std::string EnvOr(const char *key, const std::string &fallback) {
  const char *v = std::getenv(key);
  return (v && *v) ? std::string(v) : fallback;
}

/** Set an env var only if the caller has not already chosen a value. */
void SetIfUnset(const char *key, const char *value) {
  const char *cur = std::getenv(key);
  if (cur == nullptr || *cur == '\0') setenv(key, value, /*overwrite=*/0);
}

#define CUDA_OK(expr)                                                        \
  do {                                                                       \
    cudaError_t rc_ = (expr);                                                \
    if (rc_ != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __LINE__ << ": "                      \
                << cudaGetErrorString(rc_) << std::endl;                     \
      return 1;                                                              \
    }                                                                        \
  } while (0)

}  // namespace

int main(int argc, char **argv) {
  size_t num_chunks =
      std::strtoul(EnvOr("NPAUDIT_CHUNKS", "2").c_str(), nullptr, 10);
  size_t chunk_mib =
      std::strtoul(EnvOr("NPAUDIT_CHUNK_MIB", "256").c_str(), nullptr, 10);
  bool readback = EnvOr("NPAUDIT_READBACK", "1") != "0";
  // Staging on (default): generate into a private buffer, then D2D it into the
  // registered backend -- what a producer that already owns its output does.
  // Staging off: generate DIRECTLY into the registered backend, which removes
  // the copy entirely. Both are legal; the point of the flag is to measure the
  // difference rather than argue about it.
  bool stage = EnvOr("NPAUDIT_STAGE", "1") != "0";
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--chunks" && i + 1 < argc)
      num_chunks = std::strtoul(argv[++i], nullptr, 10);
    else if (a == "--chunk-mib" && i + 1 < argc)
      chunk_mib = std::strtoul(argv[++i], nullptr, 10);
    else if (a == "--no-readback")
      readback = false;
    else if (a == "--no-staging")
      stage = false;
    else if (a == "--help") {
      std::cout << "usage: " << argv[0]
                << " [--chunks N] [--chunk-mib M] [--no-readback]"
                   " [--no-staging]\n"
                   "Run under: nsys profile --trace=cuda,nvtx --stats=true\n";
      return 0;
    }
  }

  // ---- the configuration this audit claims to run under, made explicit ----
  // Ratio cost model: cost = w_io * size/(ratio*bw), no time terms.
  SetIfUnset("CLIO_NEUROPRESS_COST_W_CT", "0");
  SetIfUnset("CLIO_NEUROPRESS_COST_W_DT", "0");
  SetIfUnset("CLIO_NEUROPRESS_COST_W_IO", "1");

  const size_t chunk_bytes = chunk_mib << 20;
  const size_t num_elements = chunk_bytes / sizeof(float);
  const double total_mib = static_cast<double>(chunk_mib * num_chunks);

  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::cerr << "No CUDA device; this audits a GPU data path and needs one."
              << std::endl;
    return 77;
  }

  std::cout << "NeuroPress transfer audit\n"
            << "  chunks       : " << num_chunks << " x " << chunk_mib
            << " MiB  (total " << total_mib << " MiB)\n"
            << "  selection    : NeuroPress, inference only "
               "(exploration OFF, online learning OFF)\n"
            << "  cost model   : ratio  (w_ct="
            << EnvOr("CLIO_NEUROPRESS_COST_W_CT", "?")
            << " w_dt=" << EnvOr("CLIO_NEUROPRESS_COST_W_DT", "?")
            << " w_io=" << EnvOr("CLIO_NEUROPRESS_COST_W_IO", "?") << ")\n"
            << "  payload      : generated ON DEVICE, no host original\n"
            << "  staging      : "
            << (stage ? "ON  (generate into a private buffer, then D2D)"
                      : "OFF (generate straight into the registered backend)")
            << "\n\n";

  {
    NvtxPhase p("runtime-init");
    if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
      std::cerr << "CLIO_INIT failed" << std::endl;
      return 1;
    }
    if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
      std::cerr << "CLIO_CTE_CLIENT_INIT failed" << std::endl;
      return 1;
    }
  }
  auto *cte_client = CLIO_CTE_CLIENT;

  clio::run::PoolId compressor_pool_id(952, 1);
  clio::cte::core::TagId tag_id;
  clio::cte::compressor::Client compressor;
  {
    NvtxPhase p("clio-setup");
    auto reg = cte_client->AsyncRegisterTarget(
        "/tmp/npaudit_backend.dat", clio::run::bdev::BdevType::kFile,
        static_cast<clio::run::u64>(8) << 30, clio::run::PoolQuery::Local(),
        clio::run::PoolId(951, 0));
    reg.Wait();
    if (reg->GetReturnCode() != 0) {
      std::cerr << "RegisterTarget failed" << std::endl;
      return 1;
    }

    clio::cte::compressor::CompressorConfig cfg;
    cfg.neuropress_model_path_ = CLIO_CTP_NEUROPRESS_WEIGHTS_DIR;
    // Inference only. Both are already false by default; set explicitly so the
    // audit's configuration is in the source rather than in a default that
    // could change underneath it.
    cfg.neuropress_exploration_enabled_ = false;
    cfg.neuropress_online_learning_enabled_ = false;
    auto create = compressor.AsyncCreateCompressor(
        clio::run::PoolQuery::Local(), "npaudit_pool", compressor_pool_id, cfg);
    create.Wait();
    if (create->GetReturnCode() != 0) {
      std::cerr << "CreateCompressor failed" << std::endl;
      return 1;
    }
    compressor.Init(compressor_pool_id);

    auto tag = cte_client->AsyncGetOrCreateTag("npaudit");
    tag.Wait();
    if (tag->GetReturnCode() != 0) {
      std::cerr << "GetOrCreateTag failed" << std::endl;
      return 1;
    }
    tag_id = tag->tag_id_;
  }

  // Registration FIRST, so the no-staging path has somewhere to generate into.
  // AllocateAndRegisterGpuBackend is a plain cudaMalloc in this process plus a
  // registration record (ipc_manager.cc:3840-3849), so what comes back is an
  // ordinary device pointer this process can launch kernels against -- there is
  // no runtime-owned initialization to trample.
  std::vector<ctp::ipc::AllocatorId> alloc_ids(num_chunks);
  std::vector<char *> registered(num_chunks, nullptr);
  {
    NvtxPhase p("register-ipc-backend");
    for (size_t i = 0; i < num_chunks; ++i) {
      alloc_ids[i] = CLIO_IPC->AllocateAndRegisterGpuBackend(
          /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
          chunk_bytes, &registered[i]);
      if (alloc_ids[i].IsNull()) {
        std::cerr << "AllocateAndRegisterGpuBackend failed" << std::endl;
        return 1;
      }
    }
  }

  // ---- payload is born on the device ----
  // With staging off, src ALIASES the registered backend: the generator writes
  // the payload where the compressor will read it, and no copy exists. The
  // alias is also what makes the round-trip check still meaningful -- the
  // compressor treats its input as read-only (quantize and shuffle each
  // allocate their own destination, compressor_runtime.cc:2440-2447), so the
  // buffer still holds the original bytes at verification time.
  std::vector<float *> src(num_chunks, nullptr);
  const bool src_is_registered = !stage;
  {
    NvtxPhase p("generate-on-device");
    for (size_t i = 0; i < num_chunks; ++i) {
      if (src_is_registered) {
        src[i] = reinterpret_cast<float *>(registered[i]);
      } else {
        CUDA_OK(cudaMalloc(&src[i], chunk_bytes));
      }
      if (!NpAuditGenerateChunk(src[i], num_elements,
                                static_cast<unsigned>(i + 1))) {
        std::cerr << "generation failed" << std::endl;
        return 1;
      }
    }
  }

  if (stage) {
    // A D2D of the whole payload. Kept in its own phase because it is the
    // largest single candidate for elimination in this whole trace: a producer
    // that wrote into the registered backend directly would not make it.
    NvtxPhase p("stage-into-ipc-backend");
    for (size_t i = 0; i < num_chunks; ++i) {
      CUDA_OK(cudaMemcpy(registered[i], src[i], chunk_bytes,
                         cudaMemcpyDeviceToDevice));
    }
  }

  clio::cte::core::Context ctx;
  ctx.data_type_ = 1;  // float32
  ctx.error_bound_ = 0.0;

  std::vector<clio::run::Future<clio::cte::compressor::DynamicScheduleTask>>
      futs;
  {
    NvtxPhase p("dynamic-schedule-submit");
    for (size_t i = 0; i < num_chunks; ++i) {
      ctp::ipc::ShmPtr<> blob_data;
      blob_data.alloc_id_ = alloc_ids[i];
      blob_data.off_ = reinterpret_cast<clio::run::u64>(registered[i]);
      futs.push_back(compressor.AsyncDynamicSchedule(
          clio::run::PoolQuery::Local(), tag_id,
          "audit_chunk_" + std::to_string(i), /*offset=*/0, chunk_bytes,
          blob_data, -1.0f, ctx, 0, cte_client->pool_id_));
    }
  }
  {
    // Statistics, inference, ranking, preprocessing, codec and store all run
    // inside this wait, on runtime worker threads. Every transfer nsys
    // attributes to this NVTX range is one the compressor made.
    NvtxPhase p("compress-and-store");
    for (auto &f : futs) f.Wait();
  }

  for (size_t i = 0; i < num_chunks; ++i) {
    const auto &c = futs[i]->context_;
    const int wire = c.compress_lib_;
    std::cout << "  chunk " << i << ": rc=" << futs[i]->GetReturnCode()
              << "  compress_lib=" << wire << " ("
              << (wire == 0 ? std::string("stored raw / not beneficial")
                            : ctp::CompressionFactory::NameForWireId(wire))
              << ")  ratio=" << std::fixed << std::setprecision(3)
              << c.actual_compression_ratio_ << std::endl;
  }

  unsigned long long mismatches = 0;
  if (readback) {
    NvtxPhase p("readback-to-device");
    for (size_t i = 0; i < num_chunks; ++i) {
      char *dst_reg = nullptr;
      ctp::ipc::AllocatorId dst_id = CLIO_IPC->AllocateAndRegisterGpuBackend(
          /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
          chunk_bytes, &dst_reg);
      if (dst_id.IsNull()) {
        std::cerr << "readback registration failed" << std::endl;
        return 1;
      }
      ctp::ipc::ShmPtr<> blob_data;
      blob_data.alloc_id_ = dst_id;
      blob_data.off_ = reinterpret_cast<clio::run::u64>(dst_reg);
      auto get = compressor.AsyncDecompressExplicit(
          clio::run::PoolQuery::Local(), tag_id,
          "audit_chunk_" + std::to_string(i), /*offset=*/0, chunk_bytes, 0,
          blob_data, cte_client->pool_id_);
      get.Wait();
      if (get->GetReturnCode() != 0) {
        std::cerr << "decompress failed for chunk " << i << std::endl;
        return 1;
      }
      unsigned long long bad = 0;
      // Verification is the harness's own work, not the data path's. It is
      // inside its own NVTX range so it can be excluded from the audit.
      nvtxRangePushA("harness-verify");
      NpAuditCountMismatches(src[i],
                             reinterpret_cast<const float *>(dst_reg),
                             num_elements, &bad);
      nvtxRangePop();
      mismatches += bad;
      CLIO_IPC->FreeGpuBackend(0, dst_id);
    }
    std::cout << "\n  round trip: " << mismatches << " / "
              << (num_elements * num_chunks) << " elements differ" << std::endl;
  }

  for (size_t i = 0; i < num_chunks; ++i) {
    CLIO_IPC->FreeGpuBackend(0, alloc_ids[i]);
    // Only free src when it is a buffer of ours. With staging off it aliases
    // the registered backend, which FreeGpuBackend has just released.
    if (!src_is_registered) cudaFree(src[i]);
  }

  std::cout << "\nRun `nsys stats <report>.nsys-rep` for the transfer counts.\n"
            << "The NVTX ranges above are the attribution keys." << std::endl;

  clio::run::CLIO_RUNTIME_FINALIZE();
  return mismatches == 0 ? 0 : 1;
}
