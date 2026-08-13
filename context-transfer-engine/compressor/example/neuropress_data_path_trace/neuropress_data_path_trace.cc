/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_data_path_trace.cc
 * @brief Follow a GPU-generated chunk through Clio-NeuroPress and record every
 *        byte that moves.
 *
 * The question:
 *
 *   Data is generated in GPU memory and handed to Clio. Does it STAY on the
 *   device until it has been compressed, and only then come back to the host?
 *
 * Answering it needs a timeline, not a summary. Each phase below is a named
 * marker in the trace, every cudaMemcpy and kernel launch is recorded in order
 * via CUPTI with the thread that made it, and each end of every copy is
 * resolved against the named memory regions this driver registers -- so the
 * report can say "the chunk left the device" rather than "a 16 MiB D2H
 * happened".
 *
 * Clio's runtime is started IN-PROCESS (CLIO_INIT with default_with_runtime =
 * true), which is what makes the compressor's own worker threads visible to
 * CUPTI. Without that the interesting half of the path would happen in another
 * process and none of it would appear here.
 *
 * Run: bin/neuropress_data_path_trace [--chunks N] [--chunk-mib M] [--out DIR]
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "path_tracer.h"

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/compressor/compressor_client.h>
#include <clio_ctp/compress/compress_factory.h>

extern "C" bool NpPathGenerateChunk(float *device_buf, size_t num_elements,
                                    unsigned int seed, int regime);
extern "C" bool NpPathCountMismatches(const float *a, const float *b,
                                      size_t num_elements,
                                      unsigned long long *out);

namespace {

std::string EnvOr(const char *name, const std::string &fallback) {
  const char *v = std::getenv(name);
  return (v && v[0]) ? std::string(v) : fallback;
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
  size_t num_chunks = std::strtoul(EnvOr("NPPATH_CHUNKS", "2").c_str(), nullptr, 10);
  size_t chunk_mib = std::strtoul(EnvOr("NPPATH_CHUNK_MIB", "16").c_str(), nullptr, 10);
  std::string out_dir = EnvOr("NPPATH_OUT_DIR", ".");
  bool readback = EnvOr("NPPATH_READBACK", "1") != "0";
  // 0 = compressible (the codec wins and the blob is stored compressed),
  // 1 = incompressible (the runtime discards the codec output and stores the
  // original bytes). These are different data paths, so which one is traced
  // must be a deliberate choice.
  int regime = std::atoi(EnvOr("NPPATH_REGIME", "0").c_str());
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--chunks" && i + 1 < argc) num_chunks = std::strtoul(argv[++i], nullptr, 10);
    else if (a == "--chunk-mib" && i + 1 < argc) chunk_mib = std::strtoul(argv[++i], nullptr, 10);
    else if (a == "--out" && i + 1 < argc) out_dir = argv[++i];
    else if (a == "--no-readback") readback = false;
    else if (a == "--incompressible") regime = 1;
    else if (a == "--help") {
      std::cout << "usage: " << argv[0]
                << " [--chunks N] [--chunk-mib M] [--out DIR] [--no-readback]"
                   " [--incompressible]\n";
      return 0;
    }
  }
  const size_t chunk_bytes = chunk_mib << 20;
  const size_t num_elements = chunk_bytes / sizeof(float);

  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::cerr << "No CUDA device; this traces a GPU data path and needs one."
              << std::endl;
    return 77;
  }

  auto &tracer = nppath::Tracer::Instance();
  std::string err;
  if (!tracer.Start(&err)) {
    std::cerr << "CUPTI could not be started: " << err
              << "\nWithout it there is no transfer record, which is the entire "
                 "output of this tool."
              << std::endl;
    return 1;
  }

  std::cout << "Clio-NeuroPress data path trace\n"
            << "  chunks: " << num_chunks << " x " << chunk_mib << " MiB"
            << "  regime: "
            << (regime == 0 ? "compressible" : "incompressible") << "\n\n";

  // ---- start the runtime BEFORE generating, so its own startup traffic is
  // ---- attributed to a startup phase and not to the data path.
  {
    nppath::Phase p("runtime-init");
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

  clio::run::PoolId compressor_pool_id(950, 1);
  clio::cte::core::TagId tag_id;
  clio::cte::compressor::Client compressor;
  {
    nppath::Phase p("clio-setup");
    auto reg = cte_client->AsyncRegisterTarget(
        "/tmp/npeq_data_path_backend.dat", clio::run::bdev::BdevType::kFile,
        static_cast<clio::run::u64>(2) << 30, clio::run::PoolQuery::Local(),
        clio::run::PoolId(949, 0));
    reg.Wait();
    if (reg->GetReturnCode() != 0) {
      std::cerr << "RegisterTarget failed" << std::endl;
      return 1;
    }

    clio::cte::compressor::CompressorConfig cfg;
    cfg.neuropress_model_path_ = CLIO_CTP_NEUROPRESS_WEIGHTS_DIR;
    auto create = compressor.AsyncCreateCompressor(
        clio::run::PoolQuery::Local(), "npeq_data_path_pool",
        compressor_pool_id, cfg);
    create.Wait();
    if (create->GetReturnCode() != 0) {
      std::cerr << "CreateCompressor failed" << std::endl;
      return 1;
    }
    compressor.Init(compressor_pool_id);

    auto tag = cte_client->AsyncGetOrCreateTag("npeq_data_path");
    tag.Wait();
    if (tag->GetReturnCode() != 0) {
      std::cerr << "GetOrCreateTag failed" << std::endl;
      return 1;
    }
    tag_id = tag->tag_id_;
  }

  // ---- generate the payload directly in device memory ----
  std::vector<float *> src(num_chunks, nullptr);
  {
    nppath::Phase p("generate-on-device");
    for (size_t i = 0; i < num_chunks; ++i) {
      CUDA_OK(cudaMalloc(&src[i], chunk_bytes));
      tracer.AddRegion("chunk" + std::to_string(i) + ".src", src[i], chunk_bytes);
      if (!NpPathGenerateChunk(src[i], num_elements, static_cast<unsigned>(i + 1),
                               regime)) {
        std::cerr << "generation failed" << std::endl;
        return 1;
      }
    }
    tracer.Note("payload exists only in device memory; there is no host original");
  }

  // ---- register device memory with the runtime, then hand it over ----
  std::vector<ctp::ipc::AllocatorId> alloc_ids(num_chunks);
  std::vector<char *> registered(num_chunks, nullptr);
  {
    nppath::Phase p("register-ipc-backend");
    for (size_t i = 0; i < num_chunks; ++i) {
      alloc_ids[i] = CLIO_IPC->AllocateAndRegisterGpuBackend(
          /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
          chunk_bytes, &registered[i]);
      if (alloc_ids[i].IsNull()) {
        std::cerr << "AllocateAndRegisterGpuBackend failed" << std::endl;
        return 1;
      }
      tracer.AddRegion("chunk" + std::to_string(i) + ".ipc", registered[i],
                       chunk_bytes);
    }
  }
  {
    // This D2D is the harness handing its buffer to the runtime's registered
    // one. It is a real copy and is left as PRODUCTION rather than tagged
    // harness, because a caller using Clio this way genuinely makes it.
    nppath::Phase p("stage-into-ipc-backend");
    for (size_t i = 0; i < num_chunks; ++i) {
      CUDA_OK(cudaMemcpy(registered[i], src[i], chunk_bytes,
                         cudaMemcpyDeviceToDevice));
    }
  }

  clio::cte::core::Context ctx;
  ctx.data_type_ = 1;  // float32
  ctx.error_bound_ = 0.0;

  std::vector<clio::run::Future<clio::cte::compressor::DynamicScheduleTask>> futs;
  {
    nppath::Phase p("dynamic-schedule-submit");
    for (size_t i = 0; i < num_chunks; ++i) {
      ctp::ipc::ShmPtr<> blob_data;
      blob_data.alloc_id_ = alloc_ids[i];
      blob_data.off_ = reinterpret_cast<clio::run::u64>(registered[i]);
      futs.push_back(compressor.AsyncDynamicSchedule(
          clio::run::PoolQuery::Local(), tag_id,
          "path_chunk_" + std::to_string(i), /*offset=*/0, chunk_bytes,
          blob_data, -1.0f, ctx, 0, cte_client->pool_id_));
    }
  }
  {
    // Everything the compressor does -- statistics, inference, preprocessing,
    // the codec, and the store -- happens inside this wait, on runtime worker
    // threads. This is the phase the question is really about.
    nppath::Phase p("compress-and-store");
    for (auto &f : futs) f.Wait();
  }

  for (size_t i = 0; i < num_chunks; ++i) {
    const auto &c = futs[i]->context_;
    const int wire = c.compress_lib_;
    std::cout << "  chunk " << i << ": rc=" << futs[i]->GetReturnCode()
              << "  compress_lib=" << wire << " ("
              << (wire == 0 ? std::string("stored raw / not beneficial")
                            : ctp::CompressionFactory::NameForWireId(wire))
              << ")  ratio=" << c.actual_compression_ratio_ << std::endl;
    tracer.Note("chunk " + std::to_string(i) + " selected wire_id " +
                std::to_string(wire));
  }

  // ---- read it back into a second device buffer and verify on-device ----
  unsigned long long mismatches = 0;
  if (readback) {
    nppath::Phase p("readback-to-device");
    for (size_t i = 0; i < num_chunks; ++i) {
      char *dst_reg = nullptr;
      ctp::ipc::AllocatorId dst_id = CLIO_IPC->AllocateAndRegisterGpuBackend(
          /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
          chunk_bytes, &dst_reg);
      if (dst_id.IsNull()) {
        std::cerr << "readback registration failed" << std::endl;
        return 1;
      }
      tracer.AddRegion("chunk" + std::to_string(i) + ".readback", dst_reg,
                       chunk_bytes);
      ctp::ipc::ShmPtr<> blob_data;
      blob_data.alloc_id_ = dst_id;
      blob_data.off_ = reinterpret_cast<clio::run::u64>(dst_reg);
      auto get = compressor.AsyncDecompressExplicit(
          clio::run::PoolQuery::Local(), tag_id,
          "path_chunk_" + std::to_string(i), /*offset=*/0, chunk_bytes, 0,
          blob_data, cte_client->pool_id_);
      get.Wait();
      if (get->GetReturnCode() != 0) {
        std::cerr << "decompress failed for chunk " << i << std::endl;
        return 1;
      }
      tracer.PushHarness();
      unsigned long long bad = 0;
      NpPathCountMismatches(src[i], reinterpret_cast<const float *>(dst_reg),
                            num_elements, &bad);
      tracer.PopHarness();
      mismatches += bad;
      CLIO_IPC->FreeGpuBackend(0, dst_id);
      tracer.DropRegion(dst_reg);
    }
    std::cout << "\n  round trip: " << mismatches << " / "
              << (num_elements * num_chunks) << " elements differ" << std::endl;
  }

  for (size_t i = 0; i < num_chunks; ++i) {
    CLIO_IPC->FreeGpuBackend(0, alloc_ids[i]);
    cudaFree(src[i]);
  }

  tracer.Stop();

  const std::string report = out_dir + "/data_path_report.txt";
  const std::string json = out_dir + "/data_path_timeline.json";
  nppath::WriteReport(report, chunk_bytes, num_chunks);
  nppath::WriteTimelineJson(json);
  nppath::PrintReport(std::cout, chunk_bytes, num_chunks);
  std::cout << "\nWrote:\n  " << report << "\n  " << json << std::endl;

  clio::run::CLIO_RUNTIME_FINALIZE();
  return mismatches == 0 ? 0 : 1;
}
