/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Compress a dataset with Clio + NeuroPress and write the result to a
 * directory (issue #693).
 *
 * Generates 256 MiB on the GPU, hands it to Clio's compressor in 4 MiB
 * chunks, and lets the NeuroPress model pick the configuration for each of
 * the 64 chunks independently. Every compressed chunk is then written out as
 * its own file, alongside a manifest recording what the model chose and what
 * it achieved.
 *
 * Ranking is NeuroPress's own cost model -- w0*comp_time + w1*decomp_time +
 * w2*size/(ratio*bandwidth), all weights 1.0 at 5 GB/s -- not ratio alone,
 * which is the library default the bridge deliberately opts out of. The
 * preset is pinned to BALANCED because NeuroPress has no preset concept and
 * preset is not one of the model's 8 inputs, so enumerating FAST/BALANCED/
 * BEST would give three candidates one identical score and let list order,
 * not the model, decide. What the model actually chooses per chunk is
 * algorithm x quantize x byte-shuffle.
 *
 * INFERENCE ONLY. Online learning and exploration are both off, so the
 * weights never move and a chunk's configuration depends only on that
 * chunk's own statistics -- the same input produces the same directory every
 * run. Turning learning on would make the output depend on the order chunks
 * happened to be processed in, which is not what someone compressing a
 * dataset to disk wants.
 *
 * The files hold the blob exactly as Clio stores it: a 24-byte
 * CompressionHeader (56 when quantization ran) followed by the codec
 * payload. That is the same representation Clio's own read path decompresses,
 * which is why the verification pass at the end can hand them straight back.
 *
 * Usage:  bin/neuropress_compress_dir [output_dir]
 *         default output_dir is /tmp/neuropress_compressed
 */

#include <cuda_runtime.h>

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/compressor/compressor_client.h>

#include <sys/stat.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Defined in neuropress_e2e_kernel.cu.
void GenerateNeuroPressE2EDataset(float *d_out, unsigned long long num_elems,
                                  unsigned int elems_per_chunk,
                                  unsigned int num_chunks);

namespace {

constexpr size_t kChunkBytes = 4ull * 1024 * 1024;        // 4 MiB
constexpr size_t kDatasetBytes = 256ull * 1024 * 1024;    // 256 MiB
constexpr size_t kNumElems = kDatasetBytes / sizeof(float);
constexpr size_t kElemsPerChunk = kChunkBytes / sizeof(float);
constexpr size_t kNumChunks = kDatasetBytes / kChunkBytes;  // 64

const std::string kBackendFile = "/tmp/neuropress_compress_dir_backend.dat";

#define CUDA_CHECK(expr)                                                    \
  do {                                                                      \
    cudaError_t _rc = (expr);                                               \
    if (_rc != cudaSuccess) {                                               \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "  \
                << cudaGetErrorString(_rc) << std::endl;                    \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

/** NeuroPress's 0-7 algorithm index, recovered from the wire id, purely so
 *  the manifest reads the way its own logs do. */
const char *AlgoName(int wire_id) {
  // The compressor reports a wire id; map through the factory's own name so
  // this never drifts from what was actually applied.
  static std::string name;
  name = ctp::CompressionFactory::NameForWireId(wire_id);
  return name.c_str();
}

}  // namespace

int main(int argc, char **argv) {
  const std::string out_dir =
      (argc > 1) ? argv[1] : std::string("/tmp/neuropress_compressed");

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cerr << "No CUDA device available." << std::endl;
    return 77;
  }

  if (mkdir(out_dir.c_str(), 0755) != 0 && errno != EEXIST) {
    std::cerr << "cannot create " << out_dir << ": " << std::strerror(errno)
              << std::endl;
    return 1;
  }

  std::cout << "Dataset:    " << kDatasetBytes / (1024 * 1024) << " MiB\n"
            << "Chunk size: " << kChunkBytes / (1024 * 1024) << " MiB\n"
            << "Chunks:     " << kNumChunks << "\n"
            << "Model:      NeuroPress, inference only "
               "(no learning, no exploration)\n"
            << "Output:     " << out_dir << "\n"
            << std::endl;

  // ---- Runtime, storage target, compressor pool ----
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::cerr << "CLIO_INIT failed" << std::endl;
    return 1;
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::cerr << "CLIO_CTE_CLIENT_INIT failed" << std::endl;
    return 1;
  }
  auto *cte_client = CLIO_CTE_CLIENT;
  auto reg = cte_client->AsyncRegisterTarget(
      kBackendFile, clio::run::bdev::BdevType::kFile,
      static_cast<clio::run::u64>(1) * 1024 * 1024 * 1024,
      clio::run::PoolQuery::Local(), clio::run::PoolId(820, 0));
  reg.Wait();
  if (reg->GetReturnCode() != 0) {
    std::cerr << "RegisterTarget failed: " << reg->GetReturnCode() << std::endl;
    return 1;
  }

  const clio::run::PoolId compressor_pool_id(821, 1);
  clio::cte::compressor::CompressorConfig cfg;
  cfg.neuropress_model_path_ = CLIO_CTP_NEUROPRESS_WEIGHTS_DIR;
  cfg.neuropress_online_learning_enabled_ = false;  // inference only
  cfg.neuropress_exploration_enabled_ = false;
  clio::cte::compressor::Client compressor;
  auto create = compressor.AsyncCreateCompressor(
      clio::run::PoolQuery::Local(), "neuropress_compress_dir_pool",
      compressor_pool_id, cfg);
  create.Wait();
  if (create->GetReturnCode() != 0) {
    std::cerr << "CreateCompressor failed: " << create->GetReturnCode()
              << std::endl;
    return 1;
  }
  compressor.Init(compressor_pool_id);

  auto tag = cte_client->AsyncGetOrCreateTag("neuropress_compress_dir");
  tag.Wait();
  if (tag->GetReturnCode() != 0) {
    std::cerr << "GetOrCreateTag failed" << std::endl;
    return 1;
  }
  const auto tag_id = tag->tag_id_;

  // ---- Generate on the GPU ----
  float *d_data = nullptr;
  CUDA_CHECK(cudaMalloc(&d_data, kDatasetBytes));
  GenerateNeuroPressE2EDataset(d_data, kNumElems,
                               static_cast<unsigned>(kElemsPerChunk),
                               static_cast<unsigned>(kNumChunks));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<char> source(kDatasetBytes);
  CUDA_CHECK(cudaMemcpy(source.data(), d_data, kDatasetBytes,
                        cudaMemcpyDeviceToHost));
  std::cout << "Generated " << kDatasetBytes / (1024 * 1024)
            << " MiB on-device." << std::endl;

  // ---- Compress chunk by chunk, then write each blob out ----
  const std::string manifest_path = out_dir + "/manifest.csv";
  std::FILE *mf = std::fopen(manifest_path.c_str(), "w");
  if (!mf) {
    std::cerr << "cannot write " << manifest_path << std::endl;
    return 1;
  }
  std::fprintf(mf,
               "chunk,file,algorithm,wire_id,quantize,shuffle,preset,"
               "original_bytes,stored_bytes,ratio,compress_ms\n");

  size_t total_stored = 0;
  size_t failures = 0;
  auto t0 = std::chrono::steady_clock::now();

  for (size_t c = 0; c < kNumChunks; ++c) {
    auto in_buf = CLIO_IPC->AllocateBuffer(kChunkBytes);
    if (in_buf.IsNull()) { ++failures; continue; }
    std::memcpy(in_buf.ptr_, source.data() + c * kChunkBytes, kChunkBytes);
    ctp::ipc::ShmPtr<> blob_data = in_buf.shm_.template Cast<void>();

    const std::string blob_name = "chunk_" + std::to_string(c);
    clio::cte::core::Context ctx;
    // Dynamic: let NeuroPress choose. Nothing else needs setting here --
    // max_performance_ looks like a ratio-vs-speed knob but is inert on this
    // path. It only selects between Clio's own BestCompressTime and
    // BestCompressRatio (compressor_runtime.cc:867), and when NeuroPress
    // supplied the candidates the runtime takes its cost-model winner
    // directly (:1243) and never consults either.
    ctx.dynamic_compress_ = 0;

    auto task = compressor.AsyncDynamicSchedule(
        clio::run::PoolQuery::Local(), tag_id, blob_name, 0, kChunkBytes,
        blob_data, -1.0f, ctx, 0, cte_client->pool_id_);
    task.Wait();
    const int rc = task->return_code_;
    const auto res = task->context_;
    CLIO_IPC->FreeBuffer(in_buf);
    if (rc != 0) {
      std::cerr << "  chunk " << c << ": DynamicSchedule rc=" << rc
                << std::endl;
      ++failures;
      continue;
    }

    // Pull the stored blob back out -- this is the compressed representation
    // Clio persisted, header included, not a re-derived copy.
    auto szf = cte_client->AsyncGetBlobSize(tag_id, blob_name);
    szf.Wait();
    const size_t stored = szf->size_;
    if (stored == 0) { ++failures; continue; }

    auto out_buf = CLIO_IPC->AllocateBuffer(stored);
    if (out_buf.IsNull()) { ++failures; continue; }
    ctp::ipc::ShmPtr<> out_shm = out_buf.shm_.template Cast<void>();
    auto getf = cte_client->AsyncGetBlob(tag_id, blob_name, 0, stored, 0,
                                         out_shm);
    getf.Wait();
    if (getf->GetReturnCode() != 0) {
      CLIO_IPC->FreeBuffer(out_buf);
      ++failures;
      continue;
    }

    char fname[64];
    std::snprintf(fname, sizeof(fname), "chunk_%04zu.clioz", c);
    const std::string fpath = out_dir + "/" + fname;
    std::FILE *f = std::fopen(fpath.c_str(), "wb");
    if (!f || std::fwrite(out_buf.ptr_, 1, stored, f) != stored) {
      std::cerr << "  chunk " << c << ": write to " << fpath << " failed"
                << std::endl;
      if (f) std::fclose(f);
      CLIO_IPC->FreeBuffer(out_buf);
      ++failures;
      continue;
    }
    std::fclose(f);
    CLIO_IPC->FreeBuffer(out_buf);
    total_stored += stored;

    const uint32_t bits = static_cast<uint32_t>(res.compress_preset_);
    std::fprintf(mf, "%zu,%s,%s,%d,%d,%d,%u,%zu,%zu,%.6f,%.3f\n", c, fname,
                 AlgoName(res.compress_lib_), res.compress_lib_,
                 ((bits >> 24) & 1u) ? 1 : 0,
                 (((bits >> 8) & 0xFFu) != 0u) ? 1 : 0, bits & 0xFFu,
                 kChunkBytes, stored,
                 static_cast<double>(kChunkBytes) / static_cast<double>(stored),
                 res.actual_compress_time_ms_);
  }
  std::fclose(mf);

  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();

  std::cout << std::fixed << std::setprecision(3)
            << "\nCompressed " << (kNumChunks - failures) << " / " << kNumChunks
            << " chunks in " << elapsed * 1000.0 << " ms\n"
            << "  original : " << kDatasetBytes / (1024 * 1024) << " MiB\n"
            << "  stored   : " << total_stored / (1024.0 * 1024.0) << " MiB\n"
            << "  ratio    : "
            << static_cast<double>(kDatasetBytes) /
                   static_cast<double>(total_stored ? total_stored : 1)
            << "x\n"
            << "  manifest : " << manifest_path << std::endl;

  // ---- Verify: decompress every stored chunk and compare to the source ----
  //
  // Reads back through the compressor, which is what makes the directory
  // meaningful: the files are only useful if Clio can turn them back into the
  // original bytes. Compared byte-by-byte, not as floats -- float equality
  // would call +0.0 and -0.0 the same and would not be the guarantee a
  // lossless codec is supposed to give.
  size_t exact = 0, differing = 0, unreadable = 0, bad_bytes = 0;
  for (size_t c = 0; c < kNumChunks; ++c) {
    auto buf = CLIO_IPC->AllocateBuffer(kChunkBytes);
    if (buf.IsNull()) { ++unreadable; continue; }
    ctp::ipc::ShmPtr<> shm = buf.shm_.template Cast<void>();
    auto fut = compressor.AsyncDecompressExplicit(
        clio::run::PoolQuery::Local(), tag_id, "chunk_" + std::to_string(c), 0,
        kChunkBytes, 0, shm, cte_client->pool_id_);
    fut.Wait();
    if (fut->GetReturnCode() != 0) {
      ++unreadable;
      CLIO_IPC->FreeBuffer(buf);
      continue;
    }
    const char *got = static_cast<const char *>(buf.ptr_);
    const char *want = source.data() + c * kChunkBytes;
    if (std::memcmp(got, want, kChunkBytes) == 0) {
      ++exact;
    } else {
      ++differing;
      for (size_t i = 0; i < kChunkBytes; ++i) {
        if (got[i] != want[i]) ++bad_bytes;
      }
    }
    CLIO_IPC->FreeBuffer(buf);
  }

  std::cout << "\nVerification (decompress -> compare to source):\n"
            << "  exact      : " << exact << " / " << kNumChunks << "\n"
            << "  differing  : " << differing << " (" << bad_bytes
            << " bytes)\n"
            << "  unreadable : " << unreadable << std::endl;

  const bool ok = (exact == kNumChunks && failures == 0);
  std::cout << (ok ? "\nOK: every chunk compressed, stored, and restored "
                     "byte-for-byte."
                   : "\nFAILED: see counts above.")
            << std::endl;

  cudaFree(d_data);
  clio::run::CLIO_RUNTIME_FINALIZE();
  return ok ? 0 : 1;
}
