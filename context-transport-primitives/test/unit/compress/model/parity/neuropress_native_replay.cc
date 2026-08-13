/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_native_replay.cc
 * @brief Replay Clio's GPU->HDF5 e2e run through native NeuroPress.
 *
 * The companion to bin/neuropress_e2e. That run generates 1 GiB on the GPU,
 * writes it through the Clio HDF5 VOL in 4 MiB chunks, lets NeuroPress pick
 * a configuration per chunk with online learning on, and logs every
 * selection. It also dumps the exact bytes it compressed. This driver feeds
 * those same bytes, in the same order, to NeuroPress's own
 * gpucompress_compress() with ALGO_AUTO and online learning on, and logs its
 * selections in the same schema so the two are directly comparable.
 *
 * ORDER MATTERS and is not cosmetic. With online learning on, each chunk's
 * measured outcome updates the shared weights before later chunks are
 * scored, so the sequence of chunks IS part of the input. Clio's VOL queues
 * every chunk's DynamicSchedule asynchronously and drains at close
 * (clio_vol.cc:1899-1904), so its chunks do not necessarily complete in
 * index order. Replaying 0,1,2,... against a run that learned in some other
 * order would compare two different experiments. So this driver reads the
 * `seq` column out of Clio's log and replays in THAT order.
 *
 * Even then the two are not forced to agree: Clio can have several chunks in
 * flight at once, so one chunk's inference may read weights that a
 * concurrent chunk's update has not yet contributed to, which a strictly
 * sequential replay cannot reproduce. Run with --inference-only (and a Clio
 * run with learning disabled) for the deterministic comparison, where
 * selection is a pure function of each chunk's statistics and order drops
 * out entirely.
 *
 * NeuroPress is not a build dependency of Clio, so this is compiled against
 * a NeuroPress build out-of-tree; see the header comment in
 * parity/CMakeLists.txt for the pattern.
 */

#include <gpucompress.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr size_t kChunkBytes = 4ull * 1024 * 1024;

struct Row {
  long seq = -1;
  long chunk = -1;
};

/** Parse Clio's selection CSV for the replay order: (seq, chunk index). */
std::vector<Row> ReadClioOrder(const char *path) {
  std::vector<Row> rows;
  std::FILE *f = std::fopen(path, "r");
  if (!f) return rows;
  char line[4096];
  bool first = true;
  while (std::fgets(line, sizeof(line), f)) {
    if (first) { first = false; continue; }  // header
    // seq,blob,...  with blob ending in "/chunk_<n>"
    char *comma = std::strchr(line, ',');
    if (!comma) continue;
    *comma = '\0';
    Row r;
    r.seq = std::atol(line);
    const char *blob = comma + 1;
    const char *tail = std::strstr(blob, "/chunk_");
    if (!tail) continue;
    r.chunk = std::atol(tail + 7);
    rows.push_back(r);
  }
  std::fclose(f);
  std::sort(rows.begin(), rows.end(),
            [](const Row &a, const Row &b) { return a.seq < b.seq; });

  // Keep only each chunk's FIRST selection. A Clio run logs every chunk
  // twice: once from the write path, and once more when a later read misses
  // the cache and re-stages the chunk through DynamicSchedule
  // (clio_vol.cc:2478). The second pass is a cache-population artifact of
  // reading the file back, not a fresh decision about the data, and it runs
  // after the whole write is done -- so the write pass is what a replay
  // should reproduce.
  std::vector<Row> first_only;
  std::vector<bool> seen;
  for (const auto &r : rows) {
    if (r.chunk < 0) continue;
    if (static_cast<size_t>(r.chunk) >= seen.size()) {
      seen.resize(static_cast<size_t>(r.chunk) + 1, false);
    }
    if (seen[static_cast<size_t>(r.chunk)]) continue;
    seen[static_cast<size_t>(r.chunk)] = true;
    first_only.push_back(r);
  }
  return first_only;
}

}  // namespace

int main(int argc, char **argv) {
  const char *data_path = "/tmp/neuropress_e2e_data.bin";
  const char *clio_csv = "/tmp/neuropress_e2e_clio.csv";
  const char *out_csv = "/tmp/neuropress_e2e_native.csv";
  const char *weights = nullptr;
  bool inference_only = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--inference-only") {
      inference_only = true;
    } else if (a == "--data" && i + 1 < argc) {
      data_path = argv[++i];
    } else if (a == "--clio-csv" && i + 1 < argc) {
      clio_csv = argv[++i];
    } else if (a == "--out" && i + 1 < argc) {
      out_csv = argv[++i];
    } else if (a == "--weights" && i + 1 < argc) {
      weights = argv[++i];
    } else {
      std::fprintf(stderr,
                   "usage: %s [--data F] [--clio-csv F] [--out F] "
                   "[--weights model.nnwt] [--inference-only]\n",
                   argv[0]);
      return 2;
    }
  }
  if (!weights) {
    std::fprintf(stderr, "--weights <path to model.nnwt> is required\n");
    return 2;
  }

  if (gpucompress_init(weights) != GPUCOMPRESS_SUCCESS) {
    std::fprintf(stderr, "gpucompress_init(%s) failed\n", weights);
    return 1;
  }
  if (!gpucompress_nn_is_loaded()) {
    // init() takes a weights path, but be explicit rather than silently
    // ranking with the Q-table: without the NN this compares nothing.
    if (gpucompress_load_nn(weights) != GPUCOMPRESS_SUCCESS ||
        !gpucompress_nn_is_loaded()) {
      std::fprintf(stderr, "NN weights not loaded from %s\n", weights);
      return 1;
    }
  }
  gpucompress_set_selection_mode(GPUCOMPRESS_SELECT_NN);
  if (inference_only) {
    gpucompress_disable_online_learning();
  } else {
    gpucompress_enable_online_learning();
  }
  std::printf("native NeuroPress: NN loaded, online_learning=%d\n",
              gpucompress_online_learning_enabled());

  std::FILE *df = std::fopen(data_path, "rb");
  if (!df) {
    std::fprintf(stderr, "cannot open data file %s\n", data_path);
    return 1;
  }
  std::fseek(df, 0, SEEK_END);
  const long file_bytes = std::ftell(df);
  std::fseek(df, 0, SEEK_SET);
  const long num_chunks = file_bytes / static_cast<long>(kChunkBytes);
  std::printf("data: %ld bytes, %ld chunks of %zu\n", file_bytes, num_chunks,
              kChunkBytes);

  // Replay order: Clio's completion sequence when available, else index
  // order. Falling back silently would quietly compare two different
  // experiments, so say which one is in use.
  std::vector<Row> order = ReadClioOrder(clio_csv);
  if (order.empty()) {
    std::printf("no usable order in %s -- replaying in chunk-index order\n",
                clio_csv);
    for (long i = 0; i < num_chunks; ++i) order.push_back({i, i});
  } else {
    std::printf("replaying in Clio's completion order from %s (%zu rows)\n",
                clio_csv, order.size());
  }

  std::FILE *of = std::fopen(out_csv, "w");
  if (!of) {
    std::fprintf(stderr, "cannot open %s for writing\n", out_csv);
    return 1;
  }
  std::fprintf(of,
               "seq,chunk,action,algo_idx,quantize,shuffle,entropy,mad,"
               "second_deriv,compressed_size,ratio,pred_ratio,pred_ct_ms,"
               "pred_dt_ms,pred_psnr,actual_ct_ms,sgd_fired\n");

  std::vector<char> in(kChunkBytes);
  const size_t max_out = gpucompress_max_compressed_size(kChunkBytes);

  // Device buffers: gpucompress_compress() is a host-path stub in this
  // build, and the GPU entry point is the one the VOL uses anyway, so the
  // comparison runs through the same code path Clio's chunks take.
  void *d_in = nullptr, *d_out = nullptr;
  if (cudaMalloc(&d_in, kChunkBytes) != cudaSuccess ||
      cudaMalloc(&d_out, max_out) != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc failed\n");
    return 1;
  }

  gpucompress_config_t cfg = gpucompress_default_config();
  cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;  // NN picks per chunk
  cfg.preprocessing = GPUCOMPRESS_PREPROC_NONE;
  cfg.error_bound = 0.0;  // lossless, matching the Clio run

  long done = 0, failed = 0;
  for (const auto &r : order) {
    if (r.chunk < 0 || r.chunk >= num_chunks) continue;
    if (std::fseek(df, r.chunk * static_cast<long>(kChunkBytes), SEEK_SET) !=
        0) {
      ++failed;
      continue;
    }
    if (std::fread(in.data(), 1, kChunkBytes, df) != kChunkBytes) {
      ++failed;
      continue;
    }

    if (cudaMemcpy(d_in, in.data(), kChunkBytes, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      ++failed;
      continue;
    }
    size_t out_size = max_out;
    gpucompress_stats_t st;
    std::memset(&st, 0, sizeof(st));
    gpucompress_error_t rc =
        gpucompress_compress_gpu(d_in, kChunkBytes, d_out, &out_size, &cfg,
                                 &st, /*stream=*/nullptr);
    if (rc != GPUCOMPRESS_SUCCESS) {
      std::fprintf(stderr, "chunk %ld: gpucompress_compress rc=%d\n", r.chunk,
                   static_cast<int>(rc));
      ++failed;
      continue;
    }

    // nn_final_action is the action actually used; with exploration off it
    // equals nn_original_action. Decode it the way decodeAction does.
    const int action = st.nn_final_action;
    const int algo_idx = (action >= 0) ? (action % 8) : -1;
    const int quantize = (action >= 0) ? ((action / 8) % 2) : -1;
    const int shuffle = (action >= 0) ? ((action / 16) % 2) : -1;

    std::fprintf(of,
                 "%ld,%ld,%d,%d,%d,%d,%.10g,%.10g,%.10g,%zu,%.10g,%.10g,"
                 "%.10g,%.10g,%.10g,%.10g,%d\n",
                 r.seq, r.chunk, action, algo_idx, quantize, shuffle,
                 st.entropy_bits, st.mad, st.second_derivative,
                 st.compressed_size, st.compression_ratio,
                 st.predicted_ratio, st.predicted_comp_time_ms,
                 st.predicted_decomp_time_ms, st.predicted_psnr_db,
                 st.actual_comp_time_ms, st.sgd_fired);
    ++done;
  }
  std::fclose(of);
  cudaFree(d_in);
  cudaFree(d_out);
  std::fclose(df);

  std::printf("replayed %ld chunks (%ld failed) -> %s\n", done, failed,
              out_csv);
  gpucompress_cleanup();
  return failed == 0 ? 0 : 1;
}
