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

// NeuroPress prepends this much metadata to every compressed buffer
// ("Output includes 64-byte header with metadata", gpucompress.h).
constexpr size_t kNativeHeaderBytes = 64;

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
  bool zero_output = false;
  long dump_chunk = -1;
  bool flush_cache = false;
  const char *clio_payload_dir = nullptr;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--inference-only") {
      inference_only = true;
    } else if (a == "--dump-chunk" && i + 1 < argc) {
      dump_chunk = std::atol(argv[++i]);
    } else if (a == "--clio-payloads" && i + 1 < argc) {
      // Directory of <prefix>chunkN.bin payloads dumped by the Clio run.
      // Each is grafted behind the header native just produced for the same
      // chunk and decoded, which is the only way to establish that CLIO's
      // compressed artifact restores the original -- Clio's own read path
      // can be served by the uncompressed HDF5 copy and never decompress.
      clio_payload_dir = argv[++i];
    } else if (a == "--flush-cache") {
      // Diagnostic: evict the cached nvcomp managers before every chunk, so
      // each compress starts from a fresh manager instead of one that has
      // already processed other chunks.
      flush_cache = true;
    } else if (a == "--zero-output") {
      // Diagnostic: clear the output buffer before every compress. If the
      // payload hashes change when this is on, the codec is leaving bytes
      // in its output untouched and the "difference" against another
      // implementation is stale allocation content, not a different
      // encoding.
      zero_output = true;
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
  // `checksum` makes the shared-input claim measurable instead of assumed:
  // both sides hash the chunk they actually compressed, so a comparison can
  // prove the two consumed the same bytes rather than trusting that the
  // dump and the write path saw the same buffer.
  std::fprintf(of,
               "seq,chunk,action,algo_idx,quantize,shuffle,entropy,mad,"
               "second_deriv,compressed_size,ratio,pred_ratio,pred_ct_ms,"
               "pred_dt_ms,pred_psnr,actual_ct_ms,sgd_fired,checksum,"
               "payload_hash,payload_size,mse,mean,variance,kurtosis,"
               "restored_hash,bytes_differ,clio_decode_differ\n");

  std::vector<char> in(kChunkBytes);
  const size_t max_out = gpucompress_max_compressed_size(kChunkBytes);

  // Device buffers: gpucompress_compress() is a host-path stub in this
  // build, and the GPU entry point is the one the VOL uses anyway, so the
  // comparison runs through the same code path Clio's chunks take.
  void *d_in = nullptr, *d_out = nullptr, *d_dec = nullptr, *d_in2 = nullptr;
  if (cudaMalloc(&d_in, kChunkBytes) != cudaSuccess ||
      cudaMalloc(&d_out, max_out) != cudaSuccess ||
      cudaMalloc(&d_dec, kChunkBytes) != cudaSuccess ||
      cudaMalloc(&d_in2, max_out + 4096) != cudaSuccess) {
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
    if (flush_cache) gpucompress_flush_manager_cache();
    if (zero_output) cudaMemset(d_out, 0, max_out);
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

    // gpucompress_compress_gpu() leaves entropy/mad/second_derivative zeroed
    // in gpucompress_stats_t, so ask for them explicitly. They are pure
    // properties of the data, independent of the configuration chosen, which
    // is exactly what makes them comparable against Clio's.
    double n_entropy = 0.0, n_mad = 0.0, n_deriv = 0.0;
    if (gpucompress_compute_stats_gpu(d_in, kChunkBytes, &n_entropy, &n_mad,
                                      &n_deriv) != GPUCOMPRESS_SUCCESS) {
      n_entropy = n_mad = n_deriv = 0.0;
    }

    // FNV-1a over the chunk actually fed to the compressor.
    unsigned long long checksum = 1469598103934665603ull;
    for (size_t k = 0; k < kChunkBytes; ++k) {
      checksum ^= static_cast<unsigned char>(in[k]);
      checksum *= 1099511628211ull;
    }

    // Hash the codec payload only: NeuroPress prepends a 64-byte metadata
    // header (gpucompress.h), Clio writes its own 24-byte one separately, so
    // hashing the framed buffers would compare two different envelopes
    // around what may well be identical bytes.
    unsigned long long payload_hash = 0;
    size_t payload_size = 0;
    if (out_size > kNativeHeaderBytes) {
      payload_size = out_size - kNativeHeaderBytes;
      std::vector<char> payload(payload_size);
      if (cudaMemcpy(payload.data(),
                     static_cast<const char *>(d_out) + kNativeHeaderBytes,
                     payload_size, cudaMemcpyDeviceToHost) == cudaSuccess) {
        payload_hash = 1469598103934665603ull;
        for (size_t k = 0; k < payload_size; ++k) {
          payload_hash ^= static_cast<unsigned char>(payload[k]);
          payload_hash *= 1099511628211ull;
        }
        // Same diagnostic as the Clio side: dump one chunk for a byte diff.
        if (dump_chunk >= 0 && r.chunk == dump_chunk) {
          char dp[512];
          std::snprintf(dp, sizeof(dp), "%s.chunk%ld.bin", out_csv, r.chunk);
          if (std::FILE *df = std::fopen(dp, "wb")) {
            std::fwrite(payload.data(), 1, payload_size, df);
            std::fclose(df);
          }
          // Also dump the FULL framed buffer, so the header can be located
          // rather than assumed to be 64 bytes.
          std::vector<char> full(out_size);
          if (cudaMemcpy(full.data(), d_out, out_size,
                         cudaMemcpyDeviceToHost) == cudaSuccess) {
            std::snprintf(dp, sizeof(dp), "%s.chunk%ld.full.bin", out_csv,
                          r.chunk);
            if (std::FILE *df = std::fopen(dp, "wb")) {
              std::fwrite(full.data(), 1, out_size, df);
              std::fclose(df);
            }
          }
        }
      }
    }

    // Round trip this chunk and measure the reconstruction directly rather
    // than asserting losslessness from the configuration.
    double mse = -1.0, mean = 0.0, variance = 0.0, kurtosis = 0.0;
    unsigned long long restored_hash = 0;
    size_t bytes_differ = kChunkBytes;  // until proven otherwise
    {
      size_t dec_size = kChunkBytes;
      if (gpucompress_decompress_gpu(d_out, out_size, d_dec, &dec_size,
                                     nullptr) == GPUCOMPRESS_SUCCESS &&
          dec_size == kChunkBytes) {
        std::vector<char> dec(kChunkBytes);
        if (cudaMemcpy(dec.data(), d_dec, kChunkBytes,
                       cudaMemcpyDeviceToHost) == cudaSuccess) {
          // Byte-by-byte against the source, and a hash of the restored
          // bytes so this chunk can be matched against Clio's restored
          // chunk directly -- not just each side against the source.
          // MSE below is kept as a separate, weaker numeric view; it can
          // read 0 for buffers that are not bit-identical (-0.0 vs +0.0),
          // which is exactly why the byte compare is the one that counts.
          bytes_differ = 0;
          if (std::memcmp(dec.data(), in.data(), kChunkBytes) != 0) {
            for (size_t k = 0; k < kChunkBytes; ++k) {
              if (dec[k] != in[k]) ++bytes_differ;
            }
          }
          restored_hash = 1469598103934665603ull;
          for (size_t k = 0; k < kChunkBytes; ++k) {
            restored_hash ^= static_cast<unsigned char>(dec[k]);
            restored_hash *= 1099511628211ull;
          }
          const float *a = reinterpret_cast<const float *>(in.data());
          const float *b = reinterpret_cast<const float *>(dec.data());
          const size_t n = kChunkBytes / sizeof(float);
          double se = 0.0, sum = 0.0;
          for (size_t k = 0; k < n; ++k) {
            const double d = static_cast<double>(a[k]) - b[k];
            se += d * d;
            sum += b[k];
          }
          mse = se / static_cast<double>(n);
          mean = sum / static_cast<double>(n);
          double m2 = 0.0, m4 = 0.0;
          for (size_t k = 0; k < n; ++k) {
            const double d = b[k] - mean;
            m2 += d * d;
            m4 += d * d * d * d;
          }
          m2 /= static_cast<double>(n);
          m4 /= static_cast<double>(n);
          variance = m2;
          kurtosis = (m2 > 0.0) ? (m4 / (m2 * m2)) : 0.0;  // raw, 3 = normal
        }
      }
    }

    // Cross-decode: does CLIO's compressed payload for this chunk restore
    // the source exactly, using native's decoder?
    long clio_decode_differ = -1;  // -1 = not attempted
    if (clio_payload_dir && out_size > kNativeHeaderBytes) {
      char pp[1024];
      std::snprintf(pp, sizeof(pp), "%s%ld.bin", clio_payload_dir, r.chunk);
      if (std::FILE *pf = std::fopen(pp, "rb")) {
        std::fseek(pf, 0, SEEK_END);
        const long plen = std::ftell(pf);
        std::fseek(pf, 0, SEEK_SET);
        std::vector<char> cp(plen);
        if (std::fread(cp.data(), 1, plen, pf) == (size_t)plen) {
          std::vector<char> framed(kNativeHeaderBytes + plen);
          std::vector<char> hdr(kNativeHeaderBytes);
          cudaMemcpy(hdr.data(), d_out, kNativeHeaderBytes,
                     cudaMemcpyDeviceToHost);
          std::memcpy(framed.data(), hdr.data(), kNativeHeaderBytes);
          std::memcpy(framed.data() + kNativeHeaderBytes, cp.data(), plen);
          cudaMemcpy(d_in2, framed.data(), framed.size(),
                     cudaMemcpyHostToDevice);
          size_t ds = kChunkBytes;
          if (gpucompress_decompress_gpu(d_in2, framed.size(), d_dec, &ds,
                                         nullptr) == GPUCOMPRESS_SUCCESS &&
              ds == kChunkBytes) {
            std::vector<char> dec2(kChunkBytes);
            cudaMemcpy(dec2.data(), d_dec, kChunkBytes,
                       cudaMemcpyDeviceToHost);
            clio_decode_differ = 0;
            if (std::memcmp(dec2.data(), in.data(), kChunkBytes) != 0) {
              for (size_t k = 0; k < kChunkBytes; ++k) {
                if (dec2[k] != in[k]) ++clio_decode_differ;
              }
            }
          } else {
            clio_decode_differ = -2;  // decode failed outright
          }
        }
        std::fclose(pf);
      }
    }

    // nn_final_action is the action actually used; with exploration off it
    // equals nn_original_action. Decode it the way decodeAction does.
    const int action = st.nn_final_action;
    const int algo_idx = (action >= 0) ? (action % 8) : -1;
    const int quantize = (action >= 0) ? ((action / 8) % 2) : -1;
    const int shuffle = (action >= 0) ? ((action / 16) % 2) : -1;

    std::fprintf(of,
                 "%ld,%ld,%d,%d,%d,%d,%.10g,%.10g,%.10g,%zu,%.10g,%.10g,"
                 "%.10g,%.10g,%.10g,%.10g,%d,%llu,%llu,%zu,%.10g,%.10g,"
                 "%.10g,%.10g,%llu,%zu,%ld\n",
                 r.seq, r.chunk, action, algo_idx, quantize, shuffle,
                 n_entropy, n_mad, n_deriv,
                 st.compressed_size, st.compression_ratio,
                 st.predicted_ratio, st.predicted_comp_time_ms,
                 st.predicted_decomp_time_ms, st.predicted_psnr_db,
                 st.actual_comp_time_ms, st.sgd_fired, checksum,
                 payload_hash, payload_size, mse, mean, variance, kurtosis,
                 restored_hash, bytes_differ, clio_decode_differ);
    ++done;
  }
  std::fclose(of);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_dec);
  std::fclose(df);

  std::printf("replayed %ld chunks (%ld failed) -> %s\n", done, failed,
              out_csv);
  gpucompress_cleanup();
  return failed == 0 ? 0 : 1;
}
