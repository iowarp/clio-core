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
 * @file compress_load.cc
 * Background compression load for the DTSchedule interference benchmark.
 *
 * --threads workers each compress 1 MiB chunks of --input (cycled) with
 * --codec/--preset (the ctp codecs dtschedule uses) for --seconds, then
 * print one summary line:
 *
 *   compress_load threads=T codec=C preset=P seconds=S mb_s=X ratio=R
 *
 * where mb_s is input MB compressed per second across all threads.
 */

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <clio_ctp/compress/compress_factory.h>

namespace {

/** Command-line options. */
struct Options {
  int threads = 4;               ///< Compression threads
  std::string codec = "zstd";    ///< ctp library name
  int preset = 1;                ///< 0 fast, 1 balanced, 2 best (factory enum)
  double seconds = 30.0;         ///< How long to run
  std::string input;             ///< Data file (empty = synthetic field)
  size_t chunk = 1u << 20;       ///< Bytes per compress call
};

/**
 * Parse --key value pairs.
 * @param argc Argument count
 * @param argv Arguments
 * @return Parsed options
 */
Options Parse(int argc, char **argv) {
  Options o;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--threads") o.threads = std::atoi(v.c_str());
    else if (k == "--codec") o.codec = v;
    else if (k == "--preset") o.preset = std::atoi(v.c_str());
    else if (k == "--seconds") o.seconds = std::atof(v.c_str());
    else if (k == "--input") o.input = v;
    else if (k == "--chunk-kb") o.chunk = std::strtoull(v.c_str(), nullptr, 10) << 10;
  }
  return o;
}

/**
 * Load up to 64 MiB of input, or synthesize a smooth noisy float field.
 * @param path File to read (empty = synthetic)
 * @return Data buffer (a multiple of 1 MiB)
 */
std::vector<char> LoadData(const std::string &path) {
  constexpr size_t kMax = 64u << 20;
  std::vector<char> data;
  if (!path.empty()) {
    std::ifstream in(path, std::ios::binary);
    data.resize(kMax);
    in.read(data.data(), static_cast<std::streamsize>(kMax));
    data.resize(static_cast<size_t>(in.gcount()) & ~((1u << 20) - 1));
  }
  if (data.empty()) {
    data.resize(kMax);
    auto *f = reinterpret_cast<float *>(data.data());
    for (size_t i = 0; i < kMax / sizeof(float); ++i) {
      f[i] = 20.0f + 5.0f * static_cast<float>(i % 4096) / 4096.0f +
             0.01f * static_cast<float>(std::rand() % 100);
    }
  }
  return data;
}

/**
 * One worker: compress chunks until the deadline.
 * @param o Options
 * @param data Input
 * @param worker Worker index (staggers the start chunk)
 * @param deadline Stop time
 * @param in_bytes Accumulates input bytes compressed
 * @param out_bytes Accumulates compressed bytes
 */
void Worker(const Options &o, const std::vector<char> &data, int worker,
            std::chrono::steady_clock::time_point deadline,
            std::atomic<uint64_t> *in_bytes, std::atomic<uint64_t> *out_bytes) {
  const size_t kChunk = o.chunk;
  auto codec = ctp::CompressionFactory::GetPreset(
      o.codec, static_cast<ctp::CompressionPreset>(o.preset));
  if (!codec) return;
  std::vector<char> out(kChunk + (kChunk >> 1) + (64u << 10));
  const size_t nchunks = data.size() / kChunk;
  size_t c = static_cast<size_t>(worker) % nchunks;
  uint64_t in_local = 0, out_local = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    size_t out_size = out.size();
    char *src = const_cast<char *>(data.data() + c * kChunk);
    if (codec->Compress(out.data(), out_size, src, kChunk)) {
      in_local += kChunk;
      out_local += out_size;
    }
    c = (c + 1) % nchunks;
  }
  in_bytes->fetch_add(in_local);
  out_bytes->fetch_add(out_local);
}

}  // namespace

/**
 * Entry point: run the workers for --seconds and print the summary.
 * @param argc Argument count
 * @param argv See Parse
 * @return 0 on success, 1 if the codec is unavailable
 */
int main(int argc, char **argv) {
  Options o = Parse(argc, argv);
  if (!ctp::CompressionFactory::GetPreset(
          o.codec, static_cast<ctp::CompressionPreset>(o.preset))) {
    std::fprintf(stderr, "compress_load: codec %s unavailable\n",
                 o.codec.c_str());
    return 1;
  }
  std::vector<char> data = LoadData(o.input);
  std::atomic<uint64_t> in_bytes{0}, out_bytes{0};
  const auto t0 = std::chrono::steady_clock::now();
  const auto deadline =
      t0 + std::chrono::microseconds(static_cast<int64_t>(o.seconds * 1e6));
  std::vector<std::thread> pool;
  for (int w = 0; w < o.threads; ++w) {
    pool.emplace_back(Worker, std::cref(o), std::cref(data), w, deadline,
                      &in_bytes, &out_bytes);
  }
  for (auto &t : pool) t.join();
  const double s = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0).count();
  std::printf("compress_load threads=%d codec=%s preset=%d chunk_kb=%zu "
              "seconds=%.2f mb_s=%.1f ratio=%.2f\n",
              o.threads, o.codec.c_str(), o.preset, o.chunk >> 10, s,
              static_cast<double>(in_bytes.load()) / 1e6 / s,
              out_bytes.load() ? static_cast<double>(in_bytes.load()) /
                                     static_cast<double>(out_bytes.load())
                               : 0.0);
  return 0;
}
