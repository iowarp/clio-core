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
 * @file corpus_sweep.cu
 * @brief Measure lossless GPU codec settings on whole files: compressed size,
 * compress / decompress time, and a bit-exact round trip. One CSV row per
 * (file, setting). Built for the 9800-file synthetic corpus, where each file
 * is one chunk.
 *
 * Measurement rules:
 *  - Every input is on the GPU before anything is timed; file reads and the
 *    H2D copy are never in a measurement.
 *  - A setting is built once and runs over every file before the next one is
 *    built, so only one codec's buffers are live at a time.
 *  - Before its first timed call at each input size, a setting does one
 *    untimed round trip at that size, so per-size scratch allocation and
 *    first-launch costs never land in a timed call.
 *  - comp_ms / decomp_ms: CUDA events on the codec's stream around the
 *    compress / decompress call only -- the codec's GPU time.
 *  - comp_wall_ms: host clock from the launch until the compressed size is
 *    known on the host. decomp_wall_ms: host clock from header parsing (nvcomp
 *    configure_decompression, the SPspeed/SPratio size check) to the end of
 *    decompression. Both include launch overhead and syncs.
 *  - Each value is the median of --reps timed round trips.
 *  - comp_bytes is the whole stream the decoder needs (nvcomp's container
 *    header and chunk table included). It must be equal in every rep, or the
 *    row fails as nondeterministic.
 *  - Before each decompression the output buffer is filled with 0xA5, and
 *    after it the output is compared byte for byte with the input on the GPU.
 *    A row is ok only if every rep round-trips exactly.
 *
 * Usage:
 *   corpus_sweep --dir DIR --list FILES --configs SPECS --out CSV [--reps N]
 *                [--chunk BYTES]
 * FILES lists one path per line, relative to DIR. SPECS lists one codec spec
 * per line ("gdeflate chunk=65536 level=5", see gpu_codecs.cuh); blank lines
 * and lines starting with '#' are skipped.
 * With --chunk, every file is cut into BYTES-sized chunks (the last may be
 * shorter) and each chunk is one input, named "path#k". A list line "path k"
 * then takes only chunk k of that file; a bare "path" takes all of them.
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gpu_codecs.cuh"

namespace {

using gpu_codecs::Codec;
using gpu_codecs::CodecSpec;
using Clock = std::chrono::steady_clock;

/** @brief Command-line options. */
struct Options {
  std::string dir, list, configs, out;
  int reps = 3;      ///< timed round trips per (file, setting)
  size_t chunk = 0;  ///< bytes per input chunk; 0 = whole files
};

/** @return the options; exits with usage on a missing or unknown flag. */
Options ParseArgs(int argc, char **argv) {
  Options o;
  auto usage = [] {
    std::fprintf(stderr,
                 "usage: corpus_sweep --dir DIR --list FILES --configs SPECS "
                 "--out CSV [--reps N] [--chunk BYTES]\n");
    std::exit(1);
  };
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (i + 1 >= argc) usage();
    const char *v = argv[++i];
    if (a == "--dir") o.dir = v;
    else if (a == "--list") o.list = v;
    else if (a == "--configs") o.configs = v;
    else if (a == "--out") o.out = v;
    else if (a == "--reps") o.reps = std::atoi(v);
    else if (a == "--chunk") o.chunk = std::strtoull(v, nullptr, 10);
    else usage();
  }
  if (o.dir.empty() || o.list.empty() || o.configs.empty() || o.out.empty() ||
      o.reps < 1) {
    usage();
  }
  return o;
}

/** @return the lines of a file that are neither blank nor '#' comments. */
std::vector<std::string> ReadLines(const std::string &path) {
  std::vector<std::string> v;
  std::ifstream f(path);
  if (!f) {
    std::fprintf(stderr, "cannot read %s\n", path.c_str());
    std::exit(1);
  }
  for (std::string l; std::getline(f, l);) {
    const size_t b = l.find_first_not_of(" \t\r");
    if (b == std::string::npos || l[b] == '#') continue;
    const size_t e = l.find_last_not_of(" \t\r");
    v.push_back(l.substr(b, e - b + 1));
  }
  return v;
}

/** @brief Every input file, resident in one device allocation. */
struct Corpus {
  std::vector<std::string> names;
  std::vector<size_t> offset, bytes;
  uint8_t *d_base = nullptr;
  size_t max_bytes = 0;
};

/** @brief One input: a byte range of one file. */
struct Input {
  std::string path, name;  ///< file relative to dir; CSV label
  size_t offset = 0, bytes = 0;
};

/** @return the size of dir/path in bytes; exits when it cannot be read. */
size_t FileBytes(const std::string &dir, const std::string &path) {
  std::ifstream f(dir + "/" + path, std::ios::binary | std::ios::ate);
  const std::streamoff sz = f ? static_cast<std::streamoff>(f.tellg()) : 0;
  if (sz <= 0) {
    std::fprintf(stderr, "cannot read %s\n", path.c_str());
    std::exit(3);
  }
  return static_cast<size_t>(sz);
}

/**
 * Turn list lines into inputs: whole files, or with chunk > 0 the file's
 * chunks ("path" = every chunk, "path k" = chunk k only).
 * @param dir   directory the list is relative to
 * @param lines list lines
 * @param chunk bytes per chunk; 0 = whole files
 */
std::vector<Input> ListInputs(const std::string &dir,
                              const std::vector<std::string> &lines,
                              size_t chunk) {
  std::vector<Input> v;
  for (const std::string &line : lines) {
    std::istringstream ss(line);
    std::string path;
    ss >> path;
    const size_t size = FileBytes(dir, path);
    if (chunk == 0) {
      v.push_back({path, path, 0, size});
      continue;
    }
    const size_t nchunks = (size + chunk - 1) / chunk;
    std::vector<size_t> ks;
    for (size_t k; ss >> k;) ks.push_back(k);
    if (ks.empty()) {
      for (size_t k = 0; k < nchunks; ++k) ks.push_back(k);
    }
    for (size_t k : ks) {
      if (k >= nchunks) {
        std::fprintf(stderr, "%s has no chunk %zu\n", path.c_str(), k);
        std::exit(3);
      }
      const size_t off = k * chunk;
      v.push_back({path, path + "#" + std::to_string(k), off,
                   std::min(chunk, size - off)});
    }
  }
  return v;
}

/**
 * Read every input and copy it to the GPU (untimed).
 * @param dir    directory the inputs' paths are relative to
 * @param inputs byte ranges to load
 * @return the device-resident corpus; exits when a read fails
 */
Corpus LoadCorpus(const std::string &dir, const std::vector<Input> &inputs) {
  Corpus c;
  size_t total = 0;
  for (const Input &in : inputs) {
    c.names.push_back(in.name);
    c.offset.push_back(total);
    c.bytes.push_back(in.bytes);
    c.max_bytes = std::max(c.max_bytes, in.bytes);
    total += (in.bytes + 255) / 256 * 256;  // keep inputs 256-B aligned
  }
  CUDA_CHECK(cudaMalloc(&c.d_base, total));
  std::vector<char> buf(c.max_bytes);
  for (size_t i = 0; i < inputs.size(); ++i) {
    std::ifstream f(dir + "/" + inputs[i].path, std::ios::binary);
    f.seekg(static_cast<std::streamoff>(inputs[i].offset));
    if (!f.read(buf.data(), static_cast<std::streamsize>(inputs[i].bytes))) {
      std::fprintf(stderr, "short read %s\n", inputs[i].name.c_str());
      std::exit(3);
    }
    CUDA_CHECK(cudaMemcpy(c.d_base + c.offset[i], buf.data(), inputs[i].bytes,
                          cudaMemcpyHostToDevice));
  }
  return c;
}

/** @brief Device scratch shared by every setting. */
struct Workspace {
  uint8_t *d_comp = nullptr;  ///< compressed stream, cap bytes
  size_t cap = 0;
  uint8_t *d_dec = nullptr;   ///< decompressed output, the largest input
  unsigned int *d_bad = nullptr;
  cudaEvent_t e0 = nullptr, e1 = nullptr;
};

/** @return milliseconds of host time since t0. */
double MsSince(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/** @return milliseconds between the workspace's two recorded events. */
double EventMs(const Workspace &ws) {
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, ws.e0, ws.e1));
  return ms;
}

/** @brief One timed compress + decompress of one input. */
struct Round {
  size_t comp_bytes = 0;
  double comp_ms = 0, comp_wall_ms = 0, decomp_ms = 0, decomp_wall_ms = 0;
  bool ok = false;
  std::string note;  ///< why the round failed; empty when ok
};

/**
 * Compress one input, decompress it, and verify the output on the GPU.
 * @param c  codec, already built
 * @param in device input, n bytes
 * @return the timings; ok only for a bit-exact round trip
 */
Round RunRound(Codec *c, const uint8_t *in, size_t n, Workspace *ws) {
  Round r;
  if (!c->Accepts(n)) {
    r.note = "size not accepted";
    return r;
  }
  cudaStream_t s = c->stream();
  try {
    c->PrepareCompress(n);
    CUDA_CHECK(cudaStreamSynchronize(s));
    Clock::time_point w0 = Clock::now();
    CUDA_CHECK(cudaEventRecord(ws->e0, s));
    c->Compress(in, n, ws->d_comp, ws->cap);
    CUDA_CHECK(cudaEventRecord(ws->e1, s));
    CUDA_CHECK(cudaEventSynchronize(ws->e1));
    r.comp_bytes = c->CompressedBytes(ws->d_comp);
    r.comp_wall_ms = MsSince(w0);
    r.comp_ms = EventMs(*ws);
    if (r.comp_bytes == 0 || r.comp_bytes > ws->cap) {
      r.note = "compress failed";
      return r;
    }
    // A codec that writes nothing must not pass on stale output.
    CUDA_CHECK(cudaMemsetAsync(ws->d_dec, 0xA5, n, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    w0 = Clock::now();
    c->PrepareDecompress(ws->d_comp, r.comp_bytes);
    CUDA_CHECK(cudaEventRecord(ws->e0, s));
    c->Decompress(ws->d_comp, r.comp_bytes, ws->d_dec, n);
    CUDA_CHECK(cudaEventRecord(ws->e1, s));
    CUDA_CHECK(cudaEventSynchronize(ws->e1));
    r.decomp_wall_ms = MsSince(w0);
    r.decomp_ms = EventMs(*ws);
    if (!c->DecompressOk()) {
      r.note = "decompress failed";
      return r;
    }
    r.ok = gpu_codecs::BytesEqualOnGpu(in, ws->d_dec, n, ws->d_bad, s);
    if (!r.ok) r.note = "output differs";
  } catch (const std::exception &e) {
    r.ok = false;
    r.note = std::string("exception: ") + e.what();
    cudaGetLastError();  // clear a non-sticky error so the next setting runs
  }
  return r;
}

/** @return the median of v (mean of the middle two for an even count). */
double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

/**
 * Time reps round trips of one input and reduce them to medians.
 * @return the medians; on any failed or size-changing rep, that rep instead
 */
Round Measure(Codec *c, const uint8_t *in, size_t n, int reps,
              Workspace *ws) {
  std::vector<double> cm, cw, dm, dw;
  Round first;
  for (int i = 0; i < reps; ++i) {
    Round r = RunRound(c, in, n, ws);
    if (!r.ok) return r;
    if (i == 0) first = r;
    if (r.comp_bytes != first.comp_bytes) {
      r.ok = false;
      r.note = "compressed size differs between reps";
      return r;
    }
    cm.push_back(r.comp_ms);
    cw.push_back(r.comp_wall_ms);
    dm.push_back(r.decomp_ms);
    dw.push_back(r.decomp_wall_ms);
  }
  first.comp_ms = Median(cm);
  first.comp_wall_ms = Median(cw);
  first.decomp_ms = Median(dm);
  first.decomp_wall_ms = Median(dw);
  return first;
}

/**
 * Grow the compressed-stream buffer to the codec's bound at every input size.
 * @param c      codec about to run
 * @param sizes  the distinct input sizes
 */
void FitWorkspace(Codec *c, const std::vector<size_t> &sizes, Workspace *ws) {
  size_t need = 0;
  for (size_t n : sizes) need = std::max(need, c->Bound(n));
  if (need <= ws->cap) return;
  if (ws->d_comp) CUDA_CHECK(cudaFree(ws->d_comp));
  CUDA_CHECK(cudaMalloc(&ws->d_comp, need));
  ws->cap = need;
}

/** @return s with commas and line breaks replaced, safe in a CSV field. */
std::string CsvSafe(std::string s) {
  for (char &ch : s) {
    if (ch == ',' || ch == '\n' || ch == '\r') ch = ';';
  }
  return s;
}

/** @brief Per-setting totals for the progress line. */
struct Totals {
  size_t ok = 0, failed = 0, in_bytes = 0, out_bytes = 0;
  double comp_ms = 0, decomp_ms = 0;
};

/**
 * Run one setting over every input and write its rows.
 * @param spec   the setting
 * @param corpus device-resident inputs
 * @param sizes  distinct input sizes, for warm-up and buffer sizing
 * @return the setting's totals; every row failed when it cannot be built
 */
Totals RunSetting(const CodecSpec &spec, const Corpus &corpus,
                  const std::vector<size_t> &sizes, int reps, Workspace *ws,
                  FILE *csv) {
  Totals t;
  std::unique_ptr<Codec> c;
  std::string build_error;
  try {
    c = gpu_codecs::MakeCodec(spec);
    if (!c) build_error = "unknown codec " + spec.base;
    else FitWorkspace(c.get(), sizes, ws);
  } catch (const std::exception &e) {
    build_error = std::string("setup: ") + e.what();
    cudaGetLastError();
  }
  std::map<size_t, bool> warmed;
  for (size_t i = 0; i < corpus.names.size(); ++i) {
    const size_t n = corpus.bytes[i];
    const uint8_t *in = corpus.d_base + corpus.offset[i];
    Round r;
    if (!build_error.empty()) {
      r.note = build_error;
    } else {
      if (!warmed[n]) {
        RunRound(c.get(), in, n, ws);  // untimed; result ignored
        warmed[n] = true;
      }
      r = Measure(c.get(), in, n, reps, ws);
    }
    std::fprintf(csv, "%s,%zu,%s,%s,%zu,%.6f,%.6f,%.6f,%.6f,%d,%d,%s\n",
                 corpus.names[i].c_str(), n, spec.base.c_str(),
                 spec.Settings().c_str(), r.comp_bytes, r.comp_ms,
                 r.comp_wall_ms, r.decomp_ms, r.decomp_wall_ms, reps,
                 r.ok ? 1 : 0, CsvSafe(r.note).c_str());
    if (!r.ok) {
      ++t.failed;
      continue;
    }
    ++t.ok;
    t.in_bytes += n;
    t.out_bytes += r.comp_bytes;
    t.comp_ms += r.comp_ms;
    t.decomp_ms += r.decomp_ms;
  }
  return t;
}

}  // namespace

int main(int argc, char **argv) {
  const Options o = ParseArgs(argc, argv);
  std::vector<CodecSpec> specs;
  for (const std::string &l : ReadLines(o.configs)) {
    specs.push_back(gpu_codecs::ParseSpec(l));
  }
  const Clock::time_point t0 = Clock::now();
  const Corpus corpus =
      LoadCorpus(o.dir, ListInputs(o.dir, ReadLines(o.list), o.chunk));
  std::vector<size_t> sizes = corpus.bytes;
  std::sort(sizes.begin(), sizes.end());
  sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
  std::fprintf(stderr, "%zu files, %zu settings, %d reps; loaded in %.0f ms\n",
               corpus.names.size(), specs.size(), o.reps, MsSince(t0));

  Workspace ws;
  CUDA_CHECK(cudaMalloc(&ws.d_dec, corpus.max_bytes));
  CUDA_CHECK(cudaMalloc(&ws.d_bad, sizeof(unsigned int)));
  CUDA_CHECK(cudaEventCreate(&ws.e0));
  CUDA_CHECK(cudaEventCreate(&ws.e1));
  FILE *csv = std::fopen(o.out.c_str(), "w");
  if (!csv) {
    std::fprintf(stderr, "cannot write %s\n", o.out.c_str());
    return 1;
  }
  std::fprintf(csv, "file,bytes,algorithm,settings,comp_bytes,comp_ms,"
                    "comp_wall_ms,decomp_ms,decomp_wall_ms,reps,ok,note\n");
  size_t failed = 0;
  for (size_t k = 0; k < specs.size(); ++k) {
    const Totals t = RunSetting(specs[k], corpus, sizes, o.reps, &ws, csv);
    std::fflush(csv);
    failed += t.failed;
    std::fprintf(stderr,
                 "[%zu/%zu] %-9s %-44s ok %zu fail %zu ratio %.3f "
                 "comp %.1f ms decomp %.1f ms (t=%.0f ms)\n",
                 k + 1, specs.size(), specs[k].base.c_str(),
                 specs[k].Settings().c_str(), t.ok, t.failed,
                 t.out_bytes ? double(t.in_bytes) / t.out_bytes : 0.0,
                 t.comp_ms, t.decomp_ms, MsSince(t0));
  }
  std::fclose(csv);
  if (failed) std::printf("failed rows: %zu\n", failed);
  // Nonzero on any failed row, so a SLURM afterok chain stops at a bad run.
  return failed ? 4 : 0;
}
