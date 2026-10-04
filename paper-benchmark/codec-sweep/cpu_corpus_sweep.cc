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
 * @file cpu_corpus_sweep.cc
 * @brief The CPU counterpart of corpus_sweep.cu: measure lossless CPU codec
 * settings (cpu_codecs.h) on file chunks -- compressed size, compress /
 * decompress time, bit-exact round trip -- with the same inputs, list format
 * and CSV columns.
 *
 * Measurement rules:
 *  - Every input is in host memory before anything is timed; the output
 *    buffers are written once beforehand so page faults are never timed.
 *  - A setting is built once (contexts, per-thread instances, scratch) and
 *    runs over every input before the next one is built.
 *  - Before its first timed call at each input size, a setting does one
 *    untimed round trip at that size.
 *  - comp_ms / decomp_ms: host steady clock around the compress /
 *    decompress call only (with threads=N, the whole parallel region). A
 *    shuffle and its inverse run before / after, untimed. The *_wall_ms
 *    columns repeat the same values, for corpus_sweep compatibility.
 *  - Each value is the median of --reps timed round trips; comp_bytes must
 *    be equal in every rep.
 *  - Before each decompression the output is filled with 0xA5, and after it
 *    the output is compared byte for byte with the input.
 *
 * Usage:
 *   cpu_corpus_sweep --dir DIR --list FILES --configs SPECS --out CSV
 *                    [--reps N] [--chunk BYTES]
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "cpu_codecs.h"

namespace {

using cpu_codecs::Codec;
using cpu_codecs::CodecSpec;
using Clock = std::chrono::steady_clock;

/** @brief Command-line options. */
struct Options {
  std::string dir, list, configs, out;
  int reps = 3;      ///< timed round trips per (input, setting)
  size_t chunk = 0;  ///< bytes per input chunk; 0 = whole files
};

/** @return the options; exits with usage on a missing or unknown flag. */
Options ParseArgs(int argc, char **argv) {
  Options o;
  auto usage = [] {
    std::fprintf(stderr,
                 "usage: cpu_corpus_sweep --dir DIR --list FILES --configs "
                 "SPECS --out CSV [--reps N] [--chunk BYTES]\n");
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

/** @brief One input: a byte range of one file, loaded in host memory. */
struct Input {
  std::string path, name;  ///< file relative to dir; CSV label
  size_t offset = 0, bytes = 0;
  std::vector<uint8_t> data;
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
 * Turn list lines into inputs ("path" = every chunk, "path k" = chunk k) and
 * read each one into memory (untimed).
 * @param dir   directory the list is relative to
 * @param lines list lines
 * @param chunk bytes per chunk; 0 = whole files
 */
std::vector<Input> LoadInputs(const std::string &dir,
                              const std::vector<std::string> &lines,
                              size_t chunk) {
  std::vector<Input> v;
  for (const std::string &line : lines) {
    std::istringstream ss(line);
    std::string path;
    ss >> path;
    const size_t size = FileBytes(dir, path);
    if (chunk == 0) {
      v.push_back({path, path, 0, size, {}});
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
                   std::min(chunk, size - off), {}});
    }
  }
  for (Input &in : v) {
    in.data.resize(in.bytes);
    std::ifstream f(dir + "/" + in.path, std::ios::binary);
    f.seekg(static_cast<std::streamoff>(in.offset));
    if (!f.read(reinterpret_cast<char *>(in.data.data()),
                static_cast<std::streamsize>(in.bytes))) {
      std::fprintf(stderr, "short read %s\n", in.name.c_str());
      std::exit(3);
    }
  }
  return v;
}

/** @brief Output buffers shared by every setting, pre-faulted. */
struct Workspace {
  std::vector<uint8_t> comp, dec;
};

/** @return milliseconds of host time since t0. */
double MsSince(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/** @brief One timed compress + decompress of one input. */
struct Round {
  size_t comp_bytes = 0;
  double comp_ms = 0, decomp_ms = 0;
  bool ok = false;
  std::string note;  ///< why the round failed; empty when ok
};

/**
 * Compress one input, decompress it, and verify the output.
 * @param c  codec, already built
 * @param in input, n bytes
 * @param ws output buffers, large enough for the codec's bound
 * @return the timings; ok only for a bit-exact round trip
 */
Round RunRound(Codec *c, const uint8_t *in, size_t n, Workspace *ws) {
  Round r;
  try {
    const uint8_t *src = c->Preprocess(in, n);
    Clock::time_point t0 = Clock::now();
    r.comp_bytes = c->Compress(src, n, ws->comp.data(), ws->comp.size());
    r.comp_ms = MsSince(t0);
    if (r.comp_bytes == 0 || r.comp_bytes > ws->comp.size()) {
      r.note = "compress failed";
      return r;
    }
    uint8_t *dst = c->DecodeTarget(ws->dec.data());
    std::memset(ws->dec.data(), 0xA5, n);
    if (dst != ws->dec.data()) std::memset(dst, 0xA5, n);
    t0 = Clock::now();
    const bool dok = c->Decompress(ws->comp.data(), r.comp_bytes, dst, n);
    r.decomp_ms = MsSince(t0);
    if (!dok) {
      r.note = "decompress failed";
      return r;
    }
    c->Postprocess(ws->dec.data(), n);
    r.ok = std::memcmp(in, ws->dec.data(), n) == 0;
    if (!r.ok) r.note = "output differs";
  } catch (const std::exception &e) {
    r.ok = false;
    r.note = std::string("exception: ") + e.what();
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
  std::vector<double> cm, dm;
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
    dm.push_back(r.decomp_ms);
  }
  first.comp_ms = Median(cm);
  first.decomp_ms = Median(dm);
  return first;
}

/**
 * Grow and pre-fault the output buffers for a codec at every input size.
 * @param c     codec about to run
 * @param sizes the distinct input sizes
 */
void FitWorkspace(Codec *c, const std::vector<size_t> &sizes, Workspace *ws) {
  size_t need = 0, maxn = 0;
  for (size_t n : sizes) {
    need = std::max(need, c->Bound(n));
    maxn = std::max(maxn, n);
  }
  if (ws->comp.size() < need) ws->comp.assign(need, 0);
  if (ws->dec.size() < maxn) ws->dec.assign(maxn, 0);
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
 * @param inputs host-resident inputs
 * @param sizes  distinct input sizes, for warm-up and buffer sizing
 * @return the setting's totals; every row failed when it cannot be built
 */
Totals RunSetting(const CodecSpec &spec, const std::vector<Input> &inputs,
                  const std::vector<size_t> &sizes, int reps, Workspace *ws,
                  FILE *csv) {
  Totals t;
  std::unique_ptr<Codec> c;
  std::string build_error;
  try {
    c = cpu_codecs::MakeCodec(spec);
    if (!c) build_error = "unknown codec " + spec.base;
    else FitWorkspace(c.get(), sizes, ws);
  } catch (const std::exception &e) {
    build_error = std::string("setup: ") + e.what();
  }
  std::map<size_t, bool> warmed;
  for (const Input &in : inputs) {
    const size_t n = in.bytes;
    Round r;
    if (!build_error.empty()) {
      r.note = build_error;
    } else {
      if (!warmed[n]) {
        RunRound(c.get(), in.data.data(), n, ws);  // untimed; ignored
        warmed[n] = true;
      }
      r = Measure(c.get(), in.data.data(), n, reps, ws);
    }
    std::fprintf(csv, "%s,%zu,%s,%s,%zu,%.6f,%.6f,%.6f,%.6f,%d,%d,%s\n",
                 in.name.c_str(), n, spec.base.c_str(),
                 spec.Settings().c_str(), r.comp_bytes, r.comp_ms, r.comp_ms,
                 r.decomp_ms, r.decomp_ms, reps, r.ok ? 1 : 0,
                 CsvSafe(r.note).c_str());
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
    specs.push_back(cpu_codecs::ParseSpec(l));
  }
  const Clock::time_point t0 = Clock::now();
  const std::vector<Input> inputs =
      LoadInputs(o.dir, ReadLines(o.list), o.chunk);
  std::vector<size_t> sizes;
  for (const Input &in : inputs) sizes.push_back(in.bytes);
  std::sort(sizes.begin(), sizes.end());
  sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
  std::fprintf(stderr, "%zu inputs, %zu settings, %d reps; loaded in %.0f ms\n",
               inputs.size(), specs.size(), o.reps, MsSince(t0));
  FILE *csv = std::fopen(o.out.c_str(), "w");
  if (!csv) {
    std::fprintf(stderr, "cannot write %s\n", o.out.c_str());
    return 1;
  }
  std::fprintf(csv, "file,bytes,algorithm,settings,comp_bytes,comp_ms,"
                    "comp_wall_ms,decomp_ms,decomp_wall_ms,reps,ok,note\n");
  Workspace ws;
  size_t failed = 0;
  for (size_t k = 0; k < specs.size(); ++k) {
    const Totals t = RunSetting(specs[k], inputs, sizes, o.reps, &ws, csv);
    std::fflush(csv);
    failed += t.failed;
    std::fprintf(stderr,
                 "[%zu/%zu] %-7s %-46s ok %zu fail %zu ratio %.3f "
                 "comp %.1f ms decomp %.1f ms (t=%.0f ms)\n",
                 k + 1, specs.size(), specs[k].base.c_str(),
                 specs[k].Settings().c_str(), t.ok, t.failed,
                 t.out_bytes ? double(t.in_bytes) / t.out_bytes : 0.0,
                 t.comp_ms, t.decomp_ms, MsSince(t0));
  }
  std::fclose(csv);
  if (failed) std::printf("failed rows: %zu\n", failed);
  return failed ? 4 : 0;
}
