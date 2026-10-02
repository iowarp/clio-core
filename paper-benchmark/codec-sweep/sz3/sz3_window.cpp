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

/**
 * @file sz3_window.cpp
 * @brief SZ3 on one 4 MiB chunk across a window of T consecutive dumps, in
 * every SZ3 configuration and three layouts, timed in memory.
 *
 * Layouts (dims slowest first; the chunk is Z x Y x X, x fastest):
 *   3d        each dump on its own, Config(Z, Y, X); bytes and times summed
 *   4d-tslow  the window as one array [t][z][y][x], Config(T, Z, Y, X)
 *   4d-tfast  transposed to [z][y][x][t], Config(Z, Y, X, T); the transpose
 *             and its inverse are counted in the compress / decompress time
 * Configurations:
 *   interp_lorenzo  ALGO_INTERP_LORENZO (SZ3's default; cubic)
 *   interp_cubic    ALGO_INTERP, cubic          interp_linear  ALGO_INTERP, linear
 *   lorenzo_reg     ALGO_LORENZO_REG (1st-order Lorenzo + regression)
 *   lorenzo2_reg    ALGO_LORENZO_REG with 2nd-order Lorenzo as well
 *   lorenzo_only    ALGO_LORENZO_REG, regression off
 *   nopred          ALGO_NOPRED (quantization + entropy coding only)
 * Every run uses an absolute bound, one thread, and is decompressed and
 * checked value by value. One CSV row per (configuration, layout) on stdout:
 *   config,layout,bytes_in,bytes_out,comp_ms,decomp_ms,max_err_over_eb
 *
 *   sz3_window --files f1,..,fT --offset BYTES --shape Z,Y,X --eb ABS
 *              [--configs a,b,..] [--layouts a,b,..]
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "SZ3/api/sz.hpp"

namespace {

using Clock = std::chrono::steady_clock;

/** @return the comma-separated fields of s. */
std::vector<std::string> SplitCommas(const std::string &s) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  for (std::string t; std::getline(ss, t, ',');) {
    if (!t.empty()) v.push_back(t);
  }
  return v;
}

/** @brief Command-line options. */
struct Options {
  std::vector<std::string> files, configs, layouts;
  size_t offset = 0, z = 0, y = 0, x = 0;
  double eb = 0;
};

/** @return the options; exits with usage on anything missing. */
Options ParseArgs(int argc, char **argv) {
  Options o;
  o.configs = SplitCommas(
      "interp_lorenzo,interp_cubic,interp_linear,lorenzo_reg,lorenzo2_reg,"
      "lorenzo_only,nopred");
  o.layouts = SplitCommas("3d,4d-tslow,4d-tfast");
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--files") o.files = SplitCommas(v);
    else if (a == "--offset") o.offset = std::strtoull(v.c_str(), nullptr, 10);
    else if (a == "--eb") o.eb = std::atof(v.c_str());
    else if (a == "--configs") o.configs = SplitCommas(v);
    else if (a == "--layouts") o.layouts = SplitCommas(v);
    else if (a == "--shape") {
      const auto d = SplitCommas(v);
      if (d.size() == 3) {
        o.z = std::stoull(d[0]);
        o.y = std::stoull(d[1]);
        o.x = std::stoull(d[2]);
      }
    }
  }
  if (o.files.empty() || o.x == 0 || !(o.eb > 0)) {
    std::fprintf(stderr,
                 "usage: sz3_window --files f1,..,fT --offset BYTES "
                 "--shape Z,Y,X --eb ABS [--configs ..] [--layouts ..]\n");
    std::exit(1);
  }
  return o;
}

/**
 * Read one chunk of every dump into a [t][z][y][x] array.
 * @param o options (files, offset, shape)
 * @return T * Z * Y * X floats; exits on a short read
 */
std::vector<float> LoadWindow(const Options &o) {
  const size_t n = o.z * o.y * o.x;
  std::vector<float> w(n * o.files.size());
  for (size_t t = 0; t < o.files.size(); ++t) {
    std::ifstream f(o.files[t], std::ios::binary);
    f.seekg(static_cast<std::streamoff>(o.offset));
    if (!f.read(reinterpret_cast<char *>(w.data() + t * n),
                static_cast<std::streamsize>(n * sizeof(float)))) {
      std::fprintf(stderr, "short read %s\n", o.files[t].c_str());
      std::exit(3);
    }
  }
  return w;
}

/**
 * Apply a configuration name to an SZ3 config.
 * @return false for an unknown name
 */
bool SetAlgorithm(const std::string &name, SZ3::Config *c) {
  if (name == "interp_lorenzo") {
    c->cmprAlgo = SZ3::ALGO_INTERP_LORENZO;
  } else if (name == "interp_cubic" || name == "interp_linear") {
    c->cmprAlgo = SZ3::ALGO_INTERP;
    c->interpAlgo = name == "interp_cubic" ? SZ3::INTERP_ALGO_CUBIC
                                           : SZ3::INTERP_ALGO_LINEAR;
  } else if (name == "lorenzo_reg" || name == "lorenzo2_reg" ||
             name == "lorenzo_only") {
    c->cmprAlgo = SZ3::ALGO_LORENZO_REG;
    c->lorenzo2 = name == "lorenzo2_reg";
    c->regression = name != "lorenzo_only";
  } else if (name == "nopred") {
    c->cmprAlgo = SZ3::ALGO_NOPRED;
  } else {
    return false;
  }
  return true;
}

/** @brief Totals of one (configuration, layout) run. */
struct Result {
  size_t bytes_out = 0;
  double comp_ms = 0, decomp_ms = 0, max_err = 0;
  bool ok = true;
};

/**
 * Compress and decompress one array; adds bytes, times and the max error.
 * @param name  configuration name
 * @param dims  dims slowest first
 * @param data  the array, prod(dims) floats
 * @param eb    absolute error bound
 */
void RunOne(const std::string &name, const std::vector<size_t> &dims,
            const float *data, double eb, Result *r) {
  SZ3::Config c;
  c.setDims(dims.begin(), dims.end());
  c.errorBoundMode = SZ3::EB_ABS;
  c.absErrorBound = eb;
  c.openmp = false;
  SetAlgorithm(name, &c);
  Clock::time_point t0 = Clock::now();
  size_t cmp_size = 0;
  std::unique_ptr<char[]> cmp(SZ_compress<float>(c, data, cmp_size));
  r->comp_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0)
                    .count();
  SZ3::Config d;
  float *dec = nullptr;
  t0 = Clock::now();
  SZ_decompress<float>(d, cmp.get(), cmp_size, dec);
  r->decomp_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0)
                      .count();
  std::unique_ptr<float[]> keep(dec);
  r->bytes_out += cmp_size;
  if (d.num != c.num) {
    r->ok = false;
    return;
  }
  for (size_t i = 0; i < c.num; ++i) {
    r->max_err = std::max(r->max_err,
                          std::fabs(double(dec[i]) - double(data[i])));
  }
}

/** @return w ([t][s]) transposed to [s][t]; s = points per dump. */
std::vector<float> Transpose(const std::vector<float> &w, size_t t, size_t s) {
  std::vector<float> o(w.size());
  for (size_t i = 0; i < t; ++i) {
    for (size_t j = 0; j < s; ++j) o[j * t + i] = w[i * s + j];
  }
  return o;
}

/**
 * One configuration in one layout over the window.
 * @param w the window, [t][z][y][x]
 */
Result RunLayout(const Options &o, const std::string &cfg,
                 const std::string &layout, const std::vector<float> &w) {
  const size_t T = o.files.size(), n = o.z * o.y * o.x;
  Result r;
  if (layout == "3d") {
    for (size_t t = 0; t < T; ++t) RunOne(cfg, {o.z, o.y, o.x}, &w[t * n], o.eb, &r);
  } else if (layout == "4d-tslow") {
    RunOne(cfg, {T, o.z, o.y, o.x}, w.data(), o.eb, &r);
  } else if (layout == "4d-tfast") {
    Clock::time_point t0 = Clock::now();
    const std::vector<float> tf = Transpose(w, T, n);
    const double tr_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    RunOne(cfg, {o.z, o.y, o.x, T}, tf.data(), o.eb, &r);
    // Undoing the transpose costs the same pass again on the way out.
    r.comp_ms += tr_ms;
    r.decomp_ms += tr_ms;
  } else {
    r.ok = false;
  }
  return r;
}

}  // namespace

int main(int argc, char **argv) {
  const Options o = ParseArgs(argc, argv);
  const std::vector<float> w = LoadWindow(o);
  const size_t bytes_in = w.size() * sizeof(float);
  for (const std::string &cfg : o.configs) {
    SZ3::Config probe;
    if (!SetAlgorithm(cfg, &probe)) {
      std::fprintf(stderr, "unknown config %s\n", cfg.c_str());
      return 1;
    }
    for (const std::string &layout : o.layouts) {
      const Result r = RunLayout(o, cfg, layout, w);
      std::printf("%s,%s,%zu,%zu,%.3f,%.3f,%.6f%s\n", cfg.c_str(),
                  layout.c_str(), bytes_in, r.bytes_out, r.comp_ms,
                  r.decomp_ms, r.max_err / o.eb, r.ok ? "" : ",FAILED");
      std::fflush(stdout);
    }
  }
  return 0;
}
