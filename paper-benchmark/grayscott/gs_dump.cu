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
 * @file gs_dump.cu
 * @brief Run the GPU Gray-Scott simulation (the NeuroPress / ADIOS2 3D model,
 * context-transfer-engine/compressor/generator/grayscott) and dump U and V as
 * flat float32 files, in the layout the Nyx and VPIC dumps use, for the
 * replay benchmark (stage_full_workloads.py).
 *
 *   gs_dump --out DIR [--L 256] [--F 0.04] [--k 0.06075] [--noise 0]
 *           [--seed 42] [--steps 10000] [--dump-int 100] [--fields u,v]
 *
 * Writes DIR/plt<NNNNN>/fab0000_comp00_u.f32 and fab0000_comp01_v.f32 (frame
 * NNNNN = steps / dump-int, frame 0 = the initial condition) and DIR/gen.json
 * with every setting. One field is L^3 float32 (L = 256: 64 MiB).
 */

#include <cuda_runtime.h>
#include <sys/stat.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "grayscott_sim.h"

namespace gs = clio::cte::compressor::grayscott;

namespace {

/** @brief Command-line settings of one dump. */
struct Options {
  std::string out;
  int L = 256;
  float F = 0.04f;
  float k = 0.06075f;
  float noise = 0.0f;
  int seed = 42;
  int steps = 10000;
  int dump_int = 100;
  bool dump_u = true;
  bool dump_v = true;
};

/**
 * @brief Parse the command line.
 * @param argc argument count
 * @param argv arguments
 * @param o    the parsed settings
 * @return false on a usage error (a message was printed)
 */
bool Parse(int argc, char **argv, Options *o) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (i + 1 >= argc) {
      std::fprintf(stderr, "missing value for %s\n", a.c_str());
      return false;
    }
    const char *v = argv[++i];
    if (a == "--out") o->out = v;
    else if (a == "--L") o->L = std::atoi(v);
    else if (a == "--F") o->F = std::strtof(v, nullptr);
    else if (a == "--k") o->k = std::strtof(v, nullptr);
    else if (a == "--noise") o->noise = std::strtof(v, nullptr);
    else if (a == "--seed") o->seed = std::atoi(v);
    else if (a == "--steps") o->steps = std::atoi(v);
    else if (a == "--dump-int") o->dump_int = std::atoi(v);
    else if (a == "--fields") {
      o->dump_u = std::strstr(v, "u") != nullptr;
      o->dump_v = std::strstr(v, "v") != nullptr;
    } else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return false;
    }
  }
  if (o->out.empty() || o->L < 8 || o->steps < 0 || o->dump_int < 1 ||
      !(o->dump_u || o->dump_v)) {
    std::fprintf(stderr,
                 "usage: gs_dump --out DIR [--L 256] [--F 0.04] [--k 0.06075] "
                 "[--noise 0] [--seed 42] [--steps 10000] [--dump-int 100] "
                 "[--fields u,v]\n");
    return false;
  }
  return true;
}

/**
 * @brief Copy one device field to the host and write it as a flat file.
 * @param dev   device pointer to the field
 * @param n     elements
 * @param host  staging buffer (resized as needed)
 * @param path  output file
 * @return true when the whole field was written
 */
bool WriteField(const float *dev, size_t n, std::vector<float> *host,
                const std::string &path) {
  host->resize(n);
  if (cudaMemcpy(host->data(), dev, n * sizeof(float), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    std::fprintf(stderr, "cudaMemcpy failed for %s\n", path.c_str());
    return false;
  }
  FILE *f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) {
    std::fprintf(stderr, "cannot write %s: %s\n", path.c_str(), std::strerror(errno));
    return false;
  }
  const size_t w = std::fwrite(host->data(), sizeof(float), n, f);
  std::fclose(f);
  return w == n;
}

/**
 * @brief Write one frame's fields into OUT/plt<frame>/.
 * @return true on success
 */
bool DumpFrame(const Options &o, const gs::Simulation &sim, int frame,
               std::vector<float> *host) {
  char dir[64];
  std::snprintf(dir, sizeof(dir), "/plt%05d", frame);
  const std::string d = o.out + dir;
  if (mkdir(d.c_str(), 0755) != 0 && errno != EEXIST) return false;
  bool ok = true;
  if (o.dump_u) ok = ok && WriteField(sim.DeviceU(), sim.NumElems(), host,
                                      d + "/fab0000_comp00_u.f32");
  if (o.dump_v) ok = ok && WriteField(sim.DeviceV(), sim.NumElems(), host,
                                      d + "/fab0000_comp01_v.f32");
  return ok;
}

/** @brief Record every setting next to the dump (gen.json). */
void WriteGenJson(const Options &o, int frames) {
  FILE *f = std::fopen((o.out + "/gen.json").c_str(), "w");
  if (f == nullptr) return;
  std::fprintf(f,
               "{\"model\":\"kNeuroPress3D\",\"L\":%d,\"F\":%.9g,\"k\":%.9g,"
               "\"noise\":%.9g,\"seed\":%d,\"steps\":%d,\"dump_int\":%d,"
               "\"fields\":\"%s%s\",\"frames\":%d}\n",
               o.L, o.F, o.k, o.noise, o.seed, o.steps, o.dump_int,
               o.dump_u ? "u" : "", o.dump_v ? "v" : "", frames);
  std::fclose(f);
}

}  // namespace

/** @brief Run the simulation and dump every dump-int steps. */
int main(int argc, char **argv) {
  Options o;
  if (!Parse(argc, argv, &o)) return 2;
  if (mkdir(o.out.c_str(), 0755) != 0 && errno != EEXIST) {
    std::fprintf(stderr, "cannot create %s: %s\n", o.out.c_str(), std::strerror(errno));
    return 1;
  }
  gs::SimSettings s = gs::SimSettings::NeuroPress3D(o.L);
  s.F = o.F;
  s.k = o.k;
  s.noise = o.noise;
  s.seed = o.seed;
  s.steps = o.steps;
  gs::Simulation sim(s);
  if (!sim.Valid() || !sim.Init()) {
    std::fprintf(stderr, "simulation set-up failed (L=%d)\n", o.L);
    return 1;
  }
  std::vector<float> host;
  int frame = 0;
  if (!DumpFrame(o, sim, frame++, &host)) return 1;
  for (int done = 0; done < o.steps; done += o.dump_int) {
    const int n = (o.steps - done < o.dump_int) ? o.steps - done : o.dump_int;
    if (!sim.Run(n) || !DumpFrame(o, sim, frame++, &host)) return 1;
  }
  WriteGenJson(o, frame);
  std::printf("gs_dump: L=%d F=%g k=%g steps=%d every %d -> %d frames in %s\n", o.L,
              o.F, o.k, o.steps, o.dump_int, frame, o.out.c_str());
  return 0;
}
