/* Gray-Scott -> HDF5, GPU-resident, mirroring upstream's structure: run the
   simulation on the device, then hand H5Dwrite the device pointer directly.
   Uses the ported 3D simulation (compressor/generator/grayscott).

   Three modes, because proving the READ half of the stack needs a process
   that never saw the data:

     write   simulate, write HDF5 through Clio, and dump a plain reference
     verify  write, then read back in this same process
     read    read only -- no simulation, no writes -- and compare against the
             reference an earlier 'write' left behind

   'read' is the one that carries weight. A verify inside the writing process
   can be served from whatever that process already holds; a separate process
   has to fetch the stored bytes and invert the codec to produce anything at
   all. The reference is written with plain POSIX I/O, outside HDF5 and so
   outside Clio, because comparing the stack against itself would pass even if
   every codec were broken. */
#include <hdf5.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include "grayscott_sim.h"

namespace gs = clio::cte::compressor::grayscott;

/* Reference layout: [nsets][nelems] then nsets * (name[64] + payload). Flat and
   self-describing enough that the reader needs no argument beyond the path. */
static bool ref_write(const std::string &path,
                      const std::vector<std::string> &names,
                      const std::vector<std::vector<float>> &data) {
  FILE *fp = std::fopen(path.c_str(), "wb");
  if (!fp) return false;
  uint64_t n = names.size(), e = data.empty() ? 0 : data[0].size();
  bool ok = std::fwrite(&n, sizeof(n), 1, fp) == 1 &&
            std::fwrite(&e, sizeof(e), 1, fp) == 1;
  for (size_t i = 0; ok && i < names.size(); ++i) {
    char nm[64] = {0};
    std::snprintf(nm, sizeof(nm), "%s", names[i].c_str());
    ok = std::fwrite(nm, sizeof(nm), 1, fp) == 1 &&
         std::fwrite(data[i].data(), sizeof(float), e, fp) == e;
  }
  return (std::fclose(fp) == 0) && ok;
}

static bool ref_read(const std::string &path, std::vector<std::string> &names,
                     std::vector<std::vector<float>> &data) {
  FILE *fp = std::fopen(path.c_str(), "rb");
  if (!fp) return false;
  uint64_t n = 0, e = 0;
  bool ok = std::fread(&n, sizeof(n), 1, fp) == 1 &&
            std::fread(&e, sizeof(e), 1, fp) == 1;
  for (uint64_t i = 0; ok && i < n; ++i) {
    char nm[64] = {0};
    std::vector<float> buf(e);
    ok = std::fread(nm, sizeof(nm), 1, fp) == 1 &&
         std::fread(buf.data(), sizeof(float), e, fp) == e;
    if (ok) { names.emplace_back(nm); data.push_back(std::move(buf)); }
  }
  std::fclose(fp);
  return ok;
}

/* Bit-exact is the right bar: every codec selected here is lossless, so any
   difference is a defect rather than tolerable error. The worst delta is
   reported anyway -- "off by 1e-7 everywhere" and "one wild value" are
   different bugs and a mismatch count cannot tell them apart. */
static size_t compare(const std::vector<float> &got,
                      const std::vector<float> &want, const char *name) {
  size_t bad = 0, worst_at = 0;
  double worst = 0.0;
  for (size_t j = 0; j < got.size() && j < want.size(); ++j) {
    if (std::memcmp(&got[j], &want[j], sizeof(float)) != 0) {
      ++bad;
      double d = std::fabs((double)got[j] - (double)want[j]);
      if (d > worst) { worst = d; worst_at = j; }
    }
  }
  if (bad == 0) {
    std::printf("  verify %s: %zu elems OK (bit-exact)\n", name, got.size());
  } else {
    std::printf("  verify %s: %zu/%zu MISMATCH, worst |d|=%g at [%zu] "
                "(got %g want %g)\n", name, bad, got.size(), worst, worst_at,
                (double)got[worst_at], (double)want[worst_at]);
  }
  return bad;
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "/tmp/np_h5/gs3d.h5";
  const int L      = argc > 2 ? atoi(argv[2]) : 128;
  const int steps  = argc > 3 ? atoi(argv[3]) : 100;
  const int gap    = argc > 4 ? atoi(argv[4]) : 25;
  const char *mode = argc > 5 ? argv[5] : "write";

  const bool do_write = std::strcmp(mode, "read") != 0;
  const bool do_read  = std::strcmp(mode, "verify") == 0 ||
                        std::strcmp(mode, "read") == 0;
  /* serve: write, then stay alive holding the tier so a SEPARATE process can
     read it. Needed because a DynamicScheduleTask sent to a standalone daemon
     arrives default-constructed (blob='', bytes=0) and every write is rejected,
     so the writer has to be the runtime host -- and the tier is RAM, so it dies
     with that process. Keeping the writer parked is what lets the reader be a
     genuinely separate process. */
  const bool serve = std::strcmp(mode, "serve") == 0;
  const std::string ref_path = std::string(path) + ".ref";

  std::vector<std::string> names;
  std::vector<std::vector<float>> expected;

  if (do_write) {
    /* The regime that sustains a pattern (settings.json), not the shipped
       defaults, whose V field goes extinct by ~1000 steps. */
    auto cfg = gs::SimSettings::NeuroPress3D(L);
    cfg.Du = 0.2f; cfg.Dv = 0.1f; cfg.F = 0.02f; cfg.k = 0.048f; cfg.dt = 1.0f;
    /* GS_REGIME selects the (F, k) pair, which is what moves compressibility:
       the pattern regime IS the workload variable here, and a benchmark that
       cannot change it is measuring one dataset. Default keeps the settings
       above, so an existing run is unchanged. */
    if (const char *reg = std::getenv("GS_REGIME")) {
      const std::string r(reg);
      if (r == "stripes")   { cfg.F = 0.035f; cfg.k = 0.065f; }
      else if (r == "chaos"){ cfg.F = 0.014f; cfg.k = 0.045f; }
      else if (r == "sparse"){ cfg.F = 0.04f;  cfg.k = 0.065f; }
      else if (r != "spots" && !r.empty()) {
        std::fprintf(stderr, "unknown GS_REGIME '%s' (spots|stripes|chaos|sparse)\n", reg);
        return 1;
      }
      std::printf("regime %s: F=%.4f k=%.5f\n", reg, cfg.F, cfg.k);
    }
    gs::Simulation sim(cfg);
    if (!sim.Valid() || !sim.Init()) { std::fprintf(stderr, "sim init\n"); return 1; }
    std::printf("Gray-Scott %d^3 (%.1f MiB/field), %d steps, snapshot every %d\n",
                L, sim.NumBytes() / 1048576.0, steps, gap);

    hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (f < 0) { std::fprintf(stderr, "H5Fcreate\n"); return 1; }
    hsize_t dims[3] = {(hsize_t)L, (hsize_t)L, (hsize_t)L};
    hid_t space = H5Screate_simple(3, dims, NULL);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk[3] = {(hsize_t)L, (hsize_t)L, (hsize_t)(L / 4 ? L / 4 : 1)};
    H5Pset_chunk(dcpl, 3, chunk);

    int written = 0;
    for (int s = gap; s <= steps; s += gap) {
      if (!sim.Run(gap)) { std::fprintf(stderr, "sim run\n"); return 1; }
      char name[64];
      std::snprintf(name, sizeof(name), "V_%05d", sim.Step());
      hid_t d = H5Dcreate2(f, name, H5T_NATIVE_FLOAT, space, H5P_DEFAULT, dcpl,
                           H5P_DEFAULT);
      if (d < 0) { std::fprintf(stderr, "H5Dcreate\n"); return 1; }
      /* Device pointer straight into HDF5. */
      if (H5Dwrite(d, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   sim.DeviceV()) < 0) {
        std::fprintf(stderr, "H5Dwrite %s\n", name); return 1;
      }
      H5Dclose(d);

      /* Snapshot the exact bytes H5Dwrite was handed, straight off the device. */
      std::vector<float> host(sim.NumElems());
      if (cudaMemcpy(host.data(), sim.DeviceV(), sim.NumBytes(),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy D2H %s\n", name); return 1;
      }
      names.emplace_back(name);
      expected.push_back(std::move(host));
      std::printf("  step %5d -> %s\n", sim.Step(), name);
      ++written;
    }
    H5Pclose(dcpl); H5Sclose(space); H5Fclose(f);
    if (!ref_write(ref_path, names, expected)) {
      std::fprintf(stderr, "reference write failed: %s\n", ref_path.c_str());
      return 1;
    }
    std::printf("done: %d snapshots -> %s (reference: %s)\n", written, path,
                ref_path.c_str());
  }

  if (serve) {
    const std::string ready = std::string(path) + ".ready";
    const std::string stop  = std::string(path) + ".stop";
    FILE *rf = std::fopen(ready.c_str(), "w");
    if (rf) std::fclose(rf);
    std::printf("serving: tier held open, waiting for %s\n", stop.c_str());
    std::fflush(stdout);
    for (;;) {
      FILE *sf = std::fopen(stop.c_str(), "r");
      if (sf) { std::fclose(sf); break; }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::printf("serving: stop seen, exiting\n");
  }

  if (do_read) {
    if (names.empty() && !ref_read(ref_path, names, expected)) {
      std::fprintf(stderr, "reference read failed: %s -- run 'write' first\n",
                   ref_path.c_str());
      return 1;
    }
    std::printf("reading %zu dataset(s) from %s\n", names.size(), path);
    hid_t rf = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (rf < 0) { std::fprintf(stderr, "H5Fopen (read)\n"); return 1; }
    size_t bad_sets = 0;
    for (size_t i = 0; i < names.size(); ++i) {
      std::vector<float> got(expected[i].size(), 0.0f);
      hid_t rd = H5Dopen2(rf, names[i].c_str(), H5P_DEFAULT);
      if (rd < 0) { std::fprintf(stderr, "H5Dopen %s\n", names[i].c_str()); return 1; }
      if (H5Dread(rd, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  got.data()) < 0) {
        std::fprintf(stderr, "H5Dread %s\n", names[i].c_str()); return 1;
      }
      H5Dclose(rd);
      if (compare(got, expected[i], names[i].c_str())) ++bad_sets;
    }
    H5Fclose(rf);
    if (bad_sets) {
      std::printf("VERIFY FAILED: %zu/%zu datasets differ\n", bad_sets,
                  names.size());
      return 2;
    }
    std::printf("VERIFY OK: %zu/%zu datasets bit-exact\n", names.size(),
                names.size());
  }
  return 0;
}
