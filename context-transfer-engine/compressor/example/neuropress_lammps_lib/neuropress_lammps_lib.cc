/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_lammps_lib.cc
 * @brief LAMMPS as a LIBRARY, Clio's runtime in the SAME process.
 *
 * The two sibling LAMMPS examples reach the simulation from the outside:
 *
 *   neuropress_lammps_h5          stock `lmp` binary; its `dump h5md` goes
 *                                 through HDF5, which dlopens Clio's VOL
 *                                 connector. Zero application changes, but
 *                                 every byte is packed, sorted, written to
 *                                 the native .h5 AND staged to Clio.
 *   neuropress_lammps_gpu_direct  a patched LAMMPS with a custom `fix`
 *                                 that hands Kokkos device views to Clio.
 *                                 No host copy, but it needs a patched tree.
 *
 * This is the third shape, and the one a coupling code would actually use:
 * LAMMPS is linked in as `liblammps` and driven through its C library
 * interface (src/library.h), and Clio is hosted in this same process
 * (CLIO_WITH_RUNTIME=1, compose from CLIO_SERVER_CONF). No HDF5, no LAMMPS
 * patch, no `dump` command in the deck at all. The driver owns the coarse
 * structure of the timestep loop -- it asks LAMMPS for GAP steps at a time --
 * and between segments reads Atom::x / v / f straight out of the instance and
 * hands them to the compressor.
 *
 * WHERE THE DATA EVOLVES, AND WHEN IT IS HANDED OVER
 * --------------------------------------------------
 * Inside LAMMPS the state of the system is three contiguous float64 arrays
 * owned by the Atom class (src/atom.h:75, `double **x, **v, **f`; row
 * pointers over one nlocal*3 block, Memory::create). Verlet::run
 * (src/verlet.cpp:229) mutates them every timestep in this order:
 *
 *   initial_integrate   fix nve: v += dt/2 * f/m ; x += dt * v
 *   comm / neighbor     ghosts, (re)build lists, possibly reorder atoms
 *   force_clear + pair->compute        f recomputed from x
 *   final_integrate     fix nve: v += dt/2 * f/m
 *   end_of_step         fixes that OBSERVE the finished step (fix cliogpu,
 *                       fix ave/time, ...) run here
 *   output->write       on steps where ntimestep == output->next: dumps
 *                       (dump h5md's pack -> sort -> h5md_append -> H5Dwrite),
 *                       thermo, restarts
 *
 * So a frame is complete -- x, v and f all describe the same instant -- only
 * after final_integrate. Both of the hand-over points LAMMPS offers sit
 * there: end_of_step (fix) and output->write (dump). This driver reads
 * between `run` segments, i.e. after the last step's output->write; that is
 * the same instant, so the bytes it stores at step N are the bytes
 * `dump h5md N` would have written. On a Kokkos/GPU run the host mirror is
 * current at that point too: VerletKokkos::run syncs to host before
 * output->write on output steps and unconditionally at the end of run
 * (src/KOKKOS/verlet_kokkos.cpp:516,524).
 *
 * Run with the recipe in run.sh -- CLIO_WITH_RUNTIME=1 is load-bearing.
 */

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <library.h>  // LAMMPS C library interface (src/library.h)

#include <clio_cte/compressor/compressor_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_ctp/compress/compress_factory.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_runtime/clio_runtime.h>

/* LAMMPS' Kokkos DEVICE views, from lammps_device_view.cc -- the one
   translation unit here built with nvcc through LAMMPS' own nvcc_wrapper.
   library.h has no device accessor of any kind, so `--order device` reaches
   the GPU arrays through these and nothing else. See that file for why the
   split exists and why none of this needs a LAMMPS patch. */
extern "C" {
int clio_lmp_device_available(void *handle);
long clio_lmp_device_nlocal(void *handle);
int clio_lmp_need_sync_device(void *handle);
int clio_lmp_tags_consecutive(void *handle);
void *clio_lmp_device_scratch(size_t bytes);
void clio_lmp_device_scratch_free(void);
int clio_lmp_device_gather_id(void *handle, const char *field, double *dst_dev,
                              long natoms);
/* Gather only the slice [dst_elem_off, +dst_elem_count) of the field, so a
   chunk can be produced directly in the buffer the compressor reads instead
   of being copied out of a whole-field scratch. */
int clio_lmp_device_gather_id_window(void *handle, const char *field,
                                     double *dst_dev, long natoms,
                                     long dst_elem_off, long dst_elem_count);
/* The same two gathers writing float32. LAMMPS' state stays double; the
   narrowing happens inside the gather kernel, so --f32 --order device reads
   the same atom arrays, writes half the bytes and never leaves the GPU. */
int clio_lmp_device_gather_id_f32(void *handle, const char *field,
                                  float *dst_dev, long natoms);
int clio_lmp_device_gather_id_window_f32(void *handle, const char *field,
                                         float *dst_dev, long natoms,
                                         long dst_elem_off,
                                         long dst_elem_count);
}

namespace {

struct Options {
  int box = 10;       // lattice cells per side; natoms = 4*box^3
  int steps = 100;    // total timesteps
  int gap = 50;       // store every `gap` steps (frame 0 included)
  size_t chunk = 4u << 20;  // bytes per compressor call; 0 = whole field
  std::string fields = "position,velocity,force";
  std::string order = "id";  // id (host gather, == dump h5md)
                             // | local (host extract, LAMMPS internal order)
                             // | device (GPU gather, == dump h5md, never host)
  std::string deck;          // LAMMPS input deck (setup only, no `run`)
  // Keep the old whole-field gather + per-chunk D2D. Off by default: the
  // copy it restores does no work. Here so the two can be measured against
  // each other rather than taken on faith.
  bool device_staging = false;
  // Compare every device-gathered chunk against lammps_gather_atoms for the
  // SAME frame in the SAME process. Cross-run comparison cannot do this job:
  // LJ force summation on the GPU is not bit-reproducible between runs, so
  // two runs of identical code already disagree on `force` at step 0.
  bool crosscheck_host = false;
  std::string tag = "lammps_lib";
  std::string logfile = "none";  // LAMMPS -log
  std::string report;            // per-blob CSV
  std::string raw_dir;           // also write every staged blob's bytes here
  std::string readback;          // CSV from --report: read those blobs, no LAMMPS
  std::string decomp_dir;        // write every DECOMPRESSED blob's bytes here
  bool expect_lossy = false;     // a positive error bound: do not claim bit-exact
  bool f32 = false;              // downcast the state to float32 before staging
  bool kokkos = false;    // -k on g 1 -sf kk
  bool verify = false;    // read every blob back through the decompressor
  bool quiet = false;     // LAMMPS -screen none
  // Extra LAMMPS -var NAME VALUE pairs, so a deck can expose its physics
  // (density, temperature, cutoff, seed, timestep) without this driver
  // knowing what any of them mean. LAMMPS gives command-line -var
  // precedence over the deck's own `variable ... index` default, which is
  // what makes a parameterised deck still runnable standalone.
  std::vector<std::pair<std::string, std::string>> vars;
};

void Usage(const char *argv0) {
  std::cerr
      << "usage: " << argv0 << " --deck in.melt_lib [options]\n"
      << "  --box N          lattice cells per side (natoms = 4*N^3) [10]\n"
      << "  --steps N        timesteps [100]\n"
      << "  --gap N          store every N steps, frame 0 included [50]\n"
      << "  --chunk BYTES    bytes per compressor call, 0 = whole field "
         "[4194304]\n"
      << "  --fields LIST    any of position,velocity,force [all three]\n"
      << "  --order id|local|device\n"
      << "                   id:     atom-ID order via lammps_gather_atoms "
         "(byte-identical to dump h5md)\n"
      << "                   local:  LAMMPS' internal order via "
         "lammps_extract_atom, no gather copy\n"
      << "                   device: atom-ID order gathered ON THE GPU out of "
         "the Kokkos device views;\n"
      << "                           the payload never becomes host bytes. "
         "Requires --kokkos. Same\n"
      << "                           ordering as id, so crosscheck_h5md.sh "
         "still applies.\n"
      << "  --device-staging  on --order device, gather the whole field into\n"
         "                   scratch and copy each chunk out of it -- the old\n"
         "                   path, one payload-sized D2D per chunk. Default\n"
         "                   gathers each chunk straight into the registered\n"
         "                   backend and makes no copy at all.\n"
      << "  --crosscheck-host on --order device, also gather each frame with\n"
         "                   lammps_gather_atoms and require every chunk to\n"
         "                   match byte for byte. Same frame, same process --\n"
         "                   the only sound way to check the device gather,\n"
         "                   since GPU force summation is not reproducible\n"
         "                   across runs.\n"
      << "  --kokkos         run LAMMPS on the GPU (-k on g 1 -sf kk)\n"
      << "  --verify         read every blob back and compare\n"
      << "  --report CSV     per-blob outcome\n"
      << "  --raw DIR        also write each staged blob's raw bytes to DIR "
         "(cross-checks)\n"
      << "  --dump-decompressed DIR  write each decompressed blob's bytes "
         "to DIR (pair with --raw to compare)\n"
      << "  --f32            downcast x/v/f to float32 before staging and "
         "declare data_type_=1 (halves the payload). Works on every order:\n"
         "                   --order device narrows IN THE GATHER KERNEL\n"
         "                   (GatherIdWindowT<float>), so no float64 payload\n"
         "                   is ever materialised on the device or the host\n"
      << "  --expect-lossy   a positive CLIO_NEUROPRESS_ERROR_BOUND was set, so "
         "the bytes are NOT expected to match; report decode failures only\n"
      << "  --readback CSV   no simulation: read every blob named in a "
         "previous --report CSV back\n"
      << "                   through the decompressor and compare digests "
         "(run with CLIO_RESTART=1)\n"
      << "  --device-staging on --order device, gather the whole field into\n"
         "                   scratch and copy each chunk out of it (the old\n"
         "                   path, one payload-sized D2D per chunk). Default\n"
         "                   gathers each chunk straight into the registered\n"
         "                   backend and makes no copy at all.\n"
      << "  --tag NAME       CTE tag [lammps_lib]\n"
      << "  --log FILE       LAMMPS log file [none]\n"
      << "  --quiet          LAMMPS -screen none\n"
      << "  --var NAME=VALUE repeatable; passed to LAMMPS as -var NAME VALUE\n"
      << "                   (overrides the deck's `variable NAME index ...` "
         "default)\n";
}

bool ParseArgs(int argc, char **argv, Options *o) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char *what) -> const char * {
      if (i + 1 >= argc) {
        std::cerr << a << " needs " << what << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--box") o->box = std::atoi(need("N"));
    else if (a == "--steps") o->steps = std::atoi(need("N"));
    else if (a == "--gap") o->gap = std::atoi(need("N"));
    else if (a == "--chunk") o->chunk = std::strtoull(need("BYTES"), nullptr, 10);
    else if (a == "--fields") o->fields = need("LIST");
    else if (a == "--order") o->order = need("id|local");
    else if (a == "--device-staging") o->device_staging = true;
    else if (a == "--crosscheck-host") o->crosscheck_host = true;
    else if (a == "--deck") o->deck = need("FILE");
    else if (a == "--tag") o->tag = need("NAME");
    else if (a == "--log") o->logfile = need("FILE");
    else if (a == "--report") o->report = need("CSV");
    else if (a == "--raw") o->raw_dir = need("DIR");
    else if (a == "--readback") o->readback = need("CSV");
    else if (a == "--dump-decompressed") o->decomp_dir = need("DIR");
    else if (a == "--expect-lossy") o->expect_lossy = true;
    else if (a == "--f32") o->f32 = true;
    else if (a == "--kokkos") o->kokkos = true;
    else if (a == "--verify") o->verify = true;
    else if (a == "--var") {
      const std::string kv = need("NAME=VALUE");
      const size_t eq = kv.find('=');
      if (eq == std::string::npos || eq == 0) {
        std::cerr << "--var expects NAME=VALUE, got '" << kv << "'\n";
        return false;
      }
      o->vars.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
    }
    else if (a == "--quiet") o->quiet = true;
    else if (a == "-h" || a == "--help") { Usage(argv[0]); std::exit(0); }
    else { std::cerr << "unknown option " << a << "\n"; Usage(argv[0]); return false; }
  }
  if (o->deck.empty() && o->readback.empty()) { Usage(argv[0]); return false; }
  if (o->gap <= 0 || o->steps < 0 || o->box <= 0) return false;
  if (o->order != "id" && o->order != "local" && o->order != "device")
    return false;
  return true;
}

/* FNV-1a-64. Same digest the gpu-direct example used, so the two are
   comparable by eye. */
uint64_t Fnv1a(const void *p, size_t n) {
  const auto *b = static_cast<const unsigned char *>(p);
  uint64_t h = 14695981039346656037ull;  // 0xcbf29ce484222325
  for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
  return h;
}

/* One field of one frame, as LAMMPS names it and as it is stored. */
struct Field {
  const char *lmp_name;   // Atom::extract / gather name
  const char *blob_name;  // h5md's group name, so blobs read like its paths
};
const Field kFields[] = {{"x", "position"}, {"v", "velocity"}, {"f", "force"}};

/**
 * Absolute error bound for error-bounded lossy compression, from
 * CLIO_NEUROPRESS_ERROR_BOUND. 0 (the default) means LOSSLESS -- what this
 * driver did unconditionally before -- and masks NeuroPress's 16 quantize
 * actions; a positive bound makes them reachable. Same name and meaning as in
 * neuropress_explore_sweep.cc and gpucompress_config_t::error_bound upstream.
 */
double ErrorBoundFromEnv() {
  const char *e = std::getenv("CLIO_NEUROPRESS_ERROR_BOUND");
  if (e == nullptr || *e == '\0') return 0.0;
  const double v = std::atof(e);
  return v > 0.0 ? v : 0.0;
}

struct BlobRecord {
  std::string name;
  size_t bytes = 0;
  uint64_t digest = 0;
  // Filled in once the compressor answers.
  int lib = 0;
  double ratio = 0.0;
  size_t stored = 0;
  double ms = 0.0;
  // MEASURED decompression time (CUDA events around the codec call alone), or
  // <0 when nothing measured one. Needs CLIO_NEUROPRESS_EXPLORE_MEASURE_DT.
  // Comparable with `ms` beside it and with the exploration sweep's per
  // candidate dt -- NOT with a read-path Decompress(), which reports wall
  // clock around the whole path.
  double dt_ms = -1.0;
  bool ok = false;
};

struct Pending {
  clio::run::Future<clio::cte::compressor::DynamicScheduleTask> fut;
  // Host orders only. On --order device the staging buffer belongs to the
  // slot pool below, which outlives every individual task, so there is
  // nothing per-task to release.
  ctp::ipc::FullPtr<char> buf;
  bool on_device = false;
  size_t record;  // index into records
};

void LmpCheck(void *lmp, const char *what) {
  if (lammps_has_error(lmp)) {
    char msg[1024];
    lammps_get_last_error_message(lmp, msg, sizeof msg);
    std::cerr << "LAMMPS error during " << what << ": " << msg << "\n";
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char **argv) {
  Options opt;
  if (!ParseArgs(argc, argv, &opt)) return 2;

  const auto t_start = std::chrono::steady_clock::now();

  // ---- Clio: attach this process to the runtime. ----------------------
  // CLIO_CTE_CLIENT_INIT -> CLIO_INIT(kClient): with CLIO_WITH_RUNTIME=1 in
  // the environment the runtime comes up HERE, composed from
  // CLIO_SERVER_CONF. Same call the HDF5 VOL makes lazily on first use
  // (clio_vol.cc get_cte_client); here it is simply the first thing main does.
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::cerr << "CLIO_CTE_CLIENT_INIT failed -- is CLIO_SERVER_CONF set and "
                 "CLIO_WITH_RUNTIME=1?\n";
    return 1;
  }
  auto *cte_client = CLIO_CTE_CLIENT;

  // The compressor pool comes from the compose file (512.0 in run.sh), the
  // same way the VOL learns it from CLIO_VOL_COMPRESSOR_POOL. One-arg ctor:
  // the compressor's own next_pool_id_ decides which core pool it forwards to.
  unsigned pool_major = 512, pool_minor = 0;
  if (const char *e = std::getenv("CLIO_LMP_COMPRESSOR_POOL"); e && *e) {
    if (std::sscanf(e, "%u.%u", &pool_major, &pool_minor) < 1) {
      std::cerr << "bad CLIO_LMP_COMPRESSOR_POOL\n";
      return 1;
    }
  }
  clio::cte::compressor::Client compressor(
      clio::run::PoolId(pool_major, pool_minor));

  auto tag = cte_client->AsyncGetOrCreateTag(opt.tag);
  tag.Wait();
  if (tag->GetReturnCode() != 0) {
    std::cerr << "GetOrCreateTag failed\n";
    return 1;
  }
  const auto tag_id = tag->tag_id_;

  // Read every record back through the decompressor into a fresh host
  // buffer and compare size + digest with what was staged. Shared by
  // --verify (same process, right after the write) and --readback (a later
  // process, CLIO_RESTART=1, the tier being the only copy of the data).
  auto verify_records = [&](const std::vector<BlobRecord> &recs) -> bool {
    size_t bad = 0;
    for (const auto &r : recs) {
      auto buf = CLIO_IPC->AllocateBuffer(r.bytes);
      if (buf.IsNull()) { std::cerr << "AllocateBuffer (verify) failed\n"; return false; }
      std::memset(buf.ptr_, 0, r.bytes);
      auto get = compressor.AsyncDecompressExplicit(
          clio::run::PoolQuery::Local(), tag_id, r.name, 0, r.bytes, 0,
          buf.shm_.template Cast<void>(), cte_client->pool_id_);
      get.Wait();
      const int rc_get = get->GetReturnCode();
      // Under a positive error bound the bytes are SUPPOSED to differ, so a
      // digest test would report FAILED on a run behaving exactly as asked.
      // There `bad` counts decode failures only, and the verdict on the data
      // is whatever compares --raw against --dump-decompressed afterwards.
      const bool ok = opt.expect_lossy ? (rc_get == 0)
                                       : (rc_get == 0 &&
                                          Fnv1a(buf.ptr_, r.bytes) == r.digest);
      if (!ok) {
        ++bad;
        std::cerr << "  MISMATCH " << r.name << " rc=" << rc_get << "\n";
      }
      if (!opt.decomp_dir.empty() && rc_get == 0) {
        std::string fn = r.name;
        for (auto &ch : fn) if (ch == '/') ch = '_';
        std::ofstream(opt.decomp_dir + "/" + fn + ".bin", std::ios::binary)
            .write(static_cast<const char *>(buf.ptr_),
                   static_cast<std::streamsize>(r.bytes));
      }
      CLIO_IPC->FreeBuffer(buf);
    }
    std::cout << (bad == 0 ? "VERIFIED: " : "FAILED: ") << (recs.size() - bad)
              << " of " << recs.size()
              << (opt.expect_lossy
                      ? " blobs decompressed without error (lossy: bytes are "
                        "not expected to match)"
                      : " blobs round-tripped bit-exact through the decompressor")
              << std::endl;
    return bad == 0;
  };

  if (!opt.readback.empty()) {
    // Cold read: no LAMMPS at all. The CSV is the only thing shared with the
    // writer besides the store directory.
    std::ifstream csv(opt.readback);
    if (!csv) { std::cerr << "cannot open " << opt.readback << "\n"; return 1; }
    std::vector<BlobRecord> recs;
    std::string line;
    std::getline(csv, line);  // header
    while (std::getline(csv, line)) {
      BlobRecord r;
      size_t p1 = line.find(','), p2 = line.find(',', p1 + 1),
             p3 = line.find(',', p2 + 1);
      if (p1 == std::string::npos || p2 == std::string::npos) continue;
      r.name = line.substr(0, p1);
      r.bytes = std::strtoull(line.substr(p1 + 1, p2 - p1 - 1).c_str(), nullptr, 10);
      r.digest = std::strtoull(line.substr(p2 + 1, p3 - p2 - 1).c_str(), nullptr, 16);
      recs.push_back(r);
    }
    std::cout << "cold read-back of " << recs.size() << " blob(s) listed in "
              << opt.readback << std::endl;
    const bool ok = verify_records(recs);
    clio::run::CLIO_RUNTIME_FINALIZE();
    return ok ? 0 : 1;
  }

  // ---- LAMMPS: create an instance in this process. --------------------
  std::vector<std::string> args = {"lmp", "-log", opt.logfile,
                                   "-var", "BOX", std::to_string(opt.box),
                                   "-var", "GAP", std::to_string(opt.gap)};
  for (const auto &kv : opt.vars) {
    args.push_back("-var");
    args.push_back(kv.first);
    args.push_back(kv.second);
  }
  if (opt.quiet) { args.push_back("-screen"); args.push_back("none"); }
  if (opt.kokkos) {
    for (const char *s : {"-k", "on", "g", "1", "-sf", "kk"}) args.push_back(s);
  }
  std::vector<char *> cargs;
  for (auto &s : args) cargs.push_back(s.data());
  void *lmp = nullptr;
  lammps_open_no_mpi(static_cast<int>(cargs.size()), cargs.data(), &lmp);
  if (!lmp) { std::cerr << "lammps_open_no_mpi failed\n"; return 1; }
  LmpCheck(lmp, "open");

  // The deck builds the system and declares the fixes; it contains no `dump`
  // and no `run`. Everything after this point is the driver's.
  lammps_file(lmp, opt.deck.c_str());
  LmpCheck(lmp, "deck");
  const int64_t natoms = static_cast<int64_t>(lammps_get_natoms(lmp));
  const int nlocal = lammps_extract_setting(lmp, "nlocal");
  if (natoms <= 0) { std::cerr << "no atoms\n"; return 1; }
  if (opt.order == "local" && nlocal != natoms) {
    // lammps_extract_atom only sees THIS rank's atoms. Serial here, so the
    // two agree; under MPI you would gather or store per rank.
    std::cerr << "--order local: nlocal " << nlocal << " != natoms " << natoms
              << " (multi-rank run); use --order id\n";
    return 1;
  }

  // ---- --order device: every precondition, checked once, before anything
  // is stored. Each of these is a case where the run could have continued by
  // quietly doing something else -- reading the host mirror, gathering
  // against stale tags, storing one rank's slice -- and produced blobs that
  // pass a round-trip check while not being what was asked for.
  if (opt.order == "device") {
    if (!clio_lmp_device_available(lmp)) {
      // atomKK is null. Deliberately asked of LAMMPS rather than inferred
      // from --kokkos: the flag is only what this driver PASSES to LAMMPS,
      // and it can be passed to a liblammps built without the KOKKOS package,
      // or be overridden by the deck. What matters is whether the atom arrays
      // are actually on the device, and only LAMMPS can answer that.
      std::cerr << "--order device: LAMMPS has no Kokkos atom object "
                   "(atomKK == null), so the atom arrays are host-resident "
                   "and there is no device view to hand to Clio. Pass "
                   "--kokkos (-k on g 1 -sf kk), and check this liblammps "
                   "was built with -DPKG_KOKKOS=ON. Refusing rather than "
                   "falling back to host bytes.\n";
      return 3;
    }
    const long dev_nlocal = clio_lmp_device_nlocal(lmp);
    if (dev_nlocal != natoms) {
      // The device views hold THIS rank's atoms. Storing them under an
      // unqualified blob name would publish a fraction of the system as if
      // it were the whole of it -- the same trap --order local refuses.
      std::cerr << "--order device: nlocal " << dev_nlocal << " != natoms "
                << natoms
                << " (multi-rank run). The device views hold only this rank's "
                   "atoms; use --order id, or name blobs per rank.\n";
      return 1;
    }
    // Read-back is NOT optional on this order, and the reason is a defect,
    // not caution.
    //
    // A chunk staged as DEVICE memory whose codec actually runs can reach the
    // tier corrupted, in this process configuration. Measured on box 10, LJ
    // melt: with --static nvcomp-zstd, 5 of 6 runs stored at least one blob
    // whose header the read side rejects with "magic/version mismatch"; even
    // on the dynamic default, where only 1 chunk in 9 compresses, 3 of 10 runs
    // do. The damage is on disk -- a cold read from a separate process with
    // CLIO_RESTART=1 fails the same blobs. It is not this driver's staging:
    //   --order id  + --static (HOST staging, same GPU run)    0 of 6 failed
    //   --order device, no --static (device, 1 chunk in 9)     3 of 10 failed
    //   --order device + --static (device, all compressed)     5 of 6 failed
    //   bin/neuropress_gpu_static (device + static, NO LAMMPS) 0 of 5 failed
    // so it needs device staging AND a codec that runs AND LAMMPS/Kokkos
    // sharing the process. Remove any one and it is clean. That points at
    // Clio's device compress/store path rather than at anything here, and it
    // is reported rather than papered over.
    //
    // Until it is fixed, this order must never be able to finish quietly with
    // bad data in the tier. Forcing the read-back turns the failure into a
    // non-zero exit and a MISMATCH line per blob. Announced, not silent --
    // the whole point is that nothing about this run is implicit.
    if (!opt.verify) {
      std::cout << "--order device: enabling --verify. A device-staged chunk "
                   "whose codec runs can reach the tier corrupted in this "
                   "configuration (see the README), so this order always "
                   "reads every blob back rather than risk reporting success "
                   "over bad data."
                << std::endl;
      opt.verify = true;
    }

    // The windowed gather addresses the destination in field ELEMENTS, so a
    // chunk boundary that is not a multiple of sizeof(double) would truncate
    // to the wrong element and store a shifted field -- which round-trips
    // perfectly and is the wrong data. field_bytes is natoms*3*8, so every
    // boundary is aligned as long as the chunk size is.
    if (!opt.device_staging && opt.chunk % sizeof(double) != 0) {
      std::cerr << "--order device: --chunk " << opt.chunk
                << " is not a multiple of " << sizeof(double)
                << ". The per-chunk gather addresses whole float64 elements, "
                   "so an unaligned boundary would store a shifted field. "
                   "Use a multiple of 8, or --device-staging.\n";
      return 1;
    }

    if (!clio_lmp_tags_consecutive(lmp)) {
      // The gather writes to index tag[i]-1. Without consecutive IDs that
      // index is meaningless -- the same precondition lammps_gather_atoms
      // raises as a LAMMPS error (library.cpp:3597-3601).
      std::cerr << "--order device: atom IDs are absent or not consecutive, "
                   "so an ID-ordered gather is not defined for this system\n";
      return 1;
    }
  }

  std::vector<const Field *> fields;
  for (const auto &f : kFields)
    if (opt.fields.find(f.blob_name) != std::string::npos) fields.push_back(&f);
  if (fields.empty()) { std::cerr << "no fields selected\n"; return 1; }

  // --f32 stages a DOWNCAST copy. LAMMPS' state is double (Atom::x/v/f are
  // double**), so this is a lossy narrowing of its own, before NeuroPress sees
  // anything -- but it is what makes a NeuroPress error bound reachable at all:
  // the quantizer reads every buffer as float32 regardless of the declared
  // type, and float64 bytes read that way are largely non-finite, so it
  // declines. Handing it real float32 is the difference between an error bound
  // that applies and one that is silently inert.
  // --order device now downcasts IN THE GATHER KERNEL, so --f32 no longer
  // implies a host round trip. `raw` is still incompatible: it hands over
  // Atom::x/v/f themselves, which are double and are not ours to narrow.
  if (opt.f32 && opt.order == "raw") {
    std::cerr << "--f32 cannot be combined with --order raw: that order hands "
                 "the compressor LAMMPS' own double arrays, with no gather to "
                 "narrow in. Use --order device (GPU) or --order id (host).\n";
    return 1;
  }
  const size_t elem_size = opt.f32 ? sizeof(float) : sizeof(double);
  const size_t field_bytes = static_cast<size_t>(natoms) * 3 * elem_size;
  const size_t chunk = (opt.chunk == 0 || opt.chunk > field_bytes) ? field_bytes
                                                                    : opt.chunk;
  const size_t chunks_per_field = (field_bytes + chunk - 1) / chunk;
  const int nframes = opt.steps / opt.gap + 1;

  std::cout << "LAMMPS (library) -> Clio compressor, runtime in-process\n"
            << "  atoms=" << natoms << "  steps=" << opt.steps << "  gap="
            << opt.gap << "  frames=" << nframes << "  fields="
            << fields.size() << "\n"
            << "  field=" << field_bytes << " B  chunk=" << chunk << " B  ("
            << chunks_per_field << " chunk(s)/field)  order=" << opt.order
            << "  device=" << (opt.kokkos ? "GPU/Kokkos" : "CPU") << "\n"
            << "  payload=" << (field_bytes * fields.size() * nframes) / 1048576.0
            << " MiB\n"
            // What the compressor will actually receive. Said out loud
            // because it is the one property of this run that no return code
            // reports and that every log otherwise looks identical for.
            << "  hand-over to the compressor: "
            << (opt.order == "device"
                    ? "DEVICE pointers (CUDA IPC-registered backend)"
                    : "HOST shm (the atom arrays are read from the host "
                      "mirror)")
            << std::endl;

  // Context for every chunk: float64. data_type_ 2 is what the VOL reports for
  // an 8-byte H5T_FLOAT; the compressor converts on that flag instead of
  // reading doubles as pairs of float32 (see clio_vol.cc
  // clio_context_data_type). error_bound_ is 0 = lossless unless
  // CLIO_NEUROPRESS_ERROR_BOUND asks otherwise; see ErrorBoundFromEnv.
  clio::cte::core::Context ctx;
  ctx.data_type_ = opt.f32 ? 1 : 2;
  ctx.error_bound_ = ErrorBoundFromEnv();
  if (ctx.error_bound_ > 0.0)
    std::cout << "  error bound=" << ctx.error_bound_
              << "  LOSSY (quantize actions enabled)" << std::endl;

  std::vector<BlobRecord> records;
  std::vector<Pending> pending;
  std::vector<double> gathered;    // --order id staging, natoms*3
  std::vector<float> gathered32;   // --f32 downcast of the same
  records.reserve(static_cast<size_t>(nframes) * fields.size() * chunks_per_field);

  auto drain = [&]() {
    for (auto &p : pending) {
      p.fut.Wait();
      auto &r = records[p.record];
      const auto &c = p.fut->context_;
      r.ok = p.fut->GetReturnCode() == 0;
      r.lib = c.compress_lib_;
      r.ratio = c.actual_compression_ratio_;
      r.ms = c.actual_compress_time_ms_;
      r.dt_ms = c.actual_decompress_time_ms_;
      // compress_lib_ == 0 is how the compressor marks "stored raw": the
      // codec it tried did not shrink the chunk, so the caller's bytes went
      // to the tier as they were (compressor_runtime.cc, "Compression not
      // beneficial"). actual_compressed_size_ still reports the codec's
      // output in that case, which is NOT what was stored.
      r.stored = (r.ok && r.lib != 0) ? c.actual_compressed_size_ : r.bytes;
      // Device staging buffers are pool slots and are not freed here; see
      // DeviceSlot. Host buffers are per-chunk and are.
      if (!p.on_device) CLIO_IPC->FreeBuffer(p.buf);
    }
    pending.clear();
  };

  // Stage one frame: every selected field, chunked, one AsyncDynamicSchedule
  // per chunk. Returns bytes submitted.
  // --order device only: the bytes are pulled back to the host solely to
  // record a digest (and, with --raw, to write them out). Nothing on the
  // compressor path reads this, so it is skipped entirely when nothing needs
  // it -- otherwise a "GPU-resident" run would quietly pay a D2H per chunk.
  const bool need_bytes_on_host = opt.verify || !opt.report.empty() ||
                                  !opt.raw_dir.empty() || opt.crosscheck_host;
  // The host reference for --crosscheck-host: lammps_gather_atoms into a
  // plain buffer, per field per frame. This is the definition of the ordering
  // the device gather is supposed to reproduce (it is what `dump h5md` writes
  // after its own sort), so comparing against it checks the permutation
  // itself and not merely that a round trip is self-consistent.
  std::vector<double> host_ref;
  size_t crosscheck_ok = 0, crosscheck_bad = 0;
  std::vector<char> dev_staging;

  // --order device: one registered device backend per chunk OF A FRAME,
  // allocated on the first frame and REUSED by every frame after it. Never
  // freed and reallocated mid-run.
  //
  // Two reasons, and NEITHER of them is the corruption bug described at the
  // --order device precondition checks above -- reuse was tried as a fix for
  // it and does not fix it (5 runs of the pooled build: 4 still failed). Do
  // not read this pool as a workaround; the bug is Clio's and is unfixed.
  //
  //  1. It bounds device memory at one FRAME. The alternative that keeps a
  //     distinct backend per chunk for the whole run costs the entire payload.
  //  2. It does no free-and-reallocate churn, so no AllocatorId is ever
  //     recycled mid-run. That is worth having on its own: an id handed back
  //     out while anything still resolves through the old registration would
  //     be a second, independent way to get the wrong bytes.
  //
  // Reuse is safe because drain() waits for every task of frame N before
  // frame N+1 stages anything into the same slots. Within a frame each chunk
  // has its own slot, so nothing is overwritten while it is in flight.
  // "--order device, gathering straight into the registered backend": there
  // is no whole-field buffer to point at, but the chunk loop still has to know
  // it is on the device path. Never dereferenced, never offset -- compared
  // against, and nothing else.
  static const char kGatherPerChunkStorage = 0;
  const char *const kGatherPerChunk = &kGatherPerChunkStorage;

  struct DeviceSlot {
    ctp::ipc::AllocatorId alloc;
    char *ptr = nullptr;
    size_t bytes = 0;
  };
  std::vector<DeviceSlot> device_slots;
  size_t device_slot_next = 0;  // reset at the top of every frame

  auto store_frame = [&](int64_t step) -> size_t {
    size_t submitted = 0;
    // Frame N reuses the same slots as frame N-1, in the same order. Safe
    // only because drain() has already waited for frame N-1's tasks.
    device_slot_next = 0;
    for (const Field *f : fields) {
      // const void*, not const double*: under --f32 this points at the
      // downcast float array. Only ever read as bytes below.
      const void *src = nullptr;
      // --order device: the field, ID-ordered, sitting in device memory. Set
      // instead of `src`, and the two are never both valid.
      const char *dev_field = nullptr;
      if (opt.order == "device") {
        // Freshness, checked per field per frame rather than once at startup:
        // whether a device view is current is a property of where LAMMPS is
        // in the timestep, not of the run. Between `run` segments every mask
        // bit is clear (VerletKokkos::run has just synced to host without
        // marking the device stale). If that ever stops being true, reading
        // anyway would store a previous step's image under this step's name
        // -- a frame that round-trips perfectly and is simply the wrong data.
        // NOT repaired with sync(Device, ...): see lammps_device_view.cc.
        const int stale = clio_lmp_need_sync_device(lmp);
        if (stale != 0) {
          std::cerr << "--order device: LAMMPS' device views are stale at step "
                    << step << " (need_sync mask=" << stale
                    << ", bits 1=x 2=v 4=f 8=tag). Reading them now would "
                       "store an earlier image under this step's name. "
                       "Refusing.\n";
          std::exit(3);
        }
        if (opt.crosscheck_host) {
          host_ref.resize(static_cast<size_t>(natoms) * 3);
          lammps_gather_atoms(lmp, f->lmp_name, /*dtype=*/1, /*count=*/3,
                              host_ref.data());
          LmpCheck(lmp, "gather_atoms (crosscheck)");
        }
        if (opt.device_staging) {
          // The OLD path, kept only so it can be measured against the new
          // one: gather the whole field into scratch, then copy each chunk
          // out of it below. That copy is payload-sized and does no work.
          auto *scratch = clio_lmp_device_scratch(field_bytes);
          if (scratch == nullptr) {
            std::cerr << "--order device: could not allocate " << field_bytes
                      << " B of device scratch for the ID gather\n";
            std::exit(4);
          }
          // Same permutation lammps_gather_atoms performs on the host
          // (library.cpp:3663-3672), run on the GPU into the scratch buffer:
          // the ordering `dump h5md` writes, without the payload ever being
          // host bytes. This is what keeps crosscheck_h5md.sh meaningful for
          // this order.
          const int grc =
              opt.f32 ? clio_lmp_device_gather_id_f32(
                            lmp, f->lmp_name, static_cast<float *>(scratch),
                            natoms)
                      : clio_lmp_device_gather_id(
                            lmp, f->lmp_name, static_cast<double *>(scratch),
                            natoms);
          if (grc != 0) {
            std::cerr << "--order device: ID gather of '" << f->lmp_name
                      << "' failed (rc=" << grc << ") at step " << step << "\n";
            std::exit(3);
          }
          dev_field = reinterpret_cast<const char *>(scratch);
        } else {
          // The gather happens per chunk, directly into the registered
          // backend, in the loop below. Nothing to do here -- and no
          // whole-field scratch buffer exists at all on this path.
          dev_field = kGatherPerChunk;
        }
      } else if (opt.order == "id") {
        // Global array ordered by atom ID -- what dump h5md writes after its
        // own sort(). One copy (LAMMPS' internal order -> ID order); the
        // sibling h5md path does the same copy inside Dump::write, then a
        // second one into the dump_* arrays, then a third in the VOL.
        gathered.resize(static_cast<size_t>(natoms) * 3);
        lammps_gather_atoms(lmp, f->lmp_name, /*dtype=*/1, /*count=*/3,
                            gathered.data());
        LmpCheck(lmp, "gather_atoms");
        src = gathered.data();
        if (opt.f32) {
          gathered32.resize(gathered.size());
          for (size_t i = 0; i < gathered.size(); ++i)
            gathered32[i] = static_cast<float>(gathered[i]);
          src = gathered32.data();
        }
      } else {
        // Atom::x / v / f themselves: row pointers over one contiguous
        // nlocal*3 block, so [0] is the whole field. Order is whatever
        // LAMMPS currently holds -- spatially re-sorted every
        // atom_modify sort interval (default 1000 steps), so frames are not
        // guaranteed atom-for-atom comparable across a sort.
        auto **arr = static_cast<double **>(lammps_extract_atom(lmp, f->lmp_name));
        LmpCheck(lmp, "extract_atom");
        if (!arr) { std::cerr << "extract_atom(" << f->lmp_name << ") null\n"; std::exit(1); }
        src = arr[0];
      }

      for (size_t ci = 0; ci < chunks_per_field; ++ci) {
        const size_t off = ci * chunk;
        const size_t n = std::min(chunk, field_bytes - off);
        ctp::ipc::ShmPtr<> blob_data;
        ctp::ipc::FullPtr<char> host_buf;
        // Bytes to digest / write with --raw. Points into the caller's host
        // array on the host orders; into dev_staging on --order device.
        const char *bytes = nullptr;

        if (dev_field != nullptr) {
          // One registered backend PER CHUNK, and the chunk's bytes copied
          // into it -- rather than one backend for the field with each chunk
          // addressed by an offset into it. The offset form works today
          // (IpcManager::ToFullPtr case 4 resolves a same-PID alloc through
          // ShmPtr::off_) but breaks silently the moment the runtime moves to
          // another process: case 5 resolves through the registered
          // backend's own device_ptr and IGNORES off_, so every chunk would
          // come back as chunk 0's bytes -- and each blob would still
          // round-trip bit-exact against the digest of what was submitted.
          // This is the shape the VOL already uses (clio_stage_chunk).
          if (device_slot_next >= device_slots.size()) {
            DeviceSlot slot;
            slot.alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(
                /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
                n, &slot.ptr);
            if (slot.alloc.IsNull()) {
              std::cerr << "--order device: AllocateAndRegisterGpuBackend("
                        << n
                        << ") failed. Refusing to stage this chunk through "
                           "host memory instead -- that would silently turn a "
                           "GPU-resident run into a host one.\n";
              std::exit(4);
            }
            slot.bytes = n;
            device_slots.push_back(slot);
          }
          DeviceSlot &slot = device_slots[device_slot_next++];
          if (slot.bytes < n) {
            // Every frame chunks the same field size the same way, so a slot
            // is always revisited with the size it was made for. If that ever
            // stops being true, growing the slot would mean a free and a
            // realloc -- the exact thing this pool exists to avoid.
            std::cerr << "--order device: staging slot is " << slot.bytes
                      << " B but this chunk is " << n
                      << " B; chunk layout changed between frames\n";
            std::exit(4);
          }
          char *registered = slot.ptr;
          if (dev_field == kGatherPerChunk) {
            // Produce the chunk WHERE THE COMPRESSOR WILL READ IT. The gather
            // is a scatter by atom id, so it used to need a whole-field
            // destination and the chunk was then copied out of it -- one
            // payload-sized D2D per chunk that moved bytes without changing
            // any. Naming the destination window removes both the copy and
            // the scratch field.
            //
            // The window is in field ELEMENTS, not bytes: `off` and `n` are
            // byte quantities and a chunk boundary is always a multiple of
            // the element size because the chunk size is and the field is an
            // array of them. elem_size, not sizeof(double), because --f32
            // makes the field an array of floats.
            const long win_off = static_cast<long>(off / elem_size);
            const long win_n = static_cast<long>(n / elem_size);
            const int grc =
                opt.f32 ? clio_lmp_device_gather_id_window_f32(
                              lmp, f->lmp_name,
                              reinterpret_cast<float *>(registered), natoms,
                              win_off, win_n)
                        : clio_lmp_device_gather_id_window(
                              lmp, f->lmp_name,
                              reinterpret_cast<double *>(registered), natoms,
                              win_off, win_n);
            if (grc != 0) {
              std::cerr << "--order device: windowed ID gather of '"
                        << f->lmp_name << "' chunk " << ci << " failed (rc="
                        << grc << ") at step " << step << "\n";
              std::exit(3);
            }
            // No Synchronize() here: the gather ends in a Kokkos::fence, so
            // the bytes are already final. The staged path below still needs
            // one because a D2D memcpy is not host-synchronous.
          } else {
            ctp::GpuApi::Memcpy(registered, dev_field + off, n);
            // The compressor reads `registered` from another thread and Clio's
            // own streams; the copy above is not host-synchronous for D2D. One
            // sync per chunk on an otherwise idle device is microseconds, and
            // it removes the class of bug where a chunk is compressed before
            // its bytes have landed -- which produces a perfectly valid blob of
            // the wrong contents.
            ctp::GpuApi::Synchronize();
          }
          if (!ctp::IsDevicePointer(registered)) {
            std::cerr << "--order device: the registered staging buffer is "
                         "not device memory. The compressor would receive "
                         "host bytes while the run reported GPU residency. "
                         "Refusing.\n";
            std::exit(5);
          }
          blob_data.alloc_id_ = slot.alloc;
          blob_data.off_ = reinterpret_cast<clio::run::u64>(registered);

          if (need_bytes_on_host) {
            // Read from whichever buffer the gather actually fenced.
            //
            // Staged: the SCRATCH, not `registered`. The gather into scratch
            // has been fenced, so those bytes are certainly final; hashing
            // the destination of an unsynchronized D2D is how the earlier
            // gpu-direct example's verifier accused correct data of being
            // wrong.
            //
            // Per-chunk: `registered` IS the gather's destination and the
            // gather ended in a Kokkos::fence, so it is equally final -- and
            // there is no scratch to read instead.
            const char *digest_src =
                (dev_field == kGatherPerChunk) ? registered : dev_field + off;
            dev_staging.resize(n);
            ctp::GpuApi::Memcpy(dev_staging.data(), digest_src, n);
            bytes = dev_staging.data();

            if (opt.crosscheck_host) {
              const char *ref =
                  reinterpret_cast<const char *>(host_ref.data()) + off;
              if (std::memcmp(bytes, ref, n) == 0) {
                ++crosscheck_ok;
              } else {
                ++crosscheck_bad;
                // Say WHICH element, not just that the chunk differs: a
                // window off by one shows up as a run starting exactly at a
                // chunk boundary, which names the bug on sight.
                size_t first = 0;
                while (first < n / sizeof(double) &&
                       reinterpret_cast<const double *>(bytes)[first] ==
                           reinterpret_cast<const double *>(ref)[first])
                  ++first;
                std::cerr << "  CROSSCHECK MISMATCH " << f->blob_name
                          << " step " << step << " chunk " << ci
                          << ": first differing element " << first << " of "
                          << (n / sizeof(double)) << " (field element "
                          << (off / sizeof(double) + first) << ")\n";
              }
            }
          }
        } else {
          bytes = reinterpret_cast<const char *>(src) + off;

          // Host staging, exactly as the VOL's clio_stage_chunk host branch:
          // a runtime-visible buffer the compressor can reach through the
          // ShmPtr.
          host_buf = CLIO_IPC->AllocateBuffer(n);
          if (host_buf.IsNull()) { std::cerr << "AllocateBuffer failed\n"; std::exit(1); }
          std::memcpy(host_buf.ptr_, bytes, n);
          blob_data = host_buf.shm_.template Cast<void>();
        }

        BlobRecord rec;
        rec.name = std::string(f->blob_name) + "/step_" + std::to_string(step) +
                   "/chunk_" + std::to_string(ci);
        rec.bytes = n;
        rec.digest = bytes ? Fnv1a(bytes, n) : 0;
        records.push_back(rec);
        if (!opt.raw_dir.empty() && bytes) {
          std::string fn = rec.name;
          for (auto &ch : fn) if (ch == '/') ch = '_';
          std::ofstream(opt.raw_dir + "/" + fn + ".bin", std::ios::binary)
              .write(bytes, static_cast<std::streamsize>(n));
        }

        Pending p;
        p.fut = compressor.AsyncDynamicSchedule(
            clio::run::PoolQuery::Local(), tag_id, rec.name, /*offset=*/0, n,
            blob_data, /*score=*/-1.0f, ctx, /*flags=*/0,
            cte_client->pool_id_);
        p.buf = host_buf;
        p.on_device = (dev_field != nullptr);
        p.record = records.size() - 1;
        pending.push_back(std::move(p));
        submitted += n;
      }
    }
    return submitted;
  };

  // ---- The loop. -------------------------------------------------------
  // `run 0` is setup: lmp->init(), Verlet::setup(1) -- neighbor lists, and
  // the FIRST force computation, so f at step 0 is meaningful. (A dump at
  // step 0 is written from inside that same setup, via output->setup.)
  const auto t_sim0 = std::chrono::steady_clock::now();
  double sim_s = 0.0, stage_s = 0.0;
  lammps_command(lmp, "run 0 post no");
  LmpCheck(lmp, "run 0");
  sim_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_sim0).count();

  size_t total_in = 0;
  {
    const auto t0 = std::chrono::steady_clock::now();
    total_in += store_frame(0);
    stage_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }

  // Each segment continues the SAME trajectory: `pre no` skips init/setup
  // (forces from the last step are still valid, neighbor-list age carries
  // over), `post no` skips the timing summary. Bit-identical to one
  // continuous `run STEPS` on the CPU. The previous frame's compressor
  // calls are in flight while LAMMPS advances, and are drained (and their
  // staging buffers freed) before the next frame is staged.
  const std::string seg_cmd = "run " + std::to_string(opt.gap) + " pre no post no";
  for (int64_t step = opt.gap; step <= opt.steps; step += opt.gap) {
    const auto t0 = std::chrono::steady_clock::now();
    lammps_command(lmp, seg_cmd.c_str());
    LmpCheck(lmp, "run segment");
    const auto t1 = std::chrono::steady_clock::now();
    drain();
    total_in += store_frame(step);
    const auto t2 = std::chrono::steady_clock::now();
    sim_s += std::chrono::duration<double>(t1 - t0).count();
    stage_s += std::chrono::duration<double>(t2 - t1).count();
  }
  {
    const auto t0 = std::chrono::steady_clock::now();
    drain();
    stage_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  const auto t_end_write = std::chrono::steady_clock::now();

  // ---- Report. ----------------------------------------------------------
  size_t stored_total = 0, kept = 0, raw = 0, failed = 0;
  std::map<std::string, std::pair<size_t, size_t>> per_field;  // in, stored
  std::map<int, size_t> per_lib;
  for (const auto &r : records) {
    if (!r.ok) { ++failed; continue; }
    stored_total += r.stored;
    if (r.lib != 0) { ++kept; ++per_lib[r.lib]; } else ++raw;
    const auto slash = r.name.find('/');
    auto &pf = per_field[r.name.substr(0, slash)];
    pf.first += r.bytes;
    pf.second += r.stored;
  }
  std::cout << std::fixed << std::setprecision(3)
            << "\nstored " << records.size() << " blob(s), " << total_in
            << " B in -> " << stored_total << " B on the tier  (ratio "
            << (stored_total ? double(total_in) / double(stored_total) : 0.0)
            << ")\n  compressed: " << kept << "   stored raw: " << raw
            << "   failed: " << failed << "\n";
  for (const auto &[name, io] : per_field)
    std::cout << "  " << std::setw(9) << name << "  " << io.first << " -> "
              << io.second << "  (" << (io.second ? double(io.first) / io.second : 0.0)
              << "x)\n";
  for (const auto &[lib, n] : per_lib)
    std::cout << "  codec " << std::setw(16)
              << (lib == 0 ? std::string("raw(not-beneficial)")
                           : ctp::CompressionFactory::NameForWireId(lib))
              << " : " << n
              << " chunk(s)\n";
  std::cout << "  time: simulate " << sim_s << " s   stage+compress(wait) "
            << stage_s << " s   total "
            << std::chrono::duration<double>(t_end_write - t_start).count()
            << " s" << std::endl;

  if (!opt.report.empty()) {
    std::ofstream csv(opt.report);
    // wire 0 is NOT brotli here. It is how the compressor marks "stored raw",
    // and brotli merely happens to occupy wire id 0 in the factory registry --
    // so NameForWireId(0) answers "brotli" and a chunk that was never
    // compressed reads as a codec that never ran. Name the outcome instead.
    csv << "blob,bytes,fnv1a64,lib,codec,ratio,stored,compress_ms,"
           "decompress_ms,rc\n";
    for (const auto &r : records)
      csv << r.name << ',' << r.bytes << ',' << std::hex << r.digest << std::dec
          << ',' << r.lib << ','
          << (r.lib == 0 ? std::string("raw(not-beneficial)")
                         : ctp::CompressionFactory::NameForWireId(r.lib))
          << ',' << r.ratio << ',' << r.stored << ',' << r.ms << ','
          << r.dt_ms << ',' << (r.ok ? 0 : 1) << '\n';
  }

  // ---- Verify: every blob back through the decompressor. ---------------
  int rc = failed ? 1 : 0;
  if (opt.verify) {
    // The digest comparison inside verify_records answers "did these bytes
    // come back unchanged". Under a positive error bound they are NOT meant
    // to, so a mismatch there says nothing: it cannot distinguish
    // quantization (expected) from the device-staging corruption the forced
    // --order device check exists to catch. Reporting it as a failure would
    // mark every correct lossy run FAILED; reporting it as a pass would claim
    // a guarantee that is not being tested.
    //
    // So: say plainly that the check does not apply, do not fail the run on
    // it, and do not pretend the corruption guard ran. The check that DOES
    // apply to lossy data is an element-wise comparison against the bound,
    // which needs the original values and so lives in the field-replay driver
    // (--check-bound), whose inputs are still on disk at verify time.
    const bool lossy = ctx.error_bound_ > 0.0;
    if (lossy) {
      verify_records(records);  // still exercises the full decompress path
      std::cout << "NOTE: error bound " << ctx.error_bound_
                << " is in force, so the bit-exact digest check above does "
                   "not apply and was not used to decide the exit code. The "
                   "--order device corruption guard is INACTIVE for this run."
                << std::endl;
    } else if (!verify_records(records)) {
      rc = 1;
    }
  }

  // Before lammps_kokkos_finalize(): the scratch lives in a Kokkos memory
  // space, and freeing it after Kokkos has torn that space down is a
  // use-after-free in the allocator.
  if (opt.crosscheck_host) {
    std::cout << (crosscheck_bad == 0 ? "CROSSCHECK: " : "CROSSCHECK FAILED: ")
              << crosscheck_ok << " of " << (crosscheck_ok + crosscheck_bad)
              << " device-gathered chunk(s) match lammps_gather_atoms byte "
                 "for byte, same frame same process"
              << std::endl;
  }

  if (opt.order == "device") {
    // After the final drain, so nothing is in flight, and nothing allocates
    // a backend after this point -- so no id can be recycled.
    for (auto &slot : device_slots) {
      if (!slot.alloc.IsNull()) CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, slot.alloc);
    }
    device_slots.clear();
    clio_lmp_device_scratch_free();
  }
  lammps_close(lmp);
  lammps_kokkos_finalize();
  lammps_mpi_finalize();
  clio::run::CLIO_RUNTIME_FINALIZE();
  // A failed crosscheck means the tier holds a correctly-round-tripping blob
  // of the WRONG bytes, which is the one outcome that must never exit 0.
  if (crosscheck_bad != 0 && rc == 0) rc = 6;
  return rc;
}
