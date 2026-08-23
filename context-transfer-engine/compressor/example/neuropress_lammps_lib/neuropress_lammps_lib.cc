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
#include <clio_runtime/clio_runtime.h>

namespace {

struct Options {
  int box = 10;       // lattice cells per side; natoms = 4*box^3
  int steps = 100;    // total timesteps
  int gap = 50;       // store every `gap` steps (frame 0 included)
  size_t chunk = 4u << 20;  // bytes per compressor call; 0 = whole field
  std::string fields = "position,velocity,force";
  std::string order = "id";  // id (gather, == dump h5md) | local (extract)
  std::string deck;          // LAMMPS input deck (setup only, no `run`)
  std::string tag = "lammps_lib";
  std::string logfile = "none";  // LAMMPS -log
  std::string report;            // per-blob CSV
  std::string raw_dir;           // also write every staged blob's bytes here
  std::string readback;          // CSV from --report: read those blobs, no LAMMPS
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
      << "  --order id|local id: atom-ID order via lammps_gather_atoms "
         "(byte-identical to dump h5md)\n"
      << "                   local: LAMMPS' internal order via "
         "lammps_extract_atom, no gather copy\n"
      << "  --kokkos         run LAMMPS on the GPU (-k on g 1 -sf kk)\n"
      << "  --verify         read every blob back and compare\n"
      << "  --report CSV     per-blob outcome\n"
      << "  --raw DIR        also write each staged blob's raw bytes to DIR "
         "(cross-checks)\n"
      << "  --readback CSV   no simulation: read every blob named in a "
         "previous --report CSV back\n"
      << "                   through the decompressor and compare digests "
         "(run with CLIO_RESTART=1)\n"
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
    else if (a == "--deck") o->deck = need("FILE");
    else if (a == "--tag") o->tag = need("NAME");
    else if (a == "--log") o->logfile = need("FILE");
    else if (a == "--report") o->report = need("CSV");
    else if (a == "--raw") o->raw_dir = need("DIR");
    else if (a == "--readback") o->readback = need("CSV");
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
  if (o->order != "id" && o->order != "local") return false;
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

struct BlobRecord {
  std::string name;
  size_t bytes = 0;
  uint64_t digest = 0;
  // Filled in once the compressor answers.
  int lib = 0;
  double ratio = 0.0;
  size_t stored = 0;
  double ms = 0.0;
  bool ok = false;
};

struct Pending {
  clio::run::Future<clio::cte::compressor::DynamicScheduleTask> fut;
  ctp::ipc::FullPtr<char> buf;
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
      const bool ok = get->GetReturnCode() == 0 &&
                      Fnv1a(buf.ptr_, r.bytes) == r.digest;
      if (!ok) {
        ++bad;
        std::cerr << "  MISMATCH " << r.name << " rc=" << get->GetReturnCode()
                  << "\n";
      }
      CLIO_IPC->FreeBuffer(buf);
    }
    std::cout << (bad == 0 ? "VERIFIED: " : "FAILED: ") << (recs.size() - bad)
              << " of " << recs.size()
              << " blobs round-tripped bit-exact through the decompressor"
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

  std::vector<const Field *> fields;
  for (const auto &f : kFields)
    if (opt.fields.find(f.blob_name) != std::string::npos) fields.push_back(&f);
  if (fields.empty()) { std::cerr << "no fields selected\n"; return 1; }

  const size_t field_bytes = static_cast<size_t>(natoms) * 3 * sizeof(double);
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
            << " MiB" << std::endl;

  // Context for every chunk: float64, lossless. data_type_ 2 is what the VOL
  // reports for an 8-byte H5T_FLOAT; the compressor converts on that flag
  // instead of reading doubles as pairs of float32 (see clio_vol.cc
  // clio_context_data_type). error_bound_ stays 0 = lossless.
  clio::cte::core::Context ctx;
  ctx.data_type_ = 2;

  std::vector<BlobRecord> records;
  std::vector<Pending> pending;
  std::vector<double> gathered;  // --order id staging, natoms*3
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
      // compress_lib_ == 0 is how the compressor marks "stored raw": the
      // codec it tried did not shrink the chunk, so the caller's bytes went
      // to the tier as they were (compressor_runtime.cc, "Compression not
      // beneficial"). actual_compressed_size_ still reports the codec's
      // output in that case, which is NOT what was stored.
      r.stored = (r.ok && r.lib != 0) ? c.actual_compressed_size_ : r.bytes;
      CLIO_IPC->FreeBuffer(p.buf);
    }
    pending.clear();
  };

  // Stage one frame: every selected field, chunked, one AsyncDynamicSchedule
  // per chunk. Returns bytes submitted.
  auto store_frame = [&](int64_t step) -> size_t {
    size_t submitted = 0;
    for (const Field *f : fields) {
      const double *src = nullptr;
      if (opt.order == "id") {
        // Global array ordered by atom ID -- what dump h5md writes after its
        // own sort(). One copy (LAMMPS' internal order -> ID order); the
        // sibling h5md path does the same copy inside Dump::write, then a
        // second one into the dump_* arrays, then a third in the VOL.
        gathered.resize(static_cast<size_t>(natoms) * 3);
        lammps_gather_atoms(lmp, f->lmp_name, /*dtype=*/1, /*count=*/3,
                            gathered.data());
        LmpCheck(lmp, "gather_atoms");
        src = gathered.data();
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
        const char *bytes = reinterpret_cast<const char *>(src) + off;

        // Host staging, exactly as the VOL's clio_stage_chunk host branch:
        // a runtime-visible buffer the compressor can reach through the
        // ShmPtr. (A device-resident producer would register a GPU backend
        // instead -- see neuropress_gpu_direct.cc.)
        auto buf = CLIO_IPC->AllocateBuffer(n);
        if (buf.IsNull()) { std::cerr << "AllocateBuffer failed\n"; std::exit(1); }
        std::memcpy(buf.ptr_, bytes, n);

        BlobRecord rec;
        rec.name = std::string(f->blob_name) + "/step_" + std::to_string(step) +
                   "/chunk_" + std::to_string(ci);
        rec.bytes = n;
        rec.digest = Fnv1a(bytes, n);
        records.push_back(rec);
        if (!opt.raw_dir.empty()) {
          std::string fn = rec.name;
          for (auto &ch : fn) if (ch == '/') ch = '_';
          std::ofstream(opt.raw_dir + "/" + fn + ".bin", std::ios::binary)
              .write(bytes, static_cast<std::streamsize>(n));
        }

        Pending p;
        p.fut = compressor.AsyncDynamicSchedule(
            clio::run::PoolQuery::Local(), tag_id, rec.name, /*offset=*/0, n,
            buf.shm_.template Cast<void>(), /*score=*/-1.0f, ctx, /*flags=*/0,
            cte_client->pool_id_);
        p.buf = buf;
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
              << ctp::CompressionFactory::NameForWireId(lib) << " : " << n
              << " chunk(s)\n";
  std::cout << "  time: simulate " << sim_s << " s   stage+compress(wait) "
            << stage_s << " s   total "
            << std::chrono::duration<double>(t_end_write - t_start).count()
            << " s" << std::endl;

  if (!opt.report.empty()) {
    std::ofstream csv(opt.report);
    csv << "blob,bytes,fnv1a64,lib,codec,ratio,stored,compress_ms,rc\n";
    for (const auto &r : records)
      csv << r.name << ',' << r.bytes << ',' << std::hex << r.digest << std::dec
          << ',' << r.lib << ',' << ctp::CompressionFactory::NameForWireId(r.lib)
          << ',' << r.ratio << ',' << r.stored << ',' << r.ms << ','
          << (r.ok ? 0 : 1) << '\n';
  }

  // ---- Verify: every blob back through the decompressor. ---------------
  int rc = failed ? 1 : 0;
  if (opt.verify && !verify_records(records)) rc = 1;

  lammps_close(lmp);
  lammps_kokkos_finalize();
  lammps_mpi_finalize();
  clio::run::CLIO_RUNTIME_FINALIZE();
  return rc;
}
