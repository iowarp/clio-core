/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_field_replay.cc
 * @brief Replay simulation field dumps through Clio's compressor.
 *
 * The LAMMPS example (../neuropress_lammps_lib) drives a live simulation and
 * hands its arrays to Clio. Some codes cannot be embedded that way -- Nyx is
 * an AMReX application with no library interface, and upstream NeuroPress's
 * own Nyx benchmark does not embed it either: it patches Nyx to dump raw
 * float32 fields per FArrayBox per component, then sweeps those files
 * offline. This driver is the Clio side of that second shape.
 *
 * It reads a directory of flat binary field files, cuts each into chunks, and
 * submits every chunk to the compressor exactly as the LAMMPS driver does --
 * same Context, same AsyncDynamicSchedule, same selection knobs from compose.
 * What changes is only where the bytes come from.
 *
 * Files are replayed in lexical order, which the dumper makes chronological
 * by zero-padding the timestep (plt00000_..., plt00001_...). That matters
 * for the learning and exploration modes, whose whole point is that a
 * decision depends on the chunks that came before it.
 *
 * ELEMENT TYPE. Default is float32 (Context::data_type_ = 1), because that is
 * what the dumps hold and what the selection model was trained on -- and,
 * unlike the float64 LAMMPS workload, it is the width NeuroPress's byte
 * shuffle actually assumes. Use --f64 for float64 dumps.
 *
 * Run through the scripts in paper-benchmark/nyx.
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <clio_cte/compressor/compressor_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_ctp/compress/compress_factory.h>
#include <clio_runtime/clio_runtime.h>

namespace fs = std::filesystem;

namespace {

struct Options {
  std::string dir;                 // directory of field files
  std::string ext = ".f32";        // which files to replay
  size_t chunk = 4u << 20;         // bytes per compressor call; 0 = whole file
  std::string tag = "field_replay";
  std::string report;              // per-chunk CSV
  size_t max_files = 0;            // 0 = all
  bool f64 = false;                // dumps are float64 rather than float32
  bool verify = false;
  bool readback = false;           // read a previous report back, no files
  std::string dump_dir;            // write decompressed bytes here (readback)
};

void Usage(const char *argv0) {
  std::cerr
      << "usage: " << argv0 << " --dir DIR [options]\n"
      << "  --dir DIR        directory of flat binary field files\n"
      << "  --ext .f32       file extension to replay [.f32]\n"
      << "  --chunk BYTES    bytes per compressor call, 0 = whole file "
         "[4194304]\n"
      << "  --max-files N    replay only the first N files (lexical order)\n"
      << "  --f64            files hold float64 (default float32)\n"
      << "  --tag NAME       CTE tag [field_replay]\n"
      << "  --report CSV     per-chunk outcome\n"
      << "  --verify         read every blob back and compare\n"
      << "  --readback CSV   no files: read the blobs a previous run listed\n"
      << "  --dump-decompressed DIR  with --readback, write each decompressed\n"
      << "                   blob to DIR so an EXTERNAL tool can compare it\n"
      << "                   against the simulation's own output files\n";
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
    if (a == "--dir") o->dir = need("DIR");
    else if (a == "--ext") o->ext = need("EXT");
    else if (a == "--chunk") o->chunk = std::strtoull(need("BYTES"), nullptr, 10);
    else if (a == "--max-files") o->max_files = std::strtoull(need("N"), nullptr, 10);
    else if (a == "--tag") o->tag = need("NAME");
    else if (a == "--report") o->report = need("CSV");
    else if (a == "--readback") { o->readback = true; o->report = need("CSV"); }
    else if (a == "--dump-decompressed") o->dump_dir = need("DIR");
    else if (a == "--f64") o->f64 = true;
    else if (a == "--verify") o->verify = true;
    else if (a == "-h" || a == "--help") { Usage(argv[0]); std::exit(0); }
    else { std::cerr << "unknown option " << a << "\n"; Usage(argv[0]); return false; }
  }
  if (o->dir.empty() && !o->readback) { Usage(argv[0]); return false; }
  return true;
}

/* FNV-1a-64, the digest the sibling examples use, so results are comparable. */
uint64_t Fnv1a(const void *p, size_t n) {
  const auto *b = static_cast<const unsigned char *>(p);
  uint64_t h = 14695981039346656037ull;  // 0xcbf29ce484222325
  for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
  return h;
}

struct BlobRecord {
  std::string name;
  size_t bytes = 0;
  uint64_t digest = 0;
  int lib = 0;
  double ratio = 0.0;
  size_t stored = 0;
  double ms = 0.0;
  bool ok = false;
};

struct Pending {
  clio::run::Future<clio::cte::compressor::DynamicScheduleTask> fut;
  ctp::ipc::FullPtr<char> buf;
  size_t record;
};

/* Group a dump filename by its physical field, so the report can say which
   fields compress. The dumper writes fab%04d_comp%02d_<name>.f32, so the
   field is whatever follows the component index. Anything that does not
   match that shape is grouped under its stem, which is still useful. */
std::string FieldOf(const std::string &stem) {
  const size_t c = stem.find("_comp");
  if (c == std::string::npos) return stem;
  const size_t u = stem.find('_', c + 5);
  if (u == std::string::npos) return stem;
  return stem.substr(u + 1);
}

}  // namespace

int main(int argc, char **argv) {
  Options opt;
  if (!ParseArgs(argc, argv, &opt)) return 2;
  const auto t_start = std::chrono::steady_clock::now();

  // ---- Clio: attach to the runtime, which CLIO_WITH_RUNTIME=1 brings up
  // in THIS process, composed from CLIO_SERVER_CONF. Same call the HDF5 VOL
  // makes lazily; here it is the first thing main does.
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::cerr << "CLIO_CTE_CLIENT_INIT failed -- is CLIO_SERVER_CONF set and "
                 "CLIO_WITH_RUNTIME=1?\n";
    return 1;
  }
  auto *cte_client = CLIO_CTE_CLIENT;

  unsigned pool_major = 512, pool_minor = 0;
  if (const char *e = std::getenv("CLIO_REPLAY_COMPRESSOR_POOL"); e && *e) {
    if (std::sscanf(e, "%u.%u", &pool_major, &pool_minor) < 1) {
      std::cerr << "bad CLIO_REPLAY_COMPRESSOR_POOL\n";
      return 1;
    }
  }
  clio::cte::compressor::Client compressor(
      clio::run::PoolId(pool_major, pool_minor));

  auto tag = cte_client->AsyncGetOrCreateTag(opt.tag);
  tag.Wait();
  if (tag->GetReturnCode() != 0) { std::cerr << "GetOrCreateTag failed\n"; return 1; }
  const auto tag_id = tag->tag_id_;

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
        std::cerr << "  MISMATCH " << r.name << " rc=" << get->GetReturnCode() << "\n";
      }
      /* Optionally hand the decompressed bytes to an external checker. The
         digest above is computed by the same program that computed the
         original one, so it proves the round trip is self-consistent; writing
         the bytes out lets something else compare them against the
         simulation's own output and remove this program from the loop. */
      if (!opt.dump_dir.empty() && get->GetReturnCode() == 0) {
        std::string fn = r.name;
        for (auto &ch : fn) if (ch == '/') ch = '_';
        std::ofstream(opt.dump_dir + "/" + fn + ".bin", std::ios::binary)
            .write(buf.ptr_, static_cast<std::streamsize>(r.bytes));
      }
      CLIO_IPC->FreeBuffer(buf);
    }
    std::cout << (bad == 0 ? "VERIFIED: " : "FAILED: ") << (recs.size() - bad)
              << " of " << recs.size()
              << " blobs round-tripped bit-exact through the decompressor"
              << std::endl;
    return bad == 0;
  };

  // ---- Cold read-back: no files touched, the tier is the only source. ----
  if (opt.readback) {
    std::ifstream csv(opt.report);
    if (!csv) { std::cerr << "cannot open " << opt.report << "\n"; return 1; }
    std::vector<BlobRecord> recs;
    std::string line;
    std::getline(csv, line);  // header
    while (std::getline(csv, line)) {
      size_t p1 = line.find(','), p2 = line.find(',', p1 + 1),
             p3 = line.find(',', p2 + 1);
      if (p1 == std::string::npos || p2 == std::string::npos) continue;
      BlobRecord r;
      r.name = line.substr(0, p1);
      r.bytes = std::strtoull(line.substr(p1 + 1, p2 - p1 - 1).c_str(), nullptr, 10);
      r.digest = std::strtoull(line.substr(p2 + 1, p3 - p2 - 1).c_str(), nullptr, 16);
      recs.push_back(r);
    }
    std::cout << "cold read-back of " << recs.size() << " blob(s) listed in "
              << opt.report << std::endl;
    const bool ok = verify_records(recs);
    clio::run::CLIO_RUNTIME_FINALIZE();
    return ok ? 0 : 1;
  }

  // ---- Collect the field files, in replay order. ----
  std::vector<fs::path> files;
  std::error_code ec;
  for (const auto &e : fs::recursive_directory_iterator(opt.dir, ec)) {
    if (e.is_regular_file() && e.path().extension() == opt.ext) {
      files.push_back(e.path());
    }
  }
  if (ec || files.empty()) {
    std::cerr << "no " << opt.ext << " files under " << opt.dir << "\n";
    return 1;
  }
  // Lexical order == chronological, because the dumper zero-pads the
  // timestep. Sorting the FULL path keeps each timestep's files together
  // even when they live in per-timestep subdirectories.
  std::sort(files.begin(), files.end());
  if (opt.max_files && files.size() > opt.max_files) files.resize(opt.max_files);

  size_t payload = 0;
  for (const auto &f : files) payload += fs::file_size(f);

  const size_t elem = opt.f64 ? 8 : 4;
  std::cout << "field replay -> Clio compressor, runtime in-process\n"
            << "  " << files.size() << " file(s) under " << opt.dir
            << "  payload=" << payload / 1048576.0 << " MiB  ("
            << (opt.f64 ? "float64" : "float32") << ")\n"
            << "  chunk=" << (opt.chunk ? std::to_string(opt.chunk)
                                        : std::string("whole file"))
            << std::endl;

  // float32 = 1, float64 = 2 -- the same encoding the HDF5 VOL assigns from
  // an H5T_FLOAT's size. Lossless: error_bound_ stays 0.
  clio::cte::core::Context ctx;
  ctx.data_type_ = opt.f64 ? 2 : 1;

  std::vector<BlobRecord> records;
  std::vector<Pending> pending;
  double read_s = 0.0, stage_s = 0.0;

  auto drain = [&]() {
    for (auto &p : pending) {
      p.fut.Wait();
      auto &r = records[p.record];
      const auto &c = p.fut->context_;
      r.ok = p.fut->GetReturnCode() == 0;
      r.lib = c.compress_lib_;
      r.ratio = c.actual_compression_ratio_;
      r.ms = c.actual_compress_time_ms_;
      // lib == 0 marks "stored raw": the codec did not shrink the chunk and
      // the caller's bytes went to the tier untouched, so the stored size is
      // the original -- actual_compressed_size_ still reports the codec's
      // output, which is NOT what was stored.
      r.stored = (r.ok && r.lib != 0) ? c.actual_compressed_size_ : r.bytes;
      CLIO_IPC->FreeBuffer(p.buf);
    }
    pending.clear();
  };

  std::vector<char> filebuf;
  for (const auto &f : files) {
    const auto t0 = std::chrono::steady_clock::now();
    const size_t sz = fs::file_size(f);
    if (sz == 0) continue;
    if (sz % elem != 0) {
      std::cerr << "warning: " << f.filename().string() << " is " << sz
                << " bytes, not a whole number of "
                << (opt.f64 ? "float64" : "float32") << " elements\n";
    }
    filebuf.resize(sz);
    std::ifstream in(f, std::ios::binary);
    if (!in.read(filebuf.data(), static_cast<std::streamsize>(sz))) {
      std::cerr << "short read on " << f << "\n";
      return 1;
    }
    const auto t1 = std::chrono::steady_clock::now();
    read_s += std::chrono::duration<double>(t1 - t0).count();

    const size_t chunk = (opt.chunk == 0 || opt.chunk > sz) ? sz : opt.chunk;
    const size_t nchunks = (sz + chunk - 1) / chunk;
    const std::string stem = f.stem().string();
    // Name blobs by the timestep directory plus the file stem, so a blob
    // name identifies the frame it came from even though every timestep
    // reuses the same component filenames.
    const std::string frame = f.parent_path().filename().string();

    for (size_t ci = 0; ci < nchunks; ++ci) {
      const size_t off = ci * chunk;
      const size_t n = std::min(chunk, sz - off);
      const char *src = filebuf.data() + off;

      auto buf = CLIO_IPC->AllocateBuffer(n);
      if (buf.IsNull()) { std::cerr << "AllocateBuffer failed\n"; return 1; }
      std::memcpy(buf.ptr_, src, n);

      BlobRecord rec;
      rec.name = frame + "/" + stem + "/chunk_" + std::to_string(ci);
      rec.bytes = n;
      rec.digest = Fnv1a(src, n);
      records.push_back(rec);

      Pending p;
      p.fut = compressor.AsyncDynamicSchedule(
          clio::run::PoolQuery::Local(), tag_id, rec.name, 0, n,
          buf.shm_.template Cast<void>(), -1.0f, ctx, 0, cte_client->pool_id_);
      p.buf = buf;
      p.record = records.size() - 1;
      pending.push_back(std::move(p));
    }
    // Drain per file: the staging buffers of a 4 MiB-chunked 8 MiB file are
    // cheap, but a whole run's worth is not, and the exploration modes make
    // each chunk expensive enough that unbounded queueing buys nothing.
    drain();
    stage_s += std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t1).count();
  }
  drain();

  // ---- Report. ----
  size_t stored_total = 0, in_total = 0, kept = 0, raw = 0, failed = 0;
  std::map<std::string, std::pair<size_t, size_t>> per_field;
  std::map<int, size_t> per_lib;
  for (const auto &r : records) {
    if (!r.ok) { ++failed; continue; }
    in_total += r.bytes;
    stored_total += r.stored;
    if (r.lib != 0) { ++kept; ++per_lib[r.lib]; } else ++raw;
    const size_t s1 = r.name.find('/');
    const size_t s2 = r.name.find('/', s1 + 1);
    auto &pf = per_field[FieldOf(r.name.substr(s1 + 1, s2 - s1 - 1))];
    pf.first += r.bytes;
    pf.second += r.stored;
  }
  std::cout << std::fixed << std::setprecision(3)
            << "\nstored " << records.size() << " blob(s), " << in_total
            << " B in -> " << stored_total << " B on the tier  (ratio "
            << (stored_total ? double(in_total) / double(stored_total) : 0.0)
            << ")\n  compressed: " << kept << "   stored raw: " << raw
            << "   failed: " << failed << "\n";
  for (const auto &[name, io] : per_field)
    std::cout << "  " << std::setw(14) << name << "  " << io.first << " -> "
              << io.second << "  ("
              << (io.second ? double(io.first) / io.second : 0.0) << "x)\n";
  for (const auto &[lib, n] : per_lib)
    std::cout << "  codec " << std::setw(16)
              << ctp::CompressionFactory::NameForWireId(lib) << " : " << n
              << " chunk(s)\n";
  std::cout << "  time: read " << read_s << " s   stage+compress " << stage_s
            << " s   total "
            << std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t_start).count()
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

  int rc = failed ? 1 : 0;
  if (opt.verify && !verify_records(records)) rc = 1;

  clio::run::CLIO_RUNTIME_FINALIZE();
  return rc;
}
