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
  // Check |original - decoded| <= error bound instead of a bit-exact digest.
  // Only meaningful with a positive CLIO_NEUROPRESS_ERROR_BOUND: under lossy
  // compression the digest MUST differ, so --verify alone can only report a
  // failure that is not one. This re-reads each chunk from its source file
  // and compares element-wise, which is the only check that actually tests
  // the guarantee the error bound makes.
  bool check_bound = false;
};

void Usage(const char *argv0) {
  std::cerr
      << "usage: " << argv0 << " --dir DIR [options]\n"
      << "  --dir DIR        directory of flat binary field files\n"
      << "  --ext .f32       file extension to replay [.f32]\n"
      << "  --check-bound    with a positive CLIO_NEUROPRESS_ERROR_BOUND,\n"
      << "                   verify |original-decoded| <= bound element-wise\n"
      << "                   against the source files instead of comparing a\n"
      << "                   digest, which lossy compression must fail\n"
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
    else if (a == "--check-bound") o->check_bound = true;
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

/**
 * Absolute error bound for error-bounded lossy compression, from
 * CLIO_NEUROPRESS_ERROR_BOUND. 0 (the default) means LOSSLESS and is what this
 * driver did unconditionally before -- NeuroPress's ranking masks all 16
 * quantize actions to -INFINITY at 0, so the search covers the 16 lossless
 * configurations only. A positive bound makes the other half reachable.
 *
 * Same name and same meaning as the knob neuropress_explore_sweep.cc already
 * reads, and as gpucompress_config_t::error_bound upstream.
 */
double ErrorBoundFromEnv() {
  const char *e = std::getenv("CLIO_NEUROPRESS_ERROR_BOUND");
  if (e == nullptr || *e == '\0') return 0.0;
  const double v = std::atof(e);
  return v > 0.0 ? v : 0.0;
}

/**
 * Source file and offset a blob came from, recovered from its NAME.
 *
 * Blobs are named "<frame>/<stem>/chunk_<i>" (see where records are built), so
 * the original bytes are at `<dir>/<frame>/<stem><ext>` + i*chunk. Deriving it
 * rather than storing it keeps the cold read-back path working too: that one
 * rebuilds its records from blobs.csv, which carries the name and nothing
 * about where the data came from.
 *
 * Returns false when the name does not have that shape (a caller that named
 * its blobs differently), which the bound check reports rather than skips.
 */
bool SourceOfBlob(const std::string &name, const std::string &dir,
                  const std::string &ext, size_t chunk,
                  std::string *path, size_t *offset) {
  const size_t c = name.rfind("/chunk_");
  if (c == std::string::npos) return false;
  const size_t slash = name.rfind('/', c - 1);
  if (slash == std::string::npos) return false;
  const std::string frame = name.substr(0, slash);
  const std::string stem = name.substr(slash + 1, c - slash - 1);
  const size_t idx = std::strtoull(name.c_str() + c + 7, nullptr, 10);
  *path = dir + "/" + frame + "/" + stem + ext;
  *offset = idx * chunk;
  return true;
}

struct BlobRecord {
  std::string name;
  size_t bytes = 0;
  uint64_t digest = 0;
  int lib = 0;
  double ratio = 0.0;
  size_t stored = 0;
  double ms = 0.0;
  // MEASURED decompression time (CUDA events around the codec call alone), or
  // <0 when nothing measured one. Needs CLIO_NEUROPRESS_EXPLORE_MEASURE_DT.
  // Same field, same clock and same caveat as the LAMMPS driver's dt_ms: NOT
  // comparable with a read-path Decompress(), which times the whole path.
  double dt_ms = -1.0;
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

  // Error bound in force for this run, so verification can pick the check that
  // actually applies: bit-exact for lossless, |orig-decoded| <= eb for lossy.
  const double verify_eb = ErrorBoundFromEnv();
  size_t bound_checked = 0, bound_exceeded = 0, bound_unreadable = 0;
  double bound_worst = 0.0;

  auto verify_records = [&](const std::vector<BlobRecord> &recs) -> bool {
    size_t bad = 0;
    std::vector<char> srcbuf;
    for (const auto &r : recs) {
      auto buf = CLIO_IPC->AllocateBuffer(r.bytes);
      if (buf.IsNull()) { std::cerr << "AllocateBuffer (verify) failed\n"; return false; }
      std::memset(buf.ptr_, 0, r.bytes);
      auto get = compressor.AsyncDecompressExplicit(
          clio::run::PoolQuery::Local(), tag_id, r.name, 0, r.bytes, 0,
          buf.shm_.template Cast<void>(), cte_client->pool_id_);
      get.Wait();
      const int rc_get = get->GetReturnCode();

      // Which check applies is decided by the error bound, not by a separate
      // flag: under lossy compression the decoded bytes are NOT the input
      // bytes by construction, so a digest comparison there reports a failure
      // that is not one. --check-bound asks for the element-wise comparison
      // that tests the guarantee the bound actually makes.
      bool ok;
      if (opt.check_bound && verify_eb > 0.0) {
        ok = rc_get == 0;
        std::string src;
        size_t off = 0;
        if (ok && SourceOfBlob(r.name, opt.dir, opt.ext, opt.chunk, &src, &off)) {
          std::ifstream in(src, std::ios::binary);
          if (in) {
            srcbuf.resize(r.bytes);
            in.seekg(static_cast<std::streamoff>(off));
            in.read(srcbuf.data(), static_cast<std::streamsize>(r.bytes));
            if (in.gcount() == static_cast<std::streamsize>(r.bytes)) {
              // Compare at the element width the run declared. A float64
              // replay quantizes as float64; reading it as float32 would
              // compare halves of values against each other.
              double worst = 0.0;
              if (opt.f64) {
                const auto *a = reinterpret_cast<const double *>(srcbuf.data());
                const auto *b = reinterpret_cast<const double *>(buf.ptr_);
                for (size_t i = 0; i < r.bytes / sizeof(double); ++i)
                  worst = std::max(worst, std::fabs(a[i] - b[i]));
              } else {
                const auto *a = reinterpret_cast<const float *>(srcbuf.data());
                const auto *b = reinterpret_cast<const float *>(buf.ptr_);
                for (size_t i = 0; i < r.bytes / sizeof(float); ++i)
                  worst = std::max(worst,
                                   std::fabs(static_cast<double>(a[i]) -
                                             static_cast<double>(b[i])));
              }
              ++bound_checked;
              bound_worst = std::max(bound_worst, worst);
              if (worst > verify_eb) {
                ++bound_exceeded;
                std::cerr << "  BOUND EXCEEDED " << r.name << " max|err|="
                          << worst << " > eb=" << verify_eb << "\n";
              }
            } else {
              ++bound_unreadable;
            }
          } else {
            ++bound_unreadable;
          }
        } else if (ok) {
          ++bound_unreadable;
        }
      } else {
        ok = rc_get == 0 && Fnv1a(buf.ptr_, r.bytes) == r.digest;
      }
      if (!ok) {
        ++bad;
        std::cerr << "  MISMATCH " << r.name << " rc=" << rc_get << "\n";
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
    // Say which check ran. In bound mode the bytes are NOT expected to match
    // bit for bit, so claiming they did would be false on every lossy run --
    // `bad` there counts decompression failures only, and the verdict on the
    // data is the BOUND line below.
    const bool bound_mode = opt.check_bound && verify_eb > 0.0;
    std::cout << (bad == 0 ? "VERIFIED: " : "FAILED: ") << (recs.size() - bad)
              << " of " << recs.size()
              << (bound_mode
                      ? " blobs decompressed without error (bound checked"
                        " separately below)"
                      : " blobs round-tripped bit-exact through the"
                        " decompressor")
              << std::endl;
    return bad == 0;
  };

  // One line the harness greps, in the same shape as VERIFIED:/FAILED: above.
  // A chunk whose source could not be re-read is reported, never counted as a
  // pass -- the whole point of this check is that it not be silently vacuous.
  auto report_bound = [&]() {
    if (!(opt.check_bound && verify_eb > 0.0)) return true;
    const bool ok = bound_exceeded == 0 && bound_checked > 0;
    std::cout << (ok ? "BOUND OK: " : "BOUND FAILED: ") << bound_checked
              << " chunk(s) checked against eb=" << verify_eb
              << ", " << bound_exceeded << " exceeded, worst max|err|="
              << bound_worst;
    if (bound_unreadable) std::cout << ", " << bound_unreadable
                                    << " source(s) unreadable";
    std::cout << std::endl;
    return ok;
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
    const bool ok = verify_records(recs) && report_bound();
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
  // an H5T_FLOAT's size. error_bound_ is 0 (lossless) unless
  // CLIO_NEUROPRESS_ERROR_BOUND asks otherwise; see ErrorBoundFromEnv.
  clio::cte::core::Context ctx;
  ctx.data_type_ = opt.f64 ? 2 : 1;
  ctx.error_bound_ = ErrorBoundFromEnv();
  if (ctx.error_bound_ > 0.0)
    std::cout << "  error bound=" << ctx.error_bound_
              << "  LOSSY (quantize actions enabled)" << std::endl;

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
      r.dt_ms = c.actual_decompress_time_ms_;
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
            << " B in -> " << stored_total << " B on the tier  (stored ratio "
            << (stored_total ? double(in_total) / double(stored_total) : 0.0)
            << ")\n  compressed: " << kept << "   stored raw: " << raw
            << "   failed: " << failed << "\n";
  for (const auto &[name, io] : per_field)
    std::cout << "  " << std::setw(14) << name << "  " << io.first << " -> "
              << io.second << "  ("
              << (io.second ? double(io.first) / io.second : 0.0) << "x)\n";
  for (const auto &[lib, n] : per_lib)
    std::cout << "  codec " << std::setw(16)
              << (lib == 0 ? std::string("raw(not-beneficial)")
                           : ctp::CompressionFactory::NameForWireId(lib))
              << " : " << n
              << " chunk(s)\n";
  std::cout << "  time: read " << read_s << " s   stage+compress " << stage_s
            << " s   total "
            << std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t_start).count()
            << " s" << std::endl;

  if (!opt.report.empty()) {
    std::ofstream csv(opt.report);
    // TWO ratios, because they answer different questions and disagreed
    // silently before this column existed:
    //   ratio        = bytes / CODEC PAYLOAD. Upstream's definition
    //                  (gpucompress_compress.cpp excludes its own header),
    //                  the NN's training label, and the number explore.csv
    //                  and selection.csv report. Do not change it: it feeds
    //                  the cost model and the SGD targets.
    //   stored_ratio = bytes / STORED. What the tier actually holds,
    //                  header included, and the basis of the aggregate
    //                  "(stored ratio N)" line printed above.
    // They differ by the 24-byte CTEC header (56 when quantized), so
    // sum(bytes)/sum(stored) never equals the mean of the `ratio` column.
    // Appended, never inserted: the first three columns are the cold
    // reader's contract.
    // wire 0 is NOT brotli here. It is how the compressor marks "stored raw",
    // and brotli merely happens to occupy wire id 0 in the factory registry --
    // so NameForWireId(0) answers "brotli" and a chunk that was never
    // compressed reads as a codec that never ran. Name the outcome instead.
    csv << "blob,bytes,fnv1a64,lib,codec,ratio,stored,compress_ms,"
           "decompress_ms,rc,stored_ratio\n";
    for (const auto &r : records)
      csv << r.name << ',' << r.bytes << ',' << std::hex << r.digest << std::dec
          << ',' << r.lib << ','
          << (r.lib == 0 ? std::string("raw(not-beneficial)")
                         : ctp::CompressionFactory::NameForWireId(r.lib))
          << ',' << r.ratio << ',' << r.stored << ',' << r.ms << ','
          << r.dt_ms << ',' << (r.ok ? 0 : 1) << ','
          << (r.stored ? double(r.bytes) / double(r.stored) : 0.0) << '\n';
  }

  int rc = failed ? 1 : 0;
  if (opt.verify && !verify_records(records)) rc = 1;
  // A bound violation is a FAILED run. Evaluated after verify_records, which
  // is what populates the counters, and independently of it: under --check-bound
  // verify_records only fails on a decompress error, so without this a run
  // whose data came back outside the bound would exit 0.
  if (opt.verify && !report_bound()) rc = 1;

  clio::run::CLIO_RUNTIME_FINALIZE();
  return rc;
}
