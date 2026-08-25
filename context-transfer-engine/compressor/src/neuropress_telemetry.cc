/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_telemetry.cc
 * @brief The three NeuroPress diagnostic logs declared in
 * neuropress_telemetry.h.
 *
 * Each log's handle, mutex and sequence counter stay in the anonymous
 * namespace below: nothing outside this file may name one, so the only way to
 * write a log is through the five documented entry points. That is what keeps
 * the generic compressor runtime free of file handles it does not own.
 */

#include "clio_cte/compressor/neuropress_telemetry.h"

#include <clio_ctp/compress/compress_factory.h>
#include <clio_ctp/compress/model/predictor.h>
#include <clio_ctp/compress/model/ranking.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace clio::cte::compressor {

namespace {
/**
 * Per-chunk NeuroPress selection trace.
 *
 * Off unless CLIO_NEUROPRESS_SELECTION_LOG names a file, so this costs a single
 * relaxed load on the normal path. A selection is otherwise invisible: it
 * happens inside the compressor runtime, several layers below whatever issued
 * the write, and the caller only learns that the write succeeded.
 *
 * The `seq` column is a COMPLETION order, not a chunk order -- the HDF5 VOL
 * queues each chunk's DynamicSchedule asynchronously and drains on close, so
 * with online learning on, completion order IS the order the model is updated
 * in. A replay reproducing this run's learning must follow seq.
 */
struct SelectionLog {
  std::mutex mutex;
  std::FILE *fp = nullptr;
  long seq = 0;
};

/**
 * Companion to the selection log: a hash of the COMPRESSED payload each
 * chunk produced, written from Compress() where those bytes actually exist.
 *
 * Logged here rather than passed up to DynamicSchedule because the two run
 * as separate tasks and need not share a thread; keying on the blob name
 * lets a comparison join them afterwards without any cross-task plumbing.
 *
 * The hash covers the codec's own output only -- not Clio's 24-byte header
 * -- so it is directly comparable with NeuroPress's payload, whose 64-byte
 * header is likewise excluded.
 *
 * A chunk can appear TWICE, distinguished by the `stage` column, and the LAST
 * row for a blob is the one that describes what is on the tier:
 *
 *   primary  what the selected codec produced, from Compress()
 *   adopted  what exploration replaced it with, from DynamicSchedule
 *
 * The second row exists because Compress() runs before exploration can
 * override its result. With only the primary row, a chunk stored at the
 * winner's size was reported at the primary's -- always the larger of the
 * two, and only ever on the modes that explore, so a comparison built on
 * this log ranked exploration below configurations it actually beat. A
 * consumer that keys a map on blob name gets the right answer for free; one
 * that SUMS the rows must filter to the last per blob.
 */
struct PayloadLog {
  std::mutex mutex;
  std::FILE *fp = nullptr;
};

PayloadLog *PayloadLogInstance() {
  static PayloadLog *log = [] {
    auto *l = new PayloadLog();  // leaked on purpose, see SelectionLogInstance
    const char *path = std::getenv("CLIO_NEUROPRESS_SELECTION_LOG");
    if (path && *path) {
      std::string p = std::string(path) + ".payload";
      l->fp = std::fopen(p.c_str(), "w");
      if (l->fp) {
        std::fprintf(l->fp,
                     "blob,compressed_size,payload_hash,beneficial,"
                     "compress_kernel_ms,stage\n");
      }
    }
    return l;
  }();
  return log;
}



/**
 * Per-candidate record of an exploration sweep.
 *
 * The selection log above records ONE row per chunk: the configuration the
 * model chose. That is the right shape for auditing selection, and the wrong
 * shape for auditing exploration -- a sweep measures up to 31 alternatives per
 * chunk and the interesting question is what the other 30 actually cost, which
 * the selection log cannot express because by the time it is written the sweep
 * has not run.
 *
 * Enabled by CLIO_NEUROPRESS_EXPLORE_LOG (a path). Off otherwise, and the
 * measurement is not taken at all -- the sweep already runs only when the
 * error gate trips, and this must not add work to a path that does.
 */
struct ExploreLog {
  std::mutex mutex;
  std::FILE *fp = nullptr;
  long seq = 0;
};


ExploreLog *ExploreLogInstance() {
  static ExploreLog *log = [] {
    auto *l = new ExploreLog();
    const char *path = std::getenv("CLIO_NEUROPRESS_EXPLORE_LOG");
    if (path && *path) {
      l->fp = std::fopen(path, "w");
      if (l->fp) {
        // `rank` is the candidate's position in the model's own ranking, so a
        // reader can tell an alternative the model rated highly from one it
        // buried. `adopted` marks the row whose bytes actually replaced the
        // primary's, which is at most one per chunk and may be none.
        std::fprintf(l->fp,
                     "seq,blob,chunk_bytes,rank,lib_name,algo_idx,preset,"
                     "quantize,shuffle,pred_ratio,pred_ct_ms,ratio,ct_ms,"
                     "psnr_db,cost,primary_cost,adopted\n");
      }
    }
    return l;
  }();
  return log;
}

/**
 * @brief One explored candidate. `primary_cost` is repeated on every row so a
 * row is self-contained: the whole point of a sweep row is the comparison
 * against what the model actually chose.
 */

/**
 * Leaked on purpose, for the reason documented on
 * neuropress_nn_gpu_kernels.cu's Registry(): worker threads can still reach
 * this while static destructors are running, and tearing down a mutex (or
 * closing the stream) underneath a thread that is mid-write is a crash with
 * no upside. Nothing here needs releasing at exit -- the process is going
 * away and the stream is flushed on every row.
 */
SelectionLog *SelectionLogInstance() {
  static SelectionLog *log = [] {
    auto *l = new SelectionLog();
    const char *path = std::getenv("CLIO_NEUROPRESS_SELECTION_LOG");
    if (path && *path) {
      l->fp = std::fopen(path, "w");
      if (l->fp) {
        std::fprintf(l->fp,
                     "seq,blob,chunk_bytes,entropy,mad,second_deriv,wire_lib,"
                     "lib_name,algo_idx,quantize,shuffle,preset,pred_ratio,"
                     "pred_ct_ms,pred_dt_ms,pred_psnr,actual_ratio,"
                     "actual_ct_ms,actual_psnr,checksum\n");
      }
    }
    return l;
  }();
  return log;
}
}  // namespace

bool SelectionLogEnabled() {
  static const bool on = [] {
    const char *p = std::getenv("CLIO_NEUROPRESS_SELECTION_LOG");
    return p && *p;
  }();
  return on;
}

bool ExploreLogEnabled() {
  static const bool on = [] {
    const char *p = std::getenv("CLIO_NEUROPRESS_EXPLORE_LOG");
    return p && *p;
  }();
  return on;
}

void LogCompressedPayload(const std::string &blob_name, const char *payload,
                          size_t payload_size, bool on_device, bool beneficial,
                          double compress_kernel_ms, const char *stage) {
  PayloadLog *log = PayloadLogInstance();
  if (!log->fp || !payload || payload_size == 0) return;

  std::vector<char> staged;
  const unsigned char *p = reinterpret_cast<const unsigned char *>(payload);
  if (on_device) {
    staged.resize(payload_size);
    ctp::DeviceAwareMemcpy(staged.data(), payload, payload_size);
    p = reinterpret_cast<const unsigned char *>(staged.data());
  }
  unsigned long long h = 14695981039346656037ull;  // 0xcbf29ce484222325
  for (size_t i = 0; i < payload_size; ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  std::lock_guard<std::mutex> lock(log->mutex);
  std::fprintf(log->fp, "%s,%zu,%llu,%d,%.10g,%s\n", blob_name.c_str(),
               payload_size, h, beneficial ? 1 : 0, compress_kernel_ms, stage);
  std::fflush(log->fp);

  // Diagnostic: dump one chunk's raw payload so it can be diffed against
  // another implementation's byte for byte. Whole-payload hashes say THAT
  // two encoders disagree; only the bytes say where and how.
  const char *want = std::getenv("CLIO_NEUROPRESS_DUMP_CHUNK");
  if (want && *want) {
    // "all" dumps every chunk, which is what a whole-dataset cross-decode
    // needs; a bare index dumps just that one for a byte diff.
    const std::string suffix = std::string("/chunk_") + want;
    const bool all = (std::strcmp(want, "all") == 0);
    if (all || (blob_name.size() >= suffix.size() &&
        blob_name.compare(blob_name.size() - suffix.size(), suffix.size(),
                          suffix) == 0)) {
      const char *base = std::getenv("CLIO_NEUROPRESS_SELECTION_LOG");
      std::string idx = want;
      if (all) {
        const size_t pos = blob_name.rfind("/chunk_");
        idx = (pos == std::string::npos) ? "x"
                                         : blob_name.substr(pos + 7);
      }
      std::string dp = std::string(base ? base : "/tmp/clio") + ".chunk" +
                       idx + ".bin";
      if (std::FILE *df = std::fopen(dp.c_str(), "wb")) {
        std::fwrite(p, 1, payload_size, df);
        std::fclose(df);
      }
    }
  }
}

void LogNeuroPressExplore(const std::string &blob_name, size_t chunk_size,
                          int rank, const std::string &lib_name,
                          uint32_t preset_id, bool quantize, uint32_t shuffle,
                          double pred_ratio, double pred_ct_ms, double ratio,
                          double ct_ms, double psnr_db, double cost,
                          double primary_cost, bool adopted) {
  ExploreLog *log = ExploreLogInstance();
  if (!log->fp) return;

  int algo_idx = -1;
  for (const auto &entry : ctp::compress::model::KnownCompressors()) {
    if (lib_name == entry.name) {
      switch (entry.base_id) {
        case 13: algo_idx = 0; break;
        case 14: algo_idx = 1; break;
        case 17: algo_idx = 2; break;
        case 16: algo_idx = 3; break;
        case 15: algo_idx = 4; break;
        case 18: algo_idx = 5; break;
        case 23: algo_idx = 6; break;
        case 24: algo_idx = 7; break;
        default: algo_idx = -1; break;
      }
      break;
    }
  }

  std::lock_guard<std::mutex> lock(log->mutex);
  std::fprintf(log->fp,
               "%ld,%s,%zu,%d,%s,%d,%u,%d,%u,%.10g,%.10g,%.10g,%.10g,"
               "%.10g,%.10g,%.10g,%d\n",
               log->seq++, blob_name.c_str(), chunk_size, rank,
               lib_name.c_str(), algo_idx, preset_id, quantize ? 1 : 0,
               shuffle, pred_ratio, pred_ct_ms, ratio, ct_ms, psnr_db, cost,
               primary_cost, adopted ? 1 : 0);
  std::fflush(log->fp);
}

void LogNeuroPressSelection(const std::string &blob_name, size_t chunk_size,
                            double entropy, double mad, double second_deriv,
                            int wire_lib, int packed_preset,
                            const CompressionStats *predicted,
                            double actual_ratio, double actual_ct_ms,
                            double actual_psnr,
                            unsigned long long checksum) {
  SelectionLog *log = SelectionLogInstance();
  if (!log->fp) return;

  std::lock_guard<std::mutex> lock(log->mutex);
  std::FILE *fp = log->fp;

  const std::string lib_name =
      ctp::CompressionFactory::NameForWireId(wire_lib);
  // The 0-7 index NeuroPress's decodeAction uses, recovered through the ML
  // base id so the two sides' logs are directly comparable.
  int algo_idx = -1;
  for (const auto &entry : ctp::compress::model::KnownCompressors()) {
    if (lib_name == entry.name) {
      switch (entry.base_id) {
        case 13: algo_idx = 0; break;
        case 14: algo_idx = 1; break;
        case 17: algo_idx = 2; break;
        case 16: algo_idx = 3; break;
        case 15: algo_idx = 4; break;
        case 18: algo_idx = 5; break;
        case 23: algo_idx = 6; break;
        case 24: algo_idx = 7; break;
        default: algo_idx = -1; break;
      }
      break;
    }
  }
  const uint32_t bits = static_cast<uint32_t>(packed_preset);
  const int quantize = ((bits >> 24) & 1u) ? 1 : 0;
  const int shuffle = (((bits >> 8) & 0xFFu) != 0u) ? 1 : 0;
  const int preset = static_cast<int>(bits & 0xFFu);

  std::fprintf(fp,
               "%ld,%s,%zu,%.10g,%.10g,%.10g,%d,%s,%d,%d,%d,%d,"
               "%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%llu\n",
               log->seq++, blob_name.c_str(), chunk_size, entropy, mad,
               second_deriv, wire_lib, lib_name.c_str(), algo_idx, quantize,
               shuffle, preset,
               predicted ? predicted->compression_ratio_ : 0.0,
               predicted ? predicted->compress_time_ms_ : 0.0,
               predicted ? predicted->decompress_time_ms_ : 0.0,
               predicted ? predicted->psnr_db_ : 0.0,
               actual_ratio, actual_ct_ms, actual_psnr, checksum);
  std::fflush(fp);
}

}  // namespace clio::cte::compressor
