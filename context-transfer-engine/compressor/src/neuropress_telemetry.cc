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
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace clio::cte::compressor {

namespace {
/** Per-chunk selection trace. `seq` is COMPLETION order, not chunk order --
 *  with online learning on it IS the order the model is updated in, so a
 *  replay must follow seq. */
struct SelectionLog {
  std::mutex mutex;
  std::FILE *fp = nullptr;
  long seq = 0;
};

/** Hash of the COMPRESSED payload, codec output only (no 24-byte header), so
 *  it compares directly with NeuroPress's. A chunk appears twice -- primary,
 *  then adopted -- and the LAST row per blob is what is on the tier. */
struct PayloadLog {
  std::mutex mutex;
  std::FILE *fp = nullptr;
};

struct QualityLog {
  std::FILE *fp = nullptr;
  std::mutex mutex;
  long seq = 0;
};

QualityLog *QualityLogInstance() {
  static QualityLog *log = [] {
    auto *l = new QualityLog();  // leaked on purpose, see SelectionLogInstance
    const char *path = std::getenv("CLIO_NEUROPRESS_SELECTION_LOG");
    if (path && *path) {
      std::string p = std::string(path) + ".quality";
      l->fp = std::fopen(p.c_str(), "w");
      if (l->fp) {
        // MEASURED, every one of them -- by reconstructing the stored chunk
        // and comparing with the original. Distinct from selection.csv's
        // actual_psnr, which is ANALYTICAL: derived from (range, error_bound),
        // capped at 120 dB, and structurally unable to see a bound violation.
        std::fprintf(l->fp,
                     "seq,blob,orig_bytes,elements,shuffle,quantized,"
                     // psnr_db keeps upstream's 120 dB cap. ssim_deviation
                     // is 1-ssim computed WITHOUT the subtraction: ssim
                     // saturates to 1 for any good reconstruction, so the
                     // deviation is where the information is and it cannot be
                     // recovered afterwards.
                     "rmse,max_error,psnr_db,ssim,ssim_deviation,"
                     "data_range\n");
      }
    }
    return l;
  }();
  return log;
}

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



/** Per-candidate sweep record (CLIO_NEUROPRESS_EXPLORE_LOG). Off means the
 *  measurement is not taken at all -- this must not add work to the gated
 *  path the sweep already runs on. */
/** The scalar fields of one explore row, bundled so the writer takes a single
 *  argument instead of twenty-two. Nothing is deferred: every row is written
 *  as soon as it is built. */
struct ExploreRowFields {
  std::string blob_name;
  size_t chunk_size = 0;
  int rank = -1;
  std::string lib_name;
  uint32_t preset_id = 0;
  bool quantize = false;
  uint32_t shuffle = 0;
  double pred_ratio = 0, pred_ct_ms = 0, pred_dt_ms = 0;
  double ratio = 0, ct_ms = 0, psnr_db = 0, cost = 0, primary_cost = 0;
  bool adopted = false;
  bool is_primary = false;
  double dt_ms = -1.0;
  double entropy = -1.0, mad = -1.0, second_deriv = -1.0, eb_encoded = -1.0;
};

struct ExploreLog {
  std::mutex mutex;
  std::FILE *fp = nullptr;
  long seq = 0;
  /** The primary's measured quality, parked by Runtime::Compress for the
   *  Runtime::DynamicSchedule site that writes its row.
   *
   *  The direction matters and is easy to get backwards: DynamicSchedule
   *  AWAITS Compress (compressor_runtime.cc:1388) and only logs afterwards, so
   *  the measurement exists BEFORE the row is written -- it just exists in a
   *  different function. Parking the quality is therefore enough; deferring
   *  the row instead would park it forever, since nothing runs after the
   *  logging site to release it.
   *
   *  Keyed by blob name, which is unique per chunk. Erased on take, so at most
   *  one entry per chunk in flight. */
  std::map<std::string, ctp::compress::preprocess::QualityMetrics>
      primary_quality;
};


ExploreLog *ExploreLogInstance() {
  static ExploreLog *log = [] {
    auto *l = new ExploreLog();
    const char *path = std::getenv("CLIO_NEUROPRESS_EXPLORE_LOG");
    if (path && *path) {
      l->fp = std::fopen(path, "w");
      if (l->fp) {
        // role=primary is the model's own pick, role=alt a swept alternative.
        // `adopted` marks whichever row's bytes are stored: exactly one per
        // chunk, the primary unless an alternative beat it.
        std::fprintf(l->fp,
                     "seq,blob,chunk_bytes,role,rank,lib_name,algo_idx,preset,"
                     "quantize,shuffle,pred_ratio,pred_ct_ms,pred_dt_ms,"
                     "ratio,ct_ms,"
                     "dt_ms,psnr_db,cost,primary_cost,adopted,"
                     // MEASURED, by reconstructing the candidate all the way
                     // back and comparing with the original. psnr_db above is
                     // ANALYTICAL and is a different quantity; meas_psnr_db is
                     // the one that can see a bound violation.
                     // quality_measured disambiguates: ssim's valid range
                     // includes -1, so a bare sentinel could not.
                     "quality_measured,meas_rmse,meas_max_error,"
                     "meas_psnr_db,meas_ssim,meas_ssim_deviation,"
                     // The model's own eight inputs, so a row explains the
                     // prediction beside it. algo_idx/quantize/shuffle and
                     // chunk_bytes are already above; these are the rest.
                     "entropy,mad,second_deriv,eb_encoded\n");
      }
    }
    return l;
  }();
  return log;
}

/** One explored candidate. `primary_cost` repeats on every row so the
 *  comparison against the model's own pick is self-contained. */

/** Leaked on purpose: worker threads can reach this during static
 *  destruction, and tearing the mutex down under a mid-write thread crashes
 *  for no gain. Flushed on every row, so nothing is lost at exit. */
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
                     "actual_ct_ms,actual_psnr,checksum,role\n");
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

void LogMeasuredQuality(const std::string &blob_name, size_t orig_bytes,
                        uint32_t shuffle, bool quantized,
                        const ctp::compress::preprocess::QualityMetrics &q) {
  QualityLog *log = QualityLogInstance();
  if (!log->fp) return;
  std::lock_guard<std::mutex> lock(log->mutex);
  std::fprintf(log->fp,
               "%ld,%s,%zu,%zu,%u,%d,%.10g,%.10g,%.10g,%.17g,%.10g,%.10g\n",
               log->seq++, blob_name.c_str(), orig_bytes, q.n, shuffle,
               quantized ? 1 : 0, q.rmse, q.max_error, q.psnr_db,
               // 17 digits: ssim near 1 needs every one of them to be
               // reconstructible, and ssim_deviation beside it is the value
               // that does not need them.
               q.ssim, q.ssim_deviation, q.data_range);
  std::fflush(log->fp);
}

namespace {

/** Write one row. Caller holds log->mutex.
 *
 * The five quality columns and quality_measured are NA whenever a measured
 * quality is not DEFINED for the row, and that is not the same as zero:
 *
 *   quantize=0            -> NA. Every nvcomp codec is lossless and the byte
 *                            shuffle is a permutation, so the round trip is
 *                            exact by construction. A 0 here would look like a
 *                            measurement of the data when it is really a
 *                            tautology about the codec.
 *   quantize=1, measured  -> the measured value.
 *   quantize=1, no value  -> NA with quality_measured=0. A real failure, and
 *                            the quantize column keeps it distinguishable from
 *                            the not-applicable case above.
 *
 * NA rather than the old -1 sentinel because -1 is inside SSIM's valid range,
 * and because every CSV reader already treats NA as missing. */
void WriteExploreRowLocked(ExploreLog *log, const ExploreRowFields &r,
                           const ctp::compress::preprocess::QualityMetrics
                               *quality) {
  int algo_idx = -1;
  for (const auto &entry : ctp::compress::model::KnownCompressors()) {
    if (r.lib_name == entry.name) {
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

  const bool measured = r.quantize && quality != nullptr;
  char q[5][40];
  const char *qc[5] = {"NA", "NA", "NA", "NA", "NA"};
  if (measured) {
    std::snprintf(q[0], sizeof(q[0]), "%.10g", quality->rmse);
    std::snprintf(q[1], sizeof(q[1]), "%.10g", quality->max_error);
    std::snprintf(q[2], sizeof(q[2]), "%.10g", quality->psnr_db);
    // 17 digits: ssim near 1 needs every one of them to be reconstructible.
    std::snprintf(q[3], sizeof(q[3]), "%.17g", quality->ssim);
    std::snprintf(q[4], sizeof(q[4]), "%.10g", quality->ssim_deviation);
    for (int i = 0; i < 5; ++i) qc[i] = q[i];
  }
  // NA, not 0, when the row cannot have one: a lossless action has no
  // measurement to report, which is different from having failed to take one.
  const char *qmeas = r.quantize ? (quality != nullptr ? "1" : "0") : "NA";

  // dt_ms is -1 unless CLIO_NEUROPRESS_EXPLORE_MEASURE_DT made the sweep
  // decompress each candidate back. -1 means "not measured", which is the
  // default and upstream's only behaviour -- distinct from a measured 0.
  //
  // pred_dt_ms has no such caveat: the dt head runs for every action in the
  // one inference that ranked the chunk, so it is always a real prediction,
  // and it is what dt_ms should be compared against.
  std::fprintf(log->fp,
               "%ld,%s,%zu,%s,%d,%s,%d,%u,%d,%u,%.10g,%.10g,%.10g,%.10g,"
               "%.10g,%.10g,%.10g,%.10g,%.10g,%d,"
               "%s,%s,%s,%s,%s,%s,"
               "%.10g,%.10g,%.10g,%.10g\n",
               log->seq++, r.blob_name.c_str(), r.chunk_size,
               r.is_primary ? "primary" : "alt", r.rank,
               r.lib_name.c_str(), algo_idx, r.preset_id, r.quantize ? 1 : 0,
               r.shuffle, r.pred_ratio, r.pred_ct_ms, r.pred_dt_ms, r.ratio,
               r.ct_ms, r.dt_ms, r.psnr_db, r.cost, r.primary_cost,
               r.adopted ? 1 : 0,
               qmeas, qc[0], qc[1], qc[2], qc[3], qc[4],
               r.entropy, r.mad, r.second_deriv, r.eb_encoded);
  std::fflush(log->fp);
}

ExploreRowFields MakeRow(const std::string &blob_name, size_t chunk_size,
                          int rank, const std::string &lib_name,
                          uint32_t preset_id, bool quantize, uint32_t shuffle,
                          double pred_ratio, double pred_ct_ms,
                          double pred_dt_ms, double ratio, double ct_ms,
                          double psnr_db, double cost, double primary_cost,
                          bool adopted, bool is_primary, double dt_ms,
                          double entropy, double mad, double second_deriv,
                          double eb_encoded) {
  ExploreRowFields r;
  r.blob_name = blob_name; r.chunk_size = chunk_size; r.rank = rank;
  r.lib_name = lib_name; r.preset_id = preset_id; r.quantize = quantize;
  r.shuffle = shuffle; r.pred_ratio = pred_ratio; r.pred_ct_ms = pred_ct_ms;
  r.pred_dt_ms = pred_dt_ms; r.ratio = ratio; r.ct_ms = ct_ms;
  r.psnr_db = psnr_db; r.cost = cost; r.primary_cost = primary_cost;
  r.adopted = adopted; r.is_primary = is_primary; r.dt_ms = dt_ms;
  r.entropy = entropy; r.mad = mad; r.second_deriv = second_deriv;
  r.eb_encoded = eb_encoded;
  return r;
}

}  // namespace

void LogNeuroPressExplore(const std::string &blob_name, size_t chunk_size,
                          int rank, const std::string &lib_name,
                          uint32_t preset_id, bool quantize, uint32_t shuffle,
                          double pred_ratio, double pred_ct_ms,
                          double pred_dt_ms, double ratio,
                          double ct_ms, double psnr_db, double cost,
                          double primary_cost, bool adopted, bool is_primary,
                          double dt_ms, double entropy, double mad,
                          double second_deriv, double eb_encoded,
                          const ctp::compress::preprocess::QualityMetrics
                              *quality) {
  ExploreLog *log = ExploreLogInstance();
  if (!log->fp) return;
  const ExploreRowFields r = MakeRow(
      blob_name, chunk_size, rank, lib_name, preset_id, quantize, shuffle,
      pred_ratio, pred_ct_ms, pred_dt_ms, ratio, ct_ms, psnr_db, cost,
      primary_cost, adopted, is_primary, dt_ms, entropy, mad, second_deriv,
      eb_encoded);
  std::lock_guard<std::mutex> lock(log->mutex);
  WriteExploreRowLocked(log, r, quality);
}

void RecordPrimaryQuality(
    const std::string &blob_name,
    const ctp::compress::preprocess::QualityMetrics &quality) {
  ExploreLog *log = ExploreLogInstance();
  if (!log->fp) return;
  std::lock_guard<std::mutex> lock(log->mutex);
  log->primary_quality[blob_name] = quality;
}

bool TakePrimaryQuality(const std::string &blob_name,
                        ctp::compress::preprocess::QualityMetrics *out) {
  ExploreLog *log = ExploreLogInstance();
  if (!log->fp || !out) return false;
  std::lock_guard<std::mutex> lock(log->mutex);
  auto it = log->primary_quality.find(blob_name);
  if (it == log->primary_quality.end()) return false;
  *out = it->second;
  log->primary_quality.erase(it);
  return true;
}

void LogNeuroPressSelection(const std::string &blob_name, size_t chunk_size,
                            double entropy, double mad, double second_deriv,
                            int wire_lib, int packed_preset,
                            const CompressionStats *predicted,
                            double actual_ratio, double actual_ct_ms,
                            double actual_psnr,
                            unsigned long long checksum, const char *role) {
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
  // The WIDTH, not a boolean. Upstream has one width so a flag would be
  // lossless there, but Clio's static codec also reaches 2 and 8 -- collapsing
  // them made a stride-8 run indistinguishable from a stride-4 one in this log.
  const int shuffle = static_cast<int>((bits >> 8) & 0xFFu);
  const int preset = static_cast<int>(bits & 0xFFu);

  std::fprintf(fp,
               "%ld,%s,%zu,%.10g,%.10g,%.10g,%d,%s,%d,%d,%d,%d,"
               "%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%llu,%s\n",
               log->seq++, blob_name.c_str(), chunk_size, entropy, mad,
               second_deriv, wire_lib, lib_name.c_str(), algo_idx, quantize,
               shuffle, preset,
               predicted ? predicted->compression_ratio_ : 0.0,
               predicted ? predicted->compress_time_ms_ : 0.0,
               predicted ? predicted->decompress_time_ms_ : 0.0,
               predicted ? predicted->psnr_db_ : 0.0,
               actual_ratio, actual_ct_ms, actual_psnr, checksum,
               role ? role : "primary");
  std::fflush(fp);
}

}  // namespace clio::cte::compressor
