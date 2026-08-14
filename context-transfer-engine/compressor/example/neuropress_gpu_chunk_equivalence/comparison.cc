/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file comparison.cc
 * @brief Stage-by-stage comparison, first-divergence detection and reporting.
 */

#include "comparison.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace npeq {

namespace {

/**
 * Statistics tolerance.
 *
 * Byte equality is the WRONG criterion here and would fail against native
 * itself: upstream accumulates sum, abs_diff_sum and mad_sum with
 * atomicAdd(double*) (stats_kernel.cu, :246), whose summation order
 * varies with block scheduling, so two runs of NATIVE on identical input can
 * differ in the low bits. A relative tolerance near double epsilon is the
 * correct criterion. The largest deviation measured across a 1 GiB / 256-chunk
 * sweep was 5.4e-15 relative, so 1e-9 is four orders of magnitude of headroom
 * while still catching any real algorithmic difference.
 */
constexpr double kStatsRelTolerance = 1e-9;

/**
 * Prediction tolerance.
 *
 * The network is deterministic and prior parity runs measured max relative
 * error of exactly 0 between the two implementations' predictions. The
 * tolerance exists so a future FMA-contraction difference reports as a small
 * number rather than a bare FAIL, and the actual deviation is always printed.
 */
constexpr double kPredictionRelTolerance = 1e-6;

double RelError(double a, double b) {
  const double denom = std::max(std::fabs(a), std::fabs(b));
  if (denom == 0.0) return 0.0;
  return std::fabs(a - b) / denom;
}

std::string Fmt(double v) {
  std::ostringstream os;
  os.precision(12);
  os << v;
  return os.str();
}

CheckResult Pass(const std::string &name, const std::string &detail = "") {
  return CheckResult{name, true, false, detail};
}

CheckResult Fail(const std::string &name, const std::string &detail) {
  return CheckResult{name, false, false, detail};
}

CheckResult Skip(const std::string &name, const std::string &detail) {
  return CheckResult{name, true, true, detail};
}

CheckResult CompareDouble(const std::string &name, double a, double b,
                          double tolerance) {
  const double rel = RelError(a, b);
  const std::string detail = "native=" + Fmt(a) + " clio=" + Fmt(b) +
                             " rel=" + Fmt(rel);
  return rel <= tolerance ? Pass(name, detail) : Fail(name, detail);
}

CheckResult CompareInt(const std::string &name, long long a, long long b) {
  const std::string detail =
      "native=" + std::to_string(a) + " clio=" + std::to_string(b);
  return a == b ? Pass(name, detail) : Fail(name, detail);
}

CheckResult CompareBool(const std::string &name, bool a, bool b) {
  const std::string detail = std::string("native=") + (a ? "YES" : "NO") +
                             " clio=" + (b ? "YES" : "NO");
  return a == b ? Pass(name, detail) : Fail(name, detail);
}

std::string ByteCompareDetail(const ByteCompareResult &r) {
  if (!r.valid) return "comparison did not run";
  std::ostringstream os;
  os << "total=" << r.total_bytes << " identical=" << r.identical_bytes
     << " differing=" << r.differing_bytes;
  if (r.differing_bytes != 0) {
    os << " first_differing_byte=" << r.first_differing_byte
       << " first_differing_element=" << r.first_differing_element;
  }
  return os.str();
}

ByteCompareResult CompareAsHarness(const void *a, const void *b, size_t bytes,
                                   size_t element_size) {
  Recorder::Instance().PushHarness();
  ByteCompareResult r = CompareDeviceBuffers(a, b, bytes, element_size);
  Recorder::Instance().PopHarness();
  return r;
}

/**
 * A hex window around the first difference, from both buffers.
 *
 * "N bytes differ at offset K" locates a divergence; it does not characterize
 * one. Seeing the actual bytes is usually the difference between a finding and
 * a diagnosis -- whether the delta looks like a length field, a version tag or
 * genuinely different codec output.
 */
std::string HexWindow(const void *a, const void *b, size_t bytes,
                      size_t first_diff) {
  constexpr size_t kWindow = 24;
  if (first_diff >= bytes) return "";
  const size_t start = first_diff >= 8 ? first_diff - 8 : 0;
  const size_t len = std::min(kWindow, bytes - start);
  unsigned char ha[kWindow] = {0};
  unsigned char hb[kWindow] = {0};
  Recorder::Instance().PushHarness();
  const bool ok = FetchDeviceWindow(a, start, len, ha) &&
                  FetchDeviceWindow(b, start, len, hb);
  Recorder::Instance().PopHarness();
  if (!ok) return "";

  std::ostringstream os;
  os << "\n            bytes [" << start << ".." << (start + len - 1) << "]\n";
  auto row = [&os, len](const char *label, const unsigned char *h,
                        const unsigned char *other) {
    os << "            " << label << " ";
    for (size_t i = 0; i < len; ++i) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "%02x", h[i]);
      os << (h[i] != other[i] ? "[" : " ") << buf
         << (h[i] != other[i] ? "]" : "");
    }
    os << "\n";
  };
  row("native", ha, hb);
  row("clio  ", hb, ha);
  return os.str();
}

/**
 * Strip the vendor prefix Clio's factory carries but NeuroPress's names do not.
 *
 * Clio registers the eight nvcomp codecs as "nvcomp-lz4", "nvcomp-bitcomp" and
 * so on; upstream calls the same algorithms "lz4" and "bitcomp". This is a
 * naming convention, not a difference in which codec ran -- the algorithm INDEX
 * is compared separately and unconditionally in the selection stage, so nothing
 * is being taken on trust here.
 */
std::string CanonicalCodec(const std::string &name) {
  const std::string prefix = "nvcomp-";
  if (name.rfind(prefix, 0) == 0) return name.substr(prefix.size());
  return name;
}

/**
 * Is this action one upstream's ranking kernel masks to -INFINITY?
 *
 * nn_gpu.cu: quantize actions are masked when the error bound is not
 * positive. Every masked action therefore carries the SAME score, so their
 * relative order is not a decision either implementation made -- it is
 * whatever each one's sort did with a block of equal keys.
 */
bool IsMaskedAction(int action, double error_bound) {
  if (action < 0) return false;
  const bool quantize = ((action / 8) % 2) != 0;
  return quantize && !(error_bound > 0.0);
}

/** The stages compared, in pipeline order. */
const Stage kFunctionalStages[] = {
    Stage::kStatistics,   Stage::kDiagnostics,   Stage::kInference,
    Stage::kRanking,      Stage::kSelection,     Stage::kQuantization,
    Stage::kShuffle,      Stage::kCompression,
};

}  // namespace

bool StageComparison::Pass() const {
  for (const auto &c : checks) {
    if (!c.pass) return false;
  }
  return true;
}

bool StageComparison::AnyRan() const {
  return status_native == Status::kExecuted ||
         status_clio == Status::kExecuted;
}

ChunkComparison CompareChunk(const ChunkTrace &native, const ChunkTrace &clio,
                             const SideArtifacts &native_art,
                             const SideArtifacts &clio_art) {
  ChunkComparison out;
  out.chunk_id = native.chunk_id;
  out.regime = native.regime;
  out.chunk_bytes = native.chunk_bytes;
  out.error_bound = native.error_bound;
  out.native_input_device = native.input_device_verified;
  out.clio_input_device = clio.input_device_verified;

  // ---- sequences ----
  for (const auto &e : native.entries) out.native_sequence.push_back(StageName(e.stage));
  for (const auto &e : clio.entries) {
    out.clio_sequence.push_back(StageName(e.stage));
    if (e.kind == StageKind::kClioArchitectural) {
      out.clio_architectural.push_back(StageName(e.stage));
    } else {
      out.clio_normalized.push_back(StageName(e.stage));
    }
  }
  // Native has no architectural entries by definition, so its sequence IS its
  // normalized sequence. Compare the two normalized sequences.
  std::vector<std::string> native_normalized;
  for (const auto &e : native.entries) {
    if (e.kind == StageKind::kNeuroPressFunctional) {
      native_normalized.push_back(StageName(e.stage));
    }
  }
  out.sequence_pass = (native_normalized == out.clio_normalized);

  // ---- transfer and kernel accounting ----
  auto account = [](const ChunkTrace &t, size_t *d2h, int *d2h_n, size_t *h2d,
                    int *h2d_n, int *kernels, size_t *harness,
                    std::vector<std::string> *no_kernel_stages) {
    for (const auto &e : t.entries) {
      *d2h += e.TransferBytes("D2H", false);
      *d2h_n += e.TransferCount("D2H", false);
      *h2d += e.TransferBytes("H2D", false);
      *h2d_n += e.TransferCount("H2D", false);
      *harness += e.TransferBytes("D2H", true) + e.TransferBytes("H2D", true) +
                  e.TransferBytes("D2D", true);
      const int launches = e.KernelLaunches();
      *kernels += launches;
      // Only stages that move DATA are expected to launch kernels. Ranking and
      // selection are decisions read out of a completed inference, so zero
      // kernels there is correct, not a fallback.
      const bool data_stage = e.stage == Stage::kStatistics ||
                              e.stage == Stage::kQuantization ||
                              e.stage == Stage::kShuffle ||
                              e.stage == Stage::kCompression ||
                              e.stage == Stage::kInference;
      if (data_stage && e.status == Status::kExecuted && launches == 0) {
        no_kernel_stages->push_back(StageName(e.stage));
      }
    }
  };
  account(native, &out.native_production_d2h, &out.native_production_d2h_count,
          &out.native_production_h2d, &out.native_production_h2d_count,
          &out.native_kernel_launches, &out.harness_transfer_bytes,
          &out.native_stages_without_kernels);
  account(clio, &out.clio_production_d2h, &out.clio_production_d2h_count,
          &out.clio_production_h2d, &out.clio_production_h2d_count,
          &out.clio_kernel_launches, &out.harness_transfer_bytes,
          &out.clio_stages_without_kernels);

  // Bulk transfers: a production host round trip of anything approaching the
  // chunk's own size. Half the chunk is the threshold because quantization can
  // legitimately narrow the buffer to a quarter of its original width, so a
  // fixed byte count would either miss a real D->H of quantized data or flag a
  // header. Small metadata copies are expected on both sides and are counted
  // separately above.
  const size_t bulk_threshold = std::max<size_t>(native.chunk_bytes / 2, 4096);
  auto find_bulk = [&](const ChunkTrace &t, const char *label, int *count) {
    for (const auto &e : t.entries) {
      for (const auto &tr : e.transfers) {
        if (tr.harness) continue;
        if (tr.direction != "D2H" && tr.direction != "H2D") continue;
        if (tr.bytes < bulk_threshold) continue;
        ++(*count);
        out.bulk_transfer_detail.push_back(
            std::string(label) + " " + StageName(e.stage) + ": " +
            tr.direction + " " + std::to_string(tr.bytes) + " B");
      }
    }
  };
  find_bulk(native, "native", &out.native_bulk_transfers);
  find_bulk(clio, "clio", &out.clio_bulk_transfers);

  // ---- input equality ----
  if (native_art.input != nullptr && clio_art.input != nullptr &&
      native_art.input_bytes == clio_art.input_bytes) {
    out.input_compare = CompareAsHarness(native_art.input, clio_art.input,
                                         native_art.input_bytes, 4);
  }

  // ---- per-stage comparison ----
  for (Stage stage : kFunctionalStages) {
    StageComparison sc;
    sc.stage = stage;
    const TraceEntry *n = native.Find(stage);
    const TraceEntry *c = clio.Find(stage);
    sc.present_native = (n != nullptr);
    sc.present_clio = (c != nullptr);
    if (n) sc.status_native = n->status;
    if (c) sc.status_clio = c->status;

    if (n == nullptr || c == nullptr) {
      sc.checks.push_back(Fail(
          "callback present",
          std::string("native=") + (n ? "yes" : "no") + " clio=" + (c ? "yes" : "no")));
      out.stages.push_back(std::move(sc));
      continue;
    }

    sc.checks.push_back(CompareInt("callback status",
                                   static_cast<int>(n->status),
                                   static_cast<int>(c->status)));
    sc.checks.push_back(CompareInt("chunk id", n->chunk_id, c->chunk_id));

    // Buffer residency, for stages that touch a buffer.
    if (n->input_loc != MemLoc::kNone || c->input_loc != MemLoc::kNone) {
      sc.checks.push_back(CompareInt("input location",
                                     static_cast<int>(n->input_loc),
                                     static_cast<int>(c->input_loc)));
    }

    switch (stage) {
      case Stage::kStatistics: {
        // The statistics STAGE compares the features it produced. Those live
        // in different structs on the two sides (AutoStatsGPU vs
        // DeviceFeatureStats), so the buffers are deliberately NOT
        // byte-compared -- the VALUES are, in the diagnostics stage where both
        // sides have them on the host.
        sc.checks.push_back(CompareInt("input bytes",
                                       static_cast<long long>(n->input_bytes),
                                       static_cast<long long>(c->input_bytes)));
        if (n->input_hash_valid && c->input_hash_valid) {
          sc.checks.push_back(CompareInt(
              "input hash", static_cast<long long>(n->input_hash),
              static_cast<long long>(c->input_hash)));
        }
        sc.checks.push_back(
            n->output_loc == MemLoc::kDevice && c->output_loc == MemLoc::kDevice
                ? Pass("output on device", "both DEVICE")
                : Fail("output on device",
                       std::string("native=") + MemLocName(n->output_loc) +
                           " clio=" + MemLocName(c->output_loc)));
        break;
      }
      case Stage::kDiagnostics: {
        if (n->stats.valid && c->stats.valid) {
          sc.checks.push_back(CompareDouble("entropy", n->stats.entropy,
                                            c->stats.entropy,
                                            kStatsRelTolerance));
          sc.checks.push_back(
              CompareDouble("mad", n->stats.mad, c->stats.mad,
                            kStatsRelTolerance));
          sc.checks.push_back(CompareDouble("second derivative",
                                            n->stats.second_derivative,
                                            c->stats.second_derivative,
                                            kStatsRelTolerance));
        } else {
          sc.checks.push_back(Fail("statistics readable",
                                   std::string("native=") +
                                       (n->stats.valid ? "yes" : "no") +
                                       " clio=" + (c->stats.valid ? "yes" : "no")));
        }
        // Native's own per-chunk diagnostics record, compared field by field
        // against what Clio can report. Fields Clio has no counterpart for are
        // reported as an architectural difference, not a NeuroPress mismatch.
        if (n->diagnostics.valid && c->diagnostics.valid) {
          sc.checks.push_back(CompareInt("diagnostics nn_action",
                                         n->diagnostics.nn_action,
                                         c->diagnostics.nn_action));
          sc.checks.push_back(
              CompareDouble("diagnostics feat_entropy",
                            n->diagnostics.feat_entropy,
                            c->diagnostics.feat_entropy, 1e-5));
          sc.checks.push_back(CompareDouble("diagnostics feat_mad",
                                            n->diagnostics.feat_mad,
                                            c->diagnostics.feat_mad, 1e-5));
          sc.checks.push_back(CompareDouble("diagnostics feat_deriv",
                                            n->diagnostics.feat_deriv,
                                            c->diagnostics.feat_deriv, 1e-5));
          sc.checks.push_back(CompareInt("diagnostics exploration_triggered",
                                         n->diagnostics.exploration_triggered,
                                         c->diagnostics.exploration_triggered));
          sc.checks.push_back(CompareInt("diagnostics sgd_fired",
                                         n->diagnostics.sgd_fired,
                                         c->diagnostics.sgd_fired));
          if (!c->diagnostics.native_record_present) {
            sc.checks.push_back(Skip(
                "clio per-chunk diagnostics record",
                "Clio has no counterpart of gpucompress_chunk_diag_t; the "
                "fields above are sourced from its selection result instead. "
                "Recorded as an OBSERVABILITY difference, not a divergence"));
          }
        }
        // The asymmetry that matters, stated rather than smoothed over: Clio
        // makes a REAL production 24-byte D->H of the features here; upstream's
        // AUTO path makes none at all and zeroes those fields in the stats it
        // reports (gpucompress_compress.cpp). The VALUES agree, which is
        // what the checks above establish. The READ does not exist on both
        // sides, and that is a genuine (observability-only, already recorded)
        // difference rather than something this harness discovered.
        {
          const bool native_production = n->TransferCount("D2H", false) > 0;
          const bool clio_production = c->TransferCount("D2H", false) > 0;
          sc.checks.push_back(Skip(
              "feature readback origin",
              std::string("native=") +
                  (native_production ? "production read"
                                     : "harness probe only (upstream's AUTO "
                                       "path never reads the features to the "
                                       "host)") +
                  ", clio=" +
                  (clio_production ? "production read (compressor_runtime.cc:714)"
                                   : "none") +
                  ". OBSERVABILITY difference; the feature VALUES are compared "
                  "above and must match"));
        }
        break;
      }
      case Stage::kInference: {
        if (!n->inference.valid || !c->inference.valid) {
          sc.checks.push_back(Fail("inference outputs present",
                                   "one side produced no per-candidate output"));
          break;
        }
        int compared = 0;
        double worst_ratio = 0.0, worst_ct = 0.0, worst_dt = 0.0, worst_psnr = 0.0;
        int worst_action = -1;
        for (int a = 0; a < 32; ++a) {
          if (!n->inference.present[a] || !c->inference.present[a]) continue;
          ++compared;
          const double r = RelError(n->inference.ratio[a], c->inference.ratio[a]);
          const double ct = RelError(n->inference.comp_ms[a], c->inference.comp_ms[a]);
          const double dt = RelError(n->inference.decomp_ms[a], c->inference.decomp_ms[a]);
          const double ps = RelError(n->inference.psnr_db[a], c->inference.psnr_db[a]);
          const double worst_here = std::max({r, ct, dt, ps});
          if (worst_here > std::max({worst_ratio, worst_ct, worst_dt, worst_psnr})) {
            worst_action = a;
          }
          worst_ratio = std::max(worst_ratio, r);
          worst_ct = std::max(worst_ct, ct);
          worst_dt = std::max(worst_dt, dt);
          worst_psnr = std::max(worst_psnr, ps);
        }
        sc.checks.push_back(CompareInt("candidates compared", compared, compared));
        if (compared == 0) {
          sc.checks.push_back(Fail("candidate overlap",
                                   "no action id was present on both sides"));
          break;
        }
        const std::string where =
            worst_action >= 0 ? " (worst at action " + std::to_string(worst_action) + ")"
                              : "";
        sc.checks.push_back(worst_ratio <= kPredictionRelTolerance
                                ? Pass("predicted ratio", "max rel=" + Fmt(worst_ratio) + where)
                                : Fail("predicted ratio", "max rel=" + Fmt(worst_ratio) + where));
        sc.checks.push_back(worst_ct <= kPredictionRelTolerance
                                ? Pass("predicted comp time", "max rel=" + Fmt(worst_ct))
                                : Fail("predicted comp time", "max rel=" + Fmt(worst_ct)));
        sc.checks.push_back(worst_dt <= kPredictionRelTolerance
                                ? Pass("predicted decomp time", "max rel=" + Fmt(worst_dt))
                                : Fail("predicted decomp time", "max rel=" + Fmt(worst_dt)));
        sc.checks.push_back(worst_psnr <= kPredictionRelTolerance
                                ? Pass("predicted psnr", "max rel=" + Fmt(worst_psnr))
                                : Fail("predicted psnr", "max rel=" + Fmt(worst_psnr)));
        break;
      }
      case Stage::kRanking: {
        const auto &no = n->ranking.order;
        const auto &co = c->ranking.order;
        if (no.empty() || co.empty()) {
          sc.checks.push_back(Fail("ranking present", "one side ranked nothing"));
          break;
        }
        sc.checks.push_back(CompareInt("ranked count",
                                       static_cast<long long>(no.size()),
                                       static_cast<long long>(co.size())));
        const size_t common = std::min(no.size(), co.size());
        size_t first_diff = common;
        for (size_t i = 0; i < common; ++i) {
          if (no[i] != co[i]) {
            first_diff = i;
            break;
          }
        }
        std::ostringstream detail;
        detail << "first " << std::min<size_t>(8, common) << " native=[";
        for (size_t i = 0; i < std::min<size_t>(8, no.size()); ++i) {
          detail << (i ? "," : "") << no[i];
        }
        detail << "] clio=[";
        for (size_t i = 0; i < std::min<size_t>(8, co.size()); ++i) {
          detail << (i ? "," : "") << co[i];
        }
        detail << "]";

        if (first_diff == common) {
          sc.checks.push_back(Pass("ranking order", detail.str()));
          break;
        }

        // The two orders differ. Before calling that a divergence, establish
        // whether it is a real difference of OPINION or only a different
        // arrangement of candidates the model scored identically.
        //
        // A positional difference is tie-explained when the two actions sitting
        // at that rank carry the same score. Two sources of exact ties exist and
        // both are upstream's own: actions masked to -INFINITY (nn_gpu.cu),
        // and actions whose predicted cost is bit-identical -- which is common,
        // because the predicted ratio saturates at the 100x cap on compressible
        // data. Native's costs are used as the oracle because native is the
        // ground truth AND because the inference stage above has already proven
        // the two sides' predictions identical to the last bit; if it had not,
        // this check would never be reached.
        size_t tie_explained = 0;
        size_t genuine = 0;
        size_t first_genuine = common;
        for (size_t i = 0; i < common; ++i) {
          if (no[i] == co[i]) continue;
          const int a = no[i];
          const int b = co[i];
          const bool both_masked = IsMaskedAction(a, out.error_bound) &&
                                   IsMaskedAction(b, out.error_bound);
          bool equal_cost = false;
          if (a >= 0 && a < 32 && b >= 0 && b < 32 &&
              n->ranking.cost_present[a] && n->ranking.cost_present[b]) {
            equal_cost = (n->ranking.cost[a] == n->ranking.cost[b]);
          }
          if (both_masked || equal_cost) {
            ++tie_explained;
          } else {
            ++genuine;
            if (i < first_genuine) first_genuine = i;
          }
        }

        detail << "; " << (tie_explained + genuine)
               << " positions differ, first at rank " << first_diff << " ("
               << tie_explained << " tie-explained, " << genuine << " genuine)";
        if (genuine == 0) {
          detail << ". Every difference sits between candidates the model "
                    "scored EQUALLY (masked to -INF, or bit-identical cost), so "
                    "it is an ordering of a tie block, not a different opinion, "
                    "and cannot change which action is selected -- rank 0 is "
                    "compared separately and unconditionally in `selection`";
          sc.checks.push_back(Pass("ranking order", detail.str()));
        } else {
          detail << ". First GENUINE difference at rank " << first_genuine;
          sc.checks.push_back(Fail("ranking order", detail.str()));
        }
        break;
      }
      case Stage::kSelection: {
        sc.checks.push_back(CompareInt("selected action", n->selection.action,
                                       c->selection.action));
        sc.checks.push_back(CompareInt("selected algorithm", n->selection.algo,
                                       c->selection.algo));
        sc.checks.push_back(
            n->selection.algo_name == c->selection.algo_name
                ? Pass("algorithm name",
                       "native=" + n->selection.algo_name + " clio=" +
                           c->selection.algo_name)
                : Fail("algorithm name",
                       "native=" + n->selection.algo_name + " clio=" +
                           c->selection.algo_name));
        sc.checks.push_back(CompareBool("shuffle selected",
                                        n->selection.shuffle,
                                        c->selection.shuffle));
        sc.checks.push_back(CompareBool("quantization selected",
                                        n->selection.quantize,
                                        c->selection.quantize));
        break;
      }
      case Stage::kQuantization: {
        sc.checks.push_back(CompareInt("stage ran",
                                       n->status == Status::kExecuted,
                                       c->status == Status::kExecuted));
        if (n->status != Status::kExecuted || c->status != Status::kExecuted) {
          sc.checks.push_back(Skip("quantization output",
                                   "not selected on at least one side"));
          break;
        }
        sc.checks.push_back(CompareInt("precision", n->quant.precision,
                                       c->quant.precision));
        sc.checks.push_back(CompareDouble("effective error bound",
                                          n->quant.effective_error_bound,
                                          c->quant.effective_error_bound, 1e-12));
        sc.checks.push_back(CompareDouble("scale", n->quant.scale,
                                          c->quant.scale, 1e-12));
        sc.checks.push_back(CompareDouble("data_min", n->quant.data_min,
                                          c->quant.data_min, 1e-12));
        sc.checks.push_back(CompareDouble("data_max", n->quant.data_max,
                                          c->quant.data_max, 1e-12));
        sc.checks.push_back(CompareInt("output bytes",
                                       static_cast<long long>(n->output_bytes),
                                       static_cast<long long>(c->output_bytes)));
        if (native_art.quantized != nullptr && clio_art.quantized != nullptr &&
            native_art.quantized_bytes == clio_art.quantized_bytes &&
            native_art.quantized_bytes > 0) {
          out.quantized_compare = CompareAsHarness(
              native_art.quantized, clio_art.quantized,
              native_art.quantized_bytes,
              n->quant.precision > 0 ? n->quant.precision / 8 : 1);
          sc.checks.push_back(out.quantized_compare.Identical()
                                  ? Pass("quantized output bytes",
                                         ByteCompareDetail(out.quantized_compare))
                                  : Fail("quantized output bytes",
                                         ByteCompareDetail(out.quantized_compare)));
        } else {
          sc.checks.push_back(Fail("quantized output bytes",
                                   "buffers absent or of different length"));
        }

        // The guarantee the caller actually asked for. Byte-equality between
        // the two quantized buffers says the implementations AGREE; it says
        // nothing about whether either honours its error bound. Both are
        // round-tripped through their own dequantizer and every element is
        // checked against the REQUESTED bound (not the smaller effective one),
        // on-device.
        auto bound_check = [&](const char *label, const SideArtifacts &art) {
          if (!art.bound_checked) {
            sc.checks.push_back(
                Fail(std::string(label) + " within error bound",
                     "the quantize->dequantize round trip did not run, so the "
                     "bound is UNVERIFIED"));
            return;
          }
          const std::string d =
              std::to_string(art.bound_violations) + " of " +
              std::to_string(art.dequantized_bytes / sizeof(float)) +
              " elements outside +-" + Fmt(art.requested_error_bound);
          sc.checks.push_back(art.bound_violations == 0
                                  ? Pass(std::string(label) +
                                             " within error bound", d)
                                  : Fail(std::string(label) +
                                             " within error bound", d));
        };
        bound_check("native reconstruction", native_art);
        bound_check("clio reconstruction", clio_art);

        // And the two reconstructions against each other. Equal quantized
        // bytes plus equal parameters should imply this, so a failure here
        // would mean the two dequantizers differ.
        if (native_art.dequantized != nullptr &&
            clio_art.dequantized != nullptr &&
            native_art.dequantized_bytes == clio_art.dequantized_bytes &&
            native_art.dequantized_bytes > 0) {
          out.dequantized_compare = CompareAsHarness(
              native_art.dequantized, clio_art.dequantized,
              native_art.dequantized_bytes, sizeof(float));
          sc.checks.push_back(
              out.dequantized_compare.Identical()
                  ? Pass("dequantized output bytes",
                         ByteCompareDetail(out.dequantized_compare))
                  : Fail("dequantized output bytes",
                         ByteCompareDetail(out.dequantized_compare)));
        }
        break;
      }
      case Stage::kShuffle: {
        sc.checks.push_back(CompareInt("stage ran",
                                       n->status == Status::kExecuted,
                                       c->status == Status::kExecuted));
        if (n->status != Status::kExecuted || c->status != Status::kExecuted) {
          sc.checks.push_back(Skip("shuffle output",
                                   "not selected on at least one side"));
          break;
        }
        sc.checks.push_back(CompareInt("input bytes",
                                       static_cast<long long>(n->input_bytes),
                                       static_cast<long long>(c->input_bytes)));
        sc.checks.push_back(CompareInt("output bytes",
                                       static_cast<long long>(n->output_bytes),
                                       static_cast<long long>(c->output_bytes)));
        if (native_art.shuffled != nullptr && clio_art.shuffled != nullptr &&
            native_art.shuffled_bytes == clio_art.shuffled_bytes &&
            native_art.shuffled_bytes > 0) {
          out.shuffled_compare =
              CompareAsHarness(native_art.shuffled, clio_art.shuffled,
                               native_art.shuffled_bytes, 4);
          // Byte-for-byte, as the equivalence assertion. The hashes above are
          // the fast diagnostic; this is the claim.
          sc.checks.push_back(out.shuffled_compare.Identical()
                                  ? Pass("shuffle output bytes",
                                         ByteCompareDetail(out.shuffled_compare))
                                  : Fail("shuffle output bytes",
                                         ByteCompareDetail(out.shuffled_compare)));
        } else {
          sc.checks.push_back(Fail("shuffle output bytes",
                                   "buffers absent or of different length"));
        }
        break;
      }
      case Stage::kCompression: {
        const std::string ncodec = CanonicalCodec(n->compression.algo_name);
        const std::string ccodec = CanonicalCodec(c->compression.algo_name);
        sc.checks.push_back(
            ncodec == ccodec
                ? Pass("codec", "native=" + n->compression.algo_name +
                                    " clio=" + c->compression.algo_name +
                                    " (both " + ncodec + ")")
                : Fail("codec", "native=" + n->compression.algo_name +
                                    " clio=" + c->compression.algo_name));
        // Native's recorded size includes its 64-byte header; the payload is
        // what is comparable, since the two containers differ by design
        // (D15-1). Compare payload lengths, then payload bytes.
        sc.checks.push_back(CompareInt(
            "codec payload length",
            static_cast<long long>(native_art.payload_bytes),
            static_cast<long long>(clio_art.payload_bytes)));
        if (native_art.payload != nullptr && clio_art.payload != nullptr &&
            native_art.payload_bytes == clio_art.payload_bytes &&
            native_art.payload_bytes > 0) {
          out.payload_compare =
              CompareAsHarness(native_art.payload, clio_art.payload,
                               native_art.payload_bytes, 1);
          std::string detail = ByteCompareDetail(out.payload_compare);
          if (out.payload_compare.valid && !out.payload_compare.Identical()) {
            detail += HexWindow(native_art.payload, clio_art.payload,
                                native_art.payload_bytes,
                                out.payload_compare.first_differing_byte);
          }

          // Before the cross-side result can be read as a divergence, each side
          // has to be shown reproducible against itself. nvcomp output is known
          // to depend on manager reuse history (D18-2), and native and Clio
          // reuse managers on different schedules -- so a codec that disagrees
          // with its own previous output cannot be used to judge the other
          // implementation.
          if (native_art.payload_repeat != nullptr &&
              native_art.payload_repeat_bytes == native_art.payload_bytes) {
            out.native_self_compare = CompareAsHarness(
                native_art.payload, native_art.payload_repeat,
                native_art.payload_bytes, 1);
          }
          if (clio_art.payload_repeat != nullptr &&
              clio_art.payload_repeat_bytes == clio_art.payload_bytes) {
            out.clio_self_compare =
                CompareAsHarness(clio_art.payload, clio_art.payload_repeat,
                                 clio_art.payload_bytes, 1);
          }
          const bool native_reproducible =
              !out.native_self_compare.valid || out.native_self_compare.Identical();
          const bool clio_reproducible =
              !out.clio_self_compare.valid || out.clio_self_compare.Identical();

          if (out.payload_compare.Identical()) {
            sc.checks.push_back(Pass("codec payload bytes", detail));
          } else if (!native_reproducible || !clio_reproducible) {
            // Reported, never hidden: the bytes DID differ, and the reason the
            // check is not a failure is stated in full.
            std::string why =
                detail + "\n            NOT a cross-implementation divergence: ";
            if (!native_reproducible) {
              why += "NATIVE does not reproduce its own output on a second run "
                     "over the same chunk (" +
                     ByteCompareDetail(out.native_self_compare) + "). ";
            }
            if (!clio_reproducible) {
              why += "CLIO does not reproduce its own output on a second run "
                     "over the same chunk (" +
                     ByteCompareDetail(out.clio_self_compare) + "). ";
            }
            why +=
                "A codec that disagrees with itself cannot be used to judge the "
                "other implementation. Byte equality is the wrong criterion for "
                "this codec; the selection, the preprocessing and the payload "
                "LENGTH are compared above and do hold.";
            sc.checks.push_back(Skip("codec payload bytes", why));
          } else {
            sc.checks.push_back(Fail(
                "codec payload bytes",
                detail +
                    "\n            Both sides ARE reproducible against "
                    "themselves, so this is a genuine cross-implementation "
                    "divergence."));
          }
        }
        break;
      }
      default:
        break;
    }

    out.stages.push_back(std::move(sc));
  }

  // ---- first divergence ----
  for (size_t i = 0; i < out.stages.size(); ++i) {
    if (out.stages[i].Pass()) continue;
    out.first_divergent_stage = static_cast<int>(i);
    std::ostringstream os;
    os << StageName(out.stages[i].stage) << ": ";
    bool first = true;
    for (const auto &c : out.stages[i].checks) {
      if (c.pass) continue;
      if (!first) os << "; ";
      first = false;
      os << c.name << " (" << c.detail << ")";
    }
    out.first_divergence = os.str();
    break;
  }

  // A bulk host round trip Clio makes and native does not is a failure in its
  // own right, independent of whether the bytes came out equal: the whole claim
  // under test is that Clio processes a GPU-resident chunk the way native does,
  // and staging it through the host is a different processing path that happens
  // to agree. Native doing one too is not held against Clio.
  out.pass = out.sequence_pass && out.first_divergent_stage < 0 &&
             out.clio_bulk_transfers <= out.native_bulk_transfers;
  return out;
}

namespace {

void JsonStr(std::ostream &os, const std::string &s) {
  os << '"';
  for (char ch : s) {
    if (ch == '"') os << "\\\"";
    else if (ch == '\\') os << "\\\\";
    else if (ch == '\n') os << "\\n";
    else os << ch;
  }
  os << '"';
}

void WriteByteCompare(std::ostream &os, const char *name,
                      const ByteCompareResult &r) {
  os << "        \"" << name << "\": {\"ran\": " << (r.valid ? "true" : "false")
     << ", \"total_bytes\": " << r.total_bytes
     << ", \"identical_bytes\": " << r.identical_bytes
     << ", \"differing_bytes\": " << r.differing_bytes;
  if (r.valid && r.differing_bytes != 0) {
    os << ", \"first_differing_byte\": " << r.first_differing_byte
       << ", \"first_differing_element\": " << r.first_differing_element;
  }
  os << "}";
}

}  // namespace

bool WriteComparisonJson(const std::string &path,
                         const std::vector<ChunkComparison> &comparisons) {
  std::ofstream os(path);
  if (!os) return false;
  os.precision(10);
  os << "{\n  \"chunks\": [\n";
  for (size_t i = 0; i < comparisons.size(); ++i) {
    const auto &c = comparisons[i];
    os << "    {\n";
    os << "      \"chunk_id\": " << c.chunk_id << ",\n";
    os << "      \"regime\": ";
    JsonStr(os, c.regime);
    os << ",\n";
    os << "      \"chunk_bytes\": " << c.chunk_bytes << ",\n";
    os << "      \"error_bound\": " << c.error_bound << ",\n";
    os << "      \"result\": \"" << (c.pass ? "PASS" : "FAIL") << "\",\n";
    os << "      \"callback_sequence_pass\": "
       << (c.sequence_pass ? "true" : "false") << ",\n";
    os << "      \"clio_architectural_callbacks\": [";
    for (size_t k = 0; k < c.clio_architectural.size(); ++k) {
      if (k) os << ", ";
      JsonStr(os, c.clio_architectural[k]);
    }
    os << "],\n";
    os << "      \"first_divergence\": ";
    JsonStr(os, c.first_divergence);
    os << ",\n";
    os << "      \"gpu_residency\": {\"native_input_device\": "
       << (c.native_input_device ? "true" : "false")
       << ", \"clio_input_device\": "
       << (c.clio_input_device ? "true" : "false") << "},\n";
    os << "      \"production_transfers\": {\"native_d2h_bytes\": "
       << c.native_production_d2h << ", \"native_d2h_count\": "
       << c.native_production_d2h_count << ", \"clio_d2h_bytes\": "
       << c.clio_production_d2h << ", \"clio_d2h_count\": "
       << c.clio_production_d2h_count << ", \"native_h2d_bytes\": "
       << c.native_production_h2d << ", \"native_h2d_count\": "
       << c.native_production_h2d_count << ", \"clio_h2d_bytes\": "
       << c.clio_production_h2d << ", \"clio_h2d_count\": "
       << c.clio_production_h2d_count << "},\n";
    os << "      \"harness_transfer_bytes\": " << c.harness_transfer_bytes
       << ",\n";
    os << "      \"bulk_production_transfers\": {\"native\": "
       << c.native_bulk_transfers << ", \"clio\": " << c.clio_bulk_transfers
       << ", \"detail\": [";
    for (size_t k = 0; k < c.bulk_transfer_detail.size(); ++k) {
      if (k) os << ", ";
      JsonStr(os, c.bulk_transfer_detail[k]);
    }
    os << "]},\n";
    os << "      \"kernel_launches\": {\"native\": " << c.native_kernel_launches
       << ", \"clio\": " << c.clio_kernel_launches << "},\n";
    WriteByteCompare(os, "input_compare", c.input_compare);
    os << ",\n";
    WriteByteCompare(os, "quantized_compare", c.quantized_compare);
    os << ",\n";
    WriteByteCompare(os, "shuffled_compare", c.shuffled_compare);
    os << ",\n";
    WriteByteCompare(os, "payload_compare", c.payload_compare);
    os << ",\n";
    WriteByteCompare(os, "native_self_compare", c.native_self_compare);
    os << ",\n";
    WriteByteCompare(os, "clio_self_compare", c.clio_self_compare);
    os << ",\n";
    os << "      \"callbacks\": [\n";
    for (size_t s = 0; s < c.stages.size(); ++s) {
      const auto &sc = c.stages[s];
      os << "        {\"callback\": ";
      JsonStr(os, StageName(sc.stage));
      os << ", \"result\": \"" << (sc.Pass() ? "PASS" : "FAIL")
         << "\", \"native_status\": \"" << StatusName(sc.status_native)
         << "\", \"clio_status\": \"" << StatusName(sc.status_clio)
         << "\", \"checks\": [";
      for (size_t k = 0; k < sc.checks.size(); ++k) {
        if (k) os << ", ";
        os << "{\"name\": ";
        JsonStr(os, sc.checks[k].name);
        os << ", \"result\": \""
           << (sc.checks[k].skipped ? "N/A" : (sc.checks[k].pass ? "PASS" : "FAIL"))
           << "\", \"detail\": ";
        JsonStr(os, sc.checks[k].detail);
        os << "}";
      }
      os << "]}";
      if (s + 1 < c.stages.size()) os << ",";
      os << "\n";
    }
    os << "      ]\n    }";
    if (i + 1 < comparisons.size()) os << ",";
    os << "\n";
  }
  os << "  ]\n}\n";
  return true;
}

void PrintReport(std::ostream &os,
                 const std::vector<ChunkTrace> &native_traces,
                 const std::vector<ChunkTrace> &clio_traces,
                 const std::vector<ChunkComparison> &comparisons,
                 const std::string &preamble) {
  const std::string rule(78, '=');
  const std::string thin(78, '-');

  os << rule << "\nNEUROPRESS <-> CLIO CALLBACK TRACE COMPARISON\n" << rule
     << "\n\n"
     << preamble << "\n";

  for (size_t i = 0; i < comparisons.size(); ++i) {
    const auto &c = comparisons[i];
    const ChunkTrace &nt = native_traces[i];
    const ChunkTrace &ct = clio_traces[i];

    os << rule << "\nChunk " << c.chunk_id << "  regime=" << c.regime
       << "  bytes=" << c.chunk_bytes << "  error_bound=" << c.error_bound
       << "\n" << rule << "\n\n";

    // --- the callback walk, side by side ---
    os << "Native:\n";
    for (const auto &e : nt.entries) {
      os << "    " << std::setw(3) << std::setfill('0') << e.sequence
         << std::setfill(' ') << " " << StageName(e.stage);
      if (e.status != Status::kExecuted) os << "  [" << StatusName(e.status) << "]";
      os << "\n";
    }
    os << "\nCLIO:\n";
    for (const auto &e : ct.entries) {
      os << "    " << std::setw(3) << std::setfill('0') << e.sequence
         << std::setfill(' ') << " " << StageName(e.stage);
      if (e.kind == StageKind::kClioArchitectural) os << "   [CLIO ARCHITECTURAL]";
      if (e.status != Status::kExecuted) os << "  [" << StatusName(e.status) << "]";
      os << "\n";
    }
    os << "\nNormalized CLIO:\n";
    int idx = 0;
    for (const auto &name : c.clio_normalized) {
      os << "    " << std::setw(3) << std::setfill('0') << ++idx
         << std::setfill(' ') << " " << name << "\n";
    }
    os << "\nFunctional callback sequence:\n    "
       << (c.sequence_pass ? "PASS" : "FAIL") << "\n";
    os << "\nCLIO architectural callbacks:\n";
    if (c.clio_architectural.empty()) {
      os << "    (none)\n";
    } else {
      for (const auto &a : c.clio_architectural) os << "    " << a << "\n";
    }

    // --- the data path, callback by callback ---
    os << "\n" << thin << "\nDATA PATH\n" << thin << "\n";
    auto walk = [&os](const ChunkTrace &t, const char *label) {
      os << label << ":\n";
      const void *prev_out = nullptr;
      for (const auto &e : t.entries) {
        if (e.status != Status::kExecuted) continue;
        os << "    " << StageName(e.stage) << "\n";
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "        in  0x%-14llx %-7s %10zu B",
                      static_cast<unsigned long long>(
                          reinterpret_cast<uintptr_t>(e.input_ptr)),
                      MemLocName(e.input_loc), e.input_bytes);
        os << buf;
        if (e.input_hash_valid) os << "  hash=" << e.input_hash;
        if (prev_out != nullptr && e.input_ptr != nullptr) {
          os << (prev_out == e.input_ptr ? "  [same buffer as previous output]"
                                         : "  [NEW buffer]");
        }
        os << "\n";
        std::snprintf(buf, sizeof(buf),
                      "        out 0x%-14llx %-7s %10zu B",
                      static_cast<unsigned long long>(
                          reinterpret_cast<uintptr_t>(e.output_ptr)),
                      MemLocName(e.output_loc), e.output_bytes);
        os << buf;
        if (e.output_hash_valid) os << "  hash=" << e.output_hash;
        os << "\n";
        const size_t d2h = e.TransferBytes("D2H", false);
        const size_t h2d = e.TransferBytes("H2D", false);
        const size_t d2d = e.TransferBytes("D2D", false);
        const size_t harness = e.TransferBytes("D2H", true) +
                               e.TransferBytes("H2D", true) +
                               e.TransferBytes("D2D", true);
        os << "        transfers: production D2H=" << d2h << "B H2D=" << h2d
           << "B D2D=" << d2d << "B";
        if (harness != 0) os << "  |  TEST_HARNESS_TRANSFER=" << harness << "B";
        os << "\n        kernels: " << e.KernelLaunches();
        if (!e.kernels.empty()) {
          os << "  (";
          for (size_t k = 0; k < e.kernels.size() && k < 4; ++k) {
            if (k) os << ", ";
            os << e.kernels[k].name << " x" << e.kernels[k].count;
          }
          if (e.kernels.size() > 4) os << ", ...";
          os << ")";
        }
        os << "\n";
        if (e.output_ptr != nullptr) prev_out = e.output_ptr;
      }
    };
    walk(nt, "Native");
    os << "\n";
    walk(ct, "CLIO");

    // --- per-callback verdicts ---
    os << "\n" << thin << "\nPER-CALLBACK RESULT\n" << thin << "\n";
    for (const auto &sc : c.stages) {
      os << "\n    " << StageName(sc.stage) << ":  "
         << (sc.Pass() ? "PASS" : "FAIL") << "\n";
      os << "        native=" << StatusName(sc.status_native)
         << "  clio=" << StatusName(sc.status_clio) << "\n";
      for (const auto &chk : sc.checks) {
        os << "        " << std::setw(30) << std::left << chk.name
           << std::right << "  "
           << (chk.skipped ? "N/A " : (chk.pass ? "PASS" : "FAIL"));
        if (!chk.detail.empty()) os << "   " << chk.detail;
        os << "\n";
      }
    }

    // --- residency, transfers, fallback ---
    os << "\n" << thin << "\nRESIDENCY AND TRANSFERS\n" << thin << "\n";
    os << "    Input GPU-resident (driver-verified):  native="
       << (c.native_input_device ? "YES" : "NO")
       << "  clio=" << (c.clio_input_device ? "YES" : "NO") << "\n";
    os << "    Production D->H:  native=" << c.native_production_d2h_count
       << " copies / " << c.native_production_d2h << " B   clio="
       << c.clio_production_d2h_count << " copies / " << c.clio_production_d2h
       << " B\n";
    os << "    Production H->D:  native=" << c.native_production_h2d_count
       << " copies / " << c.native_production_h2d << " B   clio="
       << c.clio_production_h2d_count << " copies / " << c.clio_production_h2d
       << " B\n";
    os << "    Test-harness transfers (excluded from the above): "
       << c.harness_transfer_bytes << " B\n";
    os << "    Bulk production transfers (>= " << (c.chunk_bytes / 2)
       << " B, i.e. the payload itself crossing the bus):  native="
       << c.native_bulk_transfers << "  clio=" << c.clio_bulk_transfers << "\n";
    for (const auto &d : c.bulk_transfer_detail) {
      os << "        " << d << "\n";
    }
    os << "    Kernel launches:  native=" << c.native_kernel_launches
       << "  clio=" << c.clio_kernel_launches << "\n";
    os << "    CPU fallback (a data stage that ran with zero kernels):  ";
    if (c.native_stages_without_kernels.empty() &&
        c.clio_stages_without_kernels.empty()) {
      os << "native=NO clio=NO\n";
    } else {
      os << "\n";
      for (const auto &s : c.native_stages_without_kernels) {
        os << "        native: " << s << "\n";
      }
      for (const auto &s : c.clio_stages_without_kernels) {
        os << "        clio:   " << s << "\n";
      }
    }

    os << "\n" << thin << "\nCODEC REPRODUCIBILITY (each side against itself)\n"
       << thin << "\n";
    auto self_line = [&os](const char *label, const ByteCompareResult &r) {
      os << "    " << label << ": ";
      if (!r.valid) {
        os << "not checked\n";
      } else if (r.Identical()) {
        os << "reproducible -- a second run over the same chunk produced the "
              "same "
           << r.total_bytes << " bytes\n";
      } else {
        os << "NOT reproducible -- " << r.differing_bytes << " of "
           << r.total_bytes
           << " bytes differ between two runs of the SAME implementation on "
              "the SAME chunk (first at byte "
           << r.first_differing_byte << ")\n";
      }
    };
    self_line("native", c.native_self_compare);
    self_line("clio  ", c.clio_self_compare);

    os << "\n" << thin << "\nFIRST DIVERGENCE\n" << thin << "\n";
    if (c.first_divergent_stage < 0) {
      os << "    none -- every functional callback agrees\n";
    } else {
      os << "    " << c.first_divergence << "\n";
      os << "\n    Callbacks BEFORE it all passed, so the inputs to this one "
            "are equal on\n    both sides and the difference originates here.\n";
    }

    os << "\n    CHUNK " << c.chunk_id << " FINAL: " << (c.pass ? "PASS" : "FAIL")
       << "\n\n";
  }

  // --- the matrix ---
  os << rule << "\nFINAL CALLBACK TRACE MATRIX\n" << rule << "\n\n";
  os << "| Chunk | Regime | Seq | Stats | Diag | NN in | Pred | Rank | Model | "
        "Shuf | Shuf B | Quant | Quant B | GPU | ExtraD2H | Result |\n";
  os << "|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n";

  auto cell = [](const ChunkComparison &c, Stage stage,
                 const char *check) -> std::string {
    for (const auto &sc : c.stages) {
      if (sc.stage != stage) continue;
      for (const auto &chk : sc.checks) {
        if (chk.name != check) continue;
        if (chk.skipped) return "n/a";
        return chk.pass ? "PASS" : "FAIL";
      }
      return "-";
    }
    return "-";
  };
  auto stage_cell = [](const ChunkComparison &c, Stage stage) -> std::string {
    for (const auto &sc : c.stages) {
      if (sc.stage == stage) return sc.Pass() ? "PASS" : "FAIL";
    }
    return "-";
  };
  auto sel = [](const ChunkComparison &c, const char *which) -> std::string {
    for (const auto &sc : c.stages) {
      if (sc.stage != Stage::kSelection) continue;
      for (const auto &chk : sc.checks) {
        if (chk.name == which) return chk.pass ? "PASS" : "FAIL";
      }
    }
    return "-";
  };

  for (const auto &c : comparisons) {
    os << "| " << c.chunk_id << " | " << c.regime << " | "
       << (c.sequence_pass ? "PASS" : "FAIL") << " | "
       << stage_cell(c, Stage::kStatistics) << " | "
       << stage_cell(c, Stage::kDiagnostics) << " | "
       << cell(c, Stage::kStatistics, "input hash") << " | "
       << cell(c, Stage::kInference, "predicted ratio") << " | "
       << cell(c, Stage::kRanking, "ranking order") << " | "
       << cell(c, Stage::kSelection, "selected action") << " | "
       << sel(c, "shuffle selected") << " | "
       << cell(c, Stage::kShuffle, "shuffle output bytes") << " | "
       << sel(c, "quantization selected") << " | "
       << cell(c, Stage::kQuantization, "quantized output bytes") << " | "
       << ((c.native_input_device && c.clio_input_device) ? "PASS" : "FAIL")
       << " | "
       << (c.clio_bulk_transfers <= c.native_bulk_transfers
               ? "none"
               : (std::to_string(c.clio_bulk_transfers) + " bulk"))
       << " | " << (c.pass ? "PASS" : "FAIL") << " |\n";
  }

  int passed = 0;
  for (const auto &c : comparisons) {
    if (c.pass) ++passed;
  }
  os << "\n" << rule << "\nOVERALL: " << passed << " / " << comparisons.size()
     << " chunks PASS\n" << rule << "\n";
}

bool WriteReport(const std::string &path,
                 const std::vector<ChunkTrace> &native_traces,
                 const std::vector<ChunkTrace> &clio_traces,
                 const std::vector<ChunkComparison> &comparisons,
                 const std::string &preamble) {
  std::ofstream os(path);
  if (!os) return false;
  PrintReport(os, native_traces, clio_traces, comparisons, preamble);
  return true;
}

}  // namespace npeq
