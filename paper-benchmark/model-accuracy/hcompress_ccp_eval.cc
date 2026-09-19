/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/** @file hcompress_ccp_eval.cc
 *  @brief Score HCompress's cost model over chunks a Clio run already measured.
 *
 *    hcompress_ccp_eval --seed seed.csv --eval eval.csv --out pred.csv
 *                       [--feedback-interval N] [--forget L]
 *                       [--feedback-scope self|executed|all] [--feedback-topk N] [--seed-json DIR]
 *
 *  Runs TWO instances of ctp::compress::model::HCompressCcpPredictor over the
 *  same rows in one pass:
 *
 *    seed only   fitted on --seed and never updated again.
 *    + feedback  the same fit, then updated from measured outcomes as the run
 *                proceeds, every N observations.
 *
 *  Both predict every row BEFORE that chunk's feedback is applied, so no
 *  prediction is ever scored against a measurement the model had already been
 *  shown. That ordering is the whole content of an online-learning claim, and
 *  getting it backwards is the easiest way to manufacture one.
 *
 *  --feedback-scope decides WHICH measured row the model learns from, and the
 *  choice of source matters more than the count:
 *    self      the configuration(s) THIS model would itself have chosen -- the
 *              --feedback-topk cheapest by its OWN predicted cost under the
 *              deployed policy.
 *              This is the DEFAULT and the reported setting. HCompress's
 *              design is a closed loop: it selects, the selection runs, and the
 *              outcome of ITS OWN choice comes back. Reproducing that here
 *              means the baseline learns from its own decisions and never
 *              touches NeuroPress's.
 *    executed  the configuration the CAMPAIGN actually ran, which is
 *              NeuroPress's adopted pick. Kept for the record, because it is
 *              what earlier revisions reported and the difference is large --
 *              but it is NOT an independent baseline. Fed only rows another
 *              selector chose, the model converges toward imitating that
 *              selector: measured at 56-72% per-chunk pick agreement with
 *              NeuroPress, against 22-31% for NeuroPress's own static weights.
 *    all       every configuration measured for that chunk. An upper bound on
 *              information, not a realistic budget: it is available only
 *              because exploration was forced, and no deployed selector learns
 *              31 counterfactual outcomes per operation.
 *
 *  Only `self` compares two systems learning over their own experience, which
 *  is the comparison the figure is for.
 *
 *  --feedback-topk MATCHES THE BUDGET. Source matters, but so does volume:
 *  NeuroPress's exploration SGD trains on 7 samples per chunk (4 cheapest plus
 *  3 most mispredicted -- compressor_runtime.cc, kMaxExploreSgdSamples=7 with
 *  EXPLORE_SGD_HARD=3), plus its primary. Feeding this model a single row per
 *  chunk would hand NeuroPress an ~8x information advantage and call the result
 *  a model comparison. --feedback-topk 7 gives it the same budget, drawn
 *  entirely from its OWN ranking, so neither system ever sees the other's
 *  choices. The default of 1 is the strict reading of the paper's "the executed
 *  choice"; 7 is the budget-matched setting this figure reports.
 *
 *  The tool predicts and never selects: it links no codec and compresses
 *  nothing.
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "clio_ctp/compress/model/hcompress_ccp_predictor.h"

using ctp::compress::model::CcpConfig;
using ctp::compress::model::CcpInputs;
using ctp::compress::model::CcpObservation;
using ctp::compress::model::HCompressCcpPredictor;

namespace {

/** The deployed selection cost (NeuroPressCost / RankingWeights defaults):
 *  times floored at 1 ms, ratio capped at 100x, 5e6 bytes per ms. Scoring a
 *  self-feedback pick with anything else would feed the model the outcome of a
 *  choice it would not actually have made. */
constexpr double kTimeFloorMs = 1.0;
constexpr double kRatioCap = 100.0;
constexpr double kBytesPerMs = 5e6;

double PolicyCost(double ct, double dt, double ratio, double bytes) {
  const double r = std::min(kRatioCap, std::max(ratio, 0.1));
  return std::max(kTimeFloorMs, ct) + std::max(kTimeFloorMs, dt) +
         bytes / (r * kBytesPerMs);
}

std::vector<std::string> SplitCsv(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  std::istringstream is(line);
  while (std::getline(is, cur, ',')) out.push_back(cur);
  return out;
}

double Num(const std::string& s) {
  if (s.empty()) return 0.0;
  try {
    return std::stod(s);
  } catch (...) {
    return 0.0;
  }
}

struct Table {
  std::map<std::string, size_t> col;
  std::vector<std::vector<std::string>> rows;
  bool Load(const std::string& path, std::string* err) {
    std::ifstream f(path);
    if (!f.is_open()) {
      *err = "cannot open " + path;
      return false;
    }
    std::string line;
    if (!std::getline(f, line)) {
      *err = "empty " + path;
      return false;
    }
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto head = SplitCsv(line);
    for (size_t i = 0; i < head.size(); ++i) col[head[i]] = i;
    while (std::getline(f, line)) {
      if (line.empty()) continue;
      if (line.back() == '\r') line.pop_back();
      rows.push_back(SplitCsv(line));
    }
    return true;
  }
  bool Has(const char* c) const { return col.count(c) != 0; }
  const std::string& Get(const std::vector<std::string>& r,
                         const char* c) const {
    static const std::string kEmpty;
    const auto it = col.find(c);
    if (it == col.end() || it->second >= r.size()) return kEmpty;
    return r[it->second];
  }
};

CcpInputs InputsOf(const Table& t, const std::vector<std::string>& r,
                   const std::string& constant_as) {
  CcpInputs in;
  in.data_type = t.Get(r, "data_type");
  in.data_format = t.Get(r, "data_format");
  in.library = t.Get(r, "library");
  in.distribution = t.Get(r, "distribution");
  if (!constant_as.empty() && in.distribution == "constant") {
    in.distribution = constant_as;
  }
  return in;
}

CcpObservation ObsOf(const Table& t, const std::vector<std::string>& r,
                     const std::string& constant_as) {
  CcpObservation o;
  o.inputs = InputsOf(t, r, constant_as);
  o.bytes = Num(t.Get(r, "bytes"));
  o.compress_time_ms = Num(t.Get(r, "ct_ms"));
  o.decompress_time_ms = Num(t.Get(r, "dt_ms"));
  o.compression_ratio = Num(t.Get(r, "ratio"));
  return o;
}

[[noreturn]] void Usage(int code) {
  std::cerr <<
      "hcompress_ccp_eval --seed seed.csv --eval eval.csv --out pred.csv\n"
      "                   [--feedback-interval N] [--forget L]\n"
      "                   [--feedback-scope self|executed|all] [--feedback-topk N] [--constant-as NAME]\n"
      "                   [--regularization C] [--ratio-target-cap X]\n"
      "                   [--seed-json DIR]\n\n"
      "seed.csv: data_type,data_format,library,distribution,bytes,ct_ms,dt_ms,ratio\n"
      "eval.csv: the same, plus chunk,seq,executed\n";
  std::exit(code);
}

}  // namespace

int main(int argc, char** argv) {
  std::string seed_path, eval_path, out_path, seed_json, scope = "self";
  size_t topk = 1;
  std::string constant_as;
  CcpConfig cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << what << "\n";
        Usage(2);
      }
      return argv[++i];
    };
    if (a == "--seed") seed_path = next("--seed");
    else if (a == "--eval") eval_path = next("--eval");
    else if (a == "--out") out_path = next("--out");
    else if (a == "--seed-json") seed_json = next("--seed-json");
    else if (a == "--feedback-scope") scope = next("--feedback-scope");
    else if (a == "--feedback-topk")
      topk = static_cast<size_t>(std::stoul(next("--feedback-topk")));
    else if (a == "--constant-as") constant_as = next("--constant-as");
    else if (a == "--feedback-interval")
      cfg.feedback_interval = static_cast<size_t>(std::stoul(next("--feedback-interval")));
    else if (a == "--forget") cfg.forget_factor = std::stod(next("--forget"));
    else if (a == "--ratio-target-cap")
      cfg.ratio_target_cap = std::stod(next("--ratio-target-cap"));
    else if (a == "--regularization")
      cfg.regularization = std::stod(next("--regularization"));
    else if (a == "-h" || a == "--help") Usage(0);
    else { std::cerr << "unknown flag: " << a << "\n"; Usage(2); }
  }
  if (seed_path.empty() || eval_path.empty() || out_path.empty()) Usage(2);
  if (scope != "self" && scope != "executed" && scope != "all") {
    std::cerr << "--feedback-scope must be self, executed or all\n";
    return 2;
  }

  std::string err;
  Table seed, eval;
  if (!seed.Load(seed_path, &err) || !eval.Load(eval_path, &err)) {
    std::cerr << err << "\n";
    return 1;
  }
  if (!eval.Has("chunk") || !eval.Has("library")) {
    std::cerr << "eval.csv needs at least chunk and library columns\n";
    return 1;
  }

  // ---- Seed both instances identically ------------------------------------
  std::vector<CcpObservation> rows;
  rows.reserve(seed.rows.size());
  for (const auto& r : seed.rows) rows.push_back(ObsOf(seed, r, constant_as));

  HCompressCcpPredictor seed_only(cfg), with_fb(cfg);
  seed_only.Seed(rows);
  with_fb.Seed(rows);
  std::cerr << "seeded on " << rows.size() << " row(s); encoded dimension "
            << seed_only.Dimension() << " (1 intercept + "
            << seed_only.Vocabulary("data_type").size() << " type + "
            << seed_only.Vocabulary("data_format").size() << " format + "
            << seed_only.Vocabulary("library").size() << " library + "
            << seed_only.Vocabulary("distribution").size() << " distribution)\n";
  if (!seed_json.empty() && !seed_only.Save(seed_json)) {
    std::cerr << "warning: could not write the seed JSON to " << seed_json << "\n";
  }

  // ---- One pass over the evaluation rows, chunk by chunk -------------------
  std::ofstream out(out_path);
  if (!out.is_open()) {
    std::cerr << "cannot write " << out_path << "\n";
    return 1;
  }
  out << "chunk,library,pred_ct_ms_seed,pred_dt_ms_seed,pred_ratio_seed,"
         "pred_ct_ms_fb,pred_dt_ms_fb,pred_ratio_fb,unseen_distribution\n";
  out.precision(9);

  // Stable chunk order: the file's own order, which prepare_inputs.py writes
  // in ascending seq. Re-sorting here would decouple the feedback order from
  // the order the run actually executed in.
  std::vector<std::string> order;
  std::map<std::string, std::vector<const std::vector<std::string>*>> by_chunk;
  for (const auto& r : eval.rows) {
    const std::string& c = eval.Get(r, "chunk");
    if (!by_chunk.count(c)) order.push_back(c);
    by_chunk[c].push_back(&r);
  }

  // The SEED's vocabulary, frozen here. Testing against the live one instead
  // would flag only the FIRST chunk of an unprofiled class, because feedback
  // widens the model as soon as it sees one -- turning a fixed property of the
  // seed into a "first occurrence" marker.
  const std::vector<std::string> seed_dists = seed_only.Vocabulary("distribution");

  size_t fed = 0, unseen = 0, predicted = 0, self_matched_run = 0;
  for (const auto& chunk : order) {
    const auto& group = by_chunk[chunk];
    // The distribution class is deduced ONCE per buffer, as the paper has it.
    const CcpInputs chunk_inputs = InputsOf(eval, *group.front(), constant_as);
    seed_only.SetInputs(chunk_inputs);
    with_fb.SetInputs(chunk_inputs);
    const bool dist_unseen =
        !chunk_inputs.distribution.empty() &&
        std::find(seed_dists.begin(), seed_dists.end(),
                  chunk_inputs.distribution) == seed_dists.end();
    if (dist_unseen) ++unseen;

    // The candidate THIS model would have selected, by its own predicted cost.
    // Captured during prediction, so it is decided before any of this chunk's
    // measurements are shown to it.
    std::vector<std::pair<double, const std::vector<std::string>*>> self_rank;
    self_rank.reserve(group.size());
    for (const auto* rp : group) {
      const auto& r = *rp;
      const std::string lib = eval.Get(r, "library");
      const double bytes = Num(eval.Get(r, "bytes"));
      const auto a = seed_only.PredictFor(lib, bytes);
      const auto b = with_fb.PredictFor(lib, bytes);
      const double self_cost = PolicyCost(b.compression_time_ms,
                                          b.decompression_time_ms,
                                          b.compression_ratio, bytes);
      if (std::isfinite(self_cost)) self_rank.emplace_back(self_cost, rp);
      out << chunk << ',' << lib << ',' << a.compression_time_ms << ','
          << a.decompression_time_ms << ',' << a.compression_ratio << ','
          << b.compression_time_ms << ',' << b.decompression_time_ms << ','
          << b.compression_ratio << ',' << (dist_unseen ? 1 : 0) << '\n';
      ++predicted;
    }
    // Feedback AFTER every prediction for this chunk.
    if (scope == "self") {
      const size_t k = std::min(topk, self_rank.size());
      std::partial_sort(self_rank.begin(), self_rank.begin() + k,
                        self_rank.end(),
                        [](const auto& a, const auto& b) { return a.first < b.first; });
      for (size_t i = 0; i < k; ++i) {
        with_fb.Observe(ObsOf(eval, *self_rank[i].second, constant_as));
        ++fed;
      }
      // Coincidence with the campaign is reported for the model's own ARGMIN.
      if (k > 0 && Num(eval.Get(*self_rank[0].second, "executed")) != 0.0)
        ++self_matched_run;
    } else {
      for (const auto* rp : group) {
        const auto& r = *rp;
        const bool executed = Num(eval.Get(r, "executed")) != 0.0;
        if (scope == "executed" && !executed) continue;
        with_fb.Observe(ObsOf(eval, r, constant_as));
        ++fed;
      }
    }
  }
  with_fb.ApplyFeedback();  // whatever the last partial interval left

  std::cerr << predicted << " prediction(s) over " << order.size()
            << " chunk(s); fed " << fed << " observation(s) in "
            << with_fb.FeedbackUpdates() << " update(s) at interval "
            << cfg.feedback_interval << ", forget " << cfg.forget_factor
            << ", scope " << scope
            << (scope == "self" ? " topk " + std::to_string(topk) : "") << "\n";
  if (scope == "self" && !order.empty()) {
    // How often this model's own choice coincided with the campaign's. Printed
    // so the independence of the baseline is a measured number in the log,
    // rather than a property the reader has to take on trust.
    std::cerr << "self-feedback picked what the run executed on "
              << (100.0 * static_cast<double>(self_matched_run) /
                  static_cast<double>(order.size()))
              << "% of chunk(s)\n";
  }
  if (unseen > 0) {
    std::cerr << unseen << " chunk(s) had a distribution class the seed never "
              << "profiled (reported per row as unseen_distribution=1)\n";
  }
  return 0;
}
