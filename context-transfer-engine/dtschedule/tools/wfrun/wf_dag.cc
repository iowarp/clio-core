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
 * @file wf_dag.cc
 * placement.json loading, sizing, ordering, rank split and the --dry-run
 * report of dtschedule_wfrun (see wf_dag.h).
 */

#include "wf_dag.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <tuple>

namespace dtwf {

namespace {

using json = nlohmann::json;

/**
 * Base name of a path.
 * @param path Path
 * @return Part after the last '/'
 */
std::string Basename(const std::string &path) {
  const size_t s = path.find_last_of('/');
  return s == std::string::npos ? path : path.substr(s + 1);
}

/**
 * Short host name (up to the first '.').
 * @param host Host name
 * @return Short name
 */
std::string ShortName(const std::string &host) {
  return host.substr(0, host.find('.'));
}

/**
 * Index of a node by name, appending it when new.
 * @param dag Workflow
 * @param name Node name
 * @return Index
 */
int NodeIndex(WfDag *dag, const std::string &name) {
  for (size_t i = 0; i < dag->nodes.size(); ++i) {
    if (dag->nodes[i] == name) return static_cast<int>(i);
  }
  dag->nodes.push_back(name);
  return static_cast<int>(dag->nodes.size()) - 1;
}

/**
 * Index of a file by placement path, creating it when new.
 * @param dag Workflow
 * @param index Path -> index map
 * @param path Placement path
 * @return Index
 */
int FileIndex(WfDag *dag, std::unordered_map<std::string, int> *index,
              const std::string &path) {
  auto it = index->find(path);
  if (it != index->end()) return it->second;
  WfFile f;
  f.path = path;
  f.cls = ClassifyFile(path);
  dag->files.push_back(f);
  (*index)[path] = static_cast<int>(dag->files.size()) - 1;
  return (*index)[path];
}

/**
 * Read the tasks object: node, level, type, runtime, inputs, outputs (with
 * output_bytes as a size fallback).
 * @param j Placement document
 * @param dag Workflow
 * @param index Path -> file index map
 * @param err Output: reason on failure
 * @return true on success
 */
bool ReadTasks(const json &j, WfDag *dag,
               std::unordered_map<std::string, int> *index, std::string *err) {
  for (const auto &[name, t] : j.at("tasks").items()) {
    WfTask task;
    task.name = name;
    task.node = NodeIndex(dag, t.value("node", std::string("localhost")));
    task.level = t.value("level", 0);
    task.type = t.value("type", std::string(""));
    task.runtime = t.value("runtime", 0.0);
    const int ti = static_cast<int>(dag->tasks.size());
    for (const auto &in : t.value("inputs", json::array())) {
      const int fi = FileIndex(dag, index, in.get<std::string>());
      task.inputs.push_back(fi);
      dag->files[fi].consumers.push_back(ti);
    }
    const json bytes = t.value("output_bytes", json::object());
    for (const auto &out : t.value("outputs", json::array())) {
      const std::string p = out.get<std::string>();
      const int fi = FileIndex(dag, index, p);
      if (dag->files[fi].producer >= 0) {
        *err = "file " + p + " has two producers";
        return false;
      }
      dag->files[fi].producer = ti;
      if (bytes.contains(p)) dag->files[fi].size = bytes[p].get<uint64_t>();
      task.outputs.push_back(fi);
    }
    dag->tasks.push_back(task);
  }
  return true;
}

/**
 * Read the files block (sizes, staging node) and resolve every staged
 * input's node: producer_node when given, else its first consumer's.
 * @param j Placement document
 * @param dag Workflow
 * @param index Path -> file index map
 */
void ReadFiles(const json &j, WfDag *dag,
               std::unordered_map<std::string, int> *index) {
  const json files = j.value("files", json::object());
  for (const auto &[path, f] : files.items()) {
    const int fi = FileIndex(dag, index, path);
    if (f.contains("size") && f["size"].is_number()) {
      dag->files[fi].size = f["size"].get<uint64_t>();
    }
    if (dag->files[fi].producer < 0 && f.contains("producer_node") &&
        f["producer_node"].is_string()) {
      dag->files[fi].stage_node =
          NodeIndex(dag, f["producer_node"].get<std::string>());
    }
  }
  for (auto &f : dag->files) {
    if (f.producer < 0 && f.stage_node < 0 && !f.consumers.empty()) {
      f.stage_node = dag->tasks[f.consumers.front()].node;
    }
  }
}

/**
 * Longest compute path through the DAG (task weight = runtime).
 * @param dag Workflow (ordered)
 * @param chain Output: task indices along the path
 * @return Path length in instance seconds
 */
double CriticalPath(const WfDag &dag, std::vector<int> *chain) {
  std::vector<double> finish(dag.tasks.size(), 0.0);
  std::vector<int> prev(dag.tasks.size(), -1);
  int last = -1;
  for (int t : dag.order) {
    double start = 0.0;
    for (int fi : dag.tasks[t].inputs) {
      const int p = dag.files[fi].producer;
      if (p >= 0 && finish[p] > start) {
        start = finish[p];
        prev[t] = p;
      }
    }
    finish[t] = start + dag.tasks[t].runtime;
    if (last < 0 || finish[t] > finish[last]) last = t;
  }
  chain->clear();
  for (int t = last; t >= 0; t = prev[t]) chain->insert(chain->begin(), t);
  return last < 0 ? 0.0 : finish[last];
}

/**
 * Print per-node task, level and byte counts.
 * @param dag Workflow
 * @param runtime_scale Compute seconds per instance second
 */
void PrintNodes(const WfDag &dag, double runtime_scale) {
  for (size_t n = 0; n < dag.nodes.size(); ++n) {
    std::map<int, int> levels;
    double compute = 0, out_b = 0, in_b = 0, staged_b = 0;
    for (const auto &t : dag.tasks) {
      if (t.node != static_cast<int>(n)) continue;
      ++levels[t.level];
      compute += t.runtime * runtime_scale;
      for (int fi : t.outputs) out_b += dag.files[fi].size;
      for (int fi : t.inputs) in_b += dag.files[fi].size;
    }
    for (const auto &f : dag.files) {
      if (f.producer < 0 && f.stage_node == static_cast<int>(n))
        staged_b += f.size;
    }
    int ntasks = 0;
    std::string lv;
    for (const auto &[l, c] : levels) {
      ntasks += c;
      lv += " L" + std::to_string(l) + ":" + std::to_string(c);
    }
    printf(
        "  node %-20s tasks=%-4d compute_s=%-8.1f write_gb=%-7.2f "
        "read_gb=%-7.2f stage_gb=%.2f |%s\n",
        dag.nodes[n].c_str(), ntasks, compute, out_b / 1e9, in_b / 1e9,
        staged_b / 1e9, lv.c_str());
  }
}

/**
 * Print bytes per data class and the cross-node edges.
 * @param dag Workflow
 */
void PrintFiles(const WfDag &dag) {
  double cls_b[kNumDataClasses] = {0};
  int cls_n[kNumDataClasses] = {0};
  double cross_b = 0, cross_pair_b = 0;
  int cross_e = 0, local_e = 0, cross_pairs = 0;
  for (const auto &f : dag.files) {
    cls_b[static_cast<int>(f.cls)] += f.size;
    ++cls_n[static_cast<int>(f.cls)];
    const int src = f.producer >= 0 ? dag.tasks[f.producer].node : f.stage_node;
    std::set<int> remote;
    for (int c : f.consumers) {
      if (dag.tasks[c].node == src) {
        ++local_e;
        continue;
      }
      ++cross_e;
      cross_b += f.size;
      remote.insert(dag.tasks[c].node);
    }
    cross_pairs += static_cast<int>(remote.size());
    cross_pair_b += static_cast<double>(remote.size()) * f.size;
  }
  for (int c = 0; c < kNumDataClasses; ++c) {
    if (cls_n[c] == 0) continue;
    printf("  class %-13s files=%-4d gb=%.2f\n",
           DataClassName(static_cast<DataClass>(c)), cls_n[c], cls_b[c] / 1e9);
  }
  printf(
      "  edges: cross-node=%d (%.2f GB read remotely; %d file->node "
      "pairs, %.2f GB) local=%d\n",
      cross_e, cross_b / 1e9, cross_pairs, cross_pair_b / 1e9, local_e);
}

}  // namespace

bool LoadDag(const std::string &path, const std::string &run, WfDag *dag,
             std::string *err) {
  std::ifstream in(path);
  if (!in) {
    *err = "cannot open " + path;
    return false;
  }
  json j;
  try {
    in >> j;
    for (const auto &n : j.value("nodes", json::array())) {
      NodeIndex(dag, n.get<std::string>());
    }
    std::unordered_map<std::string, int> index;
    if (!j.contains("tasks") || !ReadTasks(j, dag, &index, err)) {
      if (err->empty()) *err = "no tasks in " + path;
      return false;
    }
    ReadFiles(j, dag, &index);
    dag->recipe = j.value("recipe", std::string("workflow"));
    std::string out_dir = j.value("out_dir", std::string(""));
    while (out_dir.size() > 1 && out_dir.back() == '/') out_dir.pop_back();
    dag->run = !run.empty()
                   ? run
                   : (!out_dir.empty() ? Basename(out_dir) : dag->recipe);
  } catch (const json::exception &e) {
    *err = std::string("bad placement json: ") + e.what();
    return false;
  }
  std::set<std::string> tags;
  for (auto &f : dag->files) {
    f.tag = dag->run + "__" + Basename(f.path);
    if (!tags.insert(f.tag).second) {
      *err = "two files share the base name of " + f.path;
      return false;
    }
  }
  return true;
}

void ScaleSizes(WfDag *dag, double size_scale, double max_file_mb) {
  const double cap = max_file_mb > 0 ? max_file_mb * 1048576.0 : 0.0;
  for (auto &f : dag->files) {
    double s = static_cast<double>(f.size) * size_scale;
    if (cap > 0 && s > cap) s = cap;
    f.size = std::max<uint64_t>(1, static_cast<uint64_t>(std::llround(s)));
  }
}

bool TopoSort(WfDag *dag, std::string *err) {
  const size_t n = dag->tasks.size();
  std::vector<int> missing(n, 0);
  std::vector<std::vector<int>> next(n);
  for (size_t t = 0; t < n; ++t) {
    for (int fi : dag->tasks[t].inputs) {
      const int p = dag->files[fi].producer;
      if (p < 0) continue;
      ++missing[t];
      next[p].push_back(static_cast<int>(t));
    }
  }
  std::set<std::tuple<int, std::string, int>> ready;
  for (size_t t = 0; t < n; ++t) {
    if (missing[t] == 0) {
      ready.emplace(dag->tasks[t].level, dag->tasks[t].name,
                    static_cast<int>(t));
    }
  }
  dag->order.clear();
  while (!ready.empty()) {
    const int t = std::get<2>(*ready.begin());
    ready.erase(ready.begin());
    dag->order.push_back(t);
    for (int c : next[t]) {
      if (--missing[c] == 0)
        ready.emplace(dag->tasks[c].level, dag->tasks[c].name, c);
    }
  }
  if (dag->order.size() != n) {
    *err = "the task graph has a cycle";
    return false;
  }
  return true;
}

int MatchNode(const WfDag &dag, const std::string &host) {
  const std::string s = ShortName(host);
  int best = -1;
  for (size_t i = 0; i < dag.nodes.size(); ++i) {
    const std::string &n = dag.nodes[i];
    if (n == host) return static_cast<int>(i);
    if (ShortName(n) == s) best = static_cast<int>(i);
  }
  if (best >= 0) return best;
  for (size_t i = 0; i < dag.nodes.size(); ++i) {
    const std::string &n = dag.nodes[i];
    if (n.size() > s.size() && n.compare(0, s.size(), s) == 0 &&
        (n[s.size()] == '-' || n[s.size()] == '.') &&
        (best < 0 || n.size() > dag.nodes[best].size())) {
      best = static_cast<int>(i);
    }
  }
  return best;
}

std::vector<int> RankTasks(const WfDag &dag, int node, int local_rank,
                           int local_size) {
  std::vector<int> mine;
  int k = 0;
  for (int t : dag.order) {
    if (dag.tasks[t].node != node) continue;
    if (k++ % local_size == local_rank) mine.push_back(t);
  }
  return mine;
}

std::vector<int> RankStaged(const WfDag &dag, int node, int local_rank,
                            int local_size) {
  std::vector<int> mine;
  int k = 0;
  for (size_t f = 0; f < dag.files.size(); ++f) {
    const WfFile &file = dag.files[f];
    if (file.producer >= 0 || file.stage_node != node ||
        file.consumers.empty()) {
      continue;
    }
    if (k++ % local_size == local_rank) mine.push_back(static_cast<int>(f));
  }
  return mine;
}

void PrintDryRun(const WfDag &dag, double runtime_scale, int ppn) {
  double total = 0, staged = 0, compute = 0;
  int nstaged = 0;
  for (const auto &f : dag.files) {
    total += f.size;
    if (f.producer < 0) {
      staged += f.size;
      ++nstaged;
    }
  }
  for (const auto &t : dag.tasks) compute += t.runtime * runtime_scale;
  printf(
      "wfrun dry-run recipe=%s run=%s tasks=%zu files=%zu staged=%d "
      "nodes=%zu total_gb=%.2f staged_gb=%.2f compute_s=%.1f\n",
      dag.recipe.c_str(), dag.run.c_str(), dag.tasks.size(), dag.files.size(),
      nstaged, dag.nodes.size(), total / 1e9, staged / 1e9, compute);
  PrintNodes(dag, runtime_scale);
  PrintFiles(dag);
  std::vector<int> chain;
  const double cp = CriticalPath(dag, &chain);
  std::string types;
  for (int t : chain) {
    types +=
        (types.empty() ? "" : ">") +
        (dag.tasks[t].type.empty() ? dag.tasks[t].name : dag.tasks[t].type);
  }
  printf("  critical path: %zu tasks, %.1f s scaled (%.0f s instance): %s\n",
         chain.size(), cp * runtime_scale, cp, types.c_str());
  double worst = 0;
  for (size_t n = 0; n < dag.nodes.size(); ++n) {
    for (int r = 0; r < ppn; ++r) {
      double s = 0;
      for (int t : RankTasks(dag, static_cast<int>(n), r, ppn))
        s += dag.tasks[t].runtime;
      worst = std::max(worst, s * runtime_scale);
    }
  }
  printf("  ppn=%d: busiest rank computes %.1f s\n", ppn, worst);
}

}  // namespace dtwf
