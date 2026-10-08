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
 * @file wfrun.cc
 * dtschedule_wfrun: a limited WfCommons executor over the CTE API.
 *
 * Runs the workflow of a placement.json (tools/wfcommons/dt_translator.py
 * --api) with one mpirun over all nodes. Each rank maps its host name to a
 * placement node and runs that node's tasks, dealt round-robin over the
 * node's ranks, in topological order. Workflow inputs are first staged by
 * the ranks of their staging node. Per task: wait for every input's
 * marker, read the inputs (all chunk gets in flight), compute for the
 * instance runtime x --runtime-scale (a stencil over the input bytes; wide
 * fan-ins are read and computed in batches of --input-mem-mb), and
 * generate + put the outputs asynchronously. Files are CTE tags
 * "<run>__<basename>" of 1 MiB blobs written through the dtschedule pool by
 * tools/prodcons/pc_cte_api.cc, so dtschedule alone places every byte.
 *
 * Output (rank 0, --out DIR): makespan.json (total_ms = workflow wall time
 * after staging; per-level wall times) and tasks.csv; summary line:
 *   wfrun ranks=N tasks=T files=F stage_s=S wall_s=W read_gb=R ... rc=C
 */

#include <errno.h>
#include <mpi.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "../prodcons/pc_cte_api.h"
#include "wf_dag.h"
#include "wf_data.h"

namespace {

using dtwf::WfDag;
using dtwf::WfFile;
using dtwf::WfTask;

/** Command-line options. */
struct Options {
  std::string placement;       ///< placement.json
  std::string out = ".";       ///< Output directory (rank 0)
  std::string run;             ///< Tag prefix ("" = from placement)
  std::string payload;         ///< Directory of real data files ("" = generate)
  std::string node;            ///< Force this rank's node ("" = host name)
  double runtime_scale = 1.0;  ///< Compute seconds per instance second
  double size_scale = 1.0;     ///< Factor on every file size
  double max_file_mb = 0.0;    ///< File size cap (0 = none)
  double noise = 0.05;         ///< Float-field noise amplitude
  double timeout_s = 3600.0;   ///< Give up waiting for an input
  double input_mem_mb = 1024;  ///< Inputs held at once (batch budget)
  int write_pending = 2;       ///< Files left in flight (0 = wait stored)
  int ppn = 1;                 ///< --dry-run: ranks per node assumed
  bool dry_run = false;        ///< Parse, check and report only
  bool verify = false;         ///< Compare inputs with their expected bytes
};

/** Fields of one task's record, gathered to rank 0 as doubles. */
enum Field {
  kTask,
  kRank,
  kStart,
  kWait,
  kRead,
  kCompute,
  kGen,
  kWrite,
  kEnd,
  kInBytes,
  kOutBytes,
  kErrors,
  kNumFields
};

/** Per-rank execution state. */
struct Ctx {
  const Options *opt;        ///< Options
  const WfDag *dag;          ///< Workflow
  int rank;                  ///< World rank
  double t0;                 ///< Workflow start (after staging)
  dtwf::Payload payload;     ///< Real-data source (when --payload)
  bool use_payload = false;  ///< Whether payload is open
  std::vector<char> obuf;    ///< Output staging buffer
  std::vector<double> recs;  ///< kNumFields doubles per task run
};

/**
 * Monotonic seconds.
 * @return Seconds
 */
double Now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

/**
 * Print usage.
 * @param prog Program name
 */
void Usage(const char *prog) {
  fprintf(stderr,
          "usage: %s --placement FILE [--out DIR] [--run NAME]\n"
          "  [--runtime-scale F] [--size-scale F] [--max-file-mb N]\n"
          "  [--write-pending N] [--payload DIR] [--noise F] [--timeout-s S]\n"
          "  [--input-mem-mb N]\n"
          "  [--node NAME] [--verify] [--dry-run [--ppn N]]\n",
          prog);
}

/**
 * Parse the command line.
 * @param argc Argument count
 * @param argv Arguments
 * @param o Output options
 * @return true when valid
 */
bool ParseOptions(int argc, char **argv, Options *o) {
  if (const char *e = getenv("DT_DATA_NOISE")) o->noise = atof(e);
  if (const char *e = getenv("DT_WFRUN_NODE")) o->node = e;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if (k == "--dry-run") {
      o->dry_run = true;
      continue;
    }
    if (k == "--verify") {
      o->verify = true;
      continue;
    }
    if (i + 1 >= argc) return false;
    const char *v = argv[++i];
    if (k == "--placement")
      o->placement = v;
    else if (k == "--out")
      o->out = v;
    else if (k == "--run")
      o->run = v;
    else if (k == "--payload")
      o->payload = v;
    else if (k == "--node")
      o->node = v;
    else if (k == "--runtime-scale")
      o->runtime_scale = atof(v);
    else if (k == "--size-scale")
      o->size_scale = atof(v);
    else if (k == "--max-file-mb")
      o->max_file_mb = atof(v);
    else if (k == "--noise")
      o->noise = atof(v);
    else if (k == "--timeout-s")
      o->timeout_s = atof(v);
    else if (k == "--input-mem-mb")
      o->input_mem_mb = atof(v);
    else if (k == "--write-pending")
      o->write_pending = atoi(v);
    else if (k == "--ppn")
      o->ppn = std::max(1, atoi(v));
    else
      return false;
  }
  return !o->placement.empty();
}

/**
 * Load, size and order the workflow.
 * @param o Options
 * @param dag Output
 * @return true on success (errors are printed)
 */
bool PrepareDag(const Options &o, WfDag *dag) {
  std::string err;
  if (!dtwf::LoadDag(o.placement, o.run, dag, &err) ||
      !dtwf::TopoSort(dag, &err)) {
    fprintf(stderr, "wfrun: %s\n", err.c_str());
    return false;
  }
  dtwf::ScaleSizes(dag, o.size_scale, o.max_file_mb);
  return true;
}

/**
 * Wait for an MPI request, sleeping between tests (never busy-spins).
 * @param req Request
 */
void SleepWait(MPI_Request *req) {
  int done = 0;
  MPI_Test(req, &done, MPI_STATUS_IGNORE);
  while (!done) {
    usleep(2000);
    MPI_Test(req, &done, MPI_STATUS_IGNORE);
  }
}

/** Barrier that sleeps while waiting. */
void SleepBarrier() {
  MPI_Request req;
  MPI_Ibarrier(MPI_COMM_WORLD, &req);
  SleepWait(&req);
}

/**
 * Fill the output buffer with a file's content (payload window or
 * generated data of its class).
 * @param c Context
 * @param f File
 * @return true on success
 */
bool FillFile(Ctx *c, const WfFile &f) {
  if (c->obuf.size() < f.size) c->obuf.resize(f.size);
  if (c->use_payload) return c->payload.Window(f.tag, c->obuf.data(), f.size);
  dtwf::GenerateFile(f.tag, f.cls, c->opt->noise, c->obuf.data(), f.size);
  return true;
}

/**
 * Generate and put one file (async per --write-pending).
 * @param c Context
 * @param f File
 * @param gen_s Output: seconds generating
 * @param write_s Output: seconds submitting (and storing, if pending 0)
 * @return 0 on success, errno-style code otherwise
 */
int PutOne(Ctx *c, const WfFile &f, double *gen_s, double *write_s) {
  const double g0 = Now();
  if (!FillFile(c, f)) return EIO;
  const double w0 = Now();
  *gen_s += w0 - g0;
  const int wp = c->opt->write_pending;
  int rc = PcApiPutFile(f.tag.c_str(), c->obuf.data(), f.size, wp > 0 ? wp : 1);
  PcApiDrain(wp > 0 ? 0 : 1);
  *write_s += Now() - w0;
  if (rc != 0)
    fprintf(stderr, "wfrun: put %s: %s\n", f.tag.c_str(), strerror(rc));
  return rc;
}

/**
 * Stage this rank's workflow input files, then wait until they are stored.
 * @param c Context
 * @param files File indices
 * @param bytes Output: bytes staged
 * @return Number of failures
 */
int StageInputs(Ctx *c, const std::vector<int> &files, double *bytes) {
  int bad = 0;
  double gen = 0, write = 0;
  for (int fi : files) {
    const WfFile &f = c->dag->files[fi];
    if (PutOne(c, f, &gen, &write) != 0) ++bad;
    *bytes += static_cast<double>(f.size);
  }
  return bad + PcApiDrain(1);
}

/**
 * Wait for an input's marker, completing this rank's own in-flight files
 * between polls (a consumer of them may be what we wait behind).
 * @param c Context
 * @param f File
 * @return 0 when present, EIO when its producer failed, ETIMEDOUT
 */
int WaitInput(Ctx *c, const WfFile &f) {
  const double end = Now() + c->opt->timeout_s;
  while (true) {
    const int rc = PcApiWaitFile(f.tag.c_str(), 0.25);
    if (rc != ETIMEDOUT) return rc;
    PcApiDrain(0);
    if (Now() > end) return ETIMEDOUT;
  }
}

/**
 * Read one input into buf and check its size (and, with --verify, bytes).
 * @param c Context
 * @param f File
 * @param buf Output buffer (resized to the file)
 * @return 0 on success, errno-style code otherwise
 */
int ReadInput(Ctx *c, const WfFile &f, std::vector<char> *buf) {
  buf->resize(f.size);
  size_t got = 0;
  int rc = PcApiGetFile(f.tag.c_str(), buf->data(), f.size, &got);
  if (rc == 0 && got != f.size) rc = EIO;
  if (rc == 0 && c->opt->verify && FillFile(c, f) &&
      std::memcmp(c->obuf.data(), buf->data(), f.size) != 0) {
    rc = EILSEQ;
  }
  if (rc != 0) {
    fprintf(stderr, "wfrun: rank %d: read %s (%zu of %llu bytes): %s\n",
            c->rank, f.tag.c_str(), got,
            static_cast<unsigned long long>(f.size), strerror(rc));
  }
  return rc;
}

/**
 * Wait for every input of a task.
 * @param c Context
 * @param t Task
 * @param r Record (errors are counted)
 */
void WaitInputs(Ctx *c, const WfTask &t, std::vector<double> *r) {
  for (int fi : t.inputs) {
    const int rc = WaitInput(c, c->dag->files[fi]);
    if (rc != 0) {
      fprintf(stderr, "wfrun: %s: input %s: %s\n", t.name.c_str(),
              c->dag->files[fi].tag.c_str(), strerror(rc));
      (*r)[kErrors] += 1;
    }
  }
}

/**
 * Read a task's inputs in batches of at most --input-mem-mb (at least one
 * file each) and compute over each batch for its byte share of the
 * task's scaled runtime, so a wide fan-in never holds every input at once.
 * @param c Context
 * @param t Task
 * @param r Record (read/compute times, bytes, errors)
 */
void ReadAndCompute(Ctx *c, const WfTask &t, std::vector<double> *r) {
  const double total_s = t.runtime * c->opt->runtime_scale;
  const double budget = c->opt->input_mem_mb * 1048576.0;
  double all = 0;
  for (int fi : t.inputs) all += static_cast<double>(c->dag->files[fi].size);
  size_t i = 0;
  do {
    std::vector<std::vector<char>> in;
    std::vector<std::pair<const char *, size_t>> views;
    double held = 0;
    const double r0 = Now();
    for (; i < t.inputs.size(); ++i) {
      const WfFile &f = c->dag->files[t.inputs[i]];
      if (!in.empty() && held + f.size > budget) break;
      in.emplace_back();
      held += static_cast<double>(f.size);
      if (ReadInput(c, f, &in.back()) != 0) {
        (*r)[kErrors] += 1;
        continue;
      }
      views.emplace_back(in.back().data(), in.back().size());
      (*r)[kInBytes] += static_cast<double>(f.size);
    }
    const double c0 = Now();
    (*r)[kRead] += c0 - r0;
    dtwf::ComputeFor(views, all > 0 ? total_s * held / all : total_s);
    (*r)[kCompute] += Now() - c0;
  } while (i < t.inputs.size());
}

/**
 * Run one task: wait, read + compute, write; append its record.
 * @param c Context
 * @param ti Task index
 */
void RunTask(Ctx *c, int ti) {
  const WfTask &t = c->dag->tasks[ti];
  std::vector<double> r(kNumFields, 0.0);
  r[kTask] = ti;
  r[kRank] = c->rank;
  const double s = Now();
  r[kStart] = s - c->t0;
  WaitInputs(c, t, &r);
  r[kWait] = Now() - s;
  ReadAndCompute(c, t, &r);
  for (int fi : t.outputs) {
    if (PutOne(c, c->dag->files[fi], &r[kGen], &r[kWrite]) != 0)
      r[kErrors] += 1;
    r[kOutBytes] += static_cast<double>(c->dag->files[fi].size);
  }
  r[kEnd] = Now() - c->t0;
  c->recs.insert(c->recs.end(), r.begin(), r.end());
}

/**
 * This rank's node index (by --node / DT_WFRUN_NODE or host name).
 * @param o Options
 * @param dag Workflow
 * @return Node index, or -1 when this host is not in the placement
 */
int MyNode(const Options &o, const WfDag &dag) {
  char host[256] = {0};
  gethostname(host, sizeof(host) - 1);
  return dtwf::MatchNode(dag, o.node.empty() ? std::string(host) : o.node);
}

/**
 * Abort unless every node with work has at least one rank.
 * @param dag Workflow
 * @param node This rank's node
 * @param rank World rank
 * @param size World size
 */
void CheckCoverage(const WfDag &dag, int node, int rank, int size) {
  std::vector<int> all(size);
  MPI_Allgather(&node, 1, MPI_INT, all.data(), 1, MPI_INT, MPI_COMM_WORLD);
  std::vector<bool> has(dag.nodes.size(), false);
  for (int n : all) {
    if (n >= 0) has[n] = true;
  }
  std::vector<bool> needs(dag.nodes.size(), false);
  for (const auto &t : dag.tasks) needs[t.node] = true;
  for (const auto &f : dag.files) {
    if (f.producer < 0 && f.stage_node >= 0) needs[f.stage_node] = true;
  }
  for (size_t n = 0; n < dag.nodes.size(); ++n) {
    if (needs[n] && !has[n]) {
      if (rank == 0) {
        fprintf(stderr,
                "wfrun: no rank runs on placement node %s (check the "
                "hostfile / --node)\n",
                dag.nodes[n].c_str());
      }
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
}

/**
 * Gather every rank's task records on rank 0.
 * @param recs This rank's records
 * @param size World size
 * @return All records (rank 0), empty elsewhere
 */
std::vector<double> GatherRecords(const std::vector<double> &recs, int size) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int n = static_cast<int>(recs.size());
  std::vector<int> counts(size), displs(size);
  MPI_Gather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  int total = 0;
  for (int i = 0; i < size; ++i) {
    displs[i] = total;
    total += counts[i];
  }
  std::vector<double> all(rank == 0 ? total : 0);
  MPI_Gatherv(recs.data(), n, MPI_DOUBLE, all.data(), counts.data(),
              displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return all;
}

/**
 * Write tasks.csv (one row per task) to the output directory.
 * @param o Options
 * @param dag Workflow
 * @param all Gathered records
 */
void WriteCsv(const Options &o, const WfDag &dag,
              const std::vector<double> &all) {
  FILE *fp = fopen((o.out + "/tasks.csv").c_str(), "w");
  if (fp == nullptr) return;
  fprintf(fp,
          "task,type,node,rank,level,start,wait_s,read_s,compute_s,"
          "gen_s,write_s,end,in_bytes,out_bytes,errors\n");
  for (size_t i = 0; i + kNumFields <= all.size(); i += kNumFields) {
    const double *r = &all[i];
    const WfTask &t = dag.tasks[static_cast<int>(r[kTask])];
    fprintf(fp,
            "%s,%s,%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.0f,%.0f,%d\n",
            t.name.c_str(), t.type.c_str(), dag.nodes[t.node].c_str(),
            static_cast<int>(r[kRank]), t.level, r[kStart], r[kWait], r[kRead],
            r[kCompute], r[kGen], r[kWrite], r[kEnd], r[kInBytes], r[kOutBytes],
            static_cast<int>(r[kErrors]));
  }
  fclose(fp);
}

/** Run-wide totals reported by rank 0. */
struct Totals {
  double stage_s = 0, wall_s = 0, stage_bytes = 0;  ///< Staging and wall
  double sum[kNumFields] = {0};  ///< Column sums of all task records
  int ranks = 0, errors = 0;     ///< World size, failures (incl. staging)
};

/**
 * Write makespan.json (format read by sweep.py / the jarvis package).
 * @param o Options
 * @param dag Workflow
 * @param all Gathered records
 * @param tot Totals
 */
void WriteMakespan(const Options &o, const WfDag &dag,
                   const std::vector<double> &all, const Totals &tot) {
  std::map<int, std::pair<double, double>> lv;  // level -> (start, end)
  std::map<int, int> lv_n;
  for (size_t i = 0; i + kNumFields <= all.size(); i += kNumFields) {
    const int l = dag.tasks[static_cast<int>(all[i + kTask])].level;
    auto it = lv.find(l);
    if (it == lv.end()) {
      lv[l] = {all[i + kStart], all[i + kEnd]};
    } else {
      it->second.first = std::min(it->second.first, all[i + kStart]);
      it->second.second = std::max(it->second.second, all[i + kEnd]);
    }
    ++lv_n[l];
  }
  FILE *fp = fopen((o.out + "/makespan.json").c_str(), "w");
  if (fp == nullptr) return;
  fprintf(fp,
          "{\"total_ms\": %.1f, \"status\": \"%s\", \"recipe\": \"%s\", "
          "\"run\": \"%s\", \"ranks\": %d, \"tasks\": %zu, \"files\": %zu, "
          "\"stage_ms\": %.1f, \"stage_gb\": %.3f, \"read_gb\": %.3f, "
          "\"write_gb\": %.3f, \"wait_s\": %.2f, \"read_s\": %.2f, "
          "\"compute_s\": %.2f, \"gen_s\": %.2f, \"write_s\": %.2f, "
          "\"errors\": %d, \"runtime_scale\": %g, \"levels\": [",
          tot.wall_s * 1e3, tot.errors == 0 ? "ok" : "failed",
          dag.recipe.c_str(), dag.run.c_str(), tot.ranks, dag.tasks.size(),
          dag.files.size(), tot.stage_s * 1e3, tot.stage_bytes / 1e9,
          tot.sum[kInBytes] / 1e9, tot.sum[kOutBytes] / 1e9, tot.sum[kWait],
          tot.sum[kRead], tot.sum[kCompute], tot.sum[kGen], tot.sum[kWrite],
          tot.errors, o.runtime_scale);
  bool first = true;
  for (const auto &[l, se] : lv) {
    fprintf(fp, "%s{\"level\": %d, \"tasks\": %d, \"wall_ms\": %.1f}",
            first ? "" : ", ", l, lv_n[l], (se.second - se.first) * 1e3);
    first = false;
  }
  fprintf(fp, "]}\n");
  fclose(fp);
}

/**
 * Rank 0: aggregate, write the outputs and print the summary line.
 * @param o Options
 * @param dag Workflow
 * @param all Gathered records
 * @param tot Totals (sum is filled here)
 */
void Report(const Options &o, const WfDag &dag, const std::vector<double> &all,
            Totals *tot) {
  for (size_t i = 0; i + kNumFields <= all.size(); i += kNumFields) {
    for (int f = 0; f < kNumFields; ++f) tot->sum[f] += all[i + f];
  }
  tot->errors += static_cast<int>(tot->sum[kErrors]);
  if (all.size() / kNumFields != dag.tasks.size()) ++tot->errors;
  mkdir(o.out.c_str(), 0755);
  WriteCsv(o, dag, all);
  WriteMakespan(o, dag, all, *tot);
  printf(
      "wfrun ranks=%d tasks=%zu files=%zu stage_s=%.2f wall_s=%.2f "
      "read_gb=%.2f write_gb=%.2f wait_s=%.1f read_s=%.1f compute_s=%.1f "
      "gen_s=%.1f write_s=%.1f errors=%d rc=%d\n",
      tot->ranks, dag.tasks.size(), dag.files.size(), tot->stage_s, tot->wall_s,
      tot->sum[kInBytes] / 1e9, tot->sum[kOutBytes] / 1e9, tot->sum[kWait],
      tot->sum[kRead], tot->sum[kCompute], tot->sum[kGen], tot->sum[kWrite],
      tot->errors, tot->errors == 0 ? 0 : 2);
  fflush(stdout);
}

/**
 * Staging, workflow and reporting of one MPI rank.
 * @param o Options
 * @param dag Workflow
 * @return Process exit code
 */
int RunMpi(const Options &o, const WfDag &dag) {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int node = MyNode(o, dag);
  CheckCoverage(dag, node, rank, size);
  MPI_Comm local;
  MPI_Comm_split(MPI_COMM_WORLD, node >= 0 ? node : MPI_UNDEFINED, rank,
                 &local);
  int lrank = 0, lsize = 1;
  if (node >= 0) {
    MPI_Comm_rank(local, &lrank);
    MPI_Comm_size(local, &lsize);
    if (PcApiInit() != 0) {
      fprintf(stderr, "wfrun: rank %d: CLIO client init failed\n", rank);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  Ctx c{&o, &dag, rank, 0.0};
  c.use_payload = !o.payload.empty() && c.payload.Open(o.payload);
  if (!o.payload.empty() && !c.use_payload) {
    fprintf(stderr, "wfrun: rank %d: payload %s unusable\n", rank,
            o.payload.c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  double staged = 0;
  const double s0 = Now();
  int bad =
      node >= 0
          ? StageInputs(&c, dtwf::RankStaged(dag, node, lrank, lsize), &staged)
          : 0;
  SleepBarrier();
  c.t0 = Now();
  if (node >= 0) {
    for (int t : dtwf::RankTasks(dag, node, lrank, lsize)) RunTask(&c, t);
    bad += PcApiDrain(1);
  }
  SleepBarrier();
  Totals tot;
  tot.ranks = size;
  tot.stage_s = c.t0 - s0;
  tot.wall_s = Now() - c.t0;
  MPI_Reduce(&staged, &tot.stage_bytes, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&bad, &tot.errors, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  const std::vector<double> all = GatherRecords(c.recs, size);
  if (rank == 0) Report(o, dag, all, &tot);
  int rc = rank == 0 ? (tot.errors == 0 ? 0 : 2) : 0;
  MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (node >= 0) MPI_Comm_free(&local);
  return rc;
}

}  // namespace

/**
 * Entry point: --dry-run reports without MPI or a runtime; otherwise every
 * rank executes its share of the workflow.
 * @param argc Argument count
 * @param argv See Usage
 * @return 0 on success, 1 on bad usage/placement, 2 on I/O failures
 */
int main(int argc, char **argv) {
  Options o;
  if (!ParseOptions(argc, argv, &o)) {
    Usage(argv[0]);
    return 1;
  }
  WfDag dag;
  if (o.dry_run) {
    if (!PrepareDag(o, &dag)) return 1;
    dtwf::PrintDryRun(dag, o.runtime_scale, o.ppn);
    return 0;
  }
  MPI_Init(&argc, &argv);
  if (!PrepareDag(o, &dag)) MPI_Abort(MPI_COMM_WORLD, 1);
  const int rc = RunMpi(o, dag);
  MPI_Finalize();
  return rc;
}
