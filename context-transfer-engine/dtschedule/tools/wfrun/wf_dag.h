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
 * @file wf_dag.h
 * The workflow dtschedule_wfrun executes: placement.json as written by
 * tools/wfcommons/dt_translator.py --api (tasks with node, inputs, outputs,
 * instance runtime; files with producer, consumers, size), checked, sized,
 * topologically ordered and split over the MPI ranks of each node.
 */

#ifndef DTSCHEDULE_WFRUN_WF_DAG_H_
#define DTSCHEDULE_WFRUN_WF_DAG_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "wf_data.h"

namespace dtwf {

/** One file of the workflow (a CTE tag of 1 MiB chunk blobs). */
struct WfFile {
  std::string path;            ///< Placement key, e.g. "data/x.fits"
  std::string tag;             ///< CTE tag: "<run>__<basename>"
  uint64_t size = 0;           ///< Bytes (after --size-scale / --max-file-mb)
  int producer = -1;           ///< Producing task index (-1: staged input)
  int stage_node = -1;         ///< Node that stages it (staged inputs only)
  std::vector<int> consumers;  ///< Consuming task indices
  DataClass cls = DataClass::kMixedBinary;  ///< Content class
};

/** One task of the workflow. */
struct WfTask {
  std::string name;          ///< Task id
  std::string type;          ///< Task type (e.g. mProject)
  int node = -1;             ///< Index into WfDag::nodes
  int level = 0;             ///< DAG level from the translator
  double runtime = 0.0;      ///< Instance runtime (s), before scaling
  std::vector<int> inputs;   ///< File indices
  std::vector<int> outputs;  ///< File indices
};

/** The whole workflow. */
struct WfDag {
  std::string recipe;              ///< Recipe name
  std::string run;                 ///< Tag prefix
  std::vector<std::string> nodes;  ///< Node names from the placement
  std::vector<WfTask> tasks;       ///< Tasks
  std::vector<WfFile> files;       ///< Files
  std::vector<int> order;          ///< Task indices in topological order
};

/**
 * Load and check a placement.json; files are named "<run>__<basename>".
 * @param path placement.json
 * @param run Tag prefix ("" = basename of the placement's out_dir)
 * @param dag Output
 * @param err Output: reason on failure
 * @return true on success
 */
bool LoadDag(const std::string &path, const std::string &run, WfDag *dag,
             std::string *err);

/**
 * Scale every file size, cap it, and keep it at least one byte.
 * @param dag Workflow
 * @param size_scale Factor applied to every size
 * @param max_file_mb Cap in MiB (0 = none)
 */
void ScaleSizes(WfDag *dag, double size_scale, double max_file_mb);

/**
 * Order the tasks topologically, preferring lower level then name.
 * @param dag Workflow (order is filled)
 * @param err Output: reason on failure (a cycle)
 * @return true on success
 */
bool TopoSort(WfDag *dag, std::string *err);

/**
 * Index of the placement node a host name stands for: exact match, else
 * same short name, else the longest node name that starts with the short
 * host name followed by '-' or '.' (ares-comp-08 -> ares-comp-08-40g).
 * @param dag Workflow
 * @param host Host name
 * @return Node index, or -1
 */
int MatchNode(const WfDag &dag, const std::string &host);

/**
 * Tasks of one rank: its node's tasks in topological order, dealt
 * round-robin over the node's ranks.
 * @param dag Workflow
 * @param node Node index
 * @param local_rank Rank among the node's ranks
 * @param local_size Ranks on the node
 * @return Task indices in execution order
 */
std::vector<int> RankTasks(const WfDag &dag, int node, int local_rank,
                           int local_size);

/**
 * Staged input files of one rank: those its node stages, dealt
 * round-robin over the node's ranks.
 * @param dag Workflow
 * @param node Node index
 * @param local_rank Rank among the node's ranks
 * @param local_size Ranks on the node
 * @return File indices
 */
std::vector<int> RankStaged(const WfDag &dag, int node, int local_rank,
                            int local_size);

/**
 * Print the --dry-run report: per-node task counts, bytes per data class,
 * cross-node edges, critical path, per-rank load.
 * @param dag Workflow
 * @param runtime_scale Compute seconds per instance second
 * @param ppn Ranks per node assumed for the per-rank split
 */
void PrintDryRun(const WfDag &dag, double runtime_scale, int ppn);

}  // namespace dtwf

#endif  // DTSCHEDULE_WFRUN_WF_DAG_H_
