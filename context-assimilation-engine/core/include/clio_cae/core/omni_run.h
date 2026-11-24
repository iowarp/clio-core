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
 * OMNI `run:` steps: transform a transfer's source with a user script before
 * it is assimilated.
 *
 *   transfers:
 *     - src: file::/data/A46_xx.csv
 *       dst: iowarp::a46_parquet
 *       format: binary
 *       run: ./to_parquet.sh          # called as: to_parquet.sh <src> <run_out>
 *       run_out: /scratch/a46.parquet  # the file the script must write
 *
 * The script runs in the clio_cae client, before any transfer is submitted, as
 * the user who runs clio_cae. It never runs inside the runtime daemon. Its first
 * argument is the local path for file:: and hdf5:: sources and the source URL
 * otherwise. When it exits 0 and run_out exists, the transfer ingests run_out
 * instead of the original source. Any failure stops the whole ingest.
 */

#ifndef CLIO_CAE_CORE_OMNI_RUN_H_
#define CLIO_CAE_CORE_OMNI_RUN_H_

#include <clio_cae/core/factory/assimilation_ctx.h>

#include <cstddef>
#include <string>
#include <vector>

namespace clio::cae::core {

/** One `run:` step, as parsed from an OMNI transfer. */
struct RunStep {
  size_t transfer = 0;  // 0-based index into the transfers / contexts
  std::string script;   // script path, env-expanded, absolute
  std::string out;      // run_out path, env-expanded, absolute
};

/**
 * The argument a run script receives for a source URL: the path for file::
 * and hdf5:: URLs, the URL unchanged for every other scheme.
 */
std::string RunSourceArg(const std::string& src);

/**
 * The source URL that ingests a run step's output: hdf5:: for formats that
 * start with "hdf5" (hdf5, hdf5_vol), file:: otherwise.
 */
std::string RunOutputSrc(const std::string& format, const std::string& out);

/**
 * Run `script` with `args`, inheriting stdio and the environment, and wait.
 *
 * @return the exit status (0 on success); -1 when the script could not be
 *         started or died on a signal, with the reason in *err
 */
int RunScript(const std::string& script, const std::vector<std::string>& args,
              std::string* err);

/**
 * Run every step in order and point its transfer at the step's output.
 *
 * For each step: run `script <RunSourceArg(src)> <out>`; require exit 0 and a
 * regular file at `out`; then set the transfer's src to RunOutputSrc(...) and
 * rewrite later depends_on entries that named the old src.
 *
 * @return empty on success; otherwise a message naming the failed transfer by
 *         its 1-based position. Steps after a failure are not run.
 */
std::string ExecuteRunSteps(const std::vector<RunStep>& steps,
                            std::vector<AssimilationCtx>* contexts);

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_OMNI_RUN_H_
