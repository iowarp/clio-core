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
 * @file async_log.h
 * @brief One background thread that formats and writes the compressor's
 * per-chunk diagnostic log rows.
 *
 * The rows (v2 predictions, v2 measurements, chunk phases) used to be
 * formatted and flushed on the worker thread that compressed or read the
 * chunk, under a lock, inside the caller's timed work; a selector with a
 * long row (NeuroPress: 45 x 3 predictions) paid more for it than a single
 * codec. A caller now hands over a closure that holds copies of the values,
 * and this thread runs the closures in order, so rows of one file keep their
 * order. The queue is drained at process exit (std::atexit), before the CUDA
 * runtime's own teardown.
 */

#ifndef CLIO_CTE_COMPRESSOR_ASYNC_LOG_H_
#define CLIO_CTE_COMPRESSOR_ASYNC_LOG_H_

#include <functional>

namespace clio::cte::compressor {

/**
 * @brief Queue one piece of log work (format and write a row) for the
 * background log thread, started on the first call.
 * @param work the closure; it must own copies of everything it writes
 */
void PostLogWork(std::function<void()> work);

/**
 * @brief Block until every queued piece of log work has run. Also registered
 * with std::atexit when the log thread starts.
 */
void DrainLogWork();

}  // namespace clio::cte::compressor

#endif  // CLIO_CTE_COMPRESSOR_ASYNC_LOG_H_
