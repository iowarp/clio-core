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

#ifndef CLIO_CEE_API_CONTEXT_INTERFACE_H_
#define CLIO_CEE_API_CONTEXT_INTERFACE_H_

#include <string>
#include <vector>
#include <clio_cae/core/factory/assimilation_ctx.h>

namespace iowarp {

/**
 * ContextInterface - High-level API for context exploration and management
 *
 * Provides a unified interface for bundling, querying, retrieving, and destroying
 * contexts in the IOWarp system. Integrates with the Context Assimilation Engine (CAE)
 * and Context Transfer Engine (CTE).
 */
class ContextInterface {
public:
  /**
   * Default constructor
   */
  ContextInterface();

  /**
   * Destructor
   */
  ~ContextInterface();

  /**
   * Bundle a group of related objects together and assimilate them
   *
   * This method takes a vector of AssimilationCtx objects and calls the CAE
   * ParseOmni function to schedule assimilation tasks for each context.
   *
   * @param bundle Vector of AssimilationCtx objects to assimilate
   * @return 0 on success, non-zero error code on failure
   */
  int ContextBundle(const std::vector<clio::cae::core::AssimilationCtx> &bundle);

  /**
   * Retrieve the identities of objects matching tag and blob patterns.
   *
   * Three modes, in priority order:
   *   1. Temporal (time_begin or time_end non-zero): TemporalSearch —
   *      returns blobs whose last_modified falls within
   *      [time_begin, time_end] (epoch nanoseconds; 0 = no bound),
   *      sorted ascending by timestamp, capped at max_results.
   *   2. Semantic (prompt non-empty): BM25 keyword search via
   *      SemanticSearch — tag_re/blob_re filter the candidate set and
   *      the prompt is scored against blob payloads; the top max_results
   *      blob names (ordered by descending score) are returned.
   *   3. Regex (default): listing via BlobQuery — tag/blob names
   *      matching tag_re AND blob_re are returned.
   *
   * max_results = 0 means "unlimited" for the regex/temporal paths and
   * falls back to a default cap of 10 for the BM25 path.
   *
   * @param tag_re      Tag regex pattern to match
   * @param blob_re     Blob regex pattern to match
   * @param max_results Result cap (0 = unlimited for regex/temporal; 10 for BM25)
   * @param prompt      Optional BM25 query text. Empty = not a semantic query.
   * @param time_begin  Temporal lower bound, epoch nanoseconds (0 = no bound)
   * @param time_end    Temporal upper bound, epoch nanoseconds (0 = no bound)
   * @return Vector of matching blob names
   */
  std::vector<std::string> ContextQuery(const std::string &tag_re,
                                         const std::string &blob_re,
                                         unsigned int max_results = 0,
                                         const std::string &prompt = "",
                                         uint64_t time_begin = 0,
                                         uint64_t time_end = 0);

  /**
   * Retrieve the identities and data of objects matching patterns
   *
   * This method queries for blobs matching the specified patterns and retrieves their
   * data into a packed binary buffer. Blobs are retrieved in batches for efficiency.
   *
   * The retrieved data is packed sequentially into a buffer and returned as a single
   * packed string. Buffer is automatically allocated and freed.
   *
   * @param tag_re Tag regex pattern to match
   * @param blob_re Blob regex pattern to match
   * @param max_results Maximum number of blobs to retrieve (0 = unlimited, default: 1024)
   * @param max_context_size Maximum total context size in bytes (default: 256MB)
   * @param batch_size Number of concurrent AsyncGetBlob operations (default: 32)
   * @return Vector containing one string with packed binary context data (empty if no data)
   */
  std::vector<std::string> ContextRetrieve(const std::string &tag_re,
                                            const std::string &blob_re,
                                            unsigned int max_results = 1024,
                                            size_t max_context_size = 256 * 1024 * 1024,
                                            unsigned int batch_size = 32);

  /**
   * Split/splice objects into a new context
   *
   * NOTE: This functionality is not yet implemented and will be added in a future version.
   *
   * @param new_ctx Name of the new context to create
   * @param tag_re Tag regex pattern to match for source objects
   * @param blob_re Blob regex pattern to match for source objects
   * @return 0 on success, non-zero error code on failure
   */
  int ContextSplice(const std::string &new_ctx,
                    const std::string &tag_re,
                    const std::string &blob_re);

  /**
   * Destroy contexts by name
   *
   * Deletes the specified contexts (tags) from the CTE system. Each context name
   * is treated as a tag name and deleted using the CTE DelTag API.
   *
   * @param context_names Vector of context names to destroy
   * @return 0 on success, non-zero error code on failure
   */
  int ContextDestroy(const std::vector<std::string> &context_names);

private:
  /**
   * Ensure the interface is initialized (lazy initialization)
   * @return true if initialization succeeded, false otherwise
   */
  bool EnsureInitialized();

  bool is_initialized_;  /**< Flag indicating whether the interface is initialized */
};

}  // namespace iowarp

#endif  // CLIO_CEE_API_CONTEXT_INTERFACE_H_
