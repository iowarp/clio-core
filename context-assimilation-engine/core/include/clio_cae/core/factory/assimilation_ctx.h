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

#ifndef CLIO_CAE_CORE_ASSIMILATION_CTX_H_
#define CLIO_CAE_CORE_ASSIMILATION_CTX_H_

#include <string>
#include <vector>
#include "clio_ctp/data_structures/serialization/global_serialize.h"

namespace clio::cae::core {

/**
 * AssimilationCtx - Context for data assimilation operations
 * Contains metadata about the source, destination, format, and range
 */
struct AssimilationCtx {
  std::string src;         // Source URL (e.g., file::/path/to/file, string::blob_name)
  std::string dst;         // Destination URL (e.g., iowarp::tag_name)
  std::string format;      // Data format (e.g., binary, hdf5)
  std::string depends_on;  // Dependency identifier (empty if none)
  size_t range_off;        // Byte offset in source file
  size_t range_size;       // Number of bytes to read
  std::string src_token;   // Authentication token for source (e.g., Globus access token)
  std::string dst_token;   // Authentication token for destination

  // Inline data payload for src="string::..." sources. The bytes here are
  // stored verbatim as a single blob whose name is the path part of `src`
  // (everything after "string::"). Unused for file/hdf5/globus sources.
  std::string src_data;

  // Dataset filtering (for HDF5 and other hierarchical formats)
  std::vector<std::string> include_patterns;  // Glob patterns for datasets to include
  std::vector<std::string> exclude_patterns;  // Glob patterns for datasets to exclude

  // Default constructor
  AssimilationCtx()
      : range_off(0), range_size(0) {}

  // Full constructor
  AssimilationCtx(const std::string& src_url,
                  const std::string& dst_url,
                  const std::string& data_format,
                  const std::string& dependency = "",
                  size_t offset = 0,
                  size_t size = 0,
                  const std::string& source_token = "",
                  const std::string& dest_token = "")
      : src(src_url),
        dst(dst_url),
        format(data_format),
        depends_on(dependency),
        range_off(offset),
        range_size(size),
        src_token(source_token),
        dst_token(dest_token) {}

  // Serialization support
  template<class Archive>
  void serialize(Archive& ar) {
    ar(src, dst, format, depends_on, range_off, range_size, src_token, dst_token,
       include_patterns, exclude_patterns, src_data);
  }
};

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_ASSIMILATION_CTX_H_
