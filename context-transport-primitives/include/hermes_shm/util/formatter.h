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

#ifndef HSHM_ERROR_SERIALIZER_H
#define HSHM_ERROR_SERIALIZER_H

#include <cstring>
#include <list>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "hermes_shm/types/argpack.h"

// MSan: unpoison strings produced by uninstrumented libstdc++ stringstream
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#include <sanitizer/msan_interface.h>
#define HSHM_MSAN_UNPOISON_STRING(s) \
  __msan_unpoison((s).data(), (s).size())
#else
#define HSHM_MSAN_UNPOISON_STRING(s) ((void)0)
#endif
#else
#define HSHM_MSAN_UNPOISON_STRING(s) ((void)0)
#endif

namespace hshm {

class Formatter {
 public:
  template <typename... Args>
  static std::string format(std::string fmt, Args &&...args) {
    std::stringstream ss;
    std::vector<std::pair<size_t, size_t>> offsets = tokenize(fmt);
    size_t packlen = make_argpack(std::forward<Args>(args)...).Size();
    if (offsets.size() != packlen + 1) {
      HSHM_MSAN_UNPOISON_STRING(fmt);
      return fmt;
    }
    auto lambda = [&ss, &fmt, &offsets](auto i, auto &&arg) {
      if (i.Get() >= offsets.size()) {
        return;
      }
      auto &sub = offsets[i.Get()];
      ss << fmt.substr(sub.first, sub.second);
      ss << arg;
    };
    ForwardIterateArgpack::Apply(make_argpack(std::forward<Args>(args)...),
                                 lambda);
    if (offsets.back().second > 0) {
      auto &sub = offsets.back();
      ss << fmt.substr(sub.first, sub.second);
    }
    std::string result = ss.str();
    HSHM_MSAN_UNPOISON_STRING(result);
    return result;
  }

  static std::vector<std::pair<size_t, size_t>> tokenize(
      const std::string &fmt) {
    std::vector<std::pair<size_t, size_t>> offsets;
    size_t i = 0;
    offsets.emplace_back(std::pair<size_t, size_t>(0, fmt.size()));
    while (i < fmt.size()) {
      if (fmt[i] == '{') {
        // Set the size of the prior substring
        // E.g., "hello{}there".
        // prior.first is 0
        // i = 5 for the '{'
        // The substring's length is i - 0 = 5.
        auto &prior = offsets.back();
        prior.second = i - prior.first;

        // The token after the closing '}'
        // i = 5 for the '{'
        // i = 7 for the 't'
        // The total size is 12
        // The remaining size is: 12 - 7 = 5 (length of "there").
        i += 2;
        offsets.emplace_back(std::pair<size_t, size_t>(i, fmt.size() - i));
        continue;
      }
      ++i;
    }
    return offsets;
  }
};

}  // namespace hshm

#endif  // HSHM_ERROR_SERIALIZER_H
