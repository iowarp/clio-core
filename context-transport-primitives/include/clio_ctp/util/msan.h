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


#ifndef CTP_UTIL_MSAN_H
#define CTP_UTIL_MSAN_H

/**
 * MemorySanitizer boundary helpers.
 *
 * MSan only knows a byte is initialized if instrumented code wrote it. Every
 * dependency we do not compile ourselves -- the system libstdc++, Poco,
 * yaml-cpp, ZMQ, Catch2 -- writes bytes MSan never sees, so anything read back
 * out of those writes looks uninitialized and is reported. Those reports are
 * false positives: the bytes are real, MSan just has no record of them.
 *
 * The fix is to unpoison at the boundary -- immediately after the
 * uninstrumented library hands data back, and never wider than the bytes it
 * actually wrote. Unpoisoning a whole struct or a whole buffer "to be safe"
 * hides genuine uninitialized reads in the same memory, so each call site
 * should name the library that wrote the bytes.
 *
 * These are no-ops in every build that is not MSan-instrumented.
 */

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#define CTP_MSAN_ENABLED 1
#endif
#endif
#ifndef CTP_MSAN_ENABLED
#define CTP_MSAN_ENABLED 0
#endif

#if CTP_MSAN_ENABLED
#include <sanitizer/msan_interface.h>
/** Mark [ptr, ptr+size) as initialized. */
#define CTP_MSAN_UNPOISON(ptr, size) __msan_unpoison((ptr), (size))
#else
#define CTP_MSAN_UNPOISON(ptr, size) ((void)0)
#endif

/**
 * Mark a whole object as initialized.
 *
 * For objects an uninstrumented library both constructs and fills in place --
 * a std::ifstream whose state libstdc++.so owns, a Poco::URI -- where there is
 * no narrower set of bytes to name.
 */
#define CTP_MSAN_UNPOISON_OBJ(obj) CTP_MSAN_UNPOISON(&(obj), sizeof(obj))

/**
 * Mark a std::string as initialized: the object itself AND its characters.
 *
 * Both halves are needed, and the object has to come first -- a string an
 * uninstrumented library returned by value has a poisoned length and pointer,
 * so reading .data()/.size() is itself a use of uninitialized memory, and the
 * destructor reading that pointer is reported long after the characters have
 * been dealt with.
 */
#define CTP_MSAN_UNPOISON_STRING(s)              \
  do {                                           \
    CTP_MSAN_UNPOISON_OBJ(s);                    \
    CTP_MSAN_UNPOISON((s).data(), (s).size());   \
  } while (0)

/**
 * Mark a std::filesystem::path as initialized.
 *
 * libstdc++.so owns every path operation that touches the filesystem
 * (read_symlink, canonical, ...), so a path it returns is poisoned in full:
 * the pathname string, and the component list whose destructor walks it.
 */
#define CTP_MSAN_UNPOISON_PATH(p)                \
  do {                                           \
    CTP_MSAN_UNPOISON_OBJ(p);                    \
    CTP_MSAN_UNPOISON_STRING((p).native());      \
  } while (0)

#ifdef __cplusplus
#include <string>

namespace ctp {

/**
 * Mark every scalar in a fully-loaded yaml-cpp tree as initialized.
 *
 * libyaml-cpp.so is not instrumented, but it assembles scalar strings by
 * calling back into our instrumented std::string code, so the characters
 * arrive carrying the scanner's missing shadow rather than none at all: every
 * later comparison against a key or a value is reported. Once Load()/LoadFile()
 * has returned the tree is structurally valid, so walking it and clearing the
 * scalars is exactly as wide as the problem. Templated on the node type so
 * this header does not have to include yaml-cpp.
 */
template <typename YamlNode>
inline void MsanUnpoisonYaml(const YamlNode &node) {
#if CTP_MSAN_ENABLED
  if (!node.IsDefined() || node.IsNull()) {
    return;
  }
  if (node.IsScalar()) {
    const std::string &s = node.Scalar();
    if (!s.empty()) {
      CTP_MSAN_UNPOISON(s.data(), s.size());
    }
  } else if (node.IsSequence()) {
    for (auto it = node.begin(); it != node.end(); ++it) {
      MsanUnpoisonYaml(*it);
    }
  } else if (node.IsMap()) {
    for (auto it = node.begin(); it != node.end(); ++it) {
      MsanUnpoisonYaml(it->first);
      MsanUnpoisonYaml(it->second);
    }
  }
#else
  (void)node;
#endif
}

/**
 * Scope guard that turns off MSan's checks on interceptor arguments.
 *
 * Distinct from unpoisoning, and much narrower than it sounds: it suppresses
 * only the checks MSan performs when an INTERCEPTED libc function (memcmp,
 * read, sigaction, ...) is handed a buffer, and only on this thread, for the
 * lifetime of the guard. Instrumented code inside the scope is still checked
 * normally.
 *
 * Reach for it when an uninstrumented library runs on stack we already
 * poisoned and hands that stack to libc -- a case no amount of unpoisoning on
 * our side can reach, because the memory belongs to a frame we never see. Keep
 * the scope down to the single call that needs it.
 */
class MsanInterceptorCheckGuard {
 public:
  MsanInterceptorCheckGuard() {
#if CTP_MSAN_ENABLED
    __msan_scoped_disable_interceptor_checks();
#endif
  }
  ~MsanInterceptorCheckGuard() {
#if CTP_MSAN_ENABLED
    __msan_scoped_enable_interceptor_checks();
#endif
  }
  MsanInterceptorCheckGuard(const MsanInterceptorCheckGuard &) = delete;
  MsanInterceptorCheckGuard &operator=(const MsanInterceptorCheckGuard &) =
      delete;
};

}  // namespace ctp
#endif  // __cplusplus

#endif  // CTP_UTIL_MSAN_H
