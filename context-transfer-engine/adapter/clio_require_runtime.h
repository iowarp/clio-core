/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause License. See LICENSE file.
 */

/**
 * CLIO_REQUIRE_RUNTIME — turn silent degradation into a hard failure.
 *
 * Both connectors degrade to pure pass-through when the CLIO runtime is
 * unreachable, and that is CORRECT for production: the native file is
 * authoritative, the tier is a performance layer, and an application should not
 * lose its I/O because a cache is missing. The plan states it as a rule -- at
 * capacity, degrade, never return an error the application would not have seen
 * without CLIO.
 *
 * It is also the single most expensive failure mode in this tree for anyone
 * measuring or testing, because a degraded run does not look degraded. It looks
 * like a clean result: every staged-byte counter reads zero, every ratio is
 * 0.00, and nothing in the output distinguishes "admission staged nothing"
 * (a real finding) from "the connector never reached a runtime" (a broken
 * measurement). The causes are all quiet ones -- a CLIO_SERVER_CONF pointing at
 * a different config than the runtime started with, a plugin path that did not
 * resolve, a runtime that died between combinations -- and each has cost real
 * diagnosis time. admission_measure.py carries a hand-rolled guard for exactly
 * this, added after it happened there; hdf5_compat_suite.py's own notes warn
 * about the CLIO_SERVER_CONF case in the same terms.
 *
 * So the switch is opt-in and one-way: default behaviour is unchanged, and
 * setting CLIO_REQUIRE_RUNTIME makes "I intended to use CLIO and silently did
 * not" a failure the caller sees. Set it in tests, benchmarks and CI. It says
 * nothing about whether caching is ENABLED -- CLIO_VOL_CACHE=0 is a deliberate
 * choice and stays honoured; this is only about intending to reach a runtime
 * and not reaching one.
 */

#ifndef CLIO_ADAPTER_REQUIRE_RUNTIME_H_
#define CLIO_ADAPTER_REQUIRE_RUNTIME_H_

#include <cstdlib>
#include <cstring>

namespace clio {
namespace adapter {

/** True when the caller has declared that a missing runtime is an error. */
inline bool RequireRuntime() {
  static const bool required = []() -> bool {
    const char *v = std::getenv("CLIO_REQUIRE_RUNTIME");
    if (v == nullptr || *v == '\0') return false;
    return !(std::strcmp(v, "0") == 0 || std::strcmp(v, "off") == 0 ||
             std::strcmp(v, "false") == 0 || std::strcmp(v, "no") == 0);
  }();
  return required;
}

/** The message every refusal shares, so the cause is greppable and uniform. */
inline const char *RequireRuntimeMessage() {
  return "CLIO_REQUIRE_RUNTIME is set and the CLIO tier is not usable for this "
         "file, so the operation is refused rather than silently degraded to "
         "native-only. Usual causes: no runtime running, or CLIO_SERVER_CONF "
         "naming a different config than the runtime was started with. Unset "
         "CLIO_REQUIRE_RUNTIME to allow pass-through";
}

}  // namespace adapter
}  // namespace clio

#endif  // CLIO_ADAPTER_REQUIRE_RUNTIME_H_
