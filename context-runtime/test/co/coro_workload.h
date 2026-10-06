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
 * THE INPUT TO clio-coroc. This is what a user writes -- and it is all of it.
 *
 * Ordinary functions. Ordinary return types. Ordinary locals. One marker at
 * each call site that may suspend. No frame declarations, no per-variable
 * hoisting, no task types, no out-pointer calling convention, no dual
 * spellings, no `#if` on the backend.
 *
 * Compiled as-is, CO_AWAIT is the identity and this is a runnable synchronous
 * program -- which is the differential oracle the test uses. Run through
 * clio-coroc, it becomes a group-scoped state machine that parks and resumes
 * across kernel launches. The two must agree bit for bit.
 *
 * The call graph is deliberately awkward, because a flat one would prove
 * nothing:
 *
 *     StreamTile   depth 0   two suspend points, one inside a loop
 *       HoldPage   depth 1   one suspend point, inside an `if`
 *         Fetch    depth 2   one leaf await
 *       Flush      depth 1   one leaf await   <- second child at the same depth
 */
#ifndef CLIO_RUNTIME_TEST_CO_CORO_WORKLOAD_H_
#define CLIO_RUNTIME_TEST_CO_CORO_WORKLOAD_H_

#include "coro_types.h"

namespace clio::co::demo {

/** Wait until the host has made `page` resident. */
CLIO_CO_FUN void Fetch(PageCache c, u64 page) {
  CO_AWAIT(FetchAwaiter{c, page}.Take());
}

/** Wait until the host has written `page` back. */
CLIO_CO_FUN void Flush(PageCache c, u64 page) {
  CO_AWAIT(FlushAwaiter{c, page}.Take());
}

/** Resolve `page`, fetching it first if it is not resident. */
CLIO_CO_FUN Held HoldPage(PageCache c, u64 page) {
  float *p = c.Find(page);
  if (p == nullptr) {
    CO_AWAIT(Fetch(c, page));
    p = c.Find(page);
  }
  return Held{p};
}

/** Double every element of every page, flushing each page as it is written. */
CLIO_CO_FUN void StreamTile(PageCache c, float *out, u64 npages, u32 lane,
                            u32 nlanes, Item it) {
  for (u64 i = 0; i < npages; ++i) {
    {
      Held h = CO_AWAIT(HoldPage(c, i));
      for (u64 k = lane; k < c.per; k += nlanes) {
        out[i * c.per + k] = h.p[k] * 2.0f + static_cast<float>(i);
      }
    }  // h dies here, before the next suspend -- rule R6 in practice
    // A barrier BETWEEN two suspend points, crossed after a resume. Legal only
    // because the whole group resumes at the same point (invariant I2).
    it.Barrier();
    CO_AWAIT(Flush(c, i));
  }
}

}  // namespace clio::co::demo

#endif  // CLIO_RUNTIME_TEST_CO_CORO_WORKLOAD_H_
