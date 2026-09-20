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
 * NEGATIVE TEST. clio-coroc must reject this file.
 *
 * Rule R1: a call to a suspending function that is not wrapped in CO_AWAIT has
 * no resume point, so its park would be silently lost -- the caller would carry
 * on as though the callee had completed. This is the check that does not exist
 * in a hand-decorated world, where the equivalent mistake is a missed hoist
 * that nothing diagnoses and that surfaces as wrong data far from its cause.
 */
#ifndef CLIO_RUNTIME_TEST_CO_BAD_MISSING_AWAIT_H_
#define CLIO_RUNTIME_TEST_CO_BAD_MISSING_AWAIT_H_

#include "coro_types.h"

namespace clio::co::demo {

CLIO_CO_FUN void BadFetch(PageCache c, u64 page) {
  CO_AWAIT(FetchAwaiter{c, page}.Take());
}

CLIO_CO_FUN void BadCaller(PageCache c, u64 page) {
  BadFetch(c, page);            // <-- must be an error: R1
  CO_AWAIT(BadFetch(c, page));  // correct, for contrast
}

}  // namespace clio::co::demo

#endif  // CLIO_RUNTIME_TEST_CO_BAD_MISSING_AWAIT_H_
