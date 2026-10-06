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
 * Data and awaiters for the transpiler's end-to-end test.
 *
 * Deliberately separate from coro_workload.h: this file contains no CO_AWAIT,
 * so clio-coroc never rewrites it and both the source build and the transpiled
 * build share exactly these definitions.
 */
#ifndef CLIO_RUNTIME_TEST_CO_CORO_TYPES_H_
#define CLIO_RUNTIME_TEST_CO_CORO_TYPES_H_

#include <clio_runtime/co/coro.h>

namespace clio::co::demo {

/** A toy page cache. `resident`/`flushed` are what the host servicer flips. */
struct PageCache {
  float *data;
  unsigned *resident;
  unsigned *flushed;
  u64 npages;
  u64 per;

  CLIO_CO_FUN float *Find(u64 page) const {
    return resident[page] ? data + page * per : nullptr;
  }
};

/** The value of `CO_AWAIT(HoldPage(...))`. Trivially copyable, per rule R6. */
struct Held {
  float *p;
};

/** Leaf awaiter: the page is not resident and the host must fetch it. */
struct FetchAwaiter {
  PageCache c;
  u64 page;
  CLIO_CO_FUN bool Ready() const { return c.resident[page] != 0; }
  CLIO_CO_FUN u64 Tag() const { return page; }
  CLIO_CO_FUN void Take() const {}
};

/** Leaf awaiter: the page is dirty and the host must write it back. Tagged in
 *  a different space so the servicer can tell the two kinds of wait apart. */
struct FlushAwaiter {
  PageCache c;
  u64 page;
  CLIO_CO_FUN bool Ready() const { return c.flushed[page] != 0; }
  CLIO_CO_FUN u64 Tag() const { return page | (1ull << 32); }
  CLIO_CO_FUN void Take() const {}
};

static_assert(std::is_trivially_copyable_v<PageCache>);
static_assert(std::is_trivially_copyable_v<Held>);
static_assert(std::is_trivially_copyable_v<FetchAwaiter>);
static_assert(std::is_trivially_copyable_v<FlushAwaiter>);
/** Saved across a suspend like any other parameter, so R6 applies to it too. */
static_assert(std::is_trivially_copyable_v<Item>);

}  // namespace clio::co::demo

#endif  // CLIO_RUNTIME_TEST_CO_CORO_TYPES_H_
