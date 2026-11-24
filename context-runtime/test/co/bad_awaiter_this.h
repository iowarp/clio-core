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
 * Rule R9: an awaiter may not carry the address of frame-private storage.
 * The awaiter is pushed into the frame at a park and popped on the relaunch
 * -- the generated switch jumps to the case label, past the assignment that
 * built it -- so a pointer it holds is dereferenced in a launch other than
 * the one that formed it. `this` and the address of a parameter or local are
 * private memory, at whatever address the next launch happens to give them.
 *
 * This is how the ported gpu_vector failed on Aurora: `FlushWait{this}`
 * restored a pointer to the previous launch's by-value DeviceVector, read
 * some other block's flush state through it, and re-fired a task slot the
 * host was still writing back. CUDA had handed back the same address every
 * launch, so nothing there ever noticed.
 */
#ifndef CLIO_RUNTIME_TEST_CO_BAD_AWAITER_THIS_H_
#define CLIO_RUNTIME_TEST_CO_BAD_AWAITER_THIS_H_

#include "coro_types.h"

namespace clio::co::demo {

struct Owner {
  PageCache c;
  u64 page;

  /** Waits through a pointer back to the object -- private memory. */
  struct BadWait {
    Owner *o;
    CLIO_CO_FUN bool Ready() const { return o->c.resident[o->page] != 0; }
    CLIO_CO_FUN u64 Tag() const { return o->page; }
    CLIO_CO_FUN void Take() const {}
  };

  /** Waits through a pointer to a local -- also private memory. */
  struct BadLocalWait {
    const u64 *p;
    CLIO_CO_FUN bool Ready() const { return *p != 0; }
    CLIO_CO_FUN u64 Tag() const { return *p; }
    CLIO_CO_FUN void Take() const {}
  };

  CLIO_CO_FUN void WaitThis() {
    CO_AWAIT(BadWait{this}.Take());   // <-- must be an error: R9
  }

  CLIO_CO_FUN void WaitLocal(u64 page_in) {
    CO_AWAIT(BadLocalWait{&page_in}.Take());   // <-- must be an error: R9
  }
};

}  // namespace clio::co::demo

#endif  // CLIO_RUNTIME_TEST_CO_BAD_AWAITER_THIS_H_
