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

// Boost.Context fiber shim for the ctp-coroutine `boost-fibers` backend
// (issue #756). Exposes a 4-function C ABI over boost::context::callcc so
// Rust can create/resume/yield/destroy stackful fibers.
//
// Design notes:
//  - NO thread_local anywhere (project rule; header thread_locals also
//    duplicate per-DLL on Windows). The yield path takes the fiber handle
//    explicitly instead of consulting hidden per-thread state: each fiber
//    stores a pointer to its caller continuation, refreshed on every hop.
//  - callcc() enters the fiber immediately; the entry lambda therefore
//    yields straight back once so that "create" does not run user code —
//    the first ctp_fiber_resume() does.

#include <boost/context/continuation.hpp>
#include <boost/context/fixedsize_stack.hpp>

#include <cstddef>
#include <memory>
#include <utility>

namespace ctx = boost::context;

extern "C" {

/** Entry signature: (user_arg, fiber_handle). Rust supplies a trampoline. */
typedef void (*ctp_fiber_entry_fn)(void *arg, void *fiber);

struct CtpFiber {
  ctx::continuation callee;   // resumes INTO the fiber
  ctx::continuation *caller;  // resumes OUT of the fiber; valid while inside
  ctp_fiber_entry_fn entry;
  void *arg;
  bool done;
};

/**
 * Create a suspended fiber. User code does NOT run until the first
 * ctp_fiber_resume(). Returns an opaque handle (never null on success).
 * @param stack_size Fiber stack size in bytes (0 = Boost default).
 * @param entry Entry trampoline (called once, on the fiber stack).
 * @param arg Opaque argument forwarded to entry.
 */
void *ctp_fiber_create(size_t stack_size, ctp_fiber_entry_fn entry,
                       void *arg) {
  auto *f = new CtpFiber{};
  f->entry = entry;
  f->arg = arg;
  f->done = false;
  auto body = [f](ctx::continuation &&c) -> ctx::continuation {
    // Return to the creator immediately: creation must not run user code.
    c = c.resume();
    f->caller = &c;
    f->entry(f->arg, f);
    f->done = true;
    return std::move(c);
  };
  if (stack_size == 0) {
    f->callee = ctx::callcc(body);
  } else {
    f->callee = ctx::callcc(std::allocator_arg,
                            ctx::fixedsize_stack(stack_size), body);
  }
  return f;
}

/**
 * Resume the fiber until its next yield or completion.
 * @return 1 when the fiber has completed, 0 when it yielded.
 */
int ctp_fiber_resume(void *fiber) {
  auto *f = static_cast<CtpFiber *>(fiber);
  if (f->done) return 1;
  f->callee = f->callee.resume();
  return f->done ? 1 : 0;
}

/**
 * Yield from INSIDE the fiber back to the resumer. Must only be called on
 * the fiber's own stack (i.e. from within the entry callback), with the
 * handle that entry received.
 */
void ctp_fiber_yield(void *fiber) {
  auto *f = static_cast<CtpFiber *>(fiber);
  *f->caller = f->caller->resume();
}

/**
 * Destroy a fiber handle. The fiber must have completed (resume returned 1)
 * — destroying a suspended fiber unwinds its stack via the continuation
 * destructor, which Boost handles, but user (Rust) frames on that stack are
 * NOT unwound safely across the FFI, so the Rust wrapper enforces
 * run-to-completion before destroy.
 */
void ctp_fiber_destroy(void *fiber) {
  delete static_cast<CtpFiber *>(fiber);
}

}  // extern "C"
