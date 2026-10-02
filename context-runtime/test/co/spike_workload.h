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
 * The P0 spike workload, in both of its forms.
 *
 * `src` is what the user writes: ordinary functions, ordinary return types, one
 * marker at each call site. With CO_AWAIT defined to the identity it is a
 * runnable synchronous program, which is the differential oracle spike S5 uses.
 *
 * `gen` is what clio-coroc will emit for `src`, written by hand. Every line of
 * it is mechanical -- the five edits of the design doc section 3.2 -- and the
 * point of writing it before the tool is that it tests the OUTPUT SHAPE on real
 * backends, which is the part that can be wrong in a way no amount of tool
 * engineering would fix.
 *
 * Read the two namespaces side by side. The differences are exactly:
 *   E1  a trailing `Ctx &_cy` parameter
 *   E2  `_cy` passed at each suspending call
 *   E3  a Frame, the hoisted declarations, and the dispatching switch
 *   E4  a `case N:` + replay + park guard at each CO_AWAIT
 *   E5  the closing brace and Done()
 *
 * THE CALL GRAPH, which is what makes this spike worth running:
 *
 *     StreamTile   depth 0   two suspend points
 *       HoldPage   depth 1   one suspend point, inside an `if`
 *         Fetch    depth 2   one leaf await
 *       Flush      depth 1   one leaf await
 *
 * Three levels, a suspend point nested inside a conditional, a suspend point
 * inside a loop, and two different children at the same depth -- which together
 * exercise invariant I1 (frame offsets reproduce) in the only way that matters.
 */
#ifndef CLIO_RUNTIME_TEST_CO_SPIKE_WORKLOAD_H_
#define CLIO_RUNTIME_TEST_CO_SPIKE_WORKLOAD_H_

#include <clio_runtime/co/coro.h>

namespace clio::co::spike {

/* =========================================================================
 * The workload's data and awaiters.
 * ========================================================================= */

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
static_assert(std::is_trivially_copyable_v<PageCache>);

/** The value of `CO_AWAIT(HoldPage(...))`. Trivially copyable, per rule R6. */
struct Held {
  float *p;
};
static_assert(std::is_trivially_copyable_v<Held>);

/** Leaf awaiter: the page is not resident and the host must fetch it. */
struct FetchAwaiter {
  PageCache c;
  u64 page;
  CLIO_CO_FUN bool Ready() const { return c.resident[page] != 0; }
  CLIO_CO_FUN u64 Tag() const { return page; }
  CLIO_CO_FUN void Take() const {}
};
static_assert(std::is_trivially_copyable_v<FetchAwaiter>);

/** Leaf awaiter: the page is dirty and the host must write it back. */
struct FlushAwaiter {
  PageCache c;
  u64 page;
  CLIO_CO_FUN bool Ready() const { return c.flushed[page] != 0; }
  /** Tagged in a different space from FetchAwaiter so the servicer can tell
   *  which kind of work a group is waiting for. */
  CLIO_CO_FUN u64 Tag() const { return page | (1ull << 32); }
  CLIO_CO_FUN void Take() const {}
};
static_assert(std::is_trivially_copyable_v<FlushAwaiter>);

/* =========================================================================
 * src -- what the user writes.
 * ========================================================================= */

namespace src {

/** CO_AWAIT is the identity here, which is what makes this namespace a
 *  runnable synchronous program and therefore a free oracle. */
#define CO_AWAIT(...) (__VA_ARGS__)

CLIO_CO_FUN void Fetch(PageCache c, u64 page) {
  CO_AWAIT(FetchAwaiter{c, page}.Take());
}

CLIO_CO_FUN void Flush(PageCache c, u64 page) {
  CO_AWAIT(FlushAwaiter{c, page}.Take());
}

CLIO_CO_FUN Held HoldPage(PageCache c, u64 page) {
  float *p = c.Find(page);
  if (p == nullptr) {
    CO_AWAIT(Fetch(c, page));
    p = c.Find(page);
  }
  return Held{p};
}

/** @param it  only so that the synchronous form can reach a barrier. The
 *             generated form uses Ctx::Barrier and needs no such parameter,
 *             because the Ctx it is already given carries the Item. */
CLIO_CO_FUN void StreamTile(PageCache c, float *out, u64 npages, u32 lane,
                            u32 nlanes, Item it) {
  for (u64 i = 0; i < npages; ++i) {
    {
      Held h = CO_AWAIT(HoldPage(c, i));
      for (u64 k = lane; k < c.per; k += nlanes) {
        out[i * c.per + k] = h.p[k] * 2.0f + static_cast<float>(i);
      }
    }  // h dies here, before the next suspend -- which is rule R6 in practice
    // A barrier BETWEEN two suspend points. This is the line spike S2 exists
    // to validate: it is legal only because the group resumes as a unit.
    it.Barrier();
    CO_AWAIT(Flush(c, i));
  }
}

#undef CO_AWAIT

}  // namespace src

/* =========================================================================
 * gen -- what clio-coroc emits for src.
 * ========================================================================= */

namespace gen {

/* Frame sizes: the max over this function's suspend points of the live set
 * (design doc R3). Hand-computed here; the tool derives them. */
inline constexpr u32 kFrameBytes_Fetch = MaxOf(PackBytes<FetchAwaiter, u64>());
inline constexpr u32 kFrameBytes_Flush = MaxOf(PackBytes<FlushAwaiter, u64>());
inline constexpr u32 kFrameBytes_HoldPage = MaxOf(PackBytes<PageCache, u64>());
inline constexpr u32 kFrameBytes_StreamTile =
    MaxOf(PackBytes<PageCache, float *, u64, u32, u32, u64>(),   // point 1
          PackBytes<PageCache, float *, u64, u32, u32, u64>());  // point 2

/** Deepest chain: StreamTile -> HoldPage -> Fetch. Flush is shallower, so it
 *  does not widen the stack. This is kStackBytes_<entry>. */
inline constexpr u32 kStackBytes_StreamTile =
    kFrameBytes_StreamTile +
    MaxOf(kFrameBytes_HoldPage, kFrameBytes_Flush) + kFrameBytes_Fetch;

CLIO_CO_FUN void Fetch(PageCache c, u64 page, Ctx &_cy) {
  Frame _cy_f(_cy, kFrameBytes_Fetch);
  FetchAwaiter _cy_a0{};   // hoisted: its scope contains a suspend point
  switch (_cy_f.Resume()) {
    case 0:
      _cy_a0 = FetchAwaiter{c, page};
      [[fallthrough]];   // the fresh path falls into the await
    case 1:
      if (_cy_f.Replaying()) _cy_f.Pop(_cy_a0, page);
      // The label sits BEFORE the test, so a resume re-evaluates readiness and
      // parks again if the host has not finished. That is what makes CO_AWAIT a
      // retry loop with the host in the middle of it.
      if (_cy.Any(!_cy_a0.Ready())) {
        _cy_f.Push(1, _cy_a0, page);
        _cy.SetWaitTag(_cy_a0.Tag());
        return;
      }
      _cy_a0.Take();
  }
  _cy_f.Done();
}

CLIO_CO_FUN void Flush(PageCache c, u64 page, Ctx &_cy) {
  Frame _cy_f(_cy, kFrameBytes_Flush);
  FlushAwaiter _cy_a0{};
  switch (_cy_f.Resume()) {
    case 0:
      _cy_a0 = FlushAwaiter{c, page};
      [[fallthrough]];
    case 1:
      if (_cy_f.Replaying()) _cy_f.Pop(_cy_a0, page);
      if (_cy.Any(!_cy_a0.Ready())) {
        _cy_f.Push(1, _cy_a0, page);
        _cy.SetWaitTag(_cy_a0.Tag());
        return;
      }
      _cy_a0.Take();
  }
  _cy_f.Done();
}

CLIO_CO_FUN Held HoldPage(PageCache c, u64 page, Ctx &_cy) {
  Frame _cy_f(_cy, kFrameBytes_HoldPage);
  float *p{};   // hoisted: its block contains a case label
  switch (_cy_f.Resume()) {
    case 0:
      p = c.Find(page);
      if (p == nullptr) {
        // A case label inside an `if` body: on resume we land here directly and
        // the condition is not re-evaluated, which is exactly right -- we are
        // continuing, not re-deciding.
        case 1:
          if (_cy_f.Replaying()) _cy_f.Pop(c, page);
          // `p` is NOT in the save list: it is overwritten immediately below,
          // so it is dead across this suspend. Liveness, not scope.
          Fetch(c, page, _cy);
          if (_cy.Parked()) {
            _cy_f.Push(1, c, page);
            return Held{};   // discarded by the caller; rule R2
          }
          p = c.Find(page);
      }
  }
  _cy_f.Done();
  return Held{p};
}

CLIO_CO_FUN void StreamTile(PageCache c, float *out, u64 npages, u32 lane,
                            u32 nlanes, Ctx &_cy) {
  Frame _cy_f(_cy, kFrameBytes_StreamTile);
  u64 i{};    // hoisted out of the for-init; still a register
  Held h{};   // hoisted; live across no suspend, so in no save list
  switch (_cy_f.Resume()) {
    case 0:
      for (i = 0; i < npages; ++i) {
        {
          case 1:
            if (_cy_f.Replaying()) _cy_f.Pop(c, out, npages, lane, nlanes, i);
            h = HoldPage(c, i, _cy);
            if (_cy.Parked()) {
              _cy_f.Push(1, c, out, npages, lane, nlanes, i);
              return;
            }
          // `k` stays where it was written: its scope contains no case label,
          // so the hoisting rule does not touch it. This is the whole point --
          // the inner loop is an ordinary loop over registers.
          for (u64 k = lane; k < c.per; k += nlanes) {
            out[i * c.per + k] = h.p[k] * 2.0f + static_cast<float>(i);
          }
        }
        // The barrier from the source, reached through the Ctx. Legal because
        // every work-item of the group is at this same point (invariant I2).
        _cy.Barrier();
        case 2:
          if (_cy_f.Replaying()) _cy_f.Pop(c, out, npages, lane, nlanes, i);
          Flush(c, i, _cy);
          if (_cy.Parked()) {
            _cy_f.Push(2, c, out, npages, lane, nlanes, i);
            return;
          }
      }
  }
  _cy_f.Done();
}

}  // namespace gen

}  // namespace clio::co::spike

#endif  // CLIO_RUNTIME_TEST_CO_SPIKE_WORKLOAD_H_
