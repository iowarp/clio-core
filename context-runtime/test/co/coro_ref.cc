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
 * The ORACLE half of the transpiler's differential test.
 *
 * Includes the ORIGINAL coro_workload.h, where CO_AWAIT is the identity, so
 * the workload is an ordinary synchronous program. Every page is resident up
 * front, so nothing can suspend and there is nothing to get wrong. Whatever
 * this prints is by definition the right answer.
 */
#include <algorithm>

#include "coro_test_common.h"
#include <coro_workload.h>  // the ORIGINAL -- CO_AWAIT is the identity

int main() {
  namespace d = clio::co::demo;
  d::World w;
  std::fill(w.resident.begin(), w.resident.end(), 1u);
  std::fill(w.flushed.begin(), w.flushed.end(), 1u);
  const d::PageCache c = w.Cache();

  d::Launch(d::kGroups, d::kLanes, [&](clio::co::Item it) {
    d::StreamTile(c, w.OutFor(it.Group()), d::kPages, it.Local(), it.Size(),
                  it);
  });

  d::PrintResult(w.out);
  std::fprintf(stderr, "ref: synchronous, %zu values\n", w.out.size());
  return 0;
}
