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

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "container.h"

// Allocate() and RunAndGetResult() are defined in lib.cc, compiled into this
// executable. RunKernel (which calls c->Run() virtually) lives in lib.cc
// alongside Sum's vtable, so device-side virtual dispatch resolves correctly.
// CUDA does not reliably resolve cross-TU device vtable calls via separate
// compilation, so both the vtable definition and the call site must be in the
// same translation unit (lib.cc).
extern "C" Container* Allocate();
extern "C" int RunAndGetResult(Container *d_obj);

int main() {
  Container *d_obj = Allocate();
  if (!d_obj) {
    fprintf(stderr, "FAIL: Allocate returned nullptr\n");
    return 1;
  }

  int result = RunAndGetResult(d_obj);

  if (result == 60) {
    printf("PASS: result = %d\n", result);
  } else {
    printf("FAIL: expected 60, got %d\n", result);
  }

  cudaFree(d_obj);
  return (result == 60) ? 0 : 1;
}
