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

#ifndef CTP_TEST_UNIT_BASIC_TEST_H_
#define CTP_TEST_UNIT_BASIC_TEST_H_

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_all.hpp>

namespace cl = Catch::Clara;
cl::Parser define_options();

#include <cstdlib>
#include <iostream>

#include "clio_ctp/clio_ctp.h"

static inline bool VerifyBuffer(char *ptr, size_t size, char nonce) {
  for (size_t i = 0; i < size; ++i) {
    if (ptr[i] != nonce) {
      std::cout << (int)ptr[i] << std::endl;
      return false;
    }
  }
  return true;
}

/** var = TYPE(val) */
#define _CREATE_SET_VAR_TO_INT_OR_STRING(TYPE, VAR, TMP_VAR, VAL) \
  if constexpr (std::is_same_v<TYPE, ctp::priv::string>) {             \
    TMP_VAR = ctp::priv::string(std::to_string(VAL));                  \
  } else if constexpr (std::is_same_v<TYPE, std::string>) {       \
    TMP_VAR = std::string(std::to_string(VAL));                   \
  } else {                                                        \
    TMP_VAR = VAL;                                                \
  }                                                               \
  TYPE &VAR = TMP_VAR;                                            \
  (void)VAR;

/** TYPE VAR = TYPE(VAL) */
#define CREATE_SET_VAR_TO_INT_OR_STRING(TYPE, VAR, VAL) \
  TYPE VAR##_tmp;                                       \
  _CREATE_SET_VAR_TO_INT_OR_STRING(TYPE, VAR, VAR##_tmp, VAL);

/** RET = int(TYPE(VAR)); */
#define GET_INT_FROM_VAR(TYPE, RET, VAR)                    \
  if constexpr (std::is_same_v<TYPE, ctp::priv::string>) {       \
    RET = atoi((VAR).str().c_str());                        \
  } else if constexpr (std::is_same_v<TYPE, std::string>) { \
    RET = atoi((VAR).c_str());                              \
  } else {                                                  \
    RET = VAR;                                              \
  }

/** int RET = int(TYPE(VAR)); */
#define CREATE_GET_INT_FROM_VAR(TYPE, RET, VAR) \
  int RET;                                      \
  GET_INT_FROM_VAR(TYPE, RET, VAR)

void MainPretest();
void MainPosttest();

#define PAGE_DIVIDE(TEXT)

#endif  // CTP_TEST_UNIT_BASIC_TEST_H_
