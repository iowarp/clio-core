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

#ifndef HSHM_ERROR_H
#define HSHM_ERROR_H

// #ifdef __cplusplus

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "hermes_shm/util/formatter.h"

#define HSHM_ERROR_TYPE std::shared_ptr<hshm::Error>
#define HSHM_ERROR_HANDLE_START() try {
#define HSHM_ERROR_HANDLE_END()         \
  }                                     \
  catch (HSHM_ERROR_TYPE & err) {       \
    err->print();                       \
    exit(-1024);                        \
  }                                     \
  catch (std::exception & e) {          \
    std::cerr << e.what() << std::endl; \
    exit(-1024);                        \
  }
#define HSHM_ERROR_HANDLE_TRY try
#define HSHM_ERROR_PTR err
#define HSHM_ERROR_HANDLE_CATCH catch (HSHM_ERROR_TYPE & HSHM_ERROR_PTR)
#define HSHM_ERROR_IS(err, check) (err->get_code() == check.get_code())

#if HSHM_IS_HOST
#define HSHM_THROW_ERROR(CODE, ...) throw CODE.format(__VA_ARGS__)
#define HSHM_THROW_STD_ERROR(...) throw std::runtime_error(__VA_ARGS__);
#endif

#if HSHM_IS_GPU
#define HSHM_THROW_ERROR(CODE, ...)
#define HSHM_THROW_STD_ERROR(...)
#endif

namespace hshm {

class Error : std::exception {
 private:
  const char* fmt_;
  std::string msg_;

 public:
  Error() : fmt_() {}

  explicit Error(const char* fmt) : fmt_(fmt) {}
  ~Error() override = default;

  template <typename... Args>
  Error format(Args&&... args) const {
    Error err = Error(fmt_);
    err.msg_ = Formatter::format(fmt_, std::forward<Args>(args)...);
    return err;
  }

  const char* what() const throw() override { return msg_.c_str(); }

  void print() { std::cout << what() << std::endl; }
};

}  // namespace hshm

// #endif

#endif  // HSHM_ERROR_H
