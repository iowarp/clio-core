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

#include <cerrno>

#include "basic_test.h"
#include "clio_ctp/io/io_error.h"

TEST_CASE("io_error_classify_errno", "[io_error]") {
  REQUIRE(ctp::ClassifyErrno(0) == ctp::IoError::kOk);

  // Device / disconnect distinction is the one storage layers rely on.
#ifdef EIO
  REQUIRE(ctp::ClassifyErrno(EIO) == ctp::IoError::kDeviceFault);
  REQUIRE(ctp::IsFatalDevice(ctp::ClassifyErrno(EIO)));
#endif
#ifdef ENODEV
  REQUIRE(ctp::ClassifyErrno(ENODEV) == ctp::IoError::kDisconnected);
  REQUIRE(ctp::IsFatalDevice(ctp::ClassifyErrno(ENODEV)));
#endif
#ifdef ENXIO
  REQUIRE(ctp::ClassifyErrno(ENXIO) == ctp::IoError::kDisconnected);
#endif

  // Transient errors must NOT be treated as a device fault.
#ifdef EAGAIN
  REQUIRE(ctp::ClassifyErrno(EAGAIN) == ctp::IoError::kTransient);
  REQUIRE(ctp::IsTransient(ctp::ClassifyErrno(EAGAIN)));
  REQUIRE_FALSE(ctp::IsFatalDevice(ctp::ClassifyErrno(EAGAIN)));
#endif
#ifdef EINTR
  REQUIRE(ctp::ClassifyErrno(EINTR) == ctp::IoError::kTransient);
#endif

#ifdef ENOSPC
  REQUIRE(ctp::ClassifyErrno(ENOSPC) == ctp::IoError::kNoSpace);
#endif
#ifdef EACCES
  REQUIRE(ctp::ClassifyErrno(EACCES) == ctp::IoError::kPermission);
#endif
#ifdef ETIMEDOUT
  REQUIRE(ctp::ClassifyErrno(ETIMEDOUT) == ctp::IoError::kTimeout);
#endif
#ifdef EINVAL
  REQUIRE(ctp::ClassifyErrno(EINVAL) == ctp::IoError::kInvalid);
#endif

  // An unmapped positive code falls through to Unknown.
  REQUIRE(ctp::ClassifyErrno(999999) == ctp::IoError::kUnknown);
}

TEST_CASE("io_error_classify_last_errno", "[io_error]") {
  errno = 0;
  REQUIRE(ctp::ClassifyLastErrno() == ctp::IoError::kOk);
#ifdef EIO
  errno = EIO;
  REQUIRE(ctp::ClassifyLastErrno() == ctp::IoError::kDeviceFault);
#endif
  errno = 0;
}
