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
 * Wall-clock deadline for the lightbeam tests' receive/poll loops. Loops used
 * to give up after a fixed number of 1 ms (or 100 ms) iterations, i.e. ~5 s
 * of SLEEPING -- which on a loaded runner is much less than 5 s of the peer
 * running. A deadline bounds real time instead, and is set far above any
 * honest delivery latency: it only separates "slow" from "never arrives".
 */
#ifndef CLIO_CTP_LIGHTBEAM_TEST_LBM_TEST_DEADLINE_H_
#define CLIO_CTP_LIGHTBEAM_TEST_LBM_TEST_DEADLINE_H_

#include <chrono>

namespace lbm_test {

/** Default receive deadline (ms) for one message in these tests. */
constexpr int kRecvDeadlineMs = 60000;

/** A point in steady-clock time after which a wait loop gives up. */
class Deadline {
 public:
  /**
   * @param ms milliseconds from now until the deadline expires
   */
  explicit Deadline(int ms = kRecvDeadlineMs)
      : end_(std::chrono::steady_clock::now() + std::chrono::milliseconds(ms)) {}

  /** @return true once the deadline has passed */
  bool Expired() const { return std::chrono::steady_clock::now() >= end_; }

 private:
  std::chrono::steady_clock::time_point end_;
};

}  // namespace lbm_test

#endif  // CLIO_CTP_LIGHTBEAM_TEST_LBM_TEST_DEADLINE_H_
