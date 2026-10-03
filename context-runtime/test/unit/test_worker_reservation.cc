/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

/**
 * Worker::ReservationUs — the µs-to-integer conversion behind the #781 load
 * reservations.
 *
 * It exists because the value being converted is a float read off the task
 * (sched_reserved_us_), so it is only as trustworthy as whatever last wrote
 * that field. A float reaches 3.4e38 where long long stops at 9.2e18, and
 * converting one that does not fit is undefined — so the conversion saturates
 * rather than trusting its input. The predicates that matter are the ones no
 * scheduler run reliably produces: a non-positive duration, a NaN, and a value
 * past the integer range.
 *
 * Pure: a static member function, so no worker, no orchestrator, no runtime.
 */

#include "../simple_test.h"

#include <cmath>
#include <limits>

#include "clio_runtime/worker.h"

using clio::run::Worker;

TEST_CASE("ReservationUs - an ordinary duration truncates to whole µs",
          "[worker][reservation]") {
  REQUIRE(Worker::ReservationUs(250.0) == 250);
  REQUIRE(Worker::ReservationUs(250.7) == 250);
  REQUIRE(Worker::ReservationUs(1.0) == 1);
}

TEST_CASE("ReservationUs - anything that is not a positive duration is zero",
          "[worker][reservation]") {
  // Callers treat 0 as "nothing to account", so these never reach the atomic.
  REQUIRE(Worker::ReservationUs(0.0) == 0);
  REQUIRE(Worker::ReservationUs(-1.0) == 0);
  REQUIRE(Worker::ReservationUs(-1e18) == 0);
  // Sub-microsecond: truncation would make it 0 anyway, and it says the same
  // thing — there is no reservation worth tracking.
  REQUIRE(Worker::ReservationUs(0.4) == 0);
}

TEST_CASE("ReservationUs - NaN is not a duration", "[worker][reservation]") {
  // NaN loses EVERY comparison, including `us > 0.0`, which is what makes the
  // guard total rather than a cast that would be undefined.
  REQUIRE(Worker::ReservationUs(std::numeric_limits<double>::quiet_NaN()) == 0);
}

TEST_CASE("ReservationUs - a value past the integer range saturates",
          "[worker][reservation]") {
  // The cases the float's range allows and long long's does not. Each of these
  // would be undefined as a direct conversion.
  const long long huge = Worker::ReservationUs(3.4e38);
  REQUIRE(huge > 0);
  REQUIRE(huge <= std::numeric_limits<long long>::max());
  REQUIRE(Worker::ReservationUs(std::numeric_limits<double>::infinity()) ==
          huge);
  // A value just inside the cap still converts exactly, so saturation only
  // touches what it has to.
  REQUIRE(Worker::ReservationUs(1.0e18) == 1000000000000000000LL);
}

SIMPLE_TEST_MAIN()
