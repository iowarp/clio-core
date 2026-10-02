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
 * safe_bdev stress: an array that starts on ONE file bdev and is grown to
 * two and then four data members (file bdevs standing in for disks of one
 * type), with members unplugged -- singly and two at once, data and parity
 * -- and recovered onto fresh bdevs. Every byte ever written is verified
 * after every membership change. A second case keeps I/O running on several
 * threads while the membership changes underneath it.
 *
 * Backing files go under $CLIO_SAFE_STRESS_DIR (default: the temp dir).
 */

#include "safe_bdev_stress_util.h"

using namespace sbstress;
using namespace std::chrono_literals;


TEST_CASE("safe_bdev_stress_grow_unplug_recover", "[safe_bdev][stress]") {
  EnsureInit();
  REQUIRE(g_initialized);
  Array a("sbs_grow", 90000u + static_cast<clio::run::u32>(ctp::SystemInfo::GetPid() & 0xFFF) * 64);
  clio::run::u32 seed = 1;
  Disk d0, d1, d2, d3, p0, p1;

  // One disk, then a mirror (parity) of it.
  REQUIRE(a.NewDisk(&d0));
  REQUIRE(a.Create(d0, /*max_failures=*/2) == 0);
  WriteMany(a, &seed, 20);
  REQUIRE(a.NewDisk(&p0));
  REQUIRE(a.Add(p0, true) == 0);
  REQUIRE(a.BuildParity() == 0);
  REQUIRE(a.VerifyAll("1 data + 1 parity") == 0);

  // Upgrade to two data disks.
  REQUIRE(a.NewDisk(&d1));
  REQUIRE(a.Add(d1, false) == 0);
  WriteMany(a, &seed, 30);
  REQUIRE(a.BuildParity() == 0);
  REQUIRE(a.VerifyAll("2 data") == 0);

  // Unplug the original disk; everything must still read (degraded), then
  // rebuild it onto a new disk.
  REQUIRE(a.Unplug(d0) == 0);
  REQUIRE(a.VerifyAll("2 data, d0 unplugged") == 0);
  Disk d0b;
  REQUIRE(a.NewDisk(&d0b));
  REQUIRE(a.Recover(d0, d0b) == 0);
  REQUIRE(a.VerifyAll("d0 recovered") == 0);

  // Upgrade to four data disks and a second parity disk.
  REQUIRE(a.NewDisk(&d2));
  REQUIRE(a.Add(d2, false) == 0);
  REQUIRE(a.NewDisk(&d3));
  REQUIRE(a.Add(d3, false) == 0);
  REQUIRE(a.NewDisk(&p1));
  REQUIRE(a.Add(p1, true) == 0);
  WriteMany(a, &seed, 40);
  REQUIRE(a.BuildParity() == 0);
  REQUIRE(a.VerifyAll("4 data + 2 parity") == 0);

  // Two disks unplugged at once (the array's full failure budget).
  REQUIRE(a.Unplug(d1) == 0);
  REQUIRE(a.Unplug(d3) == 0);
  REQUIRE(a.VerifyAll("d1 + d3 unplugged") == 0);
  Disk d1b, d3b;
  REQUIRE(a.NewDisk(&d1b));
  REQUIRE(a.NewDisk(&d3b));
  REQUIRE(a.Recover(d1, d1b) == 0);
  REQUIRE(a.Recover(d3, d3b) == 0);
  REQUIRE(a.VerifyAll("d1 + d3 recovered") == 0);

  // A parity disk fails, and a data disk with it.
  REQUIRE(a.Unplug(p0) == 0);
  REQUIRE(a.Unplug(d2) == 0);
  REQUIRE(a.VerifyAll("p0 + d2 unplugged") == 0);
  Disk p0b, d2b;
  REQUIRE(a.NewDisk(&p0b));
  REQUIRE(a.NewDisk(&d2b));
  REQUIRE(a.Recover(d2, d2b) == 0);
  REQUIRE(a.Recover(p0, p0b) == 0);
  WriteMany(a, &seed, 20);
  REQUIRE(a.BuildParity() == 0);
  REQUIRE(a.VerifyAll("final") == 0);
}

/**
 * Lose one data disk and the FIRST parity disk, so every degraded read
 * depends on the second parity alone.
 * @param a array with >= 2 data disks and 2 parity disks, parity current
 * @param data a data disk to unplug
 * @param p0 the first parity disk
 * @return blocks that read wrong or failed
 */
size_t VerifyOnSecondParity(Array &a, const Disk &data, const Disk &p0) {
  REQUIRE(a.Unplug(data) == 0);
  REQUIRE(a.Unplug(p0) == 0);
  return a.VerifyAll("data + p0 unplugged (second parity only)");
}

TEST_CASE("safe_bdev_stress_second_parity_static", "[safe_bdev][stress]") {
  EnsureInit();
  REQUIRE(g_initialized);
  Array a("sbs_p2s",
          92000u + static_cast<clio::run::u32>(ctp::SystemInfo::GetPid() & 0xFFF) * 64);
  clio::run::u32 seed = 1;
  Disk d[4], p0, p1;
  REQUIRE(a.NewDisk(&d[0]));
  REQUIRE(a.Create(d[0], 2) == 0);
  for (int i = 1; i < 4; ++i) {
    REQUIRE(a.NewDisk(&d[i]));
    REQUIRE(a.Add(d[i], false) == 0);
  }
  REQUIRE(a.NewDisk(&p0));
  REQUIRE(a.Add(p0, true) == 0);
  REQUIRE(a.NewDisk(&p1));
  REQUIRE(a.Add(p1, true) == 0);
  WriteMany(a, &seed, 40);
  REQUIRE(a.BuildParity() == 0);
  REQUIRE(VerifyOnSecondParity(a, d[1], p0) == 0);
}

TEST_CASE("safe_bdev_stress_second_parity_grown", "[safe_bdev][stress]") {
  EnsureInit();
  REQUIRE(g_initialized);
  Array a("sbs_p2g",
          93000u + static_cast<clio::run::u32>(ctp::SystemInfo::GetPid() & 0xFFF) * 64);
  clio::run::u32 seed = 1;
  Disk d[4], p0, p1;
  REQUIRE(a.NewDisk(&d[0]));
  REQUIRE(a.Create(d[0], 2) == 0);
  REQUIRE(a.NewDisk(&p0));
  REQUIRE(a.Add(p0, true) == 0);
  WriteMany(a, &seed, 20);
  REQUIRE(a.BuildParity() == 0);
  for (int i = 1; i < 4; ++i) {
    REQUIRE(a.NewDisk(&d[i]));
    REQUIRE(a.Add(d[i], false) == 0);
  }
  REQUIRE(a.NewDisk(&p1));
  REQUIRE(a.Add(p1, true) == 0);
  WriteMany(a, &seed, 40);
  REQUIRE(a.BuildParity() == 0);
  REQUIRE(VerifyOnSecondParity(a, d[1], p0) == 0);
}

TEST_CASE("safe_bdev_stress_two_data_lost", "[safe_bdev][stress]") {
  EnsureInit();
  REQUIRE(g_initialized);
  Array a("sbs_2d",
          94000u + static_cast<clio::run::u32>(ctp::SystemInfo::GetPid() & 0xFFF) * 64);
  clio::run::u32 seed = 1;
  Disk d[4], p0, p1;
  REQUIRE(a.NewDisk(&d[0]));
  REQUIRE(a.Create(d[0], 2) == 0);
  for (int i = 1; i < 4; ++i) {
    REQUIRE(a.NewDisk(&d[i]));
    REQUIRE(a.Add(d[i], false) == 0);
  }
  REQUIRE(a.NewDisk(&p0));
  REQUIRE(a.Add(p0, true) == 0);
  REQUIRE(a.NewDisk(&p1));
  REQUIRE(a.Add(p1, true) == 0);
  WriteMany(a, &seed, 40);
  REQUIRE(a.BuildParity() == 0);
  REQUIRE(a.Unplug(d[1]) == 0);
  REQUIRE(a.Unplug(d[3]) == 0);
  REQUIRE(a.VerifyAll("two data disks lost") == 0);
}

TEST_CASE("safe_bdev_stress_restart_after_growth", "[safe_bdev][stress]") {
  EnsureInit();
  REQUIRE(g_initialized);
  Array a("sbs_rst",
          96000u + static_cast<clio::run::u32>(ctp::SystemInfo::GetPid() & 0xFFF) * 64);
  clio::run::u32 seed = 1;
  Disk d[4], p0, p1;
  REQUIRE(a.NewDisk(&d[0]));
  REQUIRE(a.Create(d[0], 2) == 0);
  REQUIRE(a.NewDisk(&p0));
  REQUIRE(a.Add(p0, true) == 0);
  WriteMany(a, &seed, 20);
  for (int i = 1; i < 4; ++i) {
    REQUIRE(a.NewDisk(&d[i]));
    REQUIRE(a.Add(d[i], false) == 0);
  }
  REQUIRE(a.NewDisk(&p1));
  REQUIRE(a.Add(p1, true) == 0);
  WriteMany(a, &seed, 40);
  REQUIRE(a.BuildParity() == 0);
  // One disk replaced, another left unplugged when the array stops.
  REQUIRE(a.Unplug(d[1]) == 0);
  Disk d1b;
  REQUIRE(a.NewDisk(&d1b));
  REQUIRE(a.Recover(d[1], d1b) == 0);
  REQUIRE(a.Unplug(d[2]) == 0);
  REQUIRE(a.VerifyAll("before restart") == 0);

  REQUIRE(a.Restart() == 0);
  REQUIRE(a.VerifyAll("after restart (d2 still out)") == 0);
  Disk d2b;
  REQUIRE(a.NewDisk(&d2b));
  REQUIRE(a.Recover(d[2], d2b) == 0);
  WriteMany(a, &seed, 20);
  REQUIRE(a.BuildParity() == 0);
  REQUIRE(a.VerifyAll("after restart + recovery") == 0);
}

TEST_CASE("safe_bdev_stress_overwrite_while_degraded", "[safe_bdev][stress]") {
  EnsureInit();
  REQUIRE(g_initialized);
  Array a("sbs_ovw",
          97000u + static_cast<clio::run::u32>(ctp::SystemInfo::GetPid() & 0xFFF) * 64);
  clio::run::u32 seed = 1;
  Disk d[4], p0;
  REQUIRE(a.NewDisk(&d[0]));
  REQUIRE(a.Create(d[0], 1) == 0);
  for (int i = 1; i < 4; ++i) {
    REQUIRE(a.NewDisk(&d[i]));
    REQUIRE(a.Add(d[i], false) == 0);
  }
  REQUIRE(a.NewDisk(&p0));
  REQUIRE(a.Add(p0, true) == 0);
  WriteMany(a, &seed, 40);
  REQUIRE(a.BuildParity() == 0);
  // A disk dies; the pages on it are rewritten while it is gone. The new
  // bytes must live in the parity (nothing else holds them).
  REQUIRE(a.Unplug(d[1]) == 0);
  REQUIRE(a.OverwriteAll(5000) == 0);
  REQUIRE(a.VerifyAll("overwritten while d1 down") == 0);
  Disk d1b;
  REQUIRE(a.NewDisk(&d1b));
  REQUIRE(a.Recover(d[1], d1b) == 0);
  REQUIRE(a.VerifyAll("d1 recovered after overwrites") == 0);
}

TEST_CASE("safe_bdev_stress_io_during_membership", "[safe_bdev][stress]") {
  EnsureInit();
  REQUIRE(g_initialized);
  Array a("sbs_live",
          91000u + static_cast<clio::run::u32>(ctp::SystemInfo::GetPid() & 0xFFF) * 64);
  Disk d0, p0;
  REQUIRE(a.NewDisk(&d0));
  REQUIRE(a.Create(d0, 1) == 0);
  REQUIRE(a.NewDisk(&p0));
  REQUIRE(a.Add(p0, true) == 0);

  // Writers keep adding blocks; readers keep checking what is recorded.
  std::atomic<bool> stop{false};
  std::atomic<clio::run::u32> seed{1000};
  std::atomic<size_t> write_fail{0};
  std::vector<std::thread> io;
  for (int t = 0; t < 4; ++t) {
    io.emplace_back([&] {
      while (!stop.load()) {
        if (a.Written() >= kLiveCap) {
          std::this_thread::sleep_for(5ms);
          continue;
        }
        if (!a.WriteOne(seed.fetch_add(1))) {
          ++write_fail;  // full until the next disk arrives: back off
          std::this_thread::sleep_for(20ms);
        }
      }
    });
  }
  // Membership churn under load: grow 1 -> 2 -> 4 data disks, with parity
  // brought current before each unplug (the array's recovery precondition).
  std::vector<Disk> data{d0};
  for (int k = 1; k < 4; ++k) {
    std::this_thread::sleep_for(300ms);
    Disk d;
    REQUIRE(a.NewDisk(&d));
    REQUIRE(a.Add(d, false) == 0);
    data.push_back(d);
  }
  for (int round = 0; round < 3; ++round) {
    std::this_thread::sleep_for(300ms);
    stop.store(true);  // quiesce writers so every slot gets parity
    for (auto &th : io) th.join();
    io.clear();
    REQUIRE(a.BuildParity() == 0);
    const Disk victim = data[(round + 1) % data.size()];
    REQUIRE(a.Unplug(victim) == 0);
    stop.store(false);
    for (int t = 0; t < 4; ++t) {
      io.emplace_back([&] {
        while (!stop.load()) {
          if (a.Written() < kLiveCap && !a.WriteOne(seed.fetch_add(1))) {
            ++write_fail;
          }
          std::this_thread::sleep_for(2ms);
        }
      });
    }
    Disk fresh;
    REQUIRE(a.NewDisk(&fresh));
    REQUIRE(a.Recover(victim, fresh) == 0);
    data[(round + 1) % data.size()] = fresh;
  }
  stop.store(true);
  for (auto &th : io) th.join();
  REQUIRE(a.BuildParity() == 0);
  HLOG(kInfo, "safe_bdev stress live: {} writes recorded, {} refused",
       a.Written(), write_fail.load());
  REQUIRE(a.VerifyAll("after live churn") == 0);
}

SIMPLE_TEST_MAIN()
