/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * Distributed under BSD 3-Clause license.
 */

#ifndef CLIO_RUNTIME_BDEV_BLOCKING_OFFLOAD_H_
#define CLIO_RUNTIME_BDEV_BLOCKING_OFFLOAD_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace clio::run::bdev {

/** How often a task waiting on an offloaded device call checks it (us). */
constexpr double kOffloadPollUs = 50.0;

/**
 * Threads that run the device calls that block in the kernel for an unbounded
 * time -- fdatasync/fsync of a multi-GB tier file or of an allocator log,
 * fallocate growing a backing file. They took 5-76 s under tiering load and
 * ran ON the runtime workers: with every worker parked in such a syscall the
 * node ran nothing and answered no liveness probe, and its peers declared it
 * dead (fsync EIO, tier tests hanging). A task hands the call here and
 * yields until it is done, so its worker keeps running other tasks:
 *
 *   auto job = BlockingOffload::Get().Submit([..] { return rc; });
 *   while (!job->done.load(std::memory_order_acquire))
 *     CLIO_CO_AWAIT(clio::run::yield(kOffloadPollUs));
 */
class BlockingOffload {
 public:
  struct Job {
    std::function<int()> fn;
    std::atomic<bool> done{false};
    int rc = 0;
    const char *what = "";
    std::chrono::steady_clock::time_point queued_at;
  };

  static BlockingOffload &Get() {
    static BlockingOffload *inst = new BlockingOffload();  // never destroyed
    return *inst;
  }

  std::shared_ptr<Job> Submit(std::function<int()> fn, const char *what = "") {
    auto job = std::make_shared<Job>();
    job->fn = std::move(fn);
    job->what = what;
    job->queued_at = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> g(mu_);
      q_.push_back(job);
    }
    cv_.notify_one();
    return job;
  }

 private:
  BlockingOffload() {
    // Enough that a node's concurrent syncs and growths do not queue behind
    // one another on a few slow devices; they sleep in the kernel, not spin.
    for (int i = 0; i < kThreads; ++i) {
      std::thread([this] { Loop(); }).detach();
    }
  }

  void Loop() {
    for (;;) {
      std::shared_ptr<Job> job;
      {
        std::unique_lock<std::mutex> g(mu_);
        cv_.wait(g, [this] { return !q_.empty(); });
        job = std::move(q_.front());
        q_.pop_front();
      }
      const auto t0 = std::chrono::steady_clock::now();
      job->rc = job->fn();
      const auto t1 = std::chrono::steady_clock::now();
      job->done.store(true, std::memory_order_release);
      const double wait_s =
          std::chrono::duration<double>(t0 - job->queued_at).count();
      const double run_s = std::chrono::duration<double>(t1 - t0).count();
      if (wait_s > 1.0 || run_s > 1.0) {
        // Which device calls are slow, and whether jobs queue behind them.
        std::fprintf(stderr, "[OFFLOAD] slow %s: queued %.2f s, ran %.2f s\n",
                     job->what, wait_s, run_s);
      }
    }
  }

  static constexpr int kThreads = 16;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<Job>> q_;
};

}  // namespace clio::run::bdev

#endif  // CLIO_RUNTIME_BDEV_BLOCKING_OFFLOAD_H_
