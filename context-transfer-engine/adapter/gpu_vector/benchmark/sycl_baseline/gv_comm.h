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
 * The data plane of the Aurora baseline editions, chosen at compile time.
 *
 * THE BASELINES ON THIS MACHINE. The CUDA editions in each workload
 * directory (clio_<wl>_{mpi,nccl,nvshmem}_bench.cc) cannot run on Intel
 * GPUs, so the Aurora rows of the evaluation are these SYCL editions: one
 * source per workload, the science taken from the same CUDA-free headers the
 * paged bench uses, and the communication behind the five verbs below:
 *
 *   GV_COMM_MPI     MPICH, GPU-aware (built --with-ze): device buffers go
 *                   straight to MPI. GV_MPI_HOST_STAGE=1 at run time bounces
 *                   through the host instead, which is what the CUDA MPI
 *                   edition measures on a machine whose MPI is not
 *                   device-aware.
 *   GV_COMM_CCL     oneCCL: the NCCL analogue. Collectives on device
 *                   buffers, bootstrapped over MPI; halo planes move with
 *                   ccl::send/recv.
 *   GV_COMM_ISHMEM  Intel SHMEM: the NVSHMEM analogue. Buffers on the
 *                   symmetric heap, one-sided puts for the halo, host-side
 *                   reductions and fcollect. The runtime is MPI, initialised
 *                   by ISHMEM itself.
 *
 * Like the CUDA editions, this links NOTHING from clio: a baseline that
 * shares a substrate with the thing it is benchmarked against is not a
 * baseline. MPI is present in every edition for the gates (a broadcast of
 * the exit code, a host-side sum outside the timed region); the data plane
 * under test is the substrate's own.
 *
 * Exactly one of GV_COMM_MPI, GV_COMM_CCL, GV_COMM_ISHMEM must be defined.
 */
#ifndef CLIO_GV_BENCH_SYCL_BASELINE_GV_COMM_H_
#define CLIO_GV_BENCH_SYCL_BASELINE_GV_COMM_H_

#include <mpi.h>
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if defined(GV_COMM_CCL)
#include <oneapi/ccl.hpp>
#define GV_COMM_NAME "ONECCL"
#elif defined(GV_COMM_ISHMEM)
#include <ishmem.h>
#include <ishmemx.h>
#define GV_COMM_NAME "ISHMEM"
#elif defined(GV_COMM_MPI)
#define GV_COMM_NAME "MPI"
#else
#error "define one of GV_COMM_MPI, GV_COMM_CCL, GV_COMM_ISHMEM"
#endif

namespace gvc {

using u32 = unsigned int;
using u64 = unsigned long long;

/** Wall clock in milliseconds, for the timed region of every edition. */
inline double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(clock::now()
                                                       .time_since_epoch())
      .count();
}

/** MPI datatype of a buffer element, for the host-side gate reductions and
 *  the MPI data plane. */
template <typename T> inline MPI_Datatype MpiType();
template <> inline MPI_Datatype MpiType<float>() { return MPI_FLOAT; }
template <> inline MPI_Datatype MpiType<double>() { return MPI_DOUBLE; }
template <> inline MPI_Datatype MpiType<unsigned>() { return MPI_UNSIGNED; }
template <> inline MPI_Datatype MpiType<unsigned long long>() {
  return MPI_UNSIGNED_LONG_LONG;
}
template <> inline MPI_Datatype MpiType<int>() { return MPI_INT; }

#if defined(GV_COMM_CCL)
template <typename T> inline ccl::datatype CclType();
template <> inline ccl::datatype CclType<float>() {
  return ccl::datatype::float32;
}
template <> inline ccl::datatype CclType<double>() {
  return ccl::datatype::float64;
}
template <> inline ccl::datatype CclType<unsigned>() {
  return ccl::datatype::uint32;
}
template <> inline ccl::datatype CclType<unsigned long long>() {
  return ccl::datatype::uint64;
}
template <> inline ccl::datatype CclType<int>() { return ccl::datatype::int32; }
#endif

/**
 * The substrate: one rank per process, one SYCL queue on the GPU the
 * process was mapped to (ZE_AFFINITY_MASK, set by the job script).
 */
class Comm {
 public:
  int rank = 0;
  int nranks = 1;
  sycl::queue q;

  /** Bring the substrate up. MPI first (oneCCL bootstraps over it); for
   *  ISHMEM the library initialises MPI itself and MPI_Init is skipped.
   *  @param argc,argv main's, forwarded to MPI_Init */
  void Init(int *argc, char ***argv) {
#if defined(GV_COMM_ISHMEM)
    ishmem_init();
    int inited = 0;
    MPI_Initialized(&inited);
    if (!inited) MPI_Init(argc, argv);
#else
    MPI_Init(argc, argv);
#endif
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    q = sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order());
#if defined(GV_COMM_ISHMEM)
    if (ishmem_my_pe() != rank || ishmem_n_pes() != nranks) {
      std::fprintf(stderr, "ISHMEM pe %d/%d != MPI rank %d/%d\n",
                   ishmem_my_pe(), ishmem_n_pes(), rank, nranks);
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
#endif
#if defined(GV_COMM_CCL)
    ccl::init();
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type addr;
    if (rank == 0) {
      kvs = ccl::create_main_kvs();
      addr = kvs->get_address();
    }
    MPI_Bcast(addr.data(), static_cast<int>(addr.size()), MPI_BYTE, 0,
              MPI_COMM_WORLD);
    if (rank != 0) kvs = ccl::create_kvs(addr);
    dev_ = std::make_unique<ccl::device>(ccl::create_device(q.get_device()));
    ctx_ = std::make_unique<ccl::context>(
        ccl::create_context(q.get_context()));
    comm_ = std::make_unique<ccl::communicator>(
        ccl::create_communicator(nranks, rank, *dev_, *ctx_, kvs));
    stream_ = std::make_unique<ccl::stream>(ccl::create_stream(q));
#endif
    host_stage_ = std::getenv("GV_MPI_HOST_STAGE") != nullptr;
  }

  /** Tear the substrate down, in the reverse order of Init. */
  void Finalize() {
#if defined(GV_COMM_CCL)
    stream_.reset();
    comm_.reset();
    ctx_.reset();
    dev_.reset();
#endif
#if defined(GV_COMM_ISHMEM)
    ishmem_finalize();
    int fin = 0;
    MPI_Finalized(&fin);
    if (!fin) MPI_Finalize();
#else
    MPI_Finalize();
#endif
  }

  /** Device memory a collective may name: plain device USM for MPI and
   *  oneCCL, the symmetric heap for ISHMEM (so every PE's copy sits at the
   *  same offset, which one-sided puts require).
   *  @param n elements
   *  @return the device pointer, or exit on failure */
  template <typename T> T *Alloc(u64 n) {
    T *p = nullptr;
#if defined(GV_COMM_ISHMEM)
    p = static_cast<T *>(ishmem_malloc(static_cast<size_t>(n * sizeof(T))));
#else
    p = sycl::malloc_device<T>(static_cast<size_t>(n), q);
#endif
    if (p == nullptr) {
      std::fprintf(stderr, "rank %d: device allocation of %llu bytes failed\n",
                   rank, n * sizeof(T));
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
    return p;
  }

  /** Device memory no peer ever names: plain USM in every edition, so a
   *  large local deck does not sit on ISHMEM's symmetric heap. */
  template <typename T> T *AllocLocal(u64 n) {
    T *p = sycl::malloc_device<T>(static_cast<size_t>(n), q);
    if (p == nullptr) {
      std::fprintf(stderr, "rank %d: device allocation of %llu bytes failed\n",
                   rank, n * sizeof(T));
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
    return p;
  }

  /** Release memory from AllocLocal. */
  template <typename T> void FreeLocal(T *p) {
    if (p != nullptr) sycl::free(p, q);
  }

  /** Release memory from Alloc. */
  template <typename T> void Free(T *p) {
    if (p == nullptr) return;
#if defined(GV_COMM_ISHMEM)
    ishmem_free(p);
#else
    sycl::free(p, q);
#endif
  }

  /** In-place sum over all ranks of n elements in a device buffer. */
  template <typename T> void AllreduceSum(T *buf, u64 n) {
    q.wait();
#if defined(GV_COMM_CCL)
    ccl::allreduce(buf, buf, static_cast<size_t>(n), CclType<T>(),
                   ccl::reduction::sum, *comm_, *stream_)
        .wait();
#elif defined(GV_COMM_ISHMEM)
    IshmemSumReduce(buf, buf, static_cast<size_t>(n));
#else
    if (host_stage_) {
      std::vector<T> h(static_cast<size_t>(n));
      q.memcpy(h.data(), buf, n * sizeof(T)).wait();
      MPI_Allreduce(MPI_IN_PLACE, h.data(), static_cast<int>(n), MpiType<T>(),
                    MPI_SUM, MPI_COMM_WORLD);
      q.memcpy(buf, h.data(), n * sizeof(T)).wait();
    } else {
      MPI_Allreduce(MPI_IN_PLACE, buf, static_cast<int>(n), MpiType<T>(),
                    MPI_SUM, MPI_COMM_WORLD);
    }
#endif
  }

  /** Gather every rank's `count` elements into recv, in rank order.
   *  @param send device buffer of count elements
   *  @param recv device buffer of count * nranks elements */
  template <typename T> void Allgather(const T *send, T *recv, u64 count) {
    q.wait();
#if defined(GV_COMM_CCL)
    ccl::allgather(send, recv, static_cast<size_t>(count), CclType<T>(),
                   *comm_, *stream_)
        .wait();
#elif defined(GV_COMM_ISHMEM)
    IshmemFcollect(recv, send, static_cast<size_t>(count));
#else
    if (host_stage_) {
      std::vector<T> hs(static_cast<size_t>(count));
      std::vector<T> hr(static_cast<size_t>(count) * nranks);
      q.memcpy(hs.data(), send, count * sizeof(T)).wait();
      MPI_Allgather(hs.data(), static_cast<int>(count), MpiType<T>(),
                    hr.data(), static_cast<int>(count), MpiType<T>(),
                    MPI_COMM_WORLD);
      q.memcpy(recv, hr.data(), hr.size() * sizeof(T)).wait();
    } else {
      MPI_Allgather(send, static_cast<int>(count), MpiType<T>(), recv,
                    static_cast<int>(count), MpiType<T>(), MPI_COMM_WORLD);
    }
#endif
  }

  /**
   * One halo plane each way: send `n` elements from `send` to rank `to`
   * and receive `n` elements from rank `from` into `recv`. -1 means no
   * peer on that side. `recv` must be a symmetric buffer under ISHMEM,
   * where the exchange is a one-sided put by the SENDER into the
   * receiver's `recv` at the same offset, fenced by a barrier.
   */
  template <typename T>
  void Sendrecv(const T *send, int to, T *recv, int from, u64 n) {
    SendrecvN(send, to, n, recv, from, n);
  }

  /**
   * The general form: `nsend` elements to `to`, `nrecv` elements from
   * `from` (the counts were agreed beforehand). A zero count on either
   * side is allowed and moves nothing that way. Under ISHMEM the receiver's
   * `recv` must be symmetric and the sender's put lands at the same offset.
   */
  template <typename T>
  void SendrecvN(const T *send, int to, u64 nsend, T *recv, int from,
                 u64 nrecv) {
    q.wait();
#if defined(GV_COMM_CCL)
    std::vector<ccl::event> evs;
    if (from >= 0 && nrecv > 0) {
      evs.push_back(ccl::recv(recv, static_cast<size_t>(nrecv), CclType<T>(),
                              from, *comm_, *stream_));
    }
    if (to >= 0 && nsend > 0) {
      evs.push_back(ccl::send(const_cast<T *>(send),
                              static_cast<size_t>(nsend), CclType<T>(), to,
                              *comm_, *stream_));
    }
    for (auto &e : evs) e.wait();
#elif defined(GV_COMM_ISHMEM)
    if (to >= 0 && nsend > 0) {
      ishmem_putmem(recv, send, static_cast<size_t>(nsend * sizeof(T)), to);
    }
    ishmem_barrier_all();
#else
    const int mto = (to >= 0) ? to : MPI_PROC_NULL;
    const int mfrom = (from >= 0) ? from : MPI_PROC_NULL;
    if (host_stage_) {
      std::vector<T> hs(static_cast<size_t>(nsend) + 1);
      std::vector<T> hr(static_cast<size_t>(nrecv) + 1);
      if (to >= 0 && nsend > 0) {
        q.memcpy(hs.data(), send, nsend * sizeof(T)).wait();
      }
      MPI_Sendrecv(hs.data(), static_cast<int>(nsend), MpiType<T>(), mto, 11,
                   hr.data(), static_cast<int>(nrecv), MpiType<T>(), mfrom,
                   11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (from >= 0 && nrecv > 0) {
        q.memcpy(recv, hr.data(), nrecv * sizeof(T)).wait();
      }
    } else {
      MPI_Sendrecv(send, static_cast<int>(nsend), MpiType<T>(), mto, 11,
                   recv, static_cast<int>(nrecv), MpiType<T>(), mfrom, 11,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
#endif
  }

  /** Exchange one host integer each way (tiny control traffic, e.g. how
   *  many migrants follow); MPI in every edition, deliberately. */
  unsigned HostExchange(unsigned v, int to, int from) const {
    unsigned r = 0;
    MPI_Sendrecv(&v, 1, MPI_UNSIGNED, (to >= 0) ? to : MPI_PROC_NULL, 21, &r,
                 1, MPI_UNSIGNED, (from >= 0) ? from : MPI_PROC_NULL, 21,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return r;
  }

  /** Host-side sum of n doubles across ranks, in place (gates). */
  void HostSumN(double *v, int n) const {
    MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }

  /** Everyone waits for everyone, on the substrate's own barrier. */
  void Barrier() {
    q.wait();
#if defined(GV_COMM_CCL)
    ccl::barrier(*comm_, *stream_).wait();
#elif defined(GV_COMM_ISHMEM)
    ishmem_barrier_all();
#else
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }

  /** Host-side sum across ranks of one value, for gates outside the timed
   *  region. MPI in every edition, deliberately. */
  template <typename T> T HostSum(T v) const {
    T out = v;
    MPI_Allreduce(&v, &out, 1, MpiType<T>(), MPI_SUM, MPI_COMM_WORLD);
    return out;
  }

  /** Agree on the exit code: rank 0's verdict, broadcast. */
  int Verdict(int rc) const {
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return rc;
  }

  /** The substrate's name for the report lines. */
  static const char *Name() { return GV_COMM_NAME; }

 private:
  bool host_stage_ = false;
#if defined(GV_COMM_CCL)
  std::unique_ptr<ccl::device> dev_;
  std::unique_ptr<ccl::context> ctx_;
  std::unique_ptr<ccl::communicator> comm_;
  std::unique_ptr<ccl::stream> stream_;
#endif
#if defined(GV_COMM_ISHMEM)
  static void IshmemSumReduce(float *d, const float *s, size_t n) {
    if (ishmem_float_sum_reduce(d, s, n)) Fail("ishmem_float_sum_reduce");
  }
  static void IshmemSumReduce(double *d, const double *s, size_t n) {
    if (ishmem_double_sum_reduce(d, s, n)) Fail("ishmem_double_sum_reduce");
  }
  static void IshmemSumReduce(unsigned *d, const unsigned *s, size_t n) {
    if (ishmem_uint_sum_reduce(d, s, n)) Fail("ishmem_uint_sum_reduce");
  }
  static void IshmemSumReduce(unsigned long long *d,
                              const unsigned long long *s, size_t n) {
    if (ishmem_ulonglong_sum_reduce(d, s, n)) {
      Fail("ishmem_ulonglong_sum_reduce");
    }
  }
  static void IshmemFcollect(float *d, const float *s, size_t n) {
    if (ishmem_float_fcollect(d, s, n)) Fail("ishmem_float_fcollect");
  }
  static void IshmemFcollect(double *d, const double *s, size_t n) {
    if (ishmem_double_fcollect(d, s, n)) Fail("ishmem_double_fcollect");
  }
  static void IshmemFcollect(unsigned *d, const unsigned *s, size_t n) {
    if (ishmem_uint_fcollect(d, s, n)) Fail("ishmem_uint_fcollect");
  }
  static void IshmemFcollect(unsigned long long *d,
                             const unsigned long long *s, size_t n) {
    if (ishmem_ulonglong_fcollect(d, s, n)) Fail("ishmem_ulonglong_fcollect");
  }
  static void Fail(const char *what) {
    std::fprintf(stderr, "%s failed\n", what);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
#endif
};

/** A device-scope relaxed atomic add, the SYCL spelling of atomicAdd. */
template <typename T>
inline void AtomicAdd(T *p, T v) {
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>
      a(*p);
  a.fetch_add(v);
}

}  // namespace gvc

#endif  // CLIO_GV_BENCH_SYCL_BASELINE_GV_COMM_H_
