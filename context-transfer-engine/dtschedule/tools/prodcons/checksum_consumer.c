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
 * @file checksum_consumer.c
 * Consumer half of the DTSchedule producer-consumer workload.
 *
 * Runs on nodes other than the producer's, typically with half as many
 * ranks. For every step it waits for /clio::<run>__step<s>.done, then each
 * rank reads its share of the producer's files (producer rank r goes to
 * consumer rank r % size) and makes --passes passes over each: a Fletcher-64
 * checksum plus a sum of squares, i.e. a fraction of the producer's work.
 *
 * Summary (rank 0):
 *   checksum_consumer ranks=N steps=S read_mb=M wait_s=W proc_s=P
 *     lag_p50_s=L wall_s=T bad=B checksum=X
 * where lag is the time from a step's marker appearing to the consumer
 * finishing that step, and bad counts short or unreadable files.
 */

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "prodcons_common.h"

/**
 * Wait until the step marker exists (rank 0 polls, everyone learns).
 * @param o Options
 * @param step Step index
 * @param rank This rank
 * @param timeout_s Give up after this long
 * @return 0 when the marker appeared, ETIMEDOUT otherwise
 */
static int WaitForStep(const PcOptions *o, int step, int rank,
                       double timeout_s) {
  int rc = ETIMEDOUT;
  if (rank == 0) {
    char path[512];
    PcPath(path, sizeof(path), o->run, step, -1);
    const double end = PcNow() + timeout_s;
    while (PcNow() < end) {
      int fd = open(path, O_RDONLY);
      if (fd >= 0) {
        close(fd);
        rc = 0;
        break;
      }
      usleep(20000);
    }
  }
  MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return rc;
}

/**
 * Checksum and sum of squares over a buffer, repeated passes times.
 * @param data Bytes (a multiple of 8 is expected; the tail is ignored)
 * @param bytes Length
 * @param passes Repetitions (compute weight)
 * @param sumsq Output: sum of squares of the doubles
 * @return Fletcher-64 style checksum over the 32-bit words
 */
static uint64_t Checksum(const void *data, size_t bytes, int passes,
                         double *sumsq) {
  const uint32_t *w = (const uint32_t *)data;
  const double *d = (const double *)data;
  const size_t nw = bytes / 4, nd = bytes / 8;
  uint64_t a = 0, b = 0;
  double sq = 0.0;
  for (int p = 0; p < passes; ++p) {
    a = 0;
    b = 0;
    for (size_t i = 0; i < nw; ++i) {
      a = (a + w[i]) % 0xffffffffULL;
      b = (b + a) % 0xffffffffULL;
    }
    sq = 0.0;
    for (size_t i = 0; i < nd; ++i) sq += d[i] * d[i];
  }
  *sumsq = sq;
  return (b << 32) | a;
}

/** Bad files a rank reports per step (the rest are only counted). */
static const int kMaxBadReports = 3;

/** qsort comparator for doubles. */
static int CmpD(const void *x, const void *y) {
  const double a = *(const double *)x, b = *(const double *)y;
  return (a > b) - (a < b);
}

/**
 * Process one step: read and checksum this rank's share of the files.
 * @param o Options
 * @param step Step index
 * @param rank This rank
 * @param size Consumer rank count
 * @param buf Read buffer
 * @param cap Buffer capacity (one producer file)
 * @param sum Running checksum (xor of per-file checksums)
 * @param read_bytes Running bytes read
 * @return Files that were missing or short
 */
static int ProcessStep(const PcOptions *o, int step, int rank, int size,
                       char *buf, size_t cap, uint64_t *sum,
                       double *read_bytes) {
  int bad = 0;
  for (int r = rank; r < o->producers; r += size) {
    char path[512];
    PcPath(path, sizeof(path), o->run, step, r);
    size_t got = 0;
    int err = PcReadFile(path, buf, cap, &got);
    if (err != 0 || got != cap) {
      if (bad < kMaxBadReports) {
        fprintf(stderr, "prodcons consumer rank %d: bad file %s: errno=%d "
                "(%s) got=%zu of %zu\n", rank, path, err,
                err ? strerror(err) : "short read", got, cap);
      }
      ++bad;
      continue;
    }
    double sq = 0.0;
    *sum ^= Checksum(buf, got, o->passes, &sq);
    *read_bytes += (double)got;
  }
  return bad;
}

/**
 * Entry point: consume every step, report.
 * @param argc Argument count
 * @param argv See PcParse (--run --steps --nx --ny --producers --passes)
 * @return 0 on success, 2 when files were missing or a step timed out
 */
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  PcOptions o = {"prodcons", 10, 2048, 2048, 0, 40, 0.0, 4, NULL};
  PcParse(argc, argv, &o);
  const size_t cap = (size_t)o.nx * (size_t)o.ny * sizeof(double);
  char *buf = malloc(cap);
  double *lag = malloc(sizeof(double) * (size_t)(o.steps > 0 ? o.steps : 1));
  if (buf == NULL || lag == NULL) MPI_Abort(MPI_COMM_WORLD, 1);
  uint64_t sum = 0;
  double read_bytes = 0.0, wait_s = 0.0, proc_s = 0.0;
  int bad = 0, rc = 0;
  const double t0 = PcNow();
  for (int step = 0; step < o.steps; ++step) {
    const double w0 = PcNow();
    rc = WaitForStep(&o, step, rank, 3600.0);
    if (rc != 0) break;
    const double p0 = PcNow();
    bad += ProcessStep(&o, step, rank, size, buf, cap, &sum, &read_bytes);
    MPI_Barrier(MPI_COMM_WORLD);
    wait_s += p0 - w0;
    proc_s += PcNow() - p0;
    lag[step] = PcNow() - p0;
  }
  int bad_all = 0;
  double bytes_all = 0.0;
  uint64_t sum_all = 0;
  MPI_Reduce(&bad, &bad_all, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&read_bytes, &bytes_all, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&sum, &sum_all, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    const int n = rc == 0 ? o.steps : 0;
    if (n > 0) qsort(lag, (size_t)n, sizeof(double), CmpD);
    printf("checksum_consumer ranks=%d steps=%d read_mb=%.1f wait_s=%.2f "
           "proc_s=%.2f lag_p50_s=%.2f wall_s=%.2f bad=%d checksum=%016llx "
           "rc=%d\n",
           size, o.steps, bytes_all / 1e6, wait_s, proc_s,
           n > 0 ? lag[n / 2] : -1.0, PcNow() - t0, bad_all,
           (unsigned long long)sum_all, rc);
    fflush(stdout);
  }
  free(buf);
  free(lag);
  MPI_Finalize();
  return (rc == 0 && bad_all == 0) ? 0 : 2;
}
