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

/**
 * @file mpi_stress.c
 * Co-located MPI load for the DTSchedule evaluations.
 *
 * Every rank holds a working set (--mem-mb), runs a compute phase over it
 * (fused multiply-add sweeps, so both the FPU and the memory bus are busy),
 * then exchanges a halo with its ring neighbours (--halo-kb, crosses nodes
 * when ranks are block-mapped) and joins an MPI_Allreduce, like a bulk-
 * synchronous simulation. It runs until --duration-s elapses or SIGTERM /
 * SIGINT arrives, then rank 0 prints one summary line:
 *
 *   mpi_stress ranks=<n> iters=<min..max> iter_ms_p50=<x> wall_s=<y>
 *
 * Launch one rank per hardware thread to saturate the nodes, e.g.
 *   mpirun --map-by ppr:40:node --bind-to hwthread --use-hwthread-cpus ...
 */

#include <mpi.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/** Set by the signal handler; checked once per iteration. */
static volatile sig_atomic_t g_stop = 0;

/** Request a clean stop (the summary is still printed). */
static void OnSignal(int sig) {
  (void)sig;
  g_stop = 1;
}

/** Monotonic time in seconds. */
static double Now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/** Command-line options. */
typedef struct {
  double duration_s;  /**< 0 = run until signalled */
  size_t mem_mb;      /**< Working set per rank */
  size_t halo_kb;     /**< Bytes exchanged with each neighbour per iteration */
  double compute_ms;  /**< Target length of the compute phase */
} Options;

/**
 * Parse argv into opts.
 * @param argc Argument count
 * @param argv Arguments (--duration-s, --mem-mb, --halo-kb, --compute-ms)
 * @param opts Output options, pre-filled with defaults
 */
static void ParseArgs(int argc, char **argv, Options *opts) {
  for (int i = 1; i + 1 < argc; i += 2) {
    if (strcmp(argv[i], "--duration-s") == 0) {
      opts->duration_s = atof(argv[i + 1]);
    } else if (strcmp(argv[i], "--mem-mb") == 0) {
      opts->mem_mb = (size_t)atol(argv[i + 1]);
    } else if (strcmp(argv[i], "--halo-kb") == 0) {
      opts->halo_kb = (size_t)atol(argv[i + 1]);
    } else if (strcmp(argv[i], "--compute-ms") == 0) {
      opts->compute_ms = atof(argv[i + 1]);
    }
  }
}

/**
 * Compute phase: FMA sweeps over the working set for about budget_ms.
 * @param a Working set
 * @param n Elements in a
 * @param budget_ms Target duration
 * @return A value derived from the data (keeps the loop from being elided)
 */
static double Compute(double *a, size_t n, double budget_ms) {
  const double end = Now() + budget_ms * 1e-3;
  double acc = 0.0;
  size_t i = 0;
  do {
    for (size_t k = 0; k < 65536; ++k, ++i) {
      if (i >= n) i = 0;
      a[i] = a[i] * 1.0000001 + 0.5;
      acc += a[i];
    }
  } while (Now() < end);
  return acc;
}

/**
 * Halo exchange with both ring neighbours plus one allreduce.
 * @param send Send buffer (halo bytes)
 * @param recv Receive buffer (halo bytes)
 * @param bytes Halo size
 * @param rank This rank
 * @param size Communicator size
 * @param local Value contributed to the allreduce
 * @return The allreduced sum
 */
static double Exchange(char *send, char *recv, size_t bytes, int rank,
                       int size, double local) {
  const int right = (rank + 1) % size;
  const int left = (rank + size - 1) % size;
  MPI_Sendrecv(send, (int)bytes, MPI_BYTE, right, 0, recv, (int)bytes,
               MPI_BYTE, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(send, (int)bytes, MPI_BYTE, left, 1, recv, (int)bytes,
               MPI_BYTE, right, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return global;
}

/** qsort comparator for doubles. */
static int CmpDouble(const void *x, const void *y) {
  const double a = *(const double *)x, b = *(const double *)y;
  return (a > b) - (a < b);
}

/**
 * Gather per-iteration times and print the summary on rank 0.
 * @param iters Iterations this rank completed
 * @param iter_ms Mean iteration time on this rank
 * @param wall_s Wall time of the loop
 * @param rank This rank
 * @param size Communicator size
 */
static void Report(long iters, double iter_ms, double wall_s, int rank,
                   int size) {
  long min_it = 0, max_it = 0;
  MPI_Reduce(&iters, &min_it, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&iters, &max_it, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
  double *all = rank == 0 ? malloc(sizeof(double) * (size_t)size) : NULL;
  MPI_Gather(&iter_ms, 1, MPI_DOUBLE, all, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    qsort(all, (size_t)size, sizeof(double), CmpDouble);
    printf("mpi_stress ranks=%d iters=%ld..%ld iter_ms_p50=%.2f wall_s=%.1f\n",
           size, min_it, max_it, all[size / 2], wall_s);
    fflush(stdout);
    free(all);
  }
}

/**
 * Entry point: allocate, loop compute + exchange until stopped, report.
 * @param argc Argument count
 * @param argv See ParseArgs
 * @return 0 on success
 */
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  Options opts = {0.0, 256, 256, 50.0};
  ParseArgs(argc, argv, &opts);
  signal(SIGTERM, OnSignal);
  signal(SIGINT, OnSignal);

  const size_t n = opts.mem_mb * 1024 * 1024 / sizeof(double);
  double *a = malloc(n * sizeof(double));
  const size_t halo = opts.halo_kb * 1024;
  char *send = malloc(halo), *recv = malloc(halo);
  if (a == NULL || send == NULL || recv == NULL) {
    fprintf(stderr, "mpi_stress: rank %d out of memory\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  for (size_t i = 0; i < n; ++i) a[i] = (double)(i % 1000);  // touch pages
  memset(send, rank & 0xff, halo);

  const double t0 = Now();
  long iters = 0;
  double acc = 0.0;
  int stop = 0;
  while (!stop) {
    acc += Compute(a, n, opts.compute_ms);
    acc = Exchange(send, recv, halo, rank, size, acc) * 1e-9;
    ++iters;
    // Every rank must agree to stop in the same iteration.
    int want = g_stop || (opts.duration_s > 0 && Now() - t0 >= opts.duration_s);
    MPI_Allreduce(&want, &stop, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  }
  const double wall = Now() - t0;
  Report(iters, iters ? wall * 1e3 / (double)iters : 0.0, wall, rank, size);
  free(a);
  free(send);
  free(recv);
  MPI_Finalize();
  return acc == 12345.678 ? 1 : 0;
}
