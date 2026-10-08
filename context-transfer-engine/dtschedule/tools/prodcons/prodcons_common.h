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
 * @file prodcons_common.h
 * Shared helpers for the DTSchedule producer-consumer workload
 * (heat_producer.c, checksum_consumer.c): option parsing, file naming in
 * the flat CTE namespace, timing, and whole-buffer POSIX I/O.
 */

#ifndef DTSCHEDULE_PRODCONS_COMMON_H_
#define DTSCHEDULE_PRODCONS_COMMON_H_

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/** Options shared by both programs (unused fields are ignored). */
typedef struct {
  const char *run;     /**< Run name: files are /clio::<run>__stepS_rankR.dat */
  int steps;           /**< Output steps */
  int nx;              /**< Grid columns per producer rank */
  int ny;              /**< Grid rows per producer rank */
  int iters_per_step;  /**< Jacobi sweeps between outputs */
  int producers;       /**< Producer rank count (consumer side) */
  double noise;        /**< Relative noise added to the output field */
  int passes;          /**< Consumer: compute passes over each file */
  const char *payload; /**< Producer: directory of data files to emit
                            instead of the solver field (NULL = field) */
  int write_pending;   /**< API backend: files left in flight after a step's
                            write (0 = the write returns once stored) */
} PcOptions;

/**
 * Parse --key value pairs into opts (defaults must be set by the caller).
 * @param argc Argument count
 * @param argv Arguments
 * @param opts Options to fill
 */
static inline void PcParse(int argc, char **argv, PcOptions *opts) {
  for (int i = 1; i + 1 < argc; i += 2) {
    const char *k = argv[i], *v = argv[i + 1];
    if (!strcmp(k, "--run")) opts->run = v;
    else if (!strcmp(k, "--steps")) opts->steps = atoi(v);
    else if (!strcmp(k, "--nx")) opts->nx = atoi(v);
    else if (!strcmp(k, "--ny")) opts->ny = atoi(v);
    else if (!strcmp(k, "--iters-per-step")) opts->iters_per_step = atoi(v);
    else if (!strcmp(k, "--producers")) opts->producers = atoi(v);
    else if (!strcmp(k, "--noise")) opts->noise = atof(v);
    else if (!strcmp(k, "--passes")) opts->passes = atoi(v);
    else if (!strcmp(k, "--payload")) opts->payload = v;
    else if (!strcmp(k, "--write-pending")) opts->write_pending = atoi(v);
  }
}

/** Monotonic seconds. */
static inline double PcNow(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/**
 * Path of one rank's output for one step, in the flat CTE namespace.
 * @param buf Output buffer
 * @param len Buffer length
 * @param run Run name
 * @param step Step index
 * @param rank Producer rank (-1 = the step's completion marker)
 */
static inline void PcPath(char *buf, size_t len, const char *run, int step,
                          int rank) {
  // DT_PRODCONS_PREFIX overrides the CTE root (e.g. a scratch directory to
  // measure the data offline); default is the CTE namespace root.
  const char *env = getenv("DT_PRODCONS_PREFIX");
  const char *prefix = (env != NULL && env[0] != '\0') ? env : "/clio::";
  if (rank < 0) {
    snprintf(buf, len, "%s%s__step%d.done", prefix, run, step);
  } else {
    snprintf(buf, len, "%s%s__step%d_rank%d.dat", prefix, run, step, rank);
  }
}

/**
 * Name of one rank's output for one step as a CTE tag (API backend): the
 * flat-namespace file name without the "/clio::" root.
 * @param buf Output buffer
 * @param len Buffer length
 * @param run Run name
 * @param step Step index
 * @param rank Producer rank
 */
static inline void PcName(char *buf, size_t len, const char *run, int step,
                          int rank) {
  snprintf(buf, len, "%s__step%d_rank%d.dat", run, step, rank);
}

/**
 * Write a whole buffer to path (create/truncate).
 * @param path Destination
 * @param data Bytes
 * @param bytes Length
 * @return 0 on success, errno otherwise
 */
static inline int PcWriteFile(const char *path, const void *data,
                              size_t bytes) {
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) return errno;
  const char *p = (const char *)data;
  size_t done = 0;
  while (done < bytes) {
    ssize_t n = write(fd, p + done, bytes - done);
    if (n <= 0) {
      int e = errno ? errno : EIO;
      close(fd);
      return e;
    }
    done += (size_t)n;
  }
  return close(fd) == 0 ? 0 : errno;
}

/**
 * Read up to bytes from path into data.
 * @param path Source
 * @param data Destination buffer
 * @param bytes Capacity
 * @param got Bytes actually read
 * @return 0 on success, errno otherwise (ENOENT if the file is absent)
 */
static inline int PcReadFile(const char *path, void *data, size_t bytes,
                             size_t *got) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) return errno;
  char *p = (char *)data;
  size_t done = 0;
  while (done < bytes) {
    ssize_t n = read(fd, p + done, bytes - done);
    if (n < 0) {
      int e = errno;
      close(fd);
      return e;
    }
    if (n == 0) break;
    done += (size_t)n;
  }
  *got = done;
  close(fd);
  return 0;
}

#endif  // DTSCHEDULE_PRODCONS_COMMON_H_
