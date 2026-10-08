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
 * @file heat_producer.c
 * Producer half of the DTSchedule producer-consumer workload.
 *
 * A 2D heat-diffusion solver (5-point Jacobi) decomposed by rows across MPI
 * ranks. Run one rank per hardware thread so it holds every core. After
 * every --iters-per-step sweeps each rank writes its slab (nx x ny doubles,
 * plus --noise relative noise, like sensor/numerical noise in real output)
 * to /clio::<run>__step<s>_rank<r>.dat; after a barrier rank 0 writes the
 * step marker /clio::<run>__step<s>.done that the consumer waits for.
 *
 * Summary (rank 0):
 *   heat_producer ranks=N steps=S step_mb=M compute_s=C write_s=W wall_s=T
 */

#include <dirent.h>
#include <math.h>
#include <mpi.h>
#include <sys/stat.h>

#include "prodcons_common.h"
#ifdef PC_USE_CTE_API
#include "pc_cte_api.h"
#endif

/** Slab with one halo row above and below: (ny + 2) rows of nx doubles. */
typedef struct {
  int nx, ny;
  double *cur, *next;
} Slab;

/**
 * Allocate a slab and seed it with a smooth field plus a few hot spots.
 * @param s Slab to fill
 * @param nx Columns
 * @param ny Interior rows
 * @param rank Seeds the hot spots so ranks differ
 * @return 0 on success, -1 when out of memory
 */
static int SlabInit(Slab *s, int nx, int ny, int rank) {
  const size_t n = (size_t)nx * (size_t)(ny + 2);
  s->nx = nx;
  s->ny = ny;
  s->cur = malloc(n * sizeof(double));
  s->next = malloc(n * sizeof(double));
  if (s->cur == NULL || s->next == NULL) return -1;
  srand(1234u + (unsigned)rank);
  for (int y = 0; y < ny + 2; ++y) {
    for (int x = 0; x < nx; ++x) {
      s->cur[(size_t)y * nx + x] =
          20.0 + 5.0 * sin(0.01 * x) * cos(0.013 * (y + rank * ny));
    }
  }
  for (int k = 0; k < 8; ++k) {
    const int x = rand() % nx, y = 1 + rand() % ny;
    s->cur[(size_t)y * nx + x] = 400.0;
  }
  memcpy(s->next, s->cur, n * sizeof(double));
  return 0;
}

/**
 * Exchange halo rows with the ranks above and below (non-periodic).
 * @param s Slab
 * @param rank This rank
 * @param size Rank count
 */
static void SlabHalo(Slab *s, int rank, int size) {
  const int up = rank > 0 ? rank - 1 : MPI_PROC_NULL;
  const int down = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;
  double *first = s->cur + s->nx, *last = s->cur + (size_t)s->ny * s->nx;
  double *top = s->cur, *bottom = s->cur + (size_t)(s->ny + 1) * s->nx;
  MPI_Sendrecv(first, s->nx, MPI_DOUBLE, up, 0, bottom, s->nx, MPI_DOUBLE,
               down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(last, s->nx, MPI_DOUBLE, down, 1, top, s->nx, MPI_DOUBLE, up,
               1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

/**
 * One Jacobi sweep over the interior, then swap buffers.
 * @param s Slab
 */
static void SlabSweep(Slab *s) {
  const int nx = s->nx;
  for (int y = 1; y <= s->ny; ++y) {
    const double *a = s->cur + (size_t)(y - 1) * nx;
    const double *b = s->cur + (size_t)y * nx;
    const double *c = s->cur + (size_t)(y + 1) * nx;
    double *o = s->next + (size_t)y * nx;
    o[0] = b[0];
    o[nx - 1] = b[nx - 1];
    for (int x = 1; x < nx - 1; ++x) {
      o[x] = 0.25 * (a[x] + c[x] + b[x - 1] + b[x + 1]);
    }
  }
  double *t = s->cur;
  s->cur = s->next;
  s->next = t;
}

/**
 * Copy the interior to out with multiplicative noise of amplitude noise.
 * @param s Slab
 * @param out Destination (nx * ny doubles)
 * @param noise Relative noise amplitude (0 = exact field)
 */
static void SlabSnapshot(const Slab *s, double *out, double noise) {
  const size_t n = (size_t)s->nx * (size_t)s->ny;
  const double *in = s->cur + s->nx;
  for (size_t i = 0; i < n; ++i) {
    const double r = (double)rand() / (double)RAND_MAX - 0.5;
    out[i] = in[i] * (1.0 + noise * r);
  }
}

/**
 * Load this rank's payload file (the rank-th regular file of dir, sorted by
 * name, modulo the count) into memory.
 * @param dir Directory of data files
 * @param rank This rank
 * @param len Output: bytes loaded
 * @return Malloc'd buffer, or NULL on failure
 */
static char *LoadPayload(const char *dir, int rank, size_t *len) {
  struct dirent **list = NULL;
  int n = scandir(dir, &list, NULL, alphasort);
  char path[1024] = {0};
  int files = 0;
  for (int i = 0; i < n; ++i) {
    if (list[i]->d_name[0] != '.') ++files;
  }
  for (int i = 0, k = 0; i < n && files > 0; ++i) {
    if (list[i]->d_name[0] == '.') continue;
    if (k++ == rank % files) {
      snprintf(path, sizeof(path), "%s/%s", dir, list[i]->d_name);
    }
  }
  for (int i = 0; i < n; ++i) free(list[i]);
  free(list);
  struct stat st;
  if (path[0] == '\0' || stat(path, &st) != 0) return NULL;
  char *buf = malloc((size_t)st.st_size);
  size_t got = 0;
  if (buf == NULL || PcReadFile(path, buf, (size_t)st.st_size, &got) != 0) {
    free(buf);
    return NULL;
  }
  *len = got;
  return buf;
}

/**
 * Fill out with a window of the payload, offset by step and rank so
 * consecutive steps carry different bytes.
 * @param payload Payload data
 * @param plen Payload length (>= bytes)
 * @param out Destination
 * @param bytes Window size
 * @param step Step index
 * @param rank This rank
 */
static void PayloadWindow(const char *payload, size_t plen, char *out,
                          size_t bytes, int step, int rank) {
  const size_t span = plen - bytes + 1;
  const size_t off = (((size_t)step * 7919u + (size_t)rank * 104729u) *
                      4096u) % span & ~(size_t)4095;
  memcpy(out, payload + off, bytes);
}

/**
 * Write this rank's snapshot for a step and, on rank 0, the step marker.
 * @param o Options
 * @param step Step index
 * @param rank This rank
 * @param buf Snapshot
 * @param bytes Snapshot size
 * @return 0 on success, errno otherwise
 */
static int WriteStep(const PcOptions *o, int step, int rank, const double *buf,
                     size_t bytes) {
#ifdef PC_USE_CTE_API
  // Submit and return: the puts drain behind the next compute phase, and
  // each file's own marker tells its consumer it is complete.
  char name[512];
  PcName(name, sizeof(name), o->run, step, rank);
  // write_pending 0: every chunk in flight at once, but the step's write
  // returns only once the file is stored, so data moves while the ranks wait
  // in I/O instead of competing with the next compute phase for cores.
  const int put_rc =
      PcApiPutFile(name, buf, bytes, o->write_pending > 0 ? o->write_pending : 1);
  PcApiDrain(o->write_pending > 0 ? 0 : 1);
  return put_rc;
#endif
  char path[512];
  PcPath(path, sizeof(path), o->run, step, rank);
  int rc = PcWriteFile(path, buf, bytes);
  int all = 0;
  MPI_Allreduce(&rc, &all, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (all == 0 && rank == 0) {
    PcPath(path, sizeof(path), o->run, step, -1);
    all = PcWriteFile(path, &step, sizeof(step));
  }
  MPI_Bcast(&all, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return all;
}

/**
 * Entry point: solve, write a snapshot every step, report.
 * @param argc Argument count
 * @param argv See PcParse (--run --steps --nx --ny --iters-per-step --noise
 *             --payload DIR: emit windows of the files in DIR instead of
 *             the solver field, which barely compresses losslessly)
 * @return 0 on success, 2 on I/O failure
 */
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  PcOptions o = {"prodcons", 10, 2048, 2048, 200, 0, 0.02, 0, NULL, 2};
  PcParse(argc, argv, &o);
#ifdef PC_USE_CTE_API
  if (PcApiInit() != 0) {
    fprintf(stderr, "heat_producer: rank %d: CLIO client init failed\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
#endif
  Slab s;
  const size_t bytes = (size_t)o.nx * (size_t)o.ny * sizeof(double);
  double *snap = malloc(bytes);
  if (SlabInit(&s, o.nx, o.ny, rank) != 0 || snap == NULL) {
    fprintf(stderr, "heat_producer: rank %d out of memory\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  size_t plen = 0;
  char *payload = o.payload ? LoadPayload(o.payload, rank, &plen) : NULL;
  if (o.payload && (payload == NULL || plen < bytes)) {
    fprintf(stderr, "heat_producer: rank %d: payload under %s unusable\n",
            rank, o.payload);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  double compute_s = 0.0, write_s = 0.0;
  const double t0 = PcNow();
  int rc = 0;
  for (int step = 0; step < o.steps && rc == 0; ++step) {
    const double c0 = PcNow();
    for (int it = 0; it < o.iters_per_step; ++it) {
      SlabHalo(&s, rank, size);
      SlabSweep(&s);
    }
    if (payload != NULL) {
      PayloadWindow(payload, plen, (char *)snap, bytes, step, rank);
    } else {
      SlabSnapshot(&s, snap, o.noise);
    }
    const double w0 = PcNow();
    rc = WriteStep(&o, step, rank, snap, bytes);
    compute_s += w0 - c0;
    write_s += PcNow() - w0;
    if (rc != 0 && rank == 0) {
      fprintf(stderr, "heat_producer: step %d write failed: %s\n", step,
              strerror(rc));
    }
  }
  double flush_s = 0.0;
#ifdef PC_USE_CTE_API
  const double f0 = PcNow();
  if (PcApiDrain(1) != 0 && rc == 0) rc = EIO;
  flush_s = PcNow() - f0;
#endif
  if (rank == 0) {
    printf("heat_producer ranks=%d steps=%d step_mb=%.1f compute_s=%.2f "
           "write_s=%.2f flush_s=%.2f wall_s=%.2f rc=%d\n",
           size, o.steps, (double)bytes * size / 1e6, compute_s, write_s,
           flush_s, PcNow() - t0, rc);
    fflush(stdout);
  }
  free(payload);
  free(snap);
  free(s.cur);
  free(s.next);
  MPI_Finalize();
  return rc == 0 ? 0 : 2;
}
