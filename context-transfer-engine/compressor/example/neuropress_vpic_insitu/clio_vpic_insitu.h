/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file clio_vpic_insitu.h
 * @brief C entry points a VPIC deck calls to hand its field arrays to Clio
 * from inside the simulation, with no file in between.
 *
 * The deck includes THIS header and nothing else of Clio's. That is
 * deliberate: a VPIC deck is compiled by vpic-kokkos's generated `bin/vpic`
 * script, which drives nvcc_wrapper over deck/main.cc with the deck textually
 * included. Clio's C++ headers pull in coroutines and its runtime templates,
 * which have no business going through nvcc for a physics deck. Six extern "C"
 * declarations and <stddef.h> do.
 *
 * WHAT A VPIC DECK HANDS OVER. VPIC-Kokkos keeps the field array as
 * `Kokkos::View<float*[FIELD_VAR_COUNT]>` with no explicit layout, so in a
 * CUDA build it takes the default LayoutLeft: element (voxel v, variable m)
 * sits at `v + m*n_voxels`. Every variable is therefore ALREADY a contiguous
 * device array of n_voxels floats at `k_f_d.data() + m*n_voxels`, and the
 * hand-over needs no de-interleaving at all -- measured on this tree's deck as
 * stride0=1, stride1=n_voxels. (On a host/OpenMP build the default is
 * LayoutRight and the same view is interleaved; such a build must not use
 * these entry points without transposing first, which is why stage() takes a
 * plain contiguous extent and refuses anything it cannot verify is device
 * memory.)
 *
 * REFUSALS. Every precondition below exits the process with a named code
 * rather than returning an error the deck would have to check. A deck that
 * silently fell back to host staging, or to storing nothing, would still
 * print a plausible run.
 */

#ifndef CLIO_VPIC_INSITU_H_
#define CLIO_VPIC_INSITU_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Exit codes used for refusals. Mirrors the Nyx and LAMMPS examples. */
#define CLIO_VPIC_EXIT_PRECONDITION 3 /* the caller promised something untrue */
#define CLIO_VPIC_EXIT_ALLOC 4        /* device staging could not be allocated */
#define CLIO_VPIC_EXIT_NOT_DEVICE 5   /* the memory is not device memory */
#define CLIO_VPIC_EXIT_TOPOLOGY 6     /* the runtime topology cannot carry this */

/**
 * @brief Bring up the client, the compressor and the tag. Idempotent.
 * @return 0 on success, non-zero if Clio could not be initialised.
 *
 * Reads CLIO_VPIC_TAG, CLIO_VPIC_CHUNK, CLIO_VPIC_POOL, CLIO_VPIC_REPORT,
 * CLIO_VPIC_RAW_DIR and CLIO_VPIC_VERIFY. Refuses with EXIT_TOPOLOGY unless
 * the Clio runtime is hosted in this process (CLIO_WITH_RUNTIME=1).
 */
int clio_vpic_insitu_begin(void);

/** @brief begin(), told the MPI rank and size by the deck rather than by an
 *  environment variable, so the two cannot disagree. One runtime, one store
 *  and one port per rank. */
int clio_vpic_insitu_begin_mpi(long long rank, long long nprocs);

/** @brief Open a frame. `step` and `sim_time` are recorded, not interpreted. */
int clio_vpic_insitu_frame_begin(long long step, double sim_time);

/**
 * @brief Stage one contiguous device array as blobs named
 * `<blob_prefix>/chunk_<i>`, chunked at CLIO_VPIC_CHUNK bytes.
 *
 * @param blob_prefix   e.g. "ex/step_00025"
 * @param dev_ptr       device pointer to the first element
 * @param elem_bytes    4 (float32) or 8 (float64); sets the NeuroPress type
 * @param n_elems       elements in the array
 *
 * The bytes are copied D2D into a Clio-registered staging backend and handed
 * to the compressor as device memory; they never touch host memory on the
 * path. Refuses if `dev_ptr` is not device memory.
 */
int clio_vpic_insitu_stage(const char *blob_prefix, const void *dev_ptr,
                           long long elem_bytes, long long n_elems);

/** @brief Wait for every task of the open frame. Fields may then advance. */
int clio_vpic_insitu_frame_end(void);

/** @brief Drain, report, optionally verify, and tear the client down.
 *  @return 0, or 1 if --verify found a blob that did not round-trip. */
int clio_vpic_insitu_end(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // CLIO_VPIC_INSITU_H_
