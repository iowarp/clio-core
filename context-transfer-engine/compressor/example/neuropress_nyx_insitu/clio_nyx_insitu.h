/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file clio_nyx_insitu.h
 * @brief The C ABI Nyx dlopens to hand its GPU-resident hydro state to Clio.
 *
 * This is the whole contract between the simulation and the compressor. It is
 * deliberately plain C over plain integers and a void*, for two reasons:
 *
 *   1. Nyx's side of the patch is the only code that includes AMReX headers,
 *      so libclio_nyx_insitu.so needs neither AMReX nor a matching AMReX
 *      build. It links Clio and libcudart, nothing else.
 *   2. Nyx resolves these symbols with dlopen/dlsym at first use, so the Nyx
 *      patch adds no build-system dependency at all -- no CMake change, no
 *      Clio in Nyx's link line. Exactly the shape the HDF5 VOL uses.
 *
 * The pointer handed to clio_nyx_insitu_stage() is `fab.dataPtr(comp)` --
 * AMReX device memory, allocated by The_Arena(), which in a CUDA build is a
 * plain cudaMalloc arena. The adapter checks that per blob and REFUSES if it
 * is anything else; it never stages through host memory as a fallback.
 *
 * Every entry point returns 0 on success. A non-zero return means the call
 * did nothing. Refusals that would otherwise be silently survivable -- a host
 * pointer where device memory was promised, a failed device allocation --
 * do not return at all: they print a named refusal and _exit() with a
 * distinct code, because continuing would produce blobs that pass every
 * round-trip check while not being what was asked for.
 */

#ifndef CLIO_NYX_INSITU_H_
#define CLIO_NYX_INSITU_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Exit codes used for refusals. Mirrors the LAMMPS device example. */
#define CLIO_NYX_EXIT_PRECONDITION 3 /* the caller promised something untrue */
#define CLIO_NYX_EXIT_ALLOC 4        /* device staging could not be allocated */
#define CLIO_NYX_EXIT_NOT_DEVICE 5   /* the memory is not device memory */

/**
 * Bring up the Clio client (and, with CLIO_WITH_RUNTIME=1, the runtime) in
 * the calling process and open the tag. Idempotent. Reads its configuration
 * from the environment -- see the README:
 *
 *   CLIO_NYX_TAG          CTE tag name              [nyx_insitu]
 *   CLIO_NYX_CHUNK        bytes per compressor call [4194304]; 0 = whole field
 *   CLIO_NYX_POOL         compressor pool id        [512.0]
 *   CLIO_NYX_REPORT       per-chunk CSV, the file the cold read replays
 *   CLIO_NYX_RAW_DIR      also write each staged blob's bytes here
 *   CLIO_NYX_VERIFY       1 = read every blob back in this process at the end
 *
 * @return 0 on success, non-zero if the client could not be initialised.
 */
int clio_nyx_insitu_begin(void);

/**
 * Open a frame. Everything staged until clio_nyx_insitu_frame_end() belongs
 * to timestep @p step. Frames are drained one at a time, so a frame's device
 * staging slots are reused by the next frame.
 */
int clio_nyx_insitu_frame_begin(long long step, double sim_time);

/**
 * Stage one component of one FArrayBox.
 *
 * @param blob_prefix  names the blob, e.g. "density/step_00010/fab0000";
 *                     "/chunk_N" is appended per chunk.
 * @param dev_base     fab.dataPtr(comp) -- the start of this component's
 *                     GROWN box in AMReX device memory.
 * @param elem_bytes   sizeof(amrex::Real): 4 or 8.
 * @param gx,gy,gz     dimensions of the grown (ghost-inclusive) box.
 * @param ox,oy,oz     origin of the region to store, in cells from the grown
 *                     box's low corner. Pass 0,0,0 to store the whole FAB.
 * @param nx,ny,nz     dimensions of the region to store.
 *
 * The region is extracted on the GPU into a device scratch buffer (a single
 * cudaMemcpy3DAsync on the adapter's own non-blocking stream), then each
 * chunk of it is copied into its own Clio-registered kDeviceMem backend and
 * submitted. The payload never becomes host bytes on the compressor path.
 *
 * @return 0 on success. Refusals _exit() rather than returning.
 */
int clio_nyx_insitu_stage(const char *blob_prefix, const void *dev_base,
                          long long elem_bytes, long long gx, long long gy,
                          long long gz, long long ox, long long oy,
                          long long oz, long long nx, long long ny,
                          long long nz);

/** Wait for every task of the open frame and record what the compressor did. */
int clio_nyx_insitu_frame_end(void);

/**
 * Drain, print the summary, write the CSV, optionally verify, and release the
 * device staging pool. Safe to call more than once; the second call is a
 * no-op.
 *
 * @return 0 if every blob was stored (and, with CLIO_NYX_VERIFY=1, read back
 *         bit-exact); non-zero otherwise.
 */
int clio_nyx_insitu_end(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // CLIO_NYX_INSITU_H_
