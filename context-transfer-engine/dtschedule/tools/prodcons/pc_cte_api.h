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
 * @file pc_cte_api.h
 * CTE-API I/O backend of the producer-consumer workload (non-transparent
 * variant, built with PC_USE_CTE_API). Each producer file is one CTE tag of
 * 1 MiB chunks named by index, put asynchronously through the pool named by
 * CLIO_CTE_POOL (dtschedule), so every placement and compression decision is
 * dtschedule's and no file-system layer routes the bytes. A small "done"
 * blob in the tag marks a file complete; consumers poll it per file.
 */

#ifndef DTSCHEDULE_PC_CTE_API_H_
#define DTSCHEDULE_PC_CTE_API_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Attach to the running CLIO runtime and initialise the CTE client.
 * @return 0 on success, -1 on failure
 */
int PcApiInit(void);

/**
 * Start writing a whole file: copy it into shared memory and submit every
 * chunk put without waiting. The file's marker is put once all its chunks
 * have completed (see PcApiDrain).
 * @param name Tag name of the file
 * @param data File bytes (may be reused once this returns)
 * @param bytes File size
 * @param max_pending Files allowed in flight; older ones are completed first
 * @return 0 on success, errno-style code otherwise
 */
int PcApiPutFile(const char *name, const void *data, size_t bytes,
                 int max_pending);

/**
 * Complete in-flight files: for each whose chunk puts are all done, put its
 * marker and release its buffer.
 * @param block Nonzero: wait until every file in flight is complete
 * @return Number of chunk or marker puts that failed so far
 */
int PcApiDrain(int block);

/**
 * Wait until a file's marker exists, sleeping between polls.
 * @param name Tag name of the file
 * @param timeout_s Give up after this long
 * @return 0 when present, ETIMEDOUT otherwise
 */
int PcApiWaitFile(const char *name, double timeout_s);

/**
 * Read a whole file: all chunk gets in flight at once.
 * @param name Tag name of the file
 * @param out Destination
 * @param bytes File size (the destination's capacity)
 * @param got Output: bytes read (bytes on success, 0 on failure)
 * @return 0 on success, errno-style code otherwise
 */
int PcApiGetFile(const char *name, void *out, size_t bytes, size_t *got);

#ifdef __cplusplus
}
#endif

#endif  // DTSCHEDULE_PC_CTE_API_H_
