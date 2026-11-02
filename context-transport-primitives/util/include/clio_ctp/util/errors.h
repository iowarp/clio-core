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

#ifndef CTP_ERRORS_H
#define CTP_ERRORS_H

#ifdef __cplusplus

#include "clio_ctp/util/error.h"

namespace ctp {
const Error MEMORY_BACKEND_REPEATED(
    "Attempted to register two backends "
    "with the same id");
const Error TOO_MANY_ALLOCATORS("Too many allocators");
const Error NOT_IMPLEMENTED("{} not implemented");

const Error SHMEM_CREATE_FAILED("Failed to allocate SHMEM");
const Error SHMEM_RESERVE_FAILED("Failed to reserve SHMEM");
const Error SHMEM_NOT_SUPPORTED("Attempting to deserialize a non-shm backend");
const Error MEMORY_BACKEND_CREATE_FAILED("Failed to load memory backend");
const Error MEMORY_BACKEND_NOT_FOUND("Failed to find the memory backend");
const Error OUT_OF_MEMORY(
    "could not allocate memory of size {} from heap of size {}");
const Error INVALID_FREE("could not free memory");
const Error DOUBLE_FREE("Freeing the same memory twice: {}!");

const Error IPC_ARGS_NOT_SHM_COMPATIBLE("Args are not compatible with SHM");

const Error UNORDERED_MAP_CANT_FIND("Could not find key in unordered_map");
const Error KEY_SET_OUT_OF_BOUNDS("Too many keys in the key set");

const Error ARGPACK_INDEX_OUT_OF_BOUNDS("Argpack index out of bounds");
}  // namespace ctp

#endif

#endif  // CTP_ERRORS_H
