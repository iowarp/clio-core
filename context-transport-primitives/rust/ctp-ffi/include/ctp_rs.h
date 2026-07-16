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

#ifndef CTP_RS_FFI_H_
#define CTP_RS_FFI_H_

/*
 * C ABI of the ctp-rs Rust workspace (issue #756). Hand-maintained; must
 * stay in sync with rust/ctp-ffi/src/lib.rs.
 *
 * Conventions:
 *  - strings are NUL-terminated UTF-8; Rust copies inputs immediately.
 *  - strings RETURNED by these functions are Rust-allocated and must be
 *    released with ctp_rs_string_free (never free()/delete).
 *  - no Rust panic crosses this boundary; failures surface as NULL / 0.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque child-process handle (SystemInfo::SpawnedProcess parity). */
typedef struct CtpRsChild CtpRsChild;

/** ABI version of the library (bumped on breaking changes). */
uint32_t ctp_rs_abi_version(void);

/** Free a string returned by any ctp_rs_* function. NULL is a no-op. */
void ctp_rs_string_free(char *s);

/* --- Environment ------------------------------------------------------- */
char *ctp_rs_getenv(const char *name, size_t max_size);
void ctp_rs_setenv(const char *name, const char *value, int32_t overwrite);
void ctp_rs_unsetenv(const char *name);

/* --- Host / process / thread introspection ------------------------------ */
uint64_t ctp_rs_get_cpu_count(void);
uint64_t ctp_rs_get_page_size(void);
uint32_t ctp_rs_get_pid(void);
uint64_t ctp_rs_get_tid(void);
uint64_t ctp_rs_get_ram_capacity(void);
char *ctp_rs_get_hostname(void);
char *ctp_rs_get_home_dir(void);
int32_t ctp_rs_is_process_alive(uint32_t pid);
void ctp_rs_sleep_for_us(uint64_t us);
void ctp_rs_yield_thread(void);
uint64_t ctp_rs_thread_cpu_time_ns(void);
void ctp_rs_set_current_thread_name(const char *name);

/* --- Child processes ----------------------------------------------------
 * Spawn exe with argc args; returns NULL on failure. The handle must be
 * released with ctp_rs_child_free (which does NOT terminate the child).
 */
CtpRsChild *ctp_rs_spawn_process(const char *exe, const char *const *argv,
                                 size_t argc);
uint32_t ctp_rs_child_pid(CtpRsChild *proc);
int32_t ctp_rs_is_child_running(CtpRsChild *proc);
void ctp_rs_terminate_child(CtpRsChild *proc, uint64_t grace_ms);
void ctp_rs_child_free(CtpRsChild *proc);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CTP_RS_FFI_H_ */
