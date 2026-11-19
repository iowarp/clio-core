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
 * Minimal ChiMod stand-in for test_chimod_abi_check.cc. It exports the
 * ChiMod entry points by hand (no container: alloc/new return nullptr, which
 * LoadChiMod tolerates) so one source can play three roles, picked by
 * FAKE_ABI_MODE:
 *   0  get_chimod_abi() returns the runtime's fingerprint  ("abi_fake_match")
 *   1  get_chimod_abi() returns a different fingerprint    ("abi_fake_stale")
 *   2  no get_chimod_abi() at all, like a pre-check build  ("abi_fake_none")
 *   3  same as 2, but links no module, so dlsym finds no    ("abi_fake_bare")
 *      get_chimod_abi() in any dependency either
 */

#include "clio_runtime/container.h"

extern "C" {

/** @return nullptr: the fake module has no container to allocate. */
clio::run::Container *alloc_chimod() { return nullptr; }

/**
 * @param pool_id Unused
 * @param pool_name Unused
 * @return nullptr: the fake module has no container to create.
 */
clio::run::Container *new_chimod(const clio::run::PoolId *pool_id,
                                 const char *pool_name) {
  (void)pool_id;
  (void)pool_name;
  return nullptr;
}

/** @return The module name for this FAKE_ABI_MODE. */
const char *get_chimod_name() {
#if FAKE_ABI_MODE == 0
  return "abi_fake_match";
#elif FAKE_ABI_MODE == 1
  return "abi_fake_stale";
#elif FAKE_ABI_MODE == 2
  return "abi_fake_none";
#else
  return "abi_fake_bare";
#endif
}

/** @param container Always nullptr here; nothing to free. */
void destroy_chimod(clio::run::Container *container) { (void)container; }

#if FAKE_ABI_MODE == 0
/** @return The fingerprint of the headers this library was built against. */
const char *get_chimod_abi() { return CLIO_RUNTIME_ABI_FINGERPRINT; }
#elif FAKE_ABI_MODE == 1
/** @return A fingerprint no real build produces (stale-module stand-in). */
const char *get_chimod_abi() { return "stale-0000000000"; }
#endif

}  // extern "C"
