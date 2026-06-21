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
 * Admin ChiMod singleton
 *
 * Provides global access to the admin ChiMod client via CLIO_ADMIN macro.
 * The admin container is automatically created by the runtime and accessed
 * by clients without requiring explicit Create() calls.
 */

#ifndef CLIO_RUNTIME_INCLUDE_ADMIN_H_
#define CLIO_RUNTIME_INCLUDE_ADMIN_H_

#include "clio_runtime/api.h"
#include "clio_runtime/types.h"

// Forward declaration to avoid circular dependency
namespace clio::run::admin {
class Client;
}

// Global pointer variable declaration for Admin singleton
CLIO_RUN_DEFINE_GLOBAL_PTR_VAR_H(clio::run::admin::Client, g_admin);

// Macro for accessing the Admin singleton using global pointer variable
#define CLIO_ADMIN CTP_GET_GLOBAL_PTR_VAR(::clio::run::admin::Client, g_admin)
// Backward-compat alias (clio_run rebrand). External code that still
// uses the legacy CHI_* spelling keeps working unchanged.

#endif  // CLIO_RUNTIME_INCLUDE_ADMIN_H_
