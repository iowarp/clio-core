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

/*
 * Programmer:  Kimmy Mu
 *              April 2021
 *
 * Purpose: The public header file for the Hermes driver.
 */
#ifndef H5FDhermes_H
#define H5FDhermes_H

#include <dlfcn.h>
#include <hdf5.h>
#include <stdio.h>

#include <hermes_shm/util/logging.h>

#define H5FD_WRP_CTE_NAME  "hdf5_hermes_vfd"
#define H5FD_WRP_CTE_VALUE ((H5FD_class_value_t)(3200))

#define WRP_CTE_FORWARD_DECL(func_, ret_, args_) \
  typedef ret_(*real_t_##func_##_) args_;       \
  ret_(*real_##func_##_) args_ = NULL;

#define MAP_OR_FAIL(func_)                                                  \
  if (!(real_##func_##_)) {                                                 \
    real_##func_##_ = (real_t_##func_##_)dlsym(RTLD_NEXT, #func_);          \
    if (!(real_##func_##_)) {                                               \
      HLOG(kError, "HERMES Adapter failed to map symbol: {}", #func_);      \
      exit(1);                                                              \
    }                                                                       \
  }

#ifdef __cplusplus
extern "C" {
#endif

hid_t H5FD_hermes_init();
herr_t H5Pset_fapl_hermes(hid_t fapl_id, hbool_t persistence, size_t page_size);

H5PL_type_t H5PLget_plugin_type(void);
const void* H5PLget_plugin_info(void);

WRP_CTE_FORWARD_DECL(H5_init_library, herr_t, ());
WRP_CTE_FORWARD_DECL(H5_term_library, herr_t, ());

#ifdef __cplusplus
}
#endif

#endif /* end H5FDhermes_H */
