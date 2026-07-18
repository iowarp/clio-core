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
 * Purpose: The public header file for the Clio driver.
 */
#ifndef H5FDclio_H
#define H5FDclio_H

#include <hdf5.h>

#define H5FD_CLIO_NAME  "clio_vfd"
#define H5FD_CLIO_VALUE ((H5FD_class_value_t)(3200))

#ifdef __cplusplus
extern "C" {
#endif

hid_t H5FD_clio_init();
/* Select the CLIO VFD and attach its tiering policy.
 *   cache_enabled: populate the CTE cache tier (false = native-only, which
 *                  avoids the current write-amplification of the populate-only
 *                  cache). */
herr_t H5Pset_fapl_clio(hid_t fapl_id, hbool_t cache_enabled);

H5PL_type_t H5PLget_plugin_type(void);
const void* H5PLget_plugin_info(void);

#ifdef __cplusplus
}
#endif

#endif /* end H5FDclio_H */
