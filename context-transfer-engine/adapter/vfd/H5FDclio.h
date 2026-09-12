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
/* Declares H5PLget_plugin_type/H5PLget_plugin_info with H5PLUGIN_DLL, which is
 * the export attribute HDF5's plugin loader requires on Windows. */
#include <H5PLextern.h>

#define H5FD_CLIO_NAME  "clio_vfd"
#define H5FD_CLIO_VALUE ((H5FD_class_value_t)(3200))

/* Nothing leaves a Windows DLL unless it is exported, so the driver's public
 * entry points need an attribute that flips between export (while building
 * clio_vfd) and import (while consuming it). CLIO_VFD_BUILD is defined on the
 * clio_vfd target only; everything else -- the test binaries, any application
 * calling H5Pset_fapl_clio directly -- gets the import side.
 *
 * Applies to the COUNTERS below as well, and that part is not optional: data
 * symbols are the one thing CMake's WINDOWS_EXPORT_ALL_SYMBOLS cannot rescue,
 * because a consumer has to see __declspec(dllimport) at the declaration to
 * emit an indirect reference at all. */
#ifdef _WIN32
#  ifdef CLIO_VFD_BUILD
#    define CLIO_VFD_API __declspec(dllexport)
#  else
#    define CLIO_VFD_API __declspec(dllimport)
#  endif
#else
#  define CLIO_VFD_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

CLIO_VFD_API hid_t H5FD_clio_init();
/* Select the CLIO VFD and attach its tiering policy.
 *   cache_enabled: populate the CTE cache tier (false = native-only, which
 *                  avoids the current write-amplification of the populate-only
 *                  cache, and does not require a running CLIO runtime).
 *
 * The same tier can be switched off without touching source by setting
 * CLIO_VFD_CACHE=0 (also "off"/"false"/"no") in the environment -- the shape
 * VFD_VOL_TECHNICAL_GOALS.md §1.4 asks for, and the only shape available to an
 * application loaded via HDF5_DRIVER=clio_vfd. The env var is an opt-OUT only:
 * it can force the cache off, but never on over an explicit
 * H5Pset_fapl_clio(fapl, false). */
CLIO_VFD_API herr_t H5Pset_fapl_clio(hid_t fapl_id, hbool_t cache_enabled);
/* Read back the tiering policy from a FAPL that selects this driver. Reports
 * the default when no driver-info block was attached, so it always describes
 * what an open through this FAPL would actually do. */
CLIO_VFD_API herr_t H5Pget_fapl_clio(hid_t fapl_id, hbool_t *cache_enabled /*out*/);

/* Vector-I/O counters, for tests asserting that a pattern really did coalesce
 * into read_vector/write_vector rather than degenerating into per-block calls.
 *
 * Declared HERE rather than as block-scope `extern unsigned long` at each use
 * site, which is how the tests used to reach them: a local extern cannot carry
 * __declspec(dllimport), so on Windows it links against nothing. */
CLIO_VFD_API extern unsigned long H5FDclio_read_vector_calls_g;
CLIO_VFD_API extern unsigned long H5FDclio_write_vector_calls_g;
CLIO_VFD_API extern unsigned long H5FDclio_vec_max_span_g;

/* H5PLget_plugin_type/H5PLget_plugin_info are declared by <H5PLextern.h>
 * (included above) with H5PLUGIN_DLL. Re-declaring them bare here made the
 * two declarations disagree about linkage, which MSVC rejects outright
 * (C2375) -- and an entry point without the export attribute is one HDF5's
 * plugin loader cannot find on Windows. */

#ifdef __cplusplus
}
#endif

#endif /* end H5FDclio_H */
