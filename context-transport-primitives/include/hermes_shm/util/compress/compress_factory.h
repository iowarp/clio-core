/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_UTIL_COMPRESS_COMPRESS_FACTORY_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_UTIL_COMPRESS_COMPRESS_FACTORY_H_

#if HSHM_ENABLE_COMPRESS

#include "blosc.h"
#include "brotli.h"
#include "bzip2.h"
#include "lz4.h"
#include "lzma.h"
#include "lzo.h"
#include "snappy.h"
#include "zlib.h"
#include "zstd.h"
#if HSHM_HAS_LIBPRESSIO
#include "libpressio.h"
#endif

#endif  // HSHM_ENABLE_COMPRESS

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_UTIL_COMPRESS_COMPRESS_FACTORY_H_
