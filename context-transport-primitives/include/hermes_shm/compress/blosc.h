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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Blosc_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Blosc_H_

#if HSHM_ENABLE_COMPRESS

#include <blosc2.h>

#include "compress.h"
#include "hermes_shm/util/singleton.h"

namespace hshm {

class BloscInit {
 public:
  BloscInit() { blosc2_init(); }
  ~BloscInit() { blosc2_destroy(); }
};
#define BLOSC_INIT hshm::Singleton<BloscInit>::GetInstance()

class Blosc : public Compressor {
 public:
  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    // Initialize Blosc2
    BLOSC_INIT;

    // Create a context for compression
    blosc2_context *cctx = blosc2_create_cctx(BLOSC2_CPARAMS_DEFAULTS);
    if (!cctx) {
      return false;
    }

    // Compress the data
    output_size =
        blosc2_compress_ctx(cctx, input, input_size, output, output_size);

    // Release the compression context
    blosc2_free_ctx(cctx);
    return true;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    // Create a context for decompression
    blosc2_context *dctx = blosc2_create_dctx(BLOSC2_DPARAMS_DEFAULTS);
    if (!dctx) {
      return false;
    }

    // Decompress the data
    output_size =
        blosc2_decompress_ctx(dctx, input, input_size, output, output_size);

    // Release the decompression context
    blosc2_free_ctx(dctx);
    return true;
  }
};

}  // namespace hshm

#endif  // HSHM_ENABLE_COMPRESS

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Blosc_H_
