/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Clio. The full Clio copyright notice, including      *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory.                *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef CLIO_CTE_CORE_BLOB_TRANSFORM_H_
#define CLIO_CTE_CORE_BLOB_TRANSFORM_H_

#include "clio_runtime/types.h"

namespace clio::cte::core {

/**
 * How a blob's STORED bytes relate to the LOGICAL bytes the caller put
 * (issue #818).
 *
 * AUTHORITATIVE. This is the single source of truth for "are the bytes on the
 * device still the caller's bytes?" -- deliberately NOT derived from
 * compress_lib_, which answers a different question (which codec was
 * *requested*) and is wrong in both directions:
 *
 *  - False positive: the compressor chimod used to call PutBlob with
 *    compress_lib_ still set when compression turned out not to be beneficial,
 *    zeroing it only afterwards. The blob was then recorded as compressed while
 *    holding raw bytes.
 *  - False negative: core only records compress_lib_ under
 *    `#if CTP_ENABLE_COMPRESS` (OFF by default), yet the metadata log persists
 *    it unconditionally -- so a restart can restore a genuinely compressed blob
 *    into a runtime that never populates the field.
 *
 * Compression is only the first transform; encryption and any future
 * byte-rewriting layer must set this too. Consumers must therefore ask
 * `BlobInfo::IsTransformed()` / `ShmBlobRecord::IsTransformed()`, never
 * `compress_lib_ != 0`.
 *
 * COMPILED UNCONDITIONALLY, on purpose. Putting the flag word behind
 * `#if CTP_ENABLE_COMPRESS` would reintroduce exactly the failure it exists to
 * prevent (a build with compression off silently reporting "not transformed"),
 * and would make the enclosing types change size with a compile flag -- the
 * Context/PutBlobTask ABI-skew hazard.
 *
 * FORWARD COMPATIBILITY: a reader that does not recognise a bit must still
 * treat the blob as transformed, which is why the IsTransformed() predicates
 * test the whole word against zero rather than one known bit. An unknown future
 * transform then fails safe (refuse the direct-read fast path) instead of being
 * handed back as raw bytes.
 *
 * This lives in its own header, rather than in core_tasks.h beside BlobInfo, so
 * that the deliberately lightweight shared-memory cache header can name these
 * bits without pulling in the full task/config include graph.
 */
enum BlobTransformFlags : clio::run::u32 {
  kBlobTransformNone = 0,
  /** Generic "these bytes are not the caller's bytes". ALWAYS set alongside
   *  whichever specific bit(s) apply, so a reader that understands none of the
   *  specific kinds still gets a correct yes/no answer. */
  kBlobTransformed = 1u << 0,
  /** Stored as CompressionHeader + codec output. */
  kBlobTransformCompressed = 1u << 1,
  /** Stored as ciphertext (reserved; no producer sets this yet). */
  kBlobTransformEncrypted = 1u << 2,
};

}  // namespace clio::cte::core

#endif  // CLIO_CTE_CORE_BLOB_TRANSFORM_H_
