/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_HDF5_VOL_H_
#define CLIO_HDF5_VOL_H_

#include <hdf5.h>
#include <H5VLconnector.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* VOL connector identification */
#define CLIO_VOL_CONNECTOR_NAME    "clio"
#define CLIO_VOL_CONNECTOR_VALUE   600  /* Unique connector class value */
#define CLIO_VOL_CONNECTOR_VERSION 1

/* Default chunk size for async PutBlob/GetBlob (1 MB) */
#define CLIO_VOL_DEFAULT_CHUNK_SIZE (1024 * 1024)

/* Values for clio_vol_info_t::cache_enabled.
 *
 * Tri-state because "unset" and "explicitly on" must stay distinguishable: a
 * config string that says nothing about the cache must not override an
 * environment that asked to turn it off.
 *
 * UNSET is 0 so a zero-initialized info struct means "no opinion". Disabling
 * the cache must be something you SAY, never something you arrive at by
 * forgetting to set a field -- a silently-disabled tier looks exactly like a
 * working one that cached nothing.
 */
#define CLIO_VOL_CACHE_UNSET 0  /* env/default decides */
#define CLIO_VOL_CACHE_ON    1
#define CLIO_VOL_CACHE_OFF   2

/* VOL connector info (passed via H5Pset_vol). Value-initialize it
   (`clio_vol_info_t info = {};`) so every field starts at a defined default;
   this is a C struct with no constructor, so a bare stack instance holds
   garbage. */
typedef struct clio_vol_info_t {
    hid_t  under_vol_id;    /* VOL ID for the underlying connector */
    void  *under_vol_info;  /* Info for the underlying connector */
    size_t chunk_size;      /* Chunk size for async I/O (0 = default) */
    /* Compressor chimod pool to route cacheable H5Dwrite/H5Dread through
       (0,0) = unset -- writes go straight to the CTE core pool uncompressed,
       exactly today's behavior. Mirrors clio::run::PoolId's null-pool
       convention (major 0 means "no pool"), so a zero-initialized info
       struct (the common case, e.g. this file's own makeFapl() helpers
       before this field existed) still means "no compressor". */
    uint32_t compressor_pool_major;
    uint32_t compressor_pool_minor;
    int    cache_enabled;   /* CTE tier: CLIO_VOL_CACHE_{UNSET,ON,OFF} */
} clio_vol_info_t;

/* Global VOL connector class */
extern const H5VL_class_t H5VL_clio_cls;

/* Registration / lookup */
hid_t H5VL_clio_register(void);

#ifdef __cplusplus
}
#endif

#endif /* CLIO_HDF5_VOL_H_ */
