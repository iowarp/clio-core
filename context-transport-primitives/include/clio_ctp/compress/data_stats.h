/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file data_stats.h
 * @brief Backward-compatibility shim.
 *
 * The data-statistics / feature-extraction code moved under the unified
 * compression-metric module at clio_ctp/compress/preprocess/ (issue #693).
 * This header is retained so existing includes of
 * "clio_ctp/compress/data_stats.h" keep compiling; include the new path
 * directly in new code.
 */
#ifndef CLIO_CTP_COMPRESS_DATA_STATS_SHIM_H
#define CLIO_CTP_COMPRESS_DATA_STATS_SHIM_H

#include "clio_ctp/compress/preprocess/data_stats.h"

#endif  // CLIO_CTP_COMPRESS_DATA_STATS_SHIM_H
