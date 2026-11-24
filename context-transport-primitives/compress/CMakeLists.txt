# compress: lossless and lossy compression codecs (header-only)
# NOTE: the predictor sources in compress/src are not built; the compression
# predictors live in context-transfer-engine/compressor.

if(CLIO_CORE_ENABLE_TESTS AND CLIO_CTE_ENABLE_COMPRESS)
    add_subdirectory(test)
endif()
