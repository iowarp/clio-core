#ifndef WRP_CTE_COMPRESSOR_AUTOGEN_METHODS_H_
#define WRP_CTE_COMPRESSOR_AUTOGEN_METHODS_H_

#include <chimaera/chimaera.h>

/**
 * Auto-generated method definitions for compressor
 */

namespace wrp_cte::compressor {

namespace Method {
// Inherited methods
GLOBAL_CONST chi::u32 kCreate = 0;
GLOBAL_CONST chi::u32 kDestroy = 1;
GLOBAL_CONST chi::u32 kMonitor = 9;

// compressor-specific methods
GLOBAL_CONST chi::u32 kDynamicSchedule = 10;
GLOBAL_CONST chi::u32 kCompress = 11;
GLOBAL_CONST chi::u32 kDecompress = 12;
}  // namespace Method

}  // namespace wrp_cte::compressor

#endif  // COMPRESSOR_AUTOGEN_METHODS_H_
