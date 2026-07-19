#pragma once

#ifdef __CUDACC__
#define CROSS_FUN __host__ __device__
#else
#define CROSS_FUN
#endif

#include <cuda/std/cstdint>
#include <cuda/std/cstddef>
#include <cuda/std/cassert>

namespace cstd = cuda::std;

using cstd::size_t;

namespace kvhdf5 {

using byte_t = cstd::byte;

// Maximum number of dimensions supported for datasets/chunks
constexpr size_t MAX_DIMS = 8;

template<size_t N>
struct padding {
    CROSS_FUN constexpr bool operator==(const padding&) const { return true; }
private:
    uint8_t _[N] = {};
};

} // namespace kvhdf5

#define KVHDF5_ASSERT(cond, msg) assert((cond) && (msg))
