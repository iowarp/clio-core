#pragma once

#include "defines.h"
#include <cuda/std/span>
#include <cuda/std/expected>

// The BlobBackend concept is a C++20 language feature. The host build compiles
// as C++20 (see CMAKE_CXX_STANDARD), but the GPU tests are compiled by nvcc as
// C++17, where `concept`/`requires` and <cuda/std/concepts> are unavailable.
// Under C++17 we degrade the concept to an unconstrained `typename` template
// parameter (KVHDF5_BLOB_BACKEND) so the same headers still compile in device
// tests; the constraint is still enforced on the C++20 host build. The
// <cuda/std/concepts> include MUST stay at file scope (not inside a namespace,
// or libcu++'s own declarations get nested and its internals stop compiling).
#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
#include <cuda/std/concepts>
#define KVHDF5_BLOB_BACKEND BlobBackend
#else
#define KVHDF5_BLOB_BACKEND typename
#endif

namespace kvhdf5 {

enum class BlobError : uint8_t { NotFound, NotEnoughSpace, BackendFailure };

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
// A chunk store keyed by a host-formatted blob name (chunk-coord text). The
// value is raw chunk bytes. InMemBlobBackend satisfies this for runtime-free
// CPU tests; GpuCteBlobStore (Slice 2) is the real iowarp producer path.
template<typename T>
concept BlobBackend = requires(T s,
        cstd::span<const char> name,
        cstd::span<const byte_t> in,
        cstd::span<byte_t> out) {
    { s.WriteChunk(name, in) } -> cstd::same_as<bool>;
    { s.ReadChunk(name, out) } -> cstd::same_as<cstd::expected<cstd::span<byte_t>, BlobError>>;
    { s.ChunkExists(name) } -> cstd::same_as<bool>;
};
#endif

}  // namespace kvhdf5
