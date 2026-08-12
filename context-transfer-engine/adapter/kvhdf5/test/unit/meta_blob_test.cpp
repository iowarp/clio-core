// Unit tests for the __meta store format (meta_blob.h): the dataset
// self-description that makes an HDF5 export possible.
//
// The load-bearing property here is that IsChunkBlobName/ParseChunkName are the
// exact inverse of chunking::ChunkCoordToName. If they ever disagree, the
// exporter either skips real chunks (silently producing a dataset with holes) or
// tries to parse the __meta blob as a chunk. So the round-trip is tested against
// the REAL generator, not against hand-written strings.

#include <catch2/catch_test_macros.hpp>

#include <clio_cte/kvhdf5/chunking.h>
#include <clio_cte/kvhdf5/meta_blob.h>

#include <cstdint>
#include <string>
#include <vector>

using namespace kvhdf5;

namespace {

// A valid 2-D f32 dataset: 8x6 elements in 4x3 chunks => 2x2 = 4 chunks.
MetaBlob MakeValid() {
    MetaBlob m;
    m.rank = 2;
    m.dtype = static_cast<uint32_t>(DType::kF32);
    m.elem_size = 4;
    m.dims[0] = 8;        m.dims[1] = 6;
    m.chunk_dims[0] = 4;  m.chunk_dims[1] = 3;
    return m;
}

}  // namespace

TEST_CASE("MetaBlob encode/decode round-trips", "[meta_blob]") {
    const MetaBlob in = MakeValid();
    REQUIRE(in.Valid());

    unsigned char wire[sizeof(MetaBlob)];
    EncodeMeta(in, wire);

    MetaBlob out;
    REQUIRE(DecodeMeta(wire, sizeof(wire), &out));

    CHECK(out.rank == in.rank);
    CHECK(out.Dtype() == DType::kF32);
    CHECK(out.elem_size == 4);
    CHECK(out.dims[0] == 8);
    CHECK(out.dims[1] == 6);
    CHECK(out.chunk_dims[0] == 4);
    CHECK(out.chunk_dims[1] == 3);
    CHECK(out.ChunkCount() == 4);
    CHECK(out.ChunkBytes() == 4 * 3 * 4);
}

TEST_CASE("DecodeMeta rejects malformed input", "[meta_blob]") {
    const MetaBlob good = MakeValid();
    unsigned char wire[sizeof(MetaBlob)];
    EncodeMeta(good, wire);
    MetaBlob out;

    SECTION("wrong buffer size") {
        CHECK_FALSE(DecodeMeta(wire, sizeof(wire) - 1, &out));
        CHECK_FALSE(DecodeMeta(wire, sizeof(wire) + 1, &out));
        CHECK_FALSE(DecodeMeta(wire, 0, &out));
    }
    SECTION("null pointers") {
        CHECK_FALSE(DecodeMeta(nullptr, sizeof(wire), &out));
        CHECK_FALSE(DecodeMeta(wire, sizeof(wire), nullptr));
    }
    SECTION("bad magic") {
        MetaBlob m = good; m.magic = 0xDEADBEEF;
        EncodeMeta(m, wire);
        CHECK_FALSE(DecodeMeta(wire, sizeof(wire), &out));
    }
    SECTION("unknown version") {
        MetaBlob m = good; m.version = kMetaVersion + 1;
        EncodeMeta(m, wire);
        CHECK_FALSE(DecodeMeta(wire, sizeof(wire), &out));
    }
}

TEST_CASE("MetaBlob::Valid enforces the dtype/elem_size invariant", "[meta_blob]") {
    MetaBlob m = MakeValid();

    SECTION("typed dtype must agree with elem_size") {
        m.dtype = static_cast<uint32_t>(DType::kF32);
        m.elem_size = 8;  // f32 is 4 bytes -- inconsistent
        CHECK_FALSE(m.Valid());
    }
    SECTION("kOpaque accepts any width") {
        m.dtype = static_cast<uint32_t>(DType::kOpaque);
        m.elem_size = 7;  // arbitrary, and legal for opaque bytes
        CHECK(m.Valid());
    }
    SECTION("unknown dtype rejected") {
        m.dtype = 999;
        CHECK_FALSE(m.Valid());
    }
    SECTION("zero elem_size rejected") {
        m.elem_size = 0;
        CHECK_FALSE(m.Valid());
    }
}

TEST_CASE("MetaBlob::Valid enforces geometry", "[meta_blob]") {
    MetaBlob m = MakeValid();

    SECTION("rank bounds") {
        m.rank = 0;
        CHECK_FALSE(m.Valid());
        m.rank = kMetaMaxDims + 1;
        CHECK_FALSE(m.Valid());
    }
    SECTION("zero dim rejected") {
        m.dims[1] = 0;
        CHECK_FALSE(m.Valid());
    }
    SECTION("chunk larger than dim rejected") {
        m.chunk_dims[0] = 16;  // dims[0] is 8
        CHECK_FALSE(m.Valid());
    }
    SECTION("edge chunks (non-divisible dims) rejected") {
        // The writer refuses to create these, so a __meta describing one is
        // corrupt. Rejecting it here is what stops the exporter from emitting a
        // truncated dataset.
        m.dims[0] = 9;  // 9 % 4 != 0
        CHECK_FALSE(m.Valid());
    }
}

TEST_CASE("IsChunkBlobName accepts exactly the generator's output", "[meta_blob]") {
    CHECK(IsChunkBlobName("0"));
    CHECK(IsChunkBlobName("7"));
    CHECK(IsChunkBlobName("12_47"));
    CHECK(IsChunkBlobName("0_0_0_0"));

    // The whole point: the reserved key is not a chunk.
    CHECK_FALSE(IsChunkBlobName(kMetaBlobName));

    CHECK_FALSE(IsChunkBlobName(""));
    CHECK_FALSE(IsChunkBlobName("_1"));    // leading separator
    CHECK_FALSE(IsChunkBlobName("1_"));    // trailing separator
    CHECK_FALSE(IsChunkBlobName("1__2"));  // empty field
    CHECK_FALSE(IsChunkBlobName("_"));
    CHECK_FALSE(IsChunkBlobName("a"));
    CHECK_FALSE(IsChunkBlobName("1_a"));
    CHECK_FALSE(IsChunkBlobName("-1"));
}

TEST_CASE("ParseChunkName inverts the name format", "[meta_blob]") {
    uint64_t coord[kMetaMaxDims] = {};

    REQUIRE(ParseChunkName("12_47", 2, coord));
    CHECK(coord[0] == 12);
    CHECK(coord[1] == 47);

    REQUIRE(ParseChunkName("5", 1, coord));
    CHECK(coord[0] == 5);

    SECTION("rank must match the name's field count") {
        CHECK_FALSE(ParseChunkName("12_47", 1, coord));
        CHECK_FALSE(ParseChunkName("12_47", 3, coord));
        CHECK_FALSE(ParseChunkName("5", 2, coord));
    }
    SECTION("rejects non-chunk names") {
        CHECK_FALSE(ParseChunkName(kMetaBlobName, 1, coord));
        CHECK_FALSE(ParseChunkName("", 1, coord));
    }
    SECTION("rejects u64 overflow rather than wrapping") {
        CHECK_FALSE(ParseChunkName("99999999999999999999999", 1, coord));
    }
    SECTION("rank 0 / out of range rejected") {
        CHECK_FALSE(ParseChunkName("1", 0, coord));
        CHECK_FALSE(ParseChunkName("1", kMetaMaxDims + 1, coord));
    }
}

TEST_CASE("ParseChunkName is the exact inverse of ChunkCoordToName", "[meta_blob]") {
    // Drive the REAL generator over every chunk of a 3-D layout and require that
    // parsing its output reproduces the coordinate. This is the property the
    // exporter depends on: name -> coord -> HDF5 chunk offset.
    const uint64_t dims[3]       = {8, 6, 4};
    const uint64_t chunk_dims[3] = {2, 3, 2};
    const cstd::span<const uint64_t> d{dims, 3};
    const cstd::span<const uint64_t> cd{chunk_dims, 3};

    const uint64_t n = chunking::ChunkCount(d, cd);
    REQUIRE(n == (8 / 2) * (6 / 3) * (4 / 2));  // 4 * 2 * 2 == 16

    for (uint64_t i = 0; i < n; ++i) {
        uint64_t coord[3] = {};
        chunking::ChunkIndexToCoord(i, d, cd, {coord, 3});

        char buf[chunking::kMaxBlobNameLen + 1];
        auto name_span = chunking::ChunkCoordToName({coord, 3}, buf);
        REQUIRE_FALSE(name_span.empty());
        const std::string name(name_span.data(), name_span.size());

        // The generator's output must be recognized as a chunk...
        REQUIRE(IsChunkBlobName(name));

        // ...and parse back to the coordinate it came from.
        uint64_t parsed[kMetaMaxDims] = {};
        REQUIRE(ParseChunkName(name, 3, parsed));
        CHECK(parsed[0] == coord[0]);
        CHECK(parsed[1] == coord[1]);
        CHECK(parsed[2] == coord[2]);
    }
}

TEST_CASE("meta_blob ChunkCoordToName matches the device generator", "[meta_blob]") {
    // The CUDA-free twin used by the importer must emit the SAME string as
    // chunking::ChunkCoordToName, or an imported chunk would land under a name
    // the kvhdf5 reader never looks for.
    const uint64_t dims[3]       = {8, 6, 4};
    const uint64_t chunk_dims[3] = {2, 3, 2};
    const cstd::span<const uint64_t> d{dims, 3};
    const cstd::span<const uint64_t> cd{chunk_dims, 3};

    const uint64_t n = chunking::ChunkCount(d, cd);
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t coord[3] = {};
        chunking::ChunkIndexToCoord(i, d, cd, {coord, 3});

        char buf[chunking::kMaxBlobNameLen + 1];
        auto name_span = chunking::ChunkCoordToName({coord, 3}, buf);
        REQUIRE_FALSE(name_span.empty());
        const std::string device_name(name_span.data(), name_span.size());

        std::string host_name;
        REQUIRE(ChunkCoordToName(coord, 3, &host_name));
        CHECK(host_name == device_name);
    }

    // A name past the length cap is refused, exactly as the device twin refuses it.
    const uint64_t huge[1] = {12345678901234567890ull};  // 20 digits
    std::string overflow_name(chunking::kMaxBlobNameLen, 'x');  // pre-set sentinel
    // rank 2 of the 20-digit value => "..._..." well over 31 chars.
    const uint64_t pair[2] = {huge[0], huge[0]};
    CHECK_FALSE(ChunkCoordToName(pair, 2, &overflow_name));
    CHECK(overflow_name.size() == kMetaMaxBlobNameLen);  // untouched on failure
}

TEST_CASE("ChunkCoordFromLinearIndex inverts ChunkLinearIndex", "[meta_blob]") {
    // Round-trip every chunk index of a 3-D layout through coord and back.
    MetaBlob m;
    m.rank = 3;
    m.dtype = static_cast<uint32_t>(DType::kF64);
    m.elem_size = 8;
    m.dims[0] = 8;       m.dims[1] = 6;       m.dims[2] = 4;
    m.chunk_dims[0] = 2; m.chunk_dims[1] = 3; m.chunk_dims[2] = 2;
    REQUIRE(m.Valid());

    const uint64_t n = m.ChunkCount();
    REQUIRE(n == 4u * 2u * 2u);
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t coord[kMetaMaxDims] = {};
        m.ChunkCoordFromLinearIndex(i, coord);
        CHECK(m.ChunkLinearIndex(coord) == i);
        // Every coordinate stays within its per-dim chunk count.
        for (uint32_t k = 0; k < m.rank; ++k) {
            CHECK(coord[k] < m.dims[k] / m.chunk_dims[k]);
        }
    }
}
