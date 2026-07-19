#include <catch2/catch_test_macros.hpp>
#include <clio_cte/kvhdf5/chunking.h>
#include <cstring>

using namespace kvhdf5;

TEST_CASE("ChunksInDim is ceil division", "[chunking]") {
    REQUIRE(chunking::ChunksInDim(10, 4) == 3);
    REQUIRE(chunking::ChunksInDim(8, 4) == 2);
    REQUIRE(chunking::ChunksInDim(1, 4) == 1);
}

TEST_CASE("ChunkCount is the product over dims", "[chunking]") {
    uint64_t dims[] = {10, 10};
    uint64_t cd[] = {4, 5};  // 3 x 2
    REQUIRE(chunking::ChunkCount({dims, 2}, {cd, 2}) == 6);
}

TEST_CASE("ChunkIndex<->Coord round trips", "[chunking]") {
    uint64_t dims[] = {10, 10};
    uint64_t cd[] = {4, 5};  // 3 x 2 = 6 chunks
    for (uint64_t i = 0; i < 6; ++i) {
        uint64_t coord[2];
        chunking::ChunkIndexToCoord(i, {dims, 2}, {cd, 2}, {coord, 2});
        REQUIRE(chunking::ChunkCoordToIndex({coord, 2}, {dims, 2}, {cd, 2}) == i);
    }
}

TEST_CASE("ChunkCoordToName formats coord text", "[chunking]") {
    char buf[chunking::kMaxBlobNameLen + 1];

    uint64_t c1[] = {0};
    REQUIRE(chunking::ChunkCoordToName({c1, 1}, buf).size() == 1);
    REQUIRE(std::strcmp(buf, "0") == 0);

    uint64_t c2[] = {12, 47};
    auto name = chunking::ChunkCoordToName({c2, 2}, buf);
    REQUIRE(name.size() == 5);
    REQUIRE(std::strcmp(buf, "12_47") == 0);  // span excludes the NUL; buffer keeps it
    REQUIRE(name.data() == buf);
}

TEST_CASE("ChunkCoordToName rejects names that don't fit the buffer", "[chunking]") {
    char small[3];  // room for 2 chars + NUL
    uint64_t c[] = {123};
    REQUIRE(chunking::ChunkCoordToName({c, 1}, small).empty());
}

TEST_CASE("ChunkCoordToName rejects names over the device limit", "[chunking]") {
    // Two max-width u64 coords join to 20 + 1 + 20 = 41 chars > kMaxBlobNameLen,
    // even though the buffer is large enough — exercises the device-name guard.
    char big[64];
    uint64_t c[] = {18446744073709551615ULL, 18446744073709551615ULL};
    REQUIRE(chunking::ChunkCoordToName({c, 2}, big).empty());
}
