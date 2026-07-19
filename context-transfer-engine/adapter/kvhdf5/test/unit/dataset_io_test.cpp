#include <catch2/catch_test_macros.hpp>
#include <clio_cte/kvhdf5/inmem_blob_backend.h>
#include <clio_cte/kvhdf5/cpu_file.h>
#include <cstdint>
#include <vector>

using namespace kvhdf5;

namespace {
std::vector<byte_t> Pattern(size_t n, uint8_t seed) {
    std::vector<byte_t> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = static_cast<byte_t>((seed ^ i) & 0xFFu);
    return v;
}
}  // namespace

TEST_CASE("InMemBlobBackend stores and returns raw bytes", "[backend]") {
    InMemBlobBackend be;
    const char* name = "0_0";
    auto in = Pattern(64, 0xC3);

    REQUIRE(be.WriteChunk({name, 3}, {in.data(), in.size()}));
    REQUIRE(be.ChunkExists({name, 3}));

    std::vector<byte_t> out(64);
    auto r = be.ReadChunk({name, 3}, {out.data(), out.size()});
    REQUIRE(r.has_value());
    REQUIRE(r->size() == 64);
    REQUIRE(out == in);
}

TEST_CASE("InMemBlobBackend reports NotFound / NotEnoughSpace", "[backend]") {
    InMemBlobBackend be;
    std::vector<byte_t> out(8);
    auto miss = be.ReadChunk({"nope", 4}, {out.data(), out.size()});
    REQUIRE_FALSE(miss.has_value());
    REQUIRE(miss.error() == BlobError::NotFound);

    auto in = Pattern(16, 1);
    be.WriteChunk({"k", 1}, {in.data(), in.size()});
    std::vector<byte_t> tiny(4);
    auto small = be.ReadChunk({"k", 1}, {tiny.data(), tiny.size()});
    REQUIRE_FALSE(small.has_value());
    REQUIRE(small.error() == BlobError::NotEnoughSpace);
}

TEST_CASE("Dataset single-chunk write/read round trips", "[dataset]") {
    File<InMemBlobBackend> file{InMemBlobBackend{}};
    Layout layout{{4, 4}, {4, 4}, sizeof(float)};
    auto ds = file.CreateDataset("temperature", layout);
    REQUIRE(ds.has_value());

    auto data = Pattern(layout.TotalBytes(), 0xA5);
    REQUIRE(ds->Write({data.data(), data.size()}).has_value());

    std::vector<byte_t> out(layout.TotalBytes());
    auto rd = ds->Read({out.data(), out.size()});
    REQUIRE(rd.has_value());
    REQUIRE(out == data);
}

TEST_CASE("Dataset round trips through a reopen", "[dataset]") {
    File<InMemBlobBackend> file{InMemBlobBackend{}};
    Layout layout{{8}, {8}, sizeof(double)};
    REQUIRE(file.CreateDataset("v", layout).has_value());

    auto data = Pattern(layout.TotalBytes(), 0x5C);
    REQUIRE(file.OpenDataset("v")->Write({data.data(), data.size()}).has_value());

    std::vector<byte_t> out(layout.TotalBytes());
    auto rd = file.OpenDataset("v")->Read({out.data(), out.size()});
    REQUIRE(rd.has_value());
    REQUIRE(out == data);
}

TEST_CASE("File rejects creating a duplicate dataset path", "[file]") {
    File<InMemBlobBackend> file{InMemBlobBackend{}};
    Layout layout{{4}, {4}, sizeof(float)};
    REQUIRE(file.CreateDataset("dup", layout).has_value());
    REQUIRE_FALSE(file.CreateDataset("dup", layout).has_value());
}

TEST_CASE("Dataset rejects a size mismatch", "[dataset]") {
    File<InMemBlobBackend> file{InMemBlobBackend{}};
    Layout layout{{4}, {4}, sizeof(float)};
    auto ds = file.CreateDataset("x", layout);
    REQUIRE(ds.has_value());

    std::vector<byte_t> wrong(8);
    auto w = ds->Write({wrong.data(), wrong.size()});
    REQUIRE_FALSE(w.has_value());
    REQUIRE(w.error() == IoError::SizeMismatch);
}

TEST_CASE("Dataset multi-chunk is unsupported in the MVP", "[dataset]") {
    File<InMemBlobBackend> file{InMemBlobBackend{}};
    Layout layout{{8}, {4}, sizeof(float)};  // 2 chunks
    auto ds = file.CreateDataset("multi", layout);
    REQUIRE(ds.has_value());

    auto data = Pattern(layout.TotalBytes(), 1);
    auto w = ds->Write({data.data(), data.size()});
    REQUIRE_FALSE(w.has_value());
    REQUIRE(w.error() == IoError::Unsupported);
}
