#include <catch2/catch_test_macros.hpp>
#include <clio_cte/kvhdf5/tag_path.h>
#include <clio_cte/kvhdf5/chunking.h>
#include <clio_cte/kvhdf5/cpu_dataset.h>
#include <string_view>
#include <array>

using namespace kvhdf5;

TEST_CASE("CanonicalTag joins a simple path verbatim", "[tagpath]") {
    REQUIRE(tagpath::CanonicalTag("results/snapshots/2026/temperature")
            == "results/snapshots/2026/temperature");
}

TEST_CASE("CanonicalTag collapses duplicate separators", "[tagpath]") {
    REQUIRE(tagpath::CanonicalTag("a//b///c") == "a/b/c");
}

TEST_CASE("CanonicalTag strips leading and trailing separators", "[tagpath]") {
    REQUIRE(tagpath::CanonicalTag("/a/b/") == "a/b");
}

TEST_CASE("CanonicalTag is idempotent under redundant separators", "[tagpath]") {
    REQUIRE(tagpath::CanonicalTag("/a//b/") == tagpath::CanonicalTag("a/b"));
}

TEST_CASE("CanonicalTag accepts a single component", "[tagpath]") {
    REQUIRE(tagpath::CanonicalTag("temperature") == "temperature");
}

TEST_CASE("CanonicalTag preserves case", "[tagpath]") {
    REQUIRE(tagpath::CanonicalTag("Results/Temp") == "Results/Temp");
}

TEST_CASE("CanonicalTag returns empty for empty or all-separator input", "[tagpath]") {
    REQUIRE(tagpath::CanonicalTag("").empty());
    REQUIRE(tagpath::CanonicalTag("/").empty());
    REQUIRE(tagpath::CanonicalTag("///").empty());
}

TEST_CASE("JoinTag joins hierarchy components", "[tagpath]") {
    std::array<std::string_view, 4> parts{
        "results", "snapshots", "2026", "temperature"};
    REQUIRE(tagpath::JoinTag({parts.data(), parts.size()})
            == "results/snapshots/2026/temperature");
}

TEST_CASE("JoinTag accepts a single component", "[tagpath]") {
    std::array<std::string_view, 1> parts{"temperature"};
    REQUIRE(tagpath::JoinTag({parts.data(), parts.size()}) == "temperature");
}

TEST_CASE("JoinTag rejects an empty component", "[tagpath]") {
    std::array<std::string_view, 3> parts{"a", "", "c"};
    REQUIRE(tagpath::JoinTag({parts.data(), parts.size()}).empty());
}

TEST_CASE("JoinTag rejects a component containing the separator", "[tagpath]") {
    std::array<std::string_view, 2> parts{"a/b", "c"};
    REQUIRE(tagpath::JoinTag({parts.data(), parts.size()}).empty());
}

TEST_CASE("JoinTag of no components is empty", "[tagpath]") {
    REQUIRE(tagpath::JoinTag({}).empty());
}

TEST_CASE("JoinTag output round-trips through CanonicalTag", "[tagpath]") {
    std::array<std::string_view, 3> parts{"results", "2026", "temperature"};
    auto joined = tagpath::JoinTag({parts.data(), parts.size()});
    REQUIRE(tagpath::CanonicalTag(joined) == joined);
}

// The KV address for one chunk is (tag, name) = (dataset-path, chunk-coord).
// Both halves must be independently valid and non-empty.
TEST_CASE("path tag pairs with a chunk-coord blob name as the KV address",
          "[tagpath]") {
    auto tag = tagpath::CanonicalTag("results/snapshots/2026/temperature");
    REQUIRE_FALSE(tag.empty());

    uint64_t coord[2] = {12, 47};
    char buf[chunking::kMaxBlobNameLen + 1];
    auto name = chunking::ChunkCoordToName({coord, 2}, buf);
    REQUIRE_FALSE(name.empty());
    REQUIRE(std::string_view(name.data(), name.size()) == "12_47");
}

TEST_CASE("DatasetMeta::TagName canonicalizes its path", "[tagpath]") {
    DatasetMeta meta{"/results//snapshots/2026/temperature/", Layout{}};
    REQUIRE(meta.TagName() == "results/snapshots/2026/temperature");
}
