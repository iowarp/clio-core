#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <wrp_cte/core/core_client.h>
#include "rust/cxx.h"

namespace cte_ffi {

// CteTag wraps wrp_cte::core::Tag. Mutable inner allows cxx to pass
// const CteTag& while Tag methods remain non-const.
struct CteTag {
  mutable wrp_cte::core::Tag inner;

  explicit CteTag(const std::string &name) : inner(name) {}
  explicit CteTag(const wrp_cte::core::TagId &id) : inner(id) {}
};

// Forward-declared: defined by cxx-generated code (shared struct)
struct CteTagId;

bool cte_init(rust::Str config_path);

std::unique_ptr<CteTag> tag_new(rust::Str tag_name);
std::unique_ptr<CteTag> tag_from_id(uint32_t major, uint32_t minor);

void tag_put_blob(const CteTag &tag, rust::Str name, rust::Slice<const uint8_t> data,
                  uint64_t offset, float score);
std::unique_ptr<std::vector<uint8_t>> tag_get_blob(const CteTag &tag, rust::Str name,
                                                    uint64_t size, uint64_t offset);
float tag_get_blob_score(const CteTag &tag, rust::Str name);
uint64_t tag_get_blob_size(const CteTag &tag, rust::Str name);
std::unique_ptr<std::vector<std::string>> tag_get_contained_blobs(const CteTag &tag);
void tag_reorganize_blob(const CteTag &tag, rust::Str name, float score);
CteTagId tag_get_id(const CteTag &tag);

bool client_register_target(rust::Str target_path, uint64_t size);
bool client_del_tag(rust::Str name);
std::unique_ptr<std::vector<std::string>> client_tag_query(rust::Str regex, uint32_t max_tags);
std::unique_ptr<std::vector<std::string>> client_blob_query(rust::Str tag_re, rust::Str blob_re,
                                                             uint32_t max_results);

}  // namespace cte_ffi
