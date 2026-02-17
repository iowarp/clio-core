#include <chimaera/bdev/bdev_client.h>
#include <wrp_cte/core/content_transfer_engine.h>

// cxx-generated header: defines CteTagId shared struct
#include "wrp-cte-rs/src/lib.rs.h"

namespace cte_ffi {

bool cte_init(rust::Str config_path) {
  std::string path(config_path.data(), config_path.size());
  bool ok = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
  if (!ok) return false;
  return wrp_cte::core::WRP_CTE_CLIENT_INIT(path);
}

std::unique_ptr<CteTag> tag_new(rust::Str tag_name) {
  std::string name(tag_name.data(), tag_name.size());
  return std::make_unique<CteTag>(name);
}

std::unique_ptr<CteTag> tag_from_id(uint32_t major, uint32_t minor) {
  wrp_cte::core::TagId tid(major, minor);
  return std::make_unique<CteTag>(tid);
}

void tag_put_blob(const CteTag &tag, rust::Str name,
                  rust::Slice<const uint8_t> data, uint64_t offset,
                  float score) {
  std::string blob_name(name.data(), name.size());
  tag.inner.PutBlob(blob_name, reinterpret_cast<const char *>(data.data()),
                    data.size(), static_cast<size_t>(offset), score);
}

std::unique_ptr<std::vector<uint8_t>> tag_get_blob(const CteTag &tag,
                                                    rust::Str name,
                                                    uint64_t size,
                                                    uint64_t offset) {
  std::string blob_name(name.data(), name.size());
  auto buf = std::make_unique<std::vector<uint8_t>>(size);
  tag.inner.GetBlob(blob_name, reinterpret_cast<char *>(buf->data()),
                    static_cast<size_t>(size), static_cast<size_t>(offset));
  return buf;
}

float tag_get_blob_score(const CteTag &tag, rust::Str name) {
  std::string blob_name(name.data(), name.size());
  return tag.inner.GetBlobScore(blob_name);
}

uint64_t tag_get_blob_size(const CteTag &tag, rust::Str name) {
  std::string blob_name(name.data(), name.size());
  return tag.inner.GetBlobSize(blob_name);
}

std::unique_ptr<std::vector<std::string>> tag_get_contained_blobs(
    const CteTag &tag) {
  auto blobs = tag.inner.GetContainedBlobs();
  return std::make_unique<std::vector<std::string>>(std::move(blobs));
}

void tag_reorganize_blob(const CteTag &tag, rust::Str name, float score) {
  std::string blob_name(name.data(), name.size());
  tag.inner.ReorganizeBlob(blob_name, score);
}

CteTagId tag_get_id(const CteTag &tag) {
  const auto &id = tag.inner.GetTagId();
  return CteTagId{id.major_, id.minor_};
}

bool client_register_target(rust::Str target_path, uint64_t size) {
  std::string path(target_path.data(), target_path.size());
  // Create a bdev pool for this target
  chi::PoolId bdev_pool_id(800, 0);
  chimaera::bdev::Client bdev_client(bdev_pool_id);
  auto create_task = bdev_client.AsyncCreate(
      chi::PoolQuery::Dynamic(), path, bdev_pool_id,
      chimaera::bdev::BdevType::kFile);
  create_task.Wait();
  // Register with CTE
  auto *client = WRP_CTE_CLIENT;
  auto reg_task = client->AsyncRegisterTarget(
      path, chimaera::bdev::BdevType::kFile, size,
      chi::PoolQuery::Local(), bdev_pool_id);
  reg_task.Wait();
  return true;
}

bool client_del_tag(rust::Str name) {
  std::string tag_name(name.data(), name.size());
  auto *client = WRP_CTE_CLIENT;
  auto task = client->AsyncDelTag(tag_name);
  task.Wait();
  return true;
}

std::unique_ptr<std::vector<std::string>> client_tag_query(rust::Str regex,
                                                            uint32_t max_tags) {
  std::string re(regex.data(), regex.size());
  auto *mgr = CTE_MANAGER;
  auto results = mgr->TagQuery(re, max_tags);
  return std::make_unique<std::vector<std::string>>(std::move(results));
}

std::unique_ptr<std::vector<std::string>> client_blob_query(rust::Str tag_re,
                                                             rust::Str blob_re,
                                                             uint32_t max_results) {
  std::string tre(tag_re.data(), tag_re.size());
  std::string bre(blob_re.data(), blob_re.size());
  auto *mgr = CTE_MANAGER;
  auto pairs = mgr->BlobQuery(tre, bre, max_results);
  auto out = std::make_unique<std::vector<std::string>>();
  out->reserve(pairs.size() * 2);
  for (auto &p : pairs) {
    out->push_back(std::move(p.first));
    out->push_back(std::move(p.second));
  }
  return out;
}

}  // namespace cte_ffi
