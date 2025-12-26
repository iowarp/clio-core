#ifndef WRP_CAE_CORE_ASSIMILATION_CTX_H_
#define WRP_CAE_CORE_ASSIMILATION_CTX_H_

#include <string>
#include <vector>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

namespace wrp_cae::core {

/**
 * AssimilationCtx - Context for data assimilation operations
 * Contains metadata about the source, destination, format, and range
 */
struct AssimilationCtx {
  std::string src;         // Source URL (e.g., file::/path/to/file)
  std::string dst;         // Destination URL (e.g., iowarp::tag_name)
  std::string format;      // Data format (e.g., binary, hdf5)
  std::string depends_on;  // Dependency identifier (empty if none)
  size_t range_off;        // Byte offset in source file
  size_t range_size;       // Number of bytes to read
  std::string src_token;   // Authentication token for source (e.g., Globus access token)
  std::string dst_token;   // Authentication token for destination

  // Dataset filtering (for HDF5 and other hierarchical formats)
  std::vector<std::string> include_patterns;  // Glob patterns for datasets to include
  std::vector<std::string> exclude_patterns;  // Glob patterns for datasets to exclude

  // Default constructor
  AssimilationCtx()
      : range_off(0), range_size(0) {}

  // Full constructor
  AssimilationCtx(const std::string& src_url,
                  const std::string& dst_url,
                  const std::string& data_format,
                  const std::string& dependency = "",
                  size_t offset = 0,
                  size_t size = 0,
                  const std::string& source_token = "",
                  const std::string& dest_token = "")
      : src(src_url),
        dst(dst_url),
        format(data_format),
        depends_on(dependency),
        range_off(offset),
        range_size(size),
        src_token(source_token),
        dst_token(dest_token) {}

  // Serialization support for cereal
  template<class Archive>
  void serialize(Archive& ar) {
    ar(src, dst, format, depends_on, range_off, range_size, src_token, dst_token,
       include_patterns, exclude_patterns);
  }
};

}  // namespace wrp_cae::core

#endif  // WRP_CAE_CORE_ASSIMILATION_CTX_H_
