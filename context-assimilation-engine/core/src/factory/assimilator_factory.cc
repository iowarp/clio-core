/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <clio_cae/core/factory/assimilator_factory.h>
#include <clio_cae/core/factory/binary_file_assimilator.h>
#include <clio_cae/core/factory/string_data_assimilator.h>
#ifdef CLIO_CAE_ENABLE_HDF5
#include <clio_cae/core/factory/hdf5_file_assimilator.h>
#endif
#ifdef CAE_ENABLE_GLOBUS
#include <clio_cae/core/factory/globus_file_assimilator.h>
#endif
#include <clio_runtime/clio_runtime.h>

#include <memory>

namespace clio::cae::core {

AssimilatorFactory::AssimilatorFactory(
    std::shared_ptr<clio::cte::core::Client> cte_client)
    : cte_client_(cte_client) {}

std::unique_ptr<BaseAssimilator> AssimilatorFactory::Get(
    const std::string& src) {
  HLOG(kDebug, "AssimilatorFactory::Get ENTRY: src='{}'", src);

  // Check if this is a Globus web URL first
  if (src.find("https://app.globus.org") == 0) {
#ifdef CAE_ENABLE_GLOBUS
    HLOG(kDebug,
         "AssimilatorFactory: Detected Globus web URL, creating "
         "GlobusFileAssimilator");
    return std::make_unique<GlobusFileAssimilator>(cte_client_);
#else
    HLOG(kError,
         "AssimilatorFactory: Globus web URL detected but Globus support not "
         "compiled in. "
         "Rebuild with -DCAE_ENABLE_GLOBUS=ON to enable Globus support.");
    return nullptr;
#endif
  }

  std::string protocol = GetUrlProtocol(src);
  HLOG(kDebug, "AssimilatorFactory: Extracted protocol='{}'", protocol);

  if (protocol == "file") {
    HLOG(kDebug,
         "AssimilatorFactory: Creating BinaryFileAssimilator for 'file' "
         "protocol");
    // For file protocol, return a BinaryFileAssimilator
    return std::make_unique<BinaryFileAssimilator>(cte_client_);
  } else if (protocol == "string") {
    HLOG(kDebug,
         "AssimilatorFactory: Creating StringDataAssimilator for 'string' "
         "protocol");
    // For string protocol, the payload travels in AssimilationCtx::src_data
    return std::make_unique<StringDataAssimilator>(cte_client_);
  } else if (protocol == "hdf5") {
#ifdef CLIO_CAE_ENABLE_HDF5
    HLOG(
        kInfo,
        "AssimilatorFactory: Creating Hdf5FileAssimilator for 'hdf5' protocol");
    // For hdf5 protocol, return an Hdf5FileAssimilator
    return std::make_unique<Hdf5FileAssimilator>(cte_client_);
#else
    // HDF5 support not compiled in
    HLOG(kError,
         "AssimilatorFactory: HDF5 protocol requested but HDF5 support not "
         "compiled in. "
         "Rebuild with -DWRP_CORE_ENABLE_HDF5=ON to enable HDF5 support.");
    return nullptr;
#endif
  } else if (protocol == "globus") {
#ifdef CAE_ENABLE_GLOBUS
    HLOG(kDebug,
         "AssimilatorFactory: Creating GlobusFileAssimilator for 'globus' "
         "protocol");
    // For globus protocol, return a GlobusFileAssimilator
    return std::make_unique<GlobusFileAssimilator>(cte_client_);
#else
    // Globus support not compiled in
    HLOG(kError,
         "AssimilatorFactory: Globus protocol requested but Globus support not "
         "compiled in. "
         "Rebuild with -DCAE_ENABLE_GLOBUS=ON to enable Globus support.");
    return nullptr;
#endif
  }

  // Unsupported protocol
  HLOG(kError, "AssimilatorFactory: Unsupported protocol '{}'", protocol);
  return nullptr;
}

std::string AssimilatorFactory::GetUrlProtocol(const std::string& url) {
  // Check for standard URI format first (e.g., "globus://")
  size_t pos_standard = url.find("://");
  if (pos_standard != std::string::npos) {
    return url.substr(0, pos_standard);
  }

  // Fall back to custom format (e.g., "file::")
  size_t pos_custom = url.find("::");
  if (pos_custom != std::string::npos) {
    return url.substr(0, pos_custom);
  }

  return "";
}

}  // namespace clio::cae::core
