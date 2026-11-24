/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

#ifndef CLIO_CAE_CORE_DATAVERSE_FILE_ASSIMILATOR_H_
#define CLIO_CAE_CORE_DATAVERSE_FILE_ASSIMILATOR_H_

#include <clio_cae/core/factory/base_assimilator.h>

#include <memory>
#include <string>

namespace clio::cte::core {
class Client;
}  // namespace clio::cte::core

namespace clio::cae::core {

/**
 * dataverse:// sources: files of a Dataverse dataset and members of its zip
 * files, picked by the OMNI dataset_filter patterns (dataverse.h has the
 * rules).
 *
 *   dst: file::<dir>      stage the objects at <dir>/<path> and stop
 *   dst: iowarp::<tag>    stage them in the cache ($CAE_DATAVERSE_CACHE or
 *                         ~/.clio/cae_dataverse/<pid>) and ingest each into
 *                         the tag <tag>/<path>:
 *                           format: hdf5    .h5/.hdf5/.he5 files go through the
 *                                           HDF5 assimilator (a tag per
 *                                           dataset, <tag>/<path>/<dataset>);
 *                                           other files as binary
 *                           format: binary  every file as chunked blobs
 *                         dataset_filter patterns that start with "hdf5:"
 *                         pick datasets inside the HDF5 files (hdf5:<glob>)
 *                         and are passed to the HDF5 assimilator.
 *                         The tag <tag> gets a "manifest" blob: one line per
 *                         object (path, size, Dataverse file id, zip member).
 *
 * src_token, when given, is sent as the X-Dataverse-key header (drafts and
 * restricted files). Other headers and settings for the host come from the
 * http config file (http_auth.h).
 */
class DataverseFileAssimilator : public BaseAssimilator {
 public:
  explicit DataverseFileAssimilator(
      std::shared_ptr<clio::cte::core::Client> cte_client);

  clio::run::TaskResume Schedule(const AssimilationCtx& ctx,
                                 int& error_code) override;

 private:
  std::shared_ptr<clio::cte::core::Client> cte_client_;
};

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_DATAVERSE_FILE_ASSIMILATOR_H_
