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

#ifndef CLIO_CAE_CORE_CSV_FILE_ASSIMILATOR_H_
#define CLIO_CAE_CORE_CSV_FILE_ASSIMILATOR_H_

#include <clio_cae/core/factory/base_assimilator.h>
#include <clio_cae/core/factory/csv_table.h>

#include <memory>
#include <string>
#include <vector>

// Forward declaration
namespace clio::cte::core {
class Client;
}  // namespace clio::cte::core

namespace clio::cae::core {

/**
 * CsvFileAssimilator - ingests a CSV file column by column (src csv::<path>,
 * format csv).
 *
 * Each column is treated like a root-level HDF5 dataset named "/<header>":
 * dataset_filter globs select columns by that name, and every selected
 * column lands in CTE exactly as the HDF5 assimilator lands a dataset --
 * tag "<dst tag>/<header>" holding a "description" blob (tensor string,
 * see EncodeCsvColumn) and chunk_0..chunk_N blobs (<= 1.5 MB each) whose
 * concatenation is the column's native element image. Rejected columns
 * get no tag.
 *
 * The file is read and parsed once, on the node that runs the transfer.
 */
class CsvFileAssimilator : public BaseAssimilator {
 public:
  explicit CsvFileAssimilator(
      std::shared_ptr<clio::cte::core::Client> cte_client);

  /**
   * Schedule assimilation of one CSV file.
   * Error codes: -1 dst not iowarp::, -2 no tag name, -3 no source path,
   * -4 file unreadable, -5 CSV syntax error, -9 include patterns matched no
   * column, -6 a column failed to store.
   * @param ctx Assimilation context (src csv::<path>, dst iowarp::<tag>)
   * @param error_code Output: 0 on success
   * @return TaskResume for coroutine suspension/resumption
   */
  clio::run::TaskResume Schedule(const AssimilationCtx& ctx,
                                 int& error_code) override;

  /** Dataset name a column is filtered by: "/" + header. */
  static std::string DatasetName(const std::string& header) {
    return "/" + header;
  }

 private:
  /**
   * Store one encoded column: tag, description blob, chunk blobs.
   * @param tag_name CTE tag for the column
   * @param tensor Encoded column
   * @param error_code Output: 0 on success
   */
  clio::run::TaskResume StoreColumn(const std::string& tag_name,
                                    const CsvColumnTensor& tensor,
                                    int& error_code);

  /** Read a whole file into `out`; false if it cannot be opened/read. */
  static bool ReadFile(const std::string& path, std::string* out);

  /** Part of a protocol::path URL after the "::" (or "" if none). */
  static std::string UrlPath(const std::string& url);

  /** Part of a protocol::path URL before the "::" (or "" if none). */
  static std::string UrlProtocol(const std::string& url);

  std::shared_ptr<clio::cte::core::Client> cte_client_;
};

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_CSV_FILE_ASSIMILATOR_H_
