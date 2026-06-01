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

/**
 * test_export_data.cc
 *
 * Unit tests for the CAE ExportData feature and related task structs.
 *
 * Coverage targets (>95%):
 *   - ExportDataTask: default ctor, emplace ctor, Copy, Aggregate,
 *     SerializeIn, SerializeOut
 *   - ParseOmniTask: default ctor, emplace ctor, Copy, Aggregate,
 *     SerializeIn, SerializeOut
 *   - ProcessHdf5DatasetTask: default ctor, emplace ctor, Copy, Aggregate,
 *     SerializeIn, SerializeOut
 *   - Runtime::ExportData: empty-tag, binary roundtrip, binary bad path,
 *     unknown format (falls to binary), HDF5 or not-compiled path
 *   - autogen core_lib_exec.cc: kExportData cases in Run, SaveTask, LoadTask,
 *     LocalLoadTask, LocalSaveTask, NewCopyTask, NewTask, Aggregate, DelTask
 *     (exercised implicitly via AsyncExportData / AsyncParseOmni)
 */

#include "simple_test.h"

#include <clio_cae/core/core_client.h>
#include <clio_cae/core/core_tasks.h>
#include <clio_cae/core/constants.h>
#include <clio_cae/core/factory/assimilation_ctx.h>
#include <clio_cte/core/core_client.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/admin/admin_client.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/bdev/bdev_tasks.h>

#include "clio_ctp/data_structures/serialization/global_serialize.h"
#include <fstream>
#include <thread>
#include <cstring>
#include <vector>
#include <cstdio>
#include <filesystem>

#ifdef CLIO_CAE_ENABLE_HDF5
#include <hdf5.h>
#endif

using namespace clio::cae::core;

// Build a writable temp-file path that is valid on all platforms (there is
// no /tmp on Windows). Used by the runtime-integration export tests that
// actually open the output file for writing.
static std::string ExportTmpPath(const std::string &name) {
  return (std::filesystem::temp_directory_path() / name).string();
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class ExportDataFixture {
 public:
  static inline bool g_initialized = false;

  ExportDataFixture() {
    if (g_initialized) return;

    // Step 1: CLIO Runtime client init
    bool ok = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    if (!ok) throw std::runtime_error("CHIMAERA_INIT failed");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // Step 2: CTE client + pool
    ok = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    if (!ok) throw std::runtime_error("CLIO_CTE_CLIENT_INIT failed");
    auto *cte = CLIO_CTE_CLIENT;
    cte->Init(clio::cte::core::kCtePoolId);

    clio::cte::core::CreateParams cte_params;
    auto cte_fut = cte->AsyncCreate(chi::PoolQuery::Dynamic(),
                                    clio::cte::core::kCtePoolName,
                                    clio::cte::core::kCtePoolId, cte_params);
    cte_fut.Wait();

    // Step 3: CAE client + pool
    CLIO_CAE_CLIENT_INIT();
    clio::cae::core::Client cae_client;
    clio::cae::core::CreateParams cae_params;
    auto cae_fut = cae_client.AsyncCreate(chi::PoolQuery::Local(),
                                          "test_cae_pool",
                                          clio::cae::core::kCaePoolId, cae_params);
    cae_fut.Wait();

    g_initialized = true;
  }

  /**
   * Put a blob with known data into CTE and return the tag ID.
   * Caller must ensure chimaera is initialised (fixture ctor handles this).
   */
  clio::cte::core::TagId PutBlob(const std::string &tag_name,
                                const std::string &blob_name,
                                const std::vector<uint8_t> &data) {
    auto *cte = CLIO_CTE_CLIENT;

    auto tag_fut = cte->AsyncGetOrCreateTag(tag_name);
    tag_fut.Wait();
    auto tag_id = tag_fut->tag_id_;

    auto buf = CLIO_IPC->AllocateBuffer(data.size());
    std::memcpy(buf.ptr_, data.data(), data.size());
    ctp::ipc::ShmPtr<> shm_ptr = buf.shm_.template Cast<void>();

    auto put_fut = cte->AsyncPutBlob(tag_id, blob_name, 0,
                                     static_cast<chi::u64>(data.size()),
                                     shm_ptr);
    put_fut.Wait();
    CLIO_IPC->FreeBuffer(buf);

    return tag_id;
  }
};

// ---------------------------------------------------------------------------
// ExportDataTask struct tests
// (require runtime for CTP_MALLOC, so all use ExportDataFixture)
// ---------------------------------------------------------------------------

TEST_CASE("ExportData - Task default constructor", "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CLIO_IPC;
  auto task = ipc->NewTask<ExportDataTask>();
  REQUIRE(task->result_code_ == 0);
  REQUIRE(task->bytes_exported_ == 0);
  REQUIRE(task->tag_name_.str() == "");
  REQUIRE(task->output_path_.str() == "");
  REQUIRE(task->format_.str() == "");
  ipc->DelTask(task);

  INFO("ExportDataTask default constructor OK");
}

TEST_CASE("ExportData - Task emplace constructor", "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CLIO_IPC;
  chi::TaskId tid = chi::CreateTaskId();
  auto task = ipc->NewTask<ExportDataTask>(tid, clio::cae::core::kCaePoolId,
                                           chi::PoolQuery::Local(),
                                           "my_tag", "/tmp/out.bin", "binary");

  REQUIRE(task->tag_name_.str() == "my_tag");
  REQUIRE(task->output_path_.str() == "/tmp/out.bin");
  REQUIRE(task->format_.str() == "binary");
  REQUIRE(task->result_code_ == 0);
  REQUIRE(task->bytes_exported_ == 0);
  REQUIRE(task->method_ == Method::kExportData);

  ipc->DelTask(task);
  INFO("ExportDataTask emplace constructor OK");
}

TEST_CASE("ExportData - Task Copy", "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CLIO_IPC;
  auto src = ipc->NewTask<ExportDataTask>(chi::CreateTaskId(),
                                          clio::cae::core::kCaePoolId,
                                          chi::PoolQuery::Local(),
                                          "tag_copy", "/tmp/copy.bin", "binary");
  src->bytes_exported_ = 42;
  src->result_code_ = 7;
  src->error_message_ = chi::priv::string("copy_err", CTP_MALLOC);

  auto dst = ipc->NewTask<ExportDataTask>();
  dst->Copy(src);

  REQUIRE(dst->tag_name_.str() == "tag_copy");
  REQUIRE(dst->output_path_.str() == "/tmp/copy.bin");
  REQUIRE(dst->format_.str() == "binary");
  REQUIRE(dst->bytes_exported_ == 42);
  REQUIRE(dst->result_code_ == 7);
  REQUIRE(dst->error_message_.str() == "copy_err");

  ipc->DelTask(src);
  ipc->DelTask(dst);
  INFO("ExportDataTask Copy OK");
}

TEST_CASE("ExportData - Task Aggregate", "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CLIO_IPC;
  auto orig = ipc->NewTask<ExportDataTask>(chi::CreateTaskId(),
                                           clio::cae::core::kCaePoolId,
                                           chi::PoolQuery::Local(),
                                           "tag_agg", "/tmp/agg.bin", "binary");
  orig->bytes_exported_ = 100;
  orig->result_code_ = 1;

  auto replica = ipc->NewTask<ExportDataTask>(chi::CreateTaskId(),
                                              clio::cae::core::kCaePoolId,
                                              chi::PoolQuery::Local(),
                                              "tag_agg", "/tmp/agg.bin", "binary");
  replica->bytes_exported_ = 200;
  replica->result_code_ = 5;

  orig->Aggregate(replica.template Cast<chi::Task>());

  // Aggregate calls Copy, so orig should now have replica's values
  REQUIRE(orig->bytes_exported_ == 200);
  REQUIRE(orig->result_code_ == 5);

  ipc->DelTask(orig);
  ipc->DelTask(replica);
  INFO("ExportDataTask Aggregate OK");
}

TEST_CASE("ExportData - Task SerializeIn roundtrip", "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CLIO_IPC;
  auto task = ipc->NewTask<ExportDataTask>(chi::CreateTaskId(),
                                           clio::cae::core::kCaePoolId,
                                           chi::PoolQuery::Local(),
                                           "ser_tag", "/tmp/ser.bin", "binary");

  // Write IN fields (tag_name_, output_path_, format_)
  std::vector<char> buf;
  {
    ctp::ipc::GlobalSerialize<std::vector<char>> oa(buf);
    task->SerializeIn(oa);
    oa.Finalize();
  }

  // Read them back into a fresh task
  auto t2 = ipc->NewTask<ExportDataTask>();
  {
    ctp::ipc::GlobalDeserialize<std::vector<char>> ia(buf);
    t2->SerializeIn(ia);
  }

  REQUIRE(t2->tag_name_.str() == "ser_tag");
  REQUIRE(t2->output_path_.str() == "/tmp/ser.bin");
  REQUIRE(t2->format_.str() == "binary");

  ipc->DelTask(task);
  ipc->DelTask(t2);
  INFO("ExportDataTask SerializeIn roundtrip OK");
}

TEST_CASE("ExportData - Task SerializeOut roundtrip", "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CLIO_IPC;
  auto task = ipc->NewTask<ExportDataTask>();
  task->result_code_ = 3;
  task->bytes_exported_ = 777;
  task->error_message_ = chi::priv::string("err_msg", CTP_MALLOC);

  // Write OUT fields (result_code_, error_message_, bytes_exported_)
  std::vector<char> buf;
  {
    ctp::ipc::GlobalSerialize<std::vector<char>> oa(buf);
    task->SerializeOut(oa);
    oa.Finalize();
  }

  auto t2 = ipc->NewTask<ExportDataTask>();
  {
    ctp::ipc::GlobalDeserialize<std::vector<char>> ia(buf);
    t2->SerializeOut(ia);
  }

  REQUIRE(t2->result_code_ == 3);
  REQUIRE(t2->bytes_exported_ == 777);
  REQUIRE(t2->error_message_.str() == "err_msg");

  ipc->DelTask(task);
  ipc->DelTask(t2);
  INFO("ExportDataTask SerializeOut roundtrip OK");
}

// ---------------------------------------------------------------------------
// Runtime integration tests — exercise Runtime::ExportData code paths
// ---------------------------------------------------------------------------

TEST_CASE("ExportData - Empty tag returns success with 0 bytes",
          "[cae][export][runtime]") {
  ExportDataFixture f;

  // Tag does not exist yet; GetOrCreateTag will create it with no blobs.
  clio::cae::core::Client cae(clio::cae::core::kCaePoolId);
  auto fut = cae.AsyncExportData("export_empty_tag_xyz_001",
                                 ExportTmpPath("cae_export_empty.bin"), "binary");
  fut.Wait();

  // blob_names will be empty → success, 0 bytes exported
  REQUIRE(fut->result_code_ == 0);
  REQUIRE(fut->bytes_exported_ == 0);

  INFO("ExportData empty tag: OK");
}

TEST_CASE("ExportData - Binary export roundtrip", "[cae][export][runtime][binary]") {
  ExportDataFixture f;

  const std::string tag_name  = "export_binary_roundtrip_tag";
  const std::string out_path  = ExportTmpPath("cae_export_binary_roundtrip.bin");

  // Put two blobs with known data
  std::vector<uint8_t> data_a = {0xDE, 0xAD, 0xBE, 0xEF};
  std::vector<uint8_t> data_b = {0x01, 0x02, 0x03, 0x04, 0x05};
  f.PutBlob(tag_name, "blob_a", data_a);
  f.PutBlob(tag_name, "blob_b", data_b);

  // Export to binary
  clio::cae::core::Client cae(clio::cae::core::kCaePoolId);
  auto fut = cae.AsyncExportData(tag_name, out_path, "binary");
  fut.Wait();

  REQUIRE(fut->result_code_ == 0);
  REQUIRE(fut->bytes_exported_ == data_a.size() + data_b.size());

  // Parse the output file: [name_len(u32)][name][data_len(u64)][data] ...
  std::ifstream ifs(out_path, std::ios::binary);
  REQUIRE(ifs.is_open());

  size_t blobs_found = 0;
  size_t total_bytes = 0;
  while (ifs.good()) {
    uint32_t name_len = 0;
    ifs.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));
    if (ifs.gcount() < static_cast<std::streamsize>(sizeof(name_len))) break;

    std::string blob_name(name_len, '\0');
    ifs.read(blob_name.data(), name_len);

    uint64_t data_len = 0;
    ifs.read(reinterpret_cast<char *>(&data_len), sizeof(data_len));

    std::vector<char> blob_data(data_len);
    ifs.read(blob_data.data(), static_cast<std::streamsize>(data_len));

    REQUIRE(!blob_name.empty());
    REQUIRE(data_len > 0);
    total_bytes += data_len;
    blobs_found++;
  }

  REQUIRE(blobs_found == 2);
  REQUIRE(total_bytes == data_a.size() + data_b.size());

  std::remove(out_path.c_str());
  INFO("ExportData binary roundtrip: found " << blobs_found << " blobs, "
       << total_bytes << " bytes");
}

TEST_CASE("ExportData - Binary bad output path returns -2",
          "[cae][export][runtime][binary]") {
  ExportDataFixture f;

  // Put a blob so we pass the empty-tag check and reach the file-open branch
  const std::string tag_name = "export_bad_path_tag";
  f.PutBlob(tag_name, "blob_x", {1, 2, 3});

  clio::cae::core::Client cae(clio::cae::core::kCaePoolId);
  auto fut = cae.AsyncExportData(tag_name,
                                 "/nonexistent_dir_xyz_cae/out.bin", "binary");
  fut.Wait();

  REQUIRE(fut->result_code_ == -2);
  INFO("ExportData binary bad path: result_code=" << fut->result_code_);
}

#ifdef CLIO_CAE_ENABLE_HDF5

TEST_CASE("ExportData - HDF5 export roundtrip", "[cae][export][runtime][hdf5]") {
  ExportDataFixture f;

  const std::string tag_name = "export_hdf5_roundtrip_tag";
  const std::string out_path = ExportTmpPath("cae_export_hdf5_roundtrip.h5");

  std::vector<uint8_t> data1 = {10, 20, 30, 40};
  std::vector<uint8_t> data2 = {50, 60, 70};
  f.PutBlob(tag_name, "ds1", data1);
  f.PutBlob(tag_name, "ds2", data2);

  clio::cae::core::Client cae(clio::cae::core::kCaePoolId);
  auto fut = cae.AsyncExportData(tag_name, out_path, "hdf5");
  fut.Wait();

  REQUIRE(fut->result_code_ == 0);
  REQUIRE(fut->bytes_exported_ == data1.size() + data2.size());

  // Verify the HDF5 file has the expected datasets
  hid_t fid = H5Fopen(out_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  REQUIRE(fid >= 0);
  REQUIRE(H5Lexists(fid, "ds1", H5P_DEFAULT) > 0);
  REQUIRE(H5Lexists(fid, "ds2", H5P_DEFAULT) > 0);
  H5Fclose(fid);

  std::remove(out_path.c_str());
  INFO("ExportData HDF5 roundtrip OK");
}

TEST_CASE("ExportData - HDF5 bad output path returns -2",
          "[cae][export][runtime][hdf5]") {
  ExportDataFixture f;

  const std::string tag_name = "export_hdf5_bad_path_tag";
  f.PutBlob(tag_name, "blob_y", {1, 2, 3});

  clio::cae::core::Client cae(clio::cae::core::kCaePoolId);
  auto fut = cae.AsyncExportData(tag_name,
                                 "/nonexistent_dir_xyz_cae/out.h5", "hdf5");
  fut.Wait();

  REQUIRE(fut->result_code_ == -2);
  INFO("ExportData HDF5 bad path: result_code=" << fut->result_code_);
}

#else  // !CLIO_CAE_ENABLE_HDF5

TEST_CASE("ExportData - HDF5 not compiled returns -3",
          "[cae][export][runtime][hdf5]") {
  ExportDataFixture f;

  const std::string tag_name = "export_hdf5_nocompile_tag";
  f.PutBlob(tag_name, "blob_z", {1, 2, 3});

  clio::cae::core::Client cae(clio::cae::core::kCaePoolId);
  auto fut = cae.AsyncExportData(tag_name, "/tmp/no_hdf5_test.h5", "hdf5");
  fut.Wait();

  REQUIRE(fut->result_code_ == -3);
  INFO("ExportData HDF5 not compiled: result_code=" << fut->result_code_);
}

#endif  // CLIO_CAE_ENABLE_HDF5

// ---------------------------------------------------------------------------
// ParseOmniTask struct tests
// ---------------------------------------------------------------------------

TEST_CASE("ParseOmni - Task default constructor", "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CLIO_IPC;
  auto task = ipc->NewTask<ParseOmniTask>();
  REQUIRE(task->serialized_ctx_.str() == "");
  REQUIRE(task->num_tasks_scheduled_ == 0);
  REQUIRE(task->result_code_ == 0);
  REQUIRE(task->error_message_.str() == "");
  ipc->DelTask(task);

  INFO("ParseOmniTask default constructor OK");
}

TEST_CASE("ParseOmni - Task emplace constructor", "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CLIO_IPC;
  std::vector<clio::cae::core::AssimilationCtx> contexts;
  contexts.emplace_back("file::/tmp/a.bin", "iowarp::tag_a", "binary");

  auto task = ipc->NewTask<ParseOmniTask>(chi::CreateTaskId(),
                                          clio::cae::core::kCaePoolId,
                                          chi::PoolQuery::Local(),
                                          contexts);
  REQUIRE(!task->serialized_ctx_.str().empty());
  REQUIRE(task->num_tasks_scheduled_ == 0);
  REQUIRE(task->result_code_ == 0);
  REQUIRE(task->method_ == Method::kParseOmni);

  ipc->DelTask(task);
  INFO("ParseOmniTask emplace constructor OK");
}

TEST_CASE("ParseOmni - Task Copy", "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CLIO_IPC;
  std::vector<clio::cae::core::AssimilationCtx> contexts;
  contexts.emplace_back("file::/tmp/b.bin", "iowarp::tag_b", "binary");

  auto src = ipc->NewTask<ParseOmniTask>(chi::CreateTaskId(),
                                         clio::cae::core::kCaePoolId,
                                         chi::PoolQuery::Local(),
                                         contexts);
  src->num_tasks_scheduled_ = 3;
  src->result_code_ = 2;
  src->error_message_ = chi::priv::string("omni_err", CTP_MALLOC);

  auto dst = ipc->NewTask<ParseOmniTask>();
  dst->Copy(src);

  REQUIRE(dst->serialized_ctx_.str() == src->serialized_ctx_.str());
  REQUIRE(dst->num_tasks_scheduled_ == 3);
  REQUIRE(dst->result_code_ == 2);
  REQUIRE(dst->error_message_.str() == "omni_err");

  ipc->DelTask(src);
  ipc->DelTask(dst);
  INFO("ParseOmniTask Copy OK");
}

TEST_CASE("ParseOmni - Task Aggregate", "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CLIO_IPC;
  std::vector<clio::cae::core::AssimilationCtx> ctxs;
  ctxs.emplace_back("file::/tmp/c.bin", "iowarp::tag_c", "binary");

  auto orig = ipc->NewTask<ParseOmniTask>(chi::CreateTaskId(),
                                          clio::cae::core::kCaePoolId,
                                          chi::PoolQuery::Local(), ctxs);
  orig->num_tasks_scheduled_ = 1;
  orig->result_code_ = 0;

  auto rep = ipc->NewTask<ParseOmniTask>(chi::CreateTaskId(),
                                         clio::cae::core::kCaePoolId,
                                         chi::PoolQuery::Local(), ctxs);
  rep->num_tasks_scheduled_ = 5;
  rep->result_code_ = 9;

  orig->Aggregate(rep.template Cast<chi::Task>());

  REQUIRE(orig->num_tasks_scheduled_ == 5);
  REQUIRE(orig->result_code_ == 9);

  ipc->DelTask(orig);
  ipc->DelTask(rep);
  INFO("ParseOmniTask Aggregate OK");
}

TEST_CASE("ParseOmni - Task SerializeIn roundtrip", "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CLIO_IPC;
  std::vector<clio::cae::core::AssimilationCtx> ctxs;
  ctxs.emplace_back("file::/tmp/d.bin", "iowarp::ser_tag", "binary");

  auto task = ipc->NewTask<ParseOmniTask>(chi::CreateTaskId(),
                                          clio::cae::core::kCaePoolId,
                                          chi::PoolQuery::Local(), ctxs);
  const std::string ser_data = task->serialized_ctx_.str();

  std::vector<char> buf;
  {
    ctp::ipc::GlobalSerialize<std::vector<char>> oa(buf);
    task->SerializeIn(oa);
    oa.Finalize();
  }

  auto t2 = ipc->NewTask<ParseOmniTask>();
  {
    ctp::ipc::GlobalDeserialize<std::vector<char>> ia(buf);
    t2->SerializeIn(ia);
  }

  REQUIRE(t2->serialized_ctx_.str() == ser_data);

  ipc->DelTask(task);
  ipc->DelTask(t2);
  INFO("ParseOmniTask SerializeIn roundtrip OK");
}

TEST_CASE("ParseOmni - Task SerializeOut roundtrip", "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CLIO_IPC;
  auto task = ipc->NewTask<ParseOmniTask>();
  task->num_tasks_scheduled_ = 7;
  task->result_code_ = 4;
  task->error_message_ = chi::priv::string("out_err", CTP_MALLOC);

  std::vector<char> buf;
  {
    ctp::ipc::GlobalSerialize<std::vector<char>> oa(buf);
    task->SerializeOut(oa);
    oa.Finalize();
  }

  auto t2 = ipc->NewTask<ParseOmniTask>();
  {
    ctp::ipc::GlobalDeserialize<std::vector<char>> ia(buf);
    t2->SerializeOut(ia);
  }

  REQUIRE(t2->num_tasks_scheduled_ == 7);
  REQUIRE(t2->result_code_ == 4);
  REQUIRE(t2->error_message_.str() == "out_err");

  ipc->DelTask(task);
  ipc->DelTask(t2);
  INFO("ParseOmniTask SerializeOut roundtrip OK");
}

// ---------------------------------------------------------------------------
// ProcessHdf5DatasetTask struct tests
// ---------------------------------------------------------------------------

TEST_CASE("ProcessHdf5Dataset - Task default constructor",
          "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CLIO_IPC;
  auto task = ipc->NewTask<ProcessHdf5DatasetTask>();
  REQUIRE(task->file_path_.str() == "");
  REQUIRE(task->dataset_path_.str() == "");
  REQUIRE(task->tag_prefix_.str() == "");
  REQUIRE(task->result_code_ == 0);
  REQUIRE(task->error_message_.str() == "");
  ipc->DelTask(task);

  INFO("ProcessHdf5DatasetTask default constructor OK");
}

TEST_CASE("ProcessHdf5Dataset - Task emplace constructor",
          "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CLIO_IPC;
  auto task = ipc->NewTask<ProcessHdf5DatasetTask>(chi::CreateTaskId(),
                                                    clio::cae::core::kCaePoolId,
                                                    chi::PoolQuery::Local(),
                                                    "/tmp/test.h5",
                                                    "/dataset/path",
                                                    "my_prefix");
  REQUIRE(task->file_path_.str() == "/tmp/test.h5");
  REQUIRE(task->dataset_path_.str() == "/dataset/path");
  REQUIRE(task->tag_prefix_.str() == "my_prefix");
  REQUIRE(task->result_code_ == 0);
  REQUIRE(task->method_ == Method::kProcessHdf5Dataset);

  ipc->DelTask(task);
  INFO("ProcessHdf5DatasetTask emplace constructor OK");
}

TEST_CASE("ProcessHdf5Dataset - Task Copy", "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CLIO_IPC;
  auto src = ipc->NewTask<ProcessHdf5DatasetTask>(chi::CreateTaskId(),
                                                   clio::cae::core::kCaePoolId,
                                                   chi::PoolQuery::Local(),
                                                   "/h5/file.h5",
                                                   "/ds",
                                                   "prefix_x");
  src->result_code_ = 5;
  src->error_message_ = chi::priv::string("hdf5_err", CTP_MALLOC);

  auto dst = ipc->NewTask<ProcessHdf5DatasetTask>();
  dst->Copy(src);

  REQUIRE(dst->file_path_.str() == "/h5/file.h5");
  REQUIRE(dst->dataset_path_.str() == "/ds");
  REQUIRE(dst->tag_prefix_.str() == "prefix_x");
  REQUIRE(dst->result_code_ == 5);
  REQUIRE(dst->error_message_.str() == "hdf5_err");

  ipc->DelTask(src);
  ipc->DelTask(dst);
  INFO("ProcessHdf5DatasetTask Copy OK");
}

TEST_CASE("ProcessHdf5Dataset - Task Aggregate keeps first error",
          "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CLIO_IPC;
  auto orig = ipc->NewTask<ProcessHdf5DatasetTask>(chi::CreateTaskId(),
                                                    clio::cae::core::kCaePoolId,
                                                    chi::PoolQuery::Local(),
                                                    "/f1.h5", "/d1", "p1");
  orig->result_code_ = -1;
  orig->error_message_ = chi::priv::string("first_err", CTP_MALLOC);

  auto rep = ipc->NewTask<ProcessHdf5DatasetTask>(chi::CreateTaskId(),
                                                   clio::cae::core::kCaePoolId,
                                                   chi::PoolQuery::Local(),
                                                   "/f2.h5", "/d2", "p2");
  rep->result_code_ = -2;
  rep->error_message_ = chi::priv::string("second_err", CTP_MALLOC);

  // orig already has error → keeps its own error, does not overwrite
  orig->Aggregate(rep.template Cast<chi::Task>());

  REQUIRE(orig->result_code_ == -1);
  REQUIRE(orig->error_message_.str() == "first_err");

  ipc->DelTask(orig);
  ipc->DelTask(rep);
  INFO("ProcessHdf5DatasetTask Aggregate keeps first error OK");
}

TEST_CASE("ProcessHdf5Dataset - Task Aggregate adopts replica error",
          "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CLIO_IPC;
  auto orig = ipc->NewTask<ProcessHdf5DatasetTask>(chi::CreateTaskId(),
                                                    clio::cae::core::kCaePoolId,
                                                    chi::PoolQuery::Local(),
                                                    "/f3.h5", "/d3", "p3");
  orig->result_code_ = 0;  // success so far

  auto rep = ipc->NewTask<ProcessHdf5DatasetTask>(chi::CreateTaskId(),
                                                   clio::cae::core::kCaePoolId,
                                                   chi::PoolQuery::Local(),
                                                   "/f4.h5", "/d4", "p4");
  rep->result_code_ = -3;
  rep->error_message_ = chi::priv::string("rep_err", CTP_MALLOC);

  // orig has no error → adopts replica's error
  orig->Aggregate(rep.template Cast<chi::Task>());

  REQUIRE(orig->result_code_ == -3);
  REQUIRE(orig->error_message_.str() == "rep_err");

  ipc->DelTask(orig);
  ipc->DelTask(rep);
  INFO("ProcessHdf5DatasetTask Aggregate adopts replica error OK");
}

TEST_CASE("ProcessHdf5Dataset - Task SerializeIn roundtrip",
          "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CLIO_IPC;
  auto task = ipc->NewTask<ProcessHdf5DatasetTask>(chi::CreateTaskId(),
                                                    clio::cae::core::kCaePoolId,
                                                    chi::PoolQuery::Local(),
                                                    "/ser/file.h5",
                                                    "/ser/dataset",
                                                    "ser_prefix");
  std::vector<char> buf;
  {
    ctp::ipc::GlobalSerialize<std::vector<char>> oa(buf);
    task->SerializeIn(oa);
    oa.Finalize();
  }

  auto t2 = ipc->NewTask<ProcessHdf5DatasetTask>();
  {
    ctp::ipc::GlobalDeserialize<std::vector<char>> ia(buf);
    t2->SerializeIn(ia);
  }

  REQUIRE(t2->file_path_.str() == "/ser/file.h5");
  REQUIRE(t2->dataset_path_.str() == "/ser/dataset");
  REQUIRE(t2->tag_prefix_.str() == "ser_prefix");

  ipc->DelTask(task);
  ipc->DelTask(t2);
  INFO("ProcessHdf5DatasetTask SerializeIn roundtrip OK");
}

TEST_CASE("ProcessHdf5Dataset - Task SerializeOut roundtrip",
          "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CLIO_IPC;
  auto task = ipc->NewTask<ProcessHdf5DatasetTask>();
  task->result_code_ = 8;
  task->error_message_ = chi::priv::string("ds_err", CTP_MALLOC);

  std::vector<char> buf;
  {
    ctp::ipc::GlobalSerialize<std::vector<char>> oa(buf);
    task->SerializeOut(oa);
    oa.Finalize();
  }

  auto t2 = ipc->NewTask<ProcessHdf5DatasetTask>();
  {
    ctp::ipc::GlobalDeserialize<std::vector<char>> ia(buf);
    t2->SerializeOut(ia);
  }

  REQUIRE(t2->result_code_ == 8);
  REQUIRE(t2->error_message_.str() == "ds_err");

  ipc->DelTask(task);
  ipc->DelTask(t2);
  INFO("ProcessHdf5DatasetTask SerializeOut roundtrip OK");
}

// ---------------------------------------------------------------------------
// ExportData - additional runtime paths
// ---------------------------------------------------------------------------

TEST_CASE("ExportData - Unknown format falls through to binary",
          "[cae][export][runtime]") {
  ExportDataFixture f;

  const std::string tag_name = "export_unknown_fmt_tag";
  const std::string out_path = ExportTmpPath("cae_export_unknown_fmt.bin");

  std::vector<uint8_t> data = {0xAA, 0xBB, 0xCC};
  f.PutBlob(tag_name, "blob_unk", data);

  clio::cae::core::Client cae(clio::cae::core::kCaePoolId);
  // "csv" is neither "hdf5" nor "binary" → falls to binary else-branch
  auto fut = cae.AsyncExportData(tag_name, out_path, "csv");
  fut.Wait();

  // Binary branch should succeed and write the blob
  REQUIRE(fut->result_code_ == 0);
  REQUIRE(fut->bytes_exported_ == data.size());

  std::remove(out_path.c_str());
  INFO("ExportData unknown format (csv) → binary path OK, "
       << fut->bytes_exported_ << " bytes");
}

TEST_CASE("ExportData - Binary export with multiple large blobs",
          "[cae][export][runtime][binary]") {
  ExportDataFixture f;

  const std::string tag_name = "export_multi_blob_tag";
  const std::string out_path = ExportTmpPath("cae_export_multi_blob.bin");

  // Three blobs with distinct sizes
  std::vector<uint8_t> d1(64, 0x11);
  std::vector<uint8_t> d2(128, 0x22);
  std::vector<uint8_t> d3(32, 0x33);
  f.PutBlob(tag_name, "mb1", d1);
  f.PutBlob(tag_name, "mb2", d2);
  f.PutBlob(tag_name, "mb3", d3);

  clio::cae::core::Client cae(clio::cae::core::kCaePoolId);
  auto fut = cae.AsyncExportData(tag_name, out_path, "binary");
  fut.Wait();

  REQUIRE(fut->result_code_ == 0);
  REQUIRE(fut->bytes_exported_ == d1.size() + d2.size() + d3.size());

  std::remove(out_path.c_str());
  INFO("ExportData multi-blob binary: " << fut->bytes_exported_ << " bytes");
}

SIMPLE_TEST_MAIN()
