/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Unit tests for the header-only Write-Ahead Transaction Log
 * (clio_cte/core/transaction_log.h). Exercises every Log/Deserialize
 * helper plus Open/Sync/Size/Load/Truncate/Close with temp files.
 */

#include "simple_test.h"

#include <clio_cte/core/transaction_log.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace clio::cte::core;
namespace fs = std::filesystem;

static std::string TxnTempFile(const std::string &name) {
  std::string fname = name + "_" + std::to_string(::getpid()) + ".wal";
  return (fs::temp_directory_path() / fname).string();
}

static void TxnRemove(const std::string &path) {
  std::error_code ec;
  fs::remove(path, ec);
}

TEST_CASE("TransactionLog - Open Log Sync Size", "[cte][txnlog]") {
  std::string path = TxnTempFile("txn_basic");
  TxnRemove(path);

  TransactionLog log;
  // Size() before the file exists -> 0
  REQUIRE(log.Size() == 0);
  log.Open(path, 4096);

  TxnCreateNewBlob txn;
  txn.tag_major_ = 1;
  txn.tag_minor_ = 2;
  txn.blob_name_ = "blob_a";
  txn.score_ = 0.5f;
  log.Log(TxnType::kCreateNewBlob, txn);
  log.Sync();

  // Record: 1 byte type + 4 byte size + payload(4+4+4+6+4)
  REQUIRE(log.Size() == 1 + 4 + (4 + 4 + 4 + 6 + 4));

  log.Close();
  TxnRemove(path);
}

TEST_CASE("TransactionLog - CreateNewBlob roundtrip", "[cte][txnlog]") {
  std::string path = TxnTempFile("txn_create_blob");
  TxnRemove(path);

  TransactionLog log;
  log.Open(path, 4096);
  TxnCreateNewBlob txn;
  txn.tag_major_ = 7;
  txn.tag_minor_ = 9;
  txn.blob_name_ = "my_blob";
  txn.score_ = 0.75f;
  log.Log(TxnType::kCreateNewBlob, txn);
  log.Sync();

  auto entries = log.Load();
  REQUIRE(entries.size() == 1);
  REQUIRE(entries[0].first == TxnType::kCreateNewBlob);
  TxnCreateNewBlob out =
      TransactionLog::DeserializeCreateNewBlob(entries[0].second);
  REQUIRE(out.tag_major_ == 7);
  REQUIRE(out.tag_minor_ == 9);
  REQUIRE(out.blob_name_ == "my_blob");
  REQUIRE(out.score_ > 0.74f);
  REQUIRE(out.score_ < 0.76f);

  log.Close();
  TxnRemove(path);
}

TEST_CASE("TransactionLog - ExtendBlob roundtrip with blocks", "[cte][txnlog]") {
  std::string path = TxnTempFile("txn_extend_blob");
  TxnRemove(path);

  TransactionLog log;
  log.Open(path, 4096);

  TxnExtendBlob txn;
  txn.tag_major_ = 3;
  txn.tag_minor_ = 4;
  txn.blob_name_ = "extended";
  TxnExtendBlobBlock blk0;
  blk0.bdev_major_ = 512;
  blk0.bdev_minor_ = 1;
  blk0.target_query_ = chi::PoolQuery::Local();
  blk0.target_offset_ = 4096;
  blk0.size_ = 1024;
  TxnExtendBlobBlock blk1;
  blk1.bdev_major_ = 513;
  blk1.bdev_minor_ = 2;
  blk1.target_query_ = chi::PoolQuery::Local();
  blk1.target_offset_ = 8192;
  blk1.size_ = 2048;
  txn.new_blocks_.push_back(blk0);
  txn.new_blocks_.push_back(blk1);
  log.Log(TxnType::kExtendBlob, txn);
  log.Sync();

  auto entries = log.Load();
  REQUIRE(entries.size() == 1);
  REQUIRE(entries[0].first == TxnType::kExtendBlob);
  TxnExtendBlob out = TransactionLog::DeserializeExtendBlob(entries[0].second);
  REQUIRE(out.tag_major_ == 3);
  REQUIRE(out.tag_minor_ == 4);
  REQUIRE(out.blob_name_ == "extended");
  REQUIRE(out.new_blocks_.size() == 2);
  REQUIRE(out.new_blocks_[0].bdev_major_ == 512);
  REQUIRE(out.new_blocks_[0].bdev_minor_ == 1);
  REQUIRE(out.new_blocks_[0].target_offset_ == 4096);
  REQUIRE(out.new_blocks_[0].size_ == 1024);
  REQUIRE(out.new_blocks_[1].bdev_major_ == 513);
  REQUIRE(out.new_blocks_[1].bdev_minor_ == 2);
  REQUIRE(out.new_blocks_[1].target_offset_ == 8192);
  REQUIRE(out.new_blocks_[1].size_ == 2048);
  // PoolQuery raw bytes round-trip
  REQUIRE(std::memcmp(&out.new_blocks_[0].target_query_,
                      &blk0.target_query_, sizeof(chi::PoolQuery)) == 0);

  log.Close();
  TxnRemove(path);
}

TEST_CASE("TransactionLog - ClearBlob DelBlob roundtrip", "[cte][txnlog]") {
  std::string path = TxnTempFile("txn_clear_del");
  TxnRemove(path);

  TransactionLog log;
  log.Open(path, 4096);

  TxnClearBlob clear_txn;
  clear_txn.tag_major_ = 11;
  clear_txn.tag_minor_ = 12;
  clear_txn.blob_name_ = "to_clear";
  log.Log(TxnType::kClearBlob, clear_txn);

  TxnDelBlob del_txn;
  del_txn.tag_major_ = 13;
  del_txn.tag_minor_ = 14;
  del_txn.blob_name_ = "to_delete";
  log.Log(TxnType::kDelBlob, del_txn);
  log.Sync();

  auto entries = log.Load();
  REQUIRE(entries.size() == 2);
  REQUIRE(entries[0].first == TxnType::kClearBlob);
  REQUIRE(entries[1].first == TxnType::kDelBlob);

  TxnClearBlob clear_out =
      TransactionLog::DeserializeClearBlob(entries[0].second);
  REQUIRE(clear_out.tag_major_ == 11);
  REQUIRE(clear_out.tag_minor_ == 12);
  REQUIRE(clear_out.blob_name_ == "to_clear");

  TxnDelBlob del_out = TransactionLog::DeserializeDelBlob(entries[1].second);
  REQUIRE(del_out.tag_major_ == 13);
  REQUIRE(del_out.tag_minor_ == 14);
  REQUIRE(del_out.blob_name_ == "to_delete");

  log.Close();
  TxnRemove(path);
}

TEST_CASE("TransactionLog - CreateTag DelTag roundtrip", "[cte][txnlog]") {
  std::string path = TxnTempFile("txn_tags");
  TxnRemove(path);

  TransactionLog log;
  log.Open(path, 4096);

  TxnCreateTag create_txn;
  create_txn.tag_name_ = "tag_one";
  create_txn.tag_major_ = 21;
  create_txn.tag_minor_ = 22;
  log.Log(TxnType::kCreateTag, create_txn);

  TxnDelTag del_txn;
  del_txn.tag_name_ = "tag_two";
  del_txn.tag_major_ = 23;
  del_txn.tag_minor_ = 24;
  log.Log(TxnType::kDelTag, del_txn);
  log.Sync();

  auto entries = log.Load();
  REQUIRE(entries.size() == 2);
  REQUIRE(entries[0].first == TxnType::kCreateTag);
  REQUIRE(entries[1].first == TxnType::kDelTag);

  TxnCreateTag create_out =
      TransactionLog::DeserializeCreateTag(entries[0].second);
  REQUIRE(create_out.tag_name_ == "tag_one");
  REQUIRE(create_out.tag_major_ == 21);
  REQUIRE(create_out.tag_minor_ == 22);

  TxnDelTag del_out = TransactionLog::DeserializeDelTag(entries[1].second);
  REQUIRE(del_out.tag_name_ == "tag_two");
  REQUIRE(del_out.tag_major_ == 23);
  REQUIRE(del_out.tag_minor_ == 24);

  log.Close();
  TxnRemove(path);
}

TEST_CASE("TransactionLog - Load from nonexistent file", "[cte][txnlog]") {
  TransactionLog log;
  // No Open() was performed -> file_path_ empty, fs::exists("") is false
  auto entries = log.Load();
  REQUIRE(entries.empty());

  // Logging on a closed log is a no-op (WriteRecord early-returns)
  TxnDelTag txn;
  txn.tag_name_ = "noop";
  txn.tag_major_ = 1;
  txn.tag_minor_ = 1;
  log.Log(TxnType::kDelTag, txn);
  REQUIRE(log.Size() == 0);
}

TEST_CASE("TransactionLog - Truncate resets the file", "[cte][txnlog]") {
  std::string path = TxnTempFile("txn_truncate");
  TxnRemove(path);

  TransactionLog log;
  log.Open(path, 4096);
  TxnCreateTag txn;
  txn.tag_name_ = "will_be_truncated";
  txn.tag_major_ = 1;
  txn.tag_minor_ = 2;
  log.Log(TxnType::kCreateTag, txn);
  log.Sync();
  REQUIRE(log.Size() > 0);

  log.Truncate();
  REQUIRE(log.Size() == 0);
  REQUIRE(log.Load().empty());

  // The log remains usable in append mode after Truncate
  log.Log(TxnType::kCreateTag, txn);
  log.Sync();
  auto entries = log.Load();
  REQUIRE(entries.size() == 1);
  TxnCreateTag out = TransactionLog::DeserializeCreateTag(entries[0].second);
  REQUIRE(out.tag_name_ == "will_be_truncated");

  log.Close();
  TxnRemove(path);
}

TEST_CASE("TransactionLog - Append across Open Close cycles", "[cte][txnlog]") {
  std::string path = TxnTempFile("txn_reopen");
  TxnRemove(path);

  {
    TransactionLog log;
    log.Open(path, 4096);
    TxnCreateNewBlob txn;
    txn.tag_major_ = 1;
    txn.tag_minor_ = 1;
    txn.blob_name_ = "first";
    txn.score_ = 0.1f;
    log.Log(TxnType::kCreateNewBlob, txn);
    // destructor calls Close() -> flush
  }
  {
    TransactionLog log;
    log.Open(path, 4096);
    TxnDelBlob txn;
    txn.tag_major_ = 1;
    txn.tag_minor_ = 1;
    txn.blob_name_ = "first";
    log.Log(TxnType::kDelBlob, txn);
    log.Sync();

    auto entries = log.Load();
    REQUIRE(entries.size() == 2);
    REQUIRE(entries[0].first == TxnType::kCreateNewBlob);
    REQUIRE(entries[1].first == TxnType::kDelBlob);
    log.Close();
    // Close twice is safe
    log.Close();
  }
  TxnRemove(path);
}

TEST_CASE("TransactionLog - Truncated trailing record is skipped",
          "[cte][txnlog]") {
  std::string path = TxnTempFile("txn_partial");
  TxnRemove(path);

  TransactionLog log;
  log.Open(path, 4096);
  TxnCreateTag txn;
  txn.tag_name_ = "complete";
  txn.tag_major_ = 5;
  txn.tag_minor_ = 6;
  log.Log(TxnType::kCreateTag, txn);
  log.Sync();

  // Append a corrupt half-record: type byte + size claiming more payload
  // bytes than are present.
  {
    std::ofstream ofs(path, std::ios::binary | std::ios::app);
    uint8_t type_byte = 0;
    uint32_t payload_size = 1024;  // lie: no payload follows
    ofs.write(reinterpret_cast<const char *>(&type_byte), sizeof(type_byte));
    ofs.write(reinterpret_cast<const char *>(&payload_size),
              sizeof(payload_size));
    char partial[3] = {1, 2, 3};
    ofs.write(partial, sizeof(partial));
  }

  auto entries = log.Load();
  REQUIRE(entries.size() == 1);
  REQUIRE(entries[0].first == TxnType::kCreateTag);

  log.Close();
  TxnRemove(path);
}

SIMPLE_TEST_MAIN()
