/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

package site.ycsb.db.rocksdb;

import org.rocksdb.BlockBasedTableConfig;
import org.rocksdb.BloomFilter;
import org.rocksdb.CompressionType;
import org.rocksdb.DataBlockIndexType;
import org.rocksdb.LRUCache;
import org.rocksdb.Options;
import org.rocksdb.ReadOptions;
import org.rocksdb.RocksDB;
import org.rocksdb.RocksDBException;
import org.rocksdb.WriteOptions;
import site.ycsb.ByteArrayByteIterator;
import site.ycsb.ByteIterator;
import site.ycsb.DB;
import site.ycsb.DBException;
import site.ycsb.Status;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.Vector;

/**
 * A tuned YCSB binding for RocksDB, used to compare CLIO Core against
 * RocksDB's best configuration rather than the stock YCSB 0.17 binding
 * (which is pinned to rocksdbjni 5.11.3 and exposes no tuning knobs).
 *
 * Same semantics as the stock binding: one embedded DB per JVM shared by all
 * YCSB client threads, each record serialized into one value, and update is
 * a read-modify-write of the whole record.
 *
 * Properties (all optional):
 *   rocksdb.dir              data directory (required)
 *   rocksdb.blockCacheMB     LRU block cache size, default 4096
 *   rocksdb.writeBufferMB    memtable size, default 256
 *   rocksdb.bloomBits        bloom filter bits per key, default 10 (0 = off)
 *   rocksdb.disableWAL       true = no write-ahead log, default false
 *   rocksdb.bgJobs           background flush/compaction jobs, default 8
 */
public class RocksDBTunedClient extends DB {

  private static final Object LOCK = new Object();
  private static RocksDB db = null;
  private static Options options = null;
  private static LRUCache cache = null;
  private static WriteOptions writeOptions = null;
  private static ReadOptions readOptions = null;
  private static int references = 0;

  static {
    RocksDB.loadLibrary();
  }

  /**
   * Opens the shared DB on the first init() of this JVM.
   *
   * @throws DBException if the directory is missing or RocksDB fails to open
   */
  @Override
  public void init() throws DBException {
    synchronized (LOCK) {
      if (db == null) {
        openDb(getProperties());
      }
      references++;
    }
  }

  private static long prop(Properties p, String key, long dflt) {
    return Long.parseLong(p.getProperty(key, Long.toString(dflt)));
  }

  private static void openDb(Properties p) throws DBException {
    String dir = p.getProperty("rocksdb.dir");
    if (dir == null) {
      throw new DBException("rocksdb.dir is required");
    }
    long cacheMb = prop(p, "rocksdb.blockCacheMB", 4096);
    long wbMb = prop(p, "rocksdb.writeBufferMB", 256);
    long bloomBits = prop(p, "rocksdb.bloomBits", 10);
    int bgJobs = (int) prop(p, "rocksdb.bgJobs", 8);
    boolean noWal = Boolean.parseBoolean(p.getProperty("rocksdb.disableWAL", "false"));

    cache = new LRUCache(cacheMb << 20);
    BlockBasedTableConfig table = new BlockBasedTableConfig()
        .setBlockCache(cache)
        .setBlockSize(16 << 10)
        .setCacheIndexAndFilterBlocks(true)
        .setPinL0FilterAndIndexBlocksInCache(true)
        .setDataBlockIndexType(DataBlockIndexType.kDataBlockBinaryAndHash);
    if (bloomBits > 0) {
      table.setFilterPolicy(new BloomFilter((double) bloomBits, false));
    }
    options = new Options()
        .setCreateIfMissing(true)
        .setIncreaseParallelism(Runtime.getRuntime().availableProcessors())
        .setMaxBackgroundJobs(bgJobs)
        .setWriteBufferSize(wbMb << 20)
        .setMaxWriteBufferNumber(4)
        .setLevelCompactionDynamicLevelBytes(true)
        .setCompressionType(CompressionType.NO_COMPRESSION)
        .setAllowConcurrentMemtableWrite(true)
        .setEnablePipelinedWrite(true)
        .setTableFormatConfig(table);
    writeOptions = new WriteOptions().setDisableWAL(noWal);
    readOptions = new ReadOptions().setVerifyChecksums(false);
    try {
      db = RocksDB.open(options, dir);
    } catch (RocksDBException e) {
      throw new DBException(e);
    }
  }

  /** Closes the shared DB when the last client thread exits. */
  @Override
  public void cleanup() throws DBException {
    synchronized (LOCK) {
      references--;
      if (references == 0 && db != null) {
        db.close();
        db = null;
        writeOptions.close();
        readOptions.close();
        options.close();
        cache.close();
      }
    }
  }

  private static byte[] key(String table, String key) {
    return (table + "/" + key).getBytes(StandardCharsets.UTF_8);
  }

  private static byte[] serialize(Map<String, ByteIterator> values) {
    Map<String, byte[]> raw = new HashMap<>();
    int size = 0;
    for (Map.Entry<String, ByteIterator> e : values.entrySet()) {
      byte[] v = e.getValue().toArray();
      raw.put(e.getKey(), v);
      size += 8 + e.getKey().length() + v.length;
    }
    ByteBuffer bb = ByteBuffer.allocate(size);
    for (Map.Entry<String, byte[]> e : raw.entrySet()) {
      byte[] k = e.getKey().getBytes(StandardCharsets.UTF_8);
      bb.putInt(k.length).put(k).putInt(e.getValue().length).put(e.getValue());
    }
    return bb.array();
  }

  private static void deserialize(byte[] data, Set<String> fields,
      Map<String, ByteIterator> result) {
    ByteBuffer bb = ByteBuffer.wrap(data);
    while (bb.remaining() > 0) {
      byte[] k = new byte[bb.getInt()];
      bb.get(k);
      byte[] v = new byte[bb.getInt()];
      bb.get(v);
      String name = new String(k, StandardCharsets.UTF_8);
      if (fields == null || fields.contains(name)) {
        result.put(name, new ByteArrayByteIterator(v));
      }
    }
  }

  @Override
  public Status read(String table, String key, Set<String> fields,
      Map<String, ByteIterator> result) {
    try {
      byte[] v = db.get(readOptions, key(table, key));
      if (v == null) {
        return Status.NOT_FOUND;
      }
      deserialize(v, fields, result);
      return Status.OK;
    } catch (RocksDBException e) {
      return Status.ERROR;
    }
  }

  @Override
  public Status scan(String table, String startkey, int recordcount,
      Set<String> fields, Vector<HashMap<String, ByteIterator>> result) {
    return Status.NOT_IMPLEMENTED;
  }

  /** Read-modify-write of the whole record, as in the stock binding. */
  @Override
  public Status update(String table, String key,
      Map<String, ByteIterator> values) {
    try {
      byte[] k = key(table, key);
      byte[] cur = db.get(readOptions, k);
      if (cur == null) {
        return Status.NOT_FOUND;
      }
      Map<String, ByteIterator> merged = new HashMap<>();
      deserialize(cur, null, merged);
      merged.putAll(values);
      db.put(writeOptions, k, serialize(merged));
      return Status.OK;
    } catch (RocksDBException e) {
      return Status.ERROR;
    }
  }

  @Override
  public Status insert(String table, String key,
      Map<String, ByteIterator> values) {
    try {
      db.put(writeOptions, key(table, key), serialize(values));
      return Status.OK;
    } catch (RocksDBException e) {
      return Status.ERROR;
    }
  }

  @Override
  public Status delete(String table, String key) {
    try {
      db.delete(writeOptions, key(table, key));
      return Status.OK;
    } catch (RocksDBException e) {
      return Status.ERROR;
    }
  }
}
