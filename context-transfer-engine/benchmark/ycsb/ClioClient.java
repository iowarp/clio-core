/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

package site.ycsb.db;

import site.ycsb.ByteArrayByteIterator;
import site.ycsb.ByteIterator;
import site.ycsb.DB;
import site.ycsb.DBException;
import site.ycsb.Status;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.Vector;

/**
 * YCSB binding for the Clio Context Transfer Engine (issue #862).
 *
 * One CTE tag ("ycsb") holds all records; the YCSB key is the blob name and
 * the record's field map is serialized into the blob payload (field count,
 * then UTF field name + length-prefixed bytes per field). The native side
 * (libclio_ycsb_jni.so) is a thin JNI shim over the CTE SHM client; the Clio
 * runtime daemon must be running before the workload starts.
 *
 * scan() walks a binding-maintained sorted key index (seeded from the tag's
 * blob list at init, updated on puts — the stock Redis binding keeps a zset
 * index for the same reason), enabling workloads D and E. delete() remains
 * NOT_IMPLEMENTED (no core workload deletes).
 */
public class ClioClient extends DB {

  private static native int nativeInit();
  private static native int nativePut(String key, byte[] value);
  private static native int nativePutAsync(String key, byte[] value);
  private static native int nativeDrain();
  private static native byte[] nativeGet(String key);
  private static native String[] nativeScanKeys(String startKey, int count);

  static {
    System.loadLibrary("clio_ycsb_jni");
  }

  @Override
  public void init() throws DBException {
    int rc = nativeInit();
    if (rc != 0) {
      throw new DBException("Clio native init failed rc=" + rc
          + " (is the clio_run daemon running?)");
    }
  }

  @Override
  public void cleanup() throws DBException {
    // Drain the async write window (shared across threads; idempotent) so no
    // put escapes this phase and completion errors fail the phase loudly.
    int errs = nativeDrain();
    if (errs != 0) {
      throw new DBException("Clio async writes failed: " + errs);
    }
  }

  private static byte[] serialize(Map<String, ByteIterator> values)
      throws IOException {
    ByteArrayOutputStream bos = new ByteArrayOutputStream();
    DataOutputStream out = new DataOutputStream(bos);
    out.writeInt(values.size());
    for (Map.Entry<String, ByteIterator> e : values.entrySet()) {
      out.writeUTF(e.getKey());
      byte[] v = e.getValue().toArray();
      out.writeInt(v.length);
      out.write(v);
    }
    out.flush();
    return bos.toByteArray();
  }

  private static void deserialize(byte[] data, Set<String> fields,
      Map<String, ByteIterator> result) throws IOException {
    DataInputStream in = new DataInputStream(new ByteArrayInputStream(data));
    int n = in.readInt();
    for (int i = 0; i < n; i++) {
      String field = in.readUTF();
      int len = in.readInt();
      byte[] v = new byte[len];
      in.readFully(v);
      if (fields == null || fields.contains(field)) {
        result.put(field, new ByteArrayByteIterator(v));
      }
    }
  }

  private String blobName(String table, String key) {
    return table + "/" + key;
  }

  @Override
  public Status read(String table, String key, Set<String> fields,
      Map<String, ByteIterator> result) {
    byte[] data = nativeGet(blobName(table, key));
    if (data == null) {
      return Status.NOT_FOUND;
    }
    try {
      deserialize(data, fields, result);
    } catch (IOException e) {
      return Status.ERROR;
    }
    return Status.OK;
  }

  @Override
  public Status insert(String table, String key,
      Map<String, ByteIterator> values) {
    try {
      int rc = nativePutAsync(blobName(table, key), serialize(values));
      return rc == 0 ? Status.OK : Status.ERROR;
    } catch (IOException e) {
      return Status.ERROR;
    }
  }

  @Override
  public Status update(String table, String key,
      Map<String, ByteIterator> values) {
    // Read-modify-write so partial-field updates (workloads A/B/F) preserve
    // the untouched fields, matching what the redis/rocksdb bindings do.
    byte[] existing = nativeGet(blobName(table, key));
    Map<String, ByteIterator> merged = new HashMap<>();
    if (existing != null) {
      try {
        deserialize(existing, null, merged);
      } catch (IOException e) {
        return Status.ERROR;
      }
    }
    merged.putAll(values);
    return insert(table, key, merged);
  }

  @Override
  public Status delete(String table, String key) {
    return Status.NOT_IMPLEMENTED;
  }

  @Override
  public Status scan(String table, String startkey, int recordcount,
      Set<String> fields, Vector<HashMap<String, ByteIterator>> result) {
    // Ordered scan over the binding-maintained key index (the stock Redis
    // binding does the same with a zset index); values fetched per key.
    String[] keys = nativeScanKeys(blobName(table, startkey), recordcount);
    if (keys == null) {
      return Status.ERROR;
    }
    for (String k : keys) {
      byte[] data = nativeGet(k);
      if (data == null) {
        continue;  // deleted/in-flight between index walk and fetch
      }
      HashMap<String, ByteIterator> row = new HashMap<>();
      try {
        deserialize(data, fields, row);
      } catch (IOException e) {
        return Status.ERROR;
      }
      result.add(row);
    }
    return Status.OK;
  }
}
