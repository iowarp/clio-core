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
  private static native int nativePutDirect(String key, java.nio.ByteBuffer buf, int len);
  private static native int nativeGetDirect(String key, java.nio.ByteBuffer buf);
  private static native java.nio.ByteBuffer nativeGetView(String key, long[] genOut);
  private static native boolean nativeValidateGen(String key, long gen);

  static {
    System.loadLibrary("clio_ycsb_jni");
  }

  // ---- copy-reduction plumbing (items 1-3) --------------------------------
  // Writes: a per-thread ring of direct ByteBuffers, sized comfortably above
  // the shim's async window (default 64) so a slot's put is guaranteed reaped
  // (the window is a global FIFO) before its buffer is reused. Serialization
  // writes ONCE into the direct buffer; the native put uses its address as
  // the source (runtime mode reads it directly; client mode stages at
  // submit). Reads: one per-thread direct buffer for copying reads, and a
  // zero-copy view path over the blob's RAM extent, gen-validated after the
  // row is built (fall back to the copying read on mismatch).
  // AsyncPutBlobDefer owns a copy of the bytes at submit, so one reusable
  // per-thread direct buffer is all the write path needs — no ring, no
  // lifetime coupling, no knobs.
  private static final int BUF_CAP = 64 * 1024;
  private static final ThreadLocal<java.nio.ByteBuffer> WRITE_BUF =
      ThreadLocal.withInitial(() -> java.nio.ByteBuffer.allocateDirect(BUF_CAP));
  private static final ThreadLocal<java.nio.ByteBuffer> READ_BUF =
      ThreadLocal.withInitial(() -> java.nio.ByteBuffer.allocateDirect(1024 * 1024));
  private static final ThreadLocal<long[]> GEN_OUT =
      ThreadLocal.withInitial(() -> new long[1]);

  /** Zero-copy ByteIterator over a slice of a (shared) ByteBuffer; absolute
   *  reads so many iterators can share one buffer without position races. */
  private static final class BBByteIterator extends ByteIterator {
    private final java.nio.ByteBuffer bb;
    private int pos;
    private final int end;
    BBByteIterator(java.nio.ByteBuffer bb, int start, int len) {
      this.bb = bb;
      this.pos = start;
      this.end = start + len;
    }
    @Override public boolean hasNext() { return pos < end; }
    @Override public byte nextByte() { return bb.get(pos++); }
    @Override public long bytesLeft() { return end - pos; }
  }

  /** Serialize the field map ONCE into the direct buffer, in exactly the
   *  DataOutputStream format deserialize() expects (writeUTF == u16 length +
   *  modified-UTF8; YCSB field names are ASCII, where the encodings agree). */
  private static int serializeInto(java.nio.ByteBuffer bb,
      Map<String, ByteIterator> values) throws IOException {
    try {
      bb.putInt(values.size());
      for (Map.Entry<String, ByteIterator> e : values.entrySet()) {
        byte[] name = e.getKey().getBytes(java.nio.charset.StandardCharsets.US_ASCII);
        bb.putShort((short) name.length);
        bb.put(name);
        byte[] v = e.getValue().toArray();
        bb.putInt(v.length);
        bb.put(v);
      }
      return bb.position();
    } catch (java.nio.BufferOverflowException e) {
      throw new IOException("record exceeds direct buffer capacity", e);
    }
  }

  /** Deserialize from a ByteBuffer with zero-copy field views (BBByteIterator
   *  slices). The views are valid only until the backing buffer is reused —
   *  callers must fully consume the row within the current operation. */
  private static void deserializeBB(java.nio.ByteBuffer bb, int limit,
      Set<String> fields, Map<String, ByteIterator> result) throws IOException {
    try {
      java.nio.ByteBuffer d = bb.duplicate();
      d.position(0).limit(limit);
      int count = d.getInt();
      byte[] nameBuf = new byte[256];
      for (int i = 0; i < count; i++) {
        int nlen = d.getShort() & 0xFFFF;
        if (nlen > nameBuf.length) nameBuf = new byte[nlen];
        d.get(nameBuf, 0, nlen);
        String name = new String(nameBuf, 0, nlen,
            java.nio.charset.StandardCharsets.US_ASCII);
        int vlen = d.getInt();
        int vstart = d.position();
        if (fields == null || fields.contains(name)) {
          result.put(name, new BBByteIterator(bb, vstart, vlen));
        }
        d.position(vstart + vlen);
      }
    } catch (RuntimeException e) {
      throw new IOException("malformed record", e);
    }
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
    String name = blobName(table, key);
    // Item 3: zero-copy view over the blob's RAM extent, validated by
    // placement generation AFTER the row is built (torn/moved -> retry via
    // the copying path). The field views point into the store; YCSB consumes
    // them within this operation.
    long[] gen = GEN_OUT.get();
    java.nio.ByteBuffer view = nativeGetView(name, gen);
    if (view != null) {
      try {
        deserializeBB(view, view.capacity(), fields, result);
        if (nativeValidateGen(name, gen[0])) {
          return Status.OK;
        }
        result.clear();  // placement moved mid-read: fall through and copy
      } catch (IOException e) {
        result.clear();
      }
    }
    // Item 2: one-copy read into the per-thread direct buffer.
    java.nio.ByteBuffer buf = READ_BUF.get();
    int n = nativeGetDirect(name, buf);
    if (n == 0) {
      return Status.NOT_FOUND;
    }
    if (n < 0) {
      return Status.ERROR;
    }
    try {
      deserializeBB(buf, n, fields, result);
    } catch (IOException e) {
      return Status.ERROR;
    }
    return Status.OK;
  }

  @Override
  public Status insert(String table, String key,
      Map<String, ByteIterator> values) {
    try {
      java.nio.ByteBuffer wb = (java.nio.ByteBuffer) WRITE_BUF.get().clear();
      int len = serializeInto(wb, values);
      int rc = nativePutDirect(blobName(table, key), wb, len);
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
    byte[] existing = nativeGet(blobName(table, key));  // scan/RMW keep the
    // byte[] path: the merged row must stay valid across the serialize.
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
