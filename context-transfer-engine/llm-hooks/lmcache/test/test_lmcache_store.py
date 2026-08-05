# Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
#
# This file is part of IOWarp Core.

import os
import struct

import clio_cte_lmcache_ext

_HEADER = struct.Struct("!8sIIQ")


def main() -> None:
    if os.environ.get("CLIO_LMCACHE_TEST_RUNTIME") != "1":
        return

    store = clio_cte_lmcache_ext.LMCacheStore()
    if not store.Init("", "test_python_lmcache_bytes", "local"):
        return

    payload = bytes([0, 255, 1, 128, 42])
    assert store.PutBytes("python_binary", payload)
    assert store.Exists("python_binary")
    assert store.Size("python_binary") == len(payload)
    assert store.GetBytes("python_binary") == payload
    destination = bytearray(len(payload))
    assert store.GetBytesInto("python_binary", destination)
    assert bytes(destination) == payload
    assert not store.GetBytesInto("python_binary", bytearray(len(payload) - 1))
    assert store.GetBytes("missing") is None
    assert store.Delete("python_binary")
    assert not store.Exists("python_binary")

    names = ["python_batch_a", "python_batch_b", "python_batch_c"]
    payloads = [b"\x00\x01", b"\xff\x80\x7f", b"abc"]
    assert store.PutMany(names, payloads, 2) == [True, True, True]
    assert store.ExistsMany(["python_batch_a", "missing", "python_batch_c"], 2) == [
        True,
        False,
        True,
    ]
    assert store.SizeMany(names, 2) == [len(payload) for payload in payloads]

    values = store.GetMany(["python_batch_a", "missing", "python_batch_c"], 2)
    assert values[0] == payloads[0]
    assert values[1] is None
    assert values[2] == payloads[2]

    destinations = [bytearray(len(payload)) for payload in payloads]
    assert store.GetManyInto(names, destinations, store.SizeMany(names, 2), 2) == [
        True,
        True,
        True,
    ]
    assert [bytes(destination) for destination in destinations] == payloads

    record_names = ["python_record_a", "python_record_b"]
    metadata = [
        b'{"canonical_key":"python_record_a","payload_length":3}',
        '{"canonical_key":"python_record_b","payload_length":4}',
    ]
    record_payloads = [memoryview(bytearray(b"\x00\xff\x01")), b"data"]
    assert store.PutManyRecords(record_names, metadata, record_payloads, 1) == [
        True,
        True,
    ]
    record_values = store.GetMany(record_names, 2)
    expected = [
        _HEADER.pack(b"CLIOKV1\0", 1, len(metadata[0]), 3) + metadata[0] + b"\x00\xff\x01",
        _HEADER.pack(b"CLIOKV1\0", 1, len(metadata[1]), 4)
        + metadata[1].encode("utf-8")
        + b"data",
    ]
    assert record_values == expected
    infos = store.ReadRecordInfos(["python_record_a", "missing", "python_record_b"], 2)
    assert infos[0] == (metadata[0].decode("utf-8"), _HEADER.size + len(metadata[0]), 3)
    assert infos[1] is None
    assert infos[2] == (metadata[1], _HEADER.size + len(metadata[1]), 4)
    payload_destinations = [bytearray(3), bytearray(4)]
    assert store.GetManyRangesInto(
        record_names,
        [infos[0][1], infos[2][1]],
        [infos[0][2], infos[2][2]],
        payload_destinations,
        2,
    ) == [True, True]
    assert [bytes(destination) for destination in payload_destinations] == [
        b"\x00\xff\x01",
        b"data",
    ]
    try:
        store.PutManyRecords(["bad"], [], [], 1)
        raise AssertionError("PutManyRecords should reject mismatched lengths")
    except ValueError:
        pass

    store.Close()


if __name__ == "__main__":
    main()
