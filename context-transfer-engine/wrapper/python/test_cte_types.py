#!/usr/bin/env python3
"""
Runtime-free coverage for the CTE Python binding *types*.

The existing test_bindings.py focuses on the runtime-dependent Tag/Client
data path and skips most assertions when no runtime is available. This test
exercises the pure value types and their bound methods — UniqueId/TagId,
PoolQuery factory methods, the search-result repr lambdas, the CteTelemetry
constructors/fields, and the enums — none of which need a running runtime.
That covers the binding definitions and lambda bodies (e.g. the __repr__
implementations) deterministically.
"""

import os
import sys

sys.path.insert(0, os.getcwd())
# Make sure no runtime is spun up for this pure-types test.
os.environ.pop("CHI_WITH_RUNTIME", None)
os.environ.pop("CLIO_WITH_RUNTIME", None)

import clio_cte_core_ext as cte


def test_unique_id():
    # Default + (major, minor) constructors, accessors, field read/write.
    a = cte.UniqueId()
    b = cte.UniqueId(major=7, minor=3)
    assert b.major_ == 7
    assert b.minor_ == 3
    assert b.ToU64() != 0
    null = cte.UniqueId.GetNull()
    assert null.IsNull()
    assert not b.IsNull()
    b.major_ = 11
    assert b.major_ == 11
    # TagId / BlobId / PoolId are aliases of the same class.
    assert cte.TagId is cte.UniqueId
    assert cte.BlobId is cte.UniqueId
    assert cte.PoolId is cte.UniqueId
    print("PASSED: test_unique_id")


def test_pool_query():
    # All three factory methods + their default/explicit args.
    assert cte.PoolQuery.Broadcast() is not None
    assert cte.PoolQuery.Broadcast(net_timeout=1.5) is not None
    assert cte.PoolQuery.Dynamic() is not None
    assert cte.PoolQuery.Dynamic(net_timeout=2.0) is not None
    assert cte.PoolQuery.Local() is not None
    assert cte.PoolQuery.Local(parallelism=8) is not None
    assert cte.PoolQuery() is not None
    print("PASSED: test_pool_query")


def test_search_result_repr():
    s = cte.SemanticSearchResult()
    s.tag_id = cte.UniqueId(major=2, minor=5)
    s.blob_name = "blobA"
    s.score = 0.875
    assert s.blob_name == "blobA"
    assert abs(s.score - 0.875) < 1e-6
    rep = repr(s)  # exercises the __repr__ lambda body
    assert "SemanticSearchResult" in rep
    assert "blobA" in rep

    t = cte.TemporalSearchResult()
    t.tag_id = cte.UniqueId(major=4, minor=9)
    t.blob_name = "blobB"
    t.last_modified = 12345
    rep = repr(t)
    assert "TemporalSearchResult" in rep
    assert "blobB" in rep
    print("PASSED: test_search_result_repr")


def test_telemetry_and_enums():
    # Enum values are bound.
    assert cte.CteOp.kPutBlob is not None
    assert cte.CteOp.kGetBlob is not None
    assert cte.CteOp.kDelBlob is not None

    # Default-constructed telemetry + field read/write.
    tel = cte.CteTelemetry()
    tel.off_ = 64
    tel.size_ = 4096
    tel.logical_time_ = 7
    tel.op_ = cte.CteOp.kPutBlob
    tel.tag_id_ = cte.UniqueId(major=1, minor=2)
    assert tel.off_ == 64
    assert tel.size_ == 4096
    assert tel.logical_time_ == 7
    print("PASSED: test_telemetry_and_enums")


if __name__ == "__main__":
    test_unique_id()
    test_pool_query()
    test_search_result_repr()
    test_telemetry_and_enums()
    print("\nAll CTE type-binding tests PASSED")
