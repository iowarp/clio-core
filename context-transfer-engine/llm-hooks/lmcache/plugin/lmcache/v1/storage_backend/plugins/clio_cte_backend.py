# Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
#
# This file is part of IOWarp Core.

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import struct
import sys
import threading
import time
from collections import Counter
from typing import Any, Iterable

try:
    import torch
    from lmcache.v1.memory_management import MemoryFormat
    from lmcache.v1.storage_backend.abstract_backend import (
        StoragePluginInterface,
    )
except Exception:
    class StoragePluginInterface:  # type: ignore[no-redef]
        pass
    torch = None  # type: ignore[assignment]
    MemoryFormat = None  # type: ignore[assignment]

import clio_cte_lmcache_ext


_MAGIC = b"CLIOKV1\0"
_VERSION = 1
_HEADER = struct.Struct("!8sIIQ")


class ClioCTEBackend(StoragePluginInterface):
    """LMCache storage plugin backed by CTE blobs."""

    def __init__(
        self,
        config: Any = None,
        metadata: Any = None,
        local_cpu_backend: Any = None,
        loop: Any = None,
        dst_device: str = "cpu",
        **_: Any,
    ) -> None:
        try:
            super().__init__(
                dst_device=dst_device,
                config=config,
                metadata=metadata,
                local_cpu_backend=local_cpu_backend,
                loop=loop,
            )
        except TypeError:
            self.config = config
            self.metadata = metadata
            self.local_cpu_backend = local_cpu_backend
            self.loop = loop
        self._pins: Counter[str] = Counter()
        self._put_tasks: set[str] = set()
        self._lock = threading.RLock()
        self._profile = _profile_enabled()

        extra_config = _extra_config(config)
        tag_name = extra_config.get("clio_cte.tag_name", "lmcache_kv")
        config_path = extra_config.get("clio_cte.config_path", "")
        pool_query_mode = extra_config.get("clio_cte.pool_query_mode", "local")
        # Up to 256 async CTE ops in flight before the window drains — matches
        # LMCacheStore::kDefaultMaxInflight and gives the runtime's task
        # batching a burst deep enough to merge.
        self.max_inflight = int(extra_config.get("clio_cte.max_inflight", 256))

        self.store = clio_cte_lmcache_ext.LMCacheStore()
        if not self.store.Init(config_path, tag_name, pool_query_mode):
            raise RuntimeError("ClioCTEBackend failed to initialize CTE store")

    def __str__(self) -> str:
        return "ClioCTEBackend"

    def contains(self, key: Any, pin: bool = False) -> bool:
        blob_name = _blob_name(key)
        exists = bool(self.store.Exists(blob_name))
        if exists and pin:
            self.pin(key)
        return exists

    def batched_contains(self, keys: Iterable[Any], pin: bool = False) -> int:
        key_list = list(keys)
        blob_names = [_blob_name(key) for key in key_list]
        hit_count = 0
        for key, exists in zip(
            key_list, self.store.ExistsMany(blob_names, self.max_inflight)
        ):
            if not exists:
                break
            if pin:
                self._pin_existing(_blob_name(key))
            hit_count += 1
        return hit_count

    async def async_batched_submit_put_task(
        self,
        keys: Iterable[Any],
        objs: Iterable[Any],
        transfer_spec: Any = None,
        on_complete_callback: Any = None,
    ) -> None:
        await asyncio.to_thread(
            self.batched_submit_put_task,
            list(keys),
            list(objs),
            transfer_spec,
            on_complete_callback,
        )

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: Iterable[Any],
        pin: bool = False,
    ) -> int:
        return await asyncio.to_thread(self.batched_contains, list(keys), pin)

    def batched_submit_put_task(
        self,
        keys: Iterable[Any],
        objs: Iterable[Any],
        transfer_spec: Any = None,
        on_complete_callback: Any = None,
    ) -> None:
        self._require_worker("batched_submit_put_task")
        total_start = time.perf_counter()
        pairs = list(zip(keys, objs))
        key_list = [key for key, _ in pairs]
        obj_list = [obj for _, obj in pairs]
        encode_start = time.perf_counter()
        blob_names = [_blob_name(key) for key in key_list]
        payloads = [memoryview(getattr(obj, "byte_array")) for obj in obj_list]
        metadata_jsons = [
            _encode_metadata(key, obj, payload.nbytes)
            for key, obj, payload in zip(key_list, obj_list, payloads)
        ]
        encode_elapsed = time.perf_counter() - encode_start
        for blob_name in blob_names:
            with self._lock:
                self._put_tasks.add(blob_name)
        try:
            store_start = time.perf_counter()
            results = self.store.PutManyRecords(
                blob_names, metadata_jsons, payloads, self.max_inflight
            )
            store_elapsed = time.perf_counter() - store_start
            if on_complete_callback is not None:
                callback_start = time.perf_counter()
                for key, committed in zip(key_list, results):
                    if committed:
                        on_complete_callback(key)
                callback_elapsed = time.perf_counter() - callback_start
            else:
                callback_elapsed = 0.0
            if self._profile:
                payload_bytes = sum(payload.nbytes for payload in payloads)
                record_bytes = (
                    payload_bytes
                    + sum(len(metadata) for metadata in metadata_jsons)
                    + (_HEADER.size * len(metadata_jsons))
                )
                _profile_log(
                    "python op=batched_submit_put_task "
                    f"count={len(key_list)} bytes={record_bytes} "
                    f"payload_bytes={payload_bytes} "
                    f"success={sum(1 for item in results if item)} "
                    f"total_ms={_ms(time.perf_counter() - total_start):.3f} "
                    f"encode_ms={_ms(encode_elapsed):.3f} "
                    f"store_ms={_ms(store_elapsed):.3f} "
                    f"callback_ms={_ms(callback_elapsed):.3f}"
                )
        finally:
            for blob_name in blob_names:
                with self._lock:
                    self._put_tasks.discard(blob_name)

    def exists_in_put_tasks(self, key: Any) -> bool:
        with self._lock:
            return _blob_name(key) in self._put_tasks

    def get_blocking(self, key: Any) -> Any | None:
        self._require_worker("get_blocking")
        return self._batched_get_blocking_all([key])[0]

    def batched_get_blocking(self, keys: Iterable[Any]) -> list[Any | None]:
        self._require_worker("batched_get_blocking")
        return self._batched_get_blocking_all(list(keys))

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: Iterable[Any],
        transfer_spec: Any = None,
    ) -> list[Any]:
        results = []
        for obj in await asyncio.to_thread(self.batched_get_blocking, list(keys)):
            if obj is None:
                break
            results.append(obj)
        return results

    def remove(self, key: Any, force: bool = True) -> bool:
        blob_name = _blob_name(key)
        with self._lock:
            if not force and self._pins[blob_name] > 0:
                return False
        return bool(self.store.Delete(blob_name))

    def pin(self, key: Any) -> bool:
        blob_name = _blob_name(key)
        if not self.store.Exists(blob_name):
            return False
        with self._lock:
            self._pins[blob_name] += 1
        return True

    def unpin(self, key: Any) -> bool:
        blob_name = _blob_name(key)
        with self._lock:
            if self._pins[blob_name] <= 0:
                return False
            self._pins[blob_name] -= 1
            if self._pins[blob_name] == 0:
                del self._pins[blob_name]
        return True

    def _pin_existing(self, blob_name: str) -> None:
        with self._lock:
            self._pins[blob_name] += 1

    def get_allocator_backend(self) -> Any:
        self._require_worker("get_allocator_backend")
        return self.local_cpu_backend

    def close(self) -> None:
        self.store.Close()

    def touch_cache(self) -> None:
        return None

    def _require_worker(self, method: str) -> None:
        if self.local_cpu_backend is None:
            raise RuntimeError(f"{method} requires local_cpu_backend in worker role")

    def _allocate_from_metadata(self, metadata: dict[str, Any]) -> Any:
        shapes = metadata.get("shapes")
        dtypes = metadata.get("dtypes")
        shape = _torch_size(metadata.get("shape"))
        dtype = _torch_dtype(metadata.get("dtype"))
        alloc_shapes = [_torch_size(item) for item in shapes] if shapes else shape
        alloc_dtypes = [_torch_dtype(item) for item in dtypes] if dtypes else dtype
        fmt = _memory_format(metadata.get("fmt", metadata.get("memory_format")))

        obj = self.local_cpu_backend.allocate(alloc_shapes, alloc_dtypes, fmt)
        if obj is None:
            return None
        cached_positions = metadata.get("cached_positions")
        if cached_positions is not None and hasattr(obj, "metadata"):
            obj.metadata.cached_positions = _torch_tensor(cached_positions)
        return obj

    def _allocate_and_fill(self, metadata: dict[str, Any], payload: bytes) -> Any:
        obj = self._allocate_from_metadata(metadata)
        if obj is None:
            return None
        _copy_payload(getattr(obj, "byte_array"), payload)
        return obj

    def _batched_get_blocking_all(self, keys: list[Any]) -> list[Any | None]:
        total_start = time.perf_counter()
        blob_names = [_blob_name(key) for key in keys]
        info_start = time.perf_counter()
        infos = self.store.ReadRecordInfos(blob_names, self.max_inflight)
        info_elapsed = time.perf_counter() - info_start
        results: list[Any | None] = [None] * len(keys)
        range_blob_names = []
        range_offsets = []
        range_sizes = []
        range_destinations = []
        range_indexes = []
        range_objs = []
        decode_elapsed = 0.0
        allocate_elapsed = 0.0
        payload_bytes = 0
        for index, (key, info) in enumerate(zip(keys, infos)):
            if info is None:
                continue
            try:
                decode_start = time.perf_counter()
                metadata_json, payload_offset, payload_size = info
                metadata = json.loads(metadata_json)
                decode_elapsed += time.perf_counter() - decode_start
                if metadata.get("canonical_key") != _canonical_key(key):
                    continue
                allocate_start = time.perf_counter()
                obj = self._allocate_from_metadata(metadata)
                allocate_elapsed += time.perf_counter() - allocate_start
                if obj is None:
                    continue
                range_blob_names.append(blob_names[index])
                range_offsets.append(payload_offset)
                range_sizes.append(payload_size)
                range_destinations.append(getattr(obj, "byte_array"))
                range_indexes.append(index)
                range_objs.append(obj)
                payload_bytes += int(payload_size)
            except Exception:
                continue
        get_start = time.perf_counter()
        statuses = (
            self.store.GetManyRangesInto(
                range_blob_names,
                range_offsets,
                range_sizes,
                range_destinations,
                self.max_inflight,
            )
            if range_blob_names
            else []
        )
        get_elapsed = time.perf_counter() - get_start
        for index, obj, status in zip(range_indexes, range_objs, statuses):
            if status:
                results[index] = obj
        if self._profile:
            _profile_log(
                "python op=batched_get_blocking "
                f"count={len(keys)} bytes={payload_bytes} "
                f"success={sum(1 for item in statuses if item)} "
                f"total_ms={_ms(time.perf_counter() - total_start):.3f} "
                f"info_ms={_ms(info_elapsed):.3f} "
                f"metadata_decode_ms={_ms(decode_elapsed):.3f} "
                f"allocate_ms={_ms(allocate_elapsed):.3f} "
                f"payload_get_ms={_ms(get_elapsed):.3f}"
            )
        return results


def _extra_config(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    value = getattr(config, "extra_config", None)
    if value is None and isinstance(config, dict):
        value = config.get("extra_config")
    return dict(value or {})


def _profile_enabled() -> bool:
    return os.environ.get("CLIO_LMCACHE_PROFILE", "0") not in ("", "0", "false", "False")


def _profile_log(message: str) -> None:
    print(f"CLIO_LMCACHE_PROFILE {message}", file=sys.stderr, flush=True)


def _ms(seconds: float) -> float:
    return seconds * 1000.0


def _canonical_key(key: Any) -> str:
    to_string = getattr(key, "to_string", None)
    if callable(to_string):
        return str(to_string())
    return str(key)


def _blob_name(key: Any) -> str:
    digest = hashlib.sha256(_canonical_key(key).encode("utf-8")).hexdigest()
    return f"lmcache/v1/{digest}"


def _encode_record(key: Any, memory_obj: Any) -> bytes:
    payload = bytes(getattr(memory_obj, "byte_array"))
    metadata_bytes = _encode_metadata(key, memory_obj, len(payload))
    header = _HEADER.pack(_MAGIC, _VERSION, len(metadata_bytes), len(payload))
    return header + metadata_bytes + payload


def _encode_metadata(key: Any, memory_obj: Any, payload_length: int) -> bytes:
    metadata = _memory_obj_metadata(memory_obj)
    metadata["canonical_key"] = _canonical_key(key)
    metadata["payload_length"] = payload_length
    return json.dumps(metadata, sort_keys=True).encode("utf-8")


def _decode_record(blob: bytes | bytearray | memoryview) -> tuple[dict[str, Any], Any]:
    if len(blob) < _HEADER.size:
        raise ValueError("short ClioCTE record")
    view = memoryview(blob)
    magic, version, metadata_len, payload_len = _HEADER.unpack(view[: _HEADER.size])
    if magic != _MAGIC or version != _VERSION:
        raise ValueError("invalid ClioCTE record header")
    start = _HEADER.size
    payload_start = start + metadata_len
    payload_end = payload_start + payload_len
    if payload_end != len(blob):
        raise ValueError("invalid ClioCTE payload length")
    metadata = json.loads(bytes(view[start:payload_start]).decode("utf-8"))
    return metadata, view[payload_start:payload_end]


def _memory_obj_metadata(memory_obj: Any) -> dict[str, Any]:
    raw_metadata = getattr(memory_obj, "metadata", None)
    to_dict = getattr(raw_metadata, "to_dict", None)
    if callable(to_dict):
        metadata = dict(to_dict())
    elif isinstance(raw_metadata, dict):
        metadata = dict(raw_metadata)
    else:
        metadata = {}
    if "cached_positions" not in metadata:
        cached_positions = getattr(raw_metadata, "cached_positions", None)
        if cached_positions is not None:
            metadata["cached_positions"] = _jsonable(cached_positions)
    for name in ("shape", "dtype", "fmt"):
        if name not in metadata and hasattr(memory_obj, name):
            metadata[name] = _jsonable(getattr(memory_obj, name))
    return metadata


def _jsonable(value: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _torch_size(value: Any) -> Any:
    if torch is None or value is None:
        return value
    return torch.Size(value)


def _torch_dtype(value: Any) -> Any:
    if torch is None or value is None:
        return value
    if isinstance(value, torch.dtype):
        return value
    return getattr(torch, str(value).replace("torch.", ""))


def _memory_format(value: Any) -> Any:
    if MemoryFormat is None or value is None:
        return value
    if isinstance(value, MemoryFormat):
        return value
    return MemoryFormat(value)


def _torch_tensor(value: Any) -> Any:
    if torch is None or value is None:
        return value
    return torch.tensor(value)


def _copy_payload(destination: Any, payload: Any) -> None:
    view = destination if isinstance(destination, memoryview) else memoryview(destination)
    try:
        view[: len(payload)] = payload
    except (NotImplementedError, TypeError, ValueError):
        view.cast("B")[: len(payload)] = payload
