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

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "clio_llm/lmcache/lmcache_store.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

/**
 * RAII owner for Python buffers acquired by this binding layer.
 */
class PyBufferList {
 public:
  PyBufferList() = default;

  PyBufferList(const PyBufferList &) = delete;
  PyBufferList &operator=(const PyBufferList &) = delete;

  /**
   * Release all acquired Python buffers.
   */
  ~PyBufferList() {
    for (auto &view : views_) {
      PyBuffer_Release(&view);
    }
  }

  /**
   * Acquire a read-only contiguous buffer and copy it into a string.
   *
   * @param object Python object exposing the buffer protocol.
   * @return Copied bytes preserving embedded NUL values.
   */
  std::string CopyReadonly(nb::handle object) {
    Py_buffer view;
    if (PyObject_GetBuffer(object.ptr(), &view, PyBUF_CONTIG_RO) != 0) {
      throw nb::type_error("payload must expose a contiguous buffer");
    }
    views_.push_back(view);
    const auto *ptr = static_cast<const char *>(view.buf);
    return std::string(ptr, static_cast<std::size_t>(view.len));
  }

  /**
   * Acquire a read-only contiguous buffer without copying it.
   *
   * @param object Python object exposing the buffer protocol.
   * @return Acquired buffer view.
   */
  Py_buffer AcquireReadonly(nb::handle object) {
    Py_buffer view;
    if (PyObject_GetBuffer(object.ptr(), &view, PyBUF_CONTIG_RO) != 0) {
      throw nb::type_error("payload must expose a contiguous buffer");
    }
    views_.push_back(view);
    return view;
  }

  /**
   * Acquire a writable contiguous buffer.
   *
   * @param object Python object exposing a writable buffer.
   * @return Acquired buffer view.
   */
  Py_buffer AcquireWritable(nb::handle object) {
    Py_buffer view;
    if (PyObject_GetBuffer(object.ptr(), &view, PyBUF_WRITABLE) != 0) {
      throw nb::type_error(
          "destination must expose a writable contiguous buffer");
    }
    views_.push_back(view);
    return view;
  }

 private:
  std::vector<Py_buffer> views_;
};

/**
 * Convert optional Python size sequence into optional C++ sizes.
 *
 * @param known_blob_sizes None or a sequence of integer-or-None sizes.
 * @return Empty vector when no known sizes were supplied.
 */
std::vector<std::optional<std::uint64_t>> CastKnownSizes(
    nb::object known_blob_sizes) {
  std::vector<std::optional<std::uint64_t>> sizes;
  if (known_blob_sizes.is_none()) {
    return sizes;
  }
  nb::sequence sequence = nb::cast<nb::sequence>(known_blob_sizes);
  sizes.reserve(nb::len(sequence));
  for (nb::handle item : sequence) {
    if (item.is_none()) {
      sizes.push_back(std::nullopt);
    } else {
      sizes.push_back(nb::cast<std::uint64_t>(item));
    }
  }
  return sizes;
}

/**
 * Convert a Python str or bytes-like object into metadata bytes.
 *
 * @param object Python metadata object.
 * @return Metadata bytes.
 */
std::string MetadataBytes(nb::handle object) {
  if (PyUnicode_Check(object.ptr())) {
    Py_ssize_t size = 0;
    const char *data = PyUnicode_AsUTF8AndSize(object.ptr(), &size);
    if (data == nullptr) {
      throw nb::type_error("metadata must be valid UTF-8");
    }
    return std::string(data, static_cast<std::size_t>(size));
  }

  Py_buffer view;
  if (PyObject_GetBuffer(object.ptr(), &view, PyBUF_CONTIG_RO) != 0) {
    throw nb::type_error("metadata must be str or a contiguous buffer");
  }
  const auto *data = static_cast<const char *>(view.buf);
  std::string bytes(data, static_cast<std::size_t>(view.len));
  PyBuffer_Release(&view);
  return bytes;
}

}  // namespace

NB_MODULE(clio_cte_lmcache_ext, m) {
  m.doc() = "CTE byte-store bindings for LMCache";

  nb::class_<clio_llm::lmcache::LMCacheStore>(m, "LMCacheStore")
      .def(nb::init<>())
      .def("Init", &clio_llm::lmcache::LMCacheStore::Init, "config_path"_a = "",
           "tag_name"_a = "lmcache_kv", "pool_query_mode"_a = "local")
      .def(
          "PutBytes",
          [](clio_llm::lmcache::LMCacheStore &self,
             const std::string &blob_name, nb::object data) {
            Py_buffer view;
            if (PyObject_GetBuffer(data.ptr(), &view, PyBUF_CONTIG_RO) != 0) {
              throw nb::type_error("data must expose a contiguous buffer");
            }
            const auto *ptr = static_cast<const char *>(view.buf);
            const std::string_view bytes(ptr,
                                         static_cast<std::size_t>(view.len));
            const bool ok = self.PutBytes(blob_name, bytes);
            PyBuffer_Release(&view);
            return ok;
          },
          "blob_name"_a, "data"_a)
      .def(
          "PutMany",
          [](clio_llm::lmcache::LMCacheStore &self,
             const std::vector<std::string> &blob_names, nb::sequence payloads,
             std::size_t max_inflight) {
            if (blob_names.size() != nb::len(payloads)) {
              throw nb::value_error(
                  "blob_names and payloads must have the same length");
            }
            PyBufferList buffers;
            std::vector<std::string> copied_payloads;
            copied_payloads.reserve(blob_names.size());
            for (nb::handle payload : payloads) {
              copied_payloads.push_back(buffers.CopyReadonly(payload));
            }
            nb::gil_scoped_release release;
            return self.PutMany(blob_names, copied_payloads, max_inflight);
          },
          "blob_names"_a, "payloads"_a,
          "max_inflight"_a =
              clio_llm::lmcache::LMCacheStore::kDefaultMaxInflight)
      .def(
          "PutManyRecords",
          [](clio_llm::lmcache::LMCacheStore &self,
             const std::vector<std::string> &blob_names,
             nb::sequence metadata_jsons, nb::sequence payloads,
             std::size_t max_inflight) {
            if (blob_names.size() != nb::len(metadata_jsons) ||
                blob_names.size() != nb::len(payloads)) {
              throw nb::value_error(
                  "blob_names, metadata_jsons, and payloads must have the "
                  "same length");
            }

            std::vector<std::string> metadata_bytes;
            metadata_bytes.reserve(blob_names.size());
            for (nb::handle metadata : metadata_jsons) {
              metadata_bytes.push_back(MetadataBytes(metadata));
            }

            PyBufferList buffers;
            std::vector<const void *> payload_ptrs;
            std::vector<std::size_t> payload_sizes;
            payload_ptrs.reserve(blob_names.size());
            payload_sizes.reserve(blob_names.size());
            for (nb::handle payload : payloads) {
              Py_buffer view = buffers.AcquireReadonly(payload);
              payload_ptrs.push_back(view.buf);
              payload_sizes.push_back(static_cast<std::size_t>(view.len));
            }

            nb::gil_scoped_release release;
            return self.PutManyRecords(blob_names, metadata_bytes, payload_ptrs,
                                       payload_sizes, max_inflight);
          },
          "blob_names"_a, "metadata_jsons"_a, "payloads"_a,
          "max_inflight"_a =
              clio_llm::lmcache::LMCacheStore::kDefaultMaxInflight)
      .def(
          "GetBytes",
          [](clio_llm::lmcache::LMCacheStore &self,
             const std::string &blob_name) -> nb::object {
            auto data = self.GetBytes(blob_name);
            if (!data.has_value()) {
              return nb::none();
            }
            const auto &bytes = *data;
            return nb::bytes(reinterpret_cast<const char *>(bytes.data()),
                             bytes.size());
          },
          "blob_name"_a)
      .def(
          "GetMany",
          [](clio_llm::lmcache::LMCacheStore &self,
             const std::vector<std::string> &blob_names,
             std::size_t max_inflight) {
            std::vector<std::optional<std::vector<std::uint8_t>>> results;
            {
              nb::gil_scoped_release release;
              results = self.GetMany(blob_names, max_inflight);
            }
            nb::list output;
            for (const auto &result : results) {
              if (!result.has_value()) {
                output.append(nb::none());
                continue;
              }
              const auto &bytes = *result;
              output.append(nb::bytes(
                  reinterpret_cast<const char *>(bytes.data()), bytes.size()));
            }
            return output;
          },
          "blob_names"_a,
          "max_inflight"_a =
              clio_llm::lmcache::LMCacheStore::kDefaultMaxInflight)
      .def(
          "GetBytesInto",
          [](clio_llm::lmcache::LMCacheStore &self,
             const std::string &blob_name, nb::object destination,
             nb::object known_blob_size) {
            Py_buffer view;
            if (PyObject_GetBuffer(destination.ptr(), &view, PyBUF_WRITABLE) !=
                0) {
              throw nb::type_error(
                  "destination must expose a writable contiguous buffer");
            }
            std::optional<std::uint64_t> size;
            if (!known_blob_size.is_none()) {
              size = nb::cast<std::uint64_t>(known_blob_size);
            }
            const bool ok = self.GetBytesInto(
                blob_name, view.buf, static_cast<std::size_t>(view.len), size);
            PyBuffer_Release(&view);
            return ok;
          },
          "blob_name"_a, "destination"_a, "known_blob_size"_a = nb::none())
      .def(
          "GetManyInto",
          [](clio_llm::lmcache::LMCacheStore &self,
             const std::vector<std::string> &blob_names,
             nb::sequence destinations, nb::object known_blob_sizes,
             std::size_t max_inflight) {
            if (blob_names.size() != nb::len(destinations)) {
              throw nb::value_error(
                  "blob_names and destinations must have the same length");
            }
            PyBufferList buffers;
            std::vector<void *> destination_ptrs;
            std::vector<std::size_t> destination_sizes;
            destination_ptrs.reserve(blob_names.size());
            destination_sizes.reserve(blob_names.size());
            for (nb::handle destination : destinations) {
              Py_buffer view = buffers.AcquireWritable(destination);
              destination_ptrs.push_back(view.buf);
              destination_sizes.push_back(static_cast<std::size_t>(view.len));
            }
            auto sizes = CastKnownSizes(known_blob_sizes);
            nb::gil_scoped_release release;
            return self.GetManyInto(blob_names, destination_ptrs,
                                    destination_sizes, sizes, max_inflight);
          },
          "blob_names"_a, "destinations"_a, "known_blob_sizes"_a = nb::none(),
          "max_inflight"_a =
              clio_llm::lmcache::LMCacheStore::kDefaultMaxInflight)
      .def(
          "GetManyRangesInto",
          [](clio_llm::lmcache::LMCacheStore &self,
             const std::vector<std::string> &blob_names,
             const std::vector<std::uint64_t> &offsets,
             const std::vector<std::uint64_t> &sizes,
             nb::sequence destinations, std::size_t max_inflight) {
            if (blob_names.size() != offsets.size() ||
                blob_names.size() != sizes.size() ||
                blob_names.size() != nb::len(destinations)) {
              throw nb::value_error(
                  "blob_names, offsets, sizes, and destinations must have the "
                  "same length");
            }
            PyBufferList buffers;
            std::vector<void *> destination_ptrs;
            std::vector<std::size_t> destination_sizes;
            destination_ptrs.reserve(blob_names.size());
            destination_sizes.reserve(blob_names.size());
            for (nb::handle destination : destinations) {
              Py_buffer view = buffers.AcquireWritable(destination);
              destination_ptrs.push_back(view.buf);
              destination_sizes.push_back(static_cast<std::size_t>(view.len));
            }
            nb::gil_scoped_release release;
            return self.GetManyRangesInto(blob_names, offsets, sizes,
                                          destination_ptrs, destination_sizes,
                                          max_inflight);
          },
          "blob_names"_a, "offsets"_a, "sizes"_a, "destinations"_a,
          "max_inflight"_a =
              clio_llm::lmcache::LMCacheStore::kDefaultMaxInflight)
      .def(
          "ReadRecordInfos",
          [](clio_llm::lmcache::LMCacheStore &self,
             const std::vector<std::string> &blob_names,
             std::size_t max_inflight) {
            std::vector<std::optional<
                clio_llm::lmcache::LMCacheStore::RecordInfo>>
                infos;
            {
              nb::gil_scoped_release release;
              infos = self.ReadRecordInfos(blob_names, max_inflight);
            }
            nb::list output;
            for (const auto &info : infos) {
              if (!info.has_value()) {
                output.append(nb::none());
                continue;
              }
              output.append(nb::make_tuple(info->metadata_json,
                                           info->payload_offset,
                                           info->payload_size));
            }
            return output;
          },
          "blob_names"_a,
          "max_inflight"_a =
              clio_llm::lmcache::LMCacheStore::kDefaultMaxInflight)
      .def("Exists", &clio_llm::lmcache::LMCacheStore::Exists, "blob_name"_a)
      .def("ExistsMany", &clio_llm::lmcache::LMCacheStore::ExistsMany,
           "blob_names"_a,
           "max_inflight"_a =
               clio_llm::lmcache::LMCacheStore::kDefaultMaxInflight,
           nb::call_guard<nb::gil_scoped_release>())
      .def(
          "Size",
          [](clio_llm::lmcache::LMCacheStore &self,
             const std::string &blob_name) -> nb::object {
            auto size = self.Size(blob_name);
            if (!size.has_value()) {
              return nb::none();
            }
            return nb::int_(*size);
          },
          "blob_name"_a)
      .def("SizeMany", &clio_llm::lmcache::LMCacheStore::SizeMany,
           "blob_names"_a,
           "max_inflight"_a =
               clio_llm::lmcache::LMCacheStore::kDefaultMaxInflight,
           nb::call_guard<nb::gil_scoped_release>())
      .def("Delete", &clio_llm::lmcache::LMCacheStore::Delete, "blob_name"_a)
      .def("Close", &clio_llm::lmcache::LMCacheStore::Close)
      .def("IsReady", &clio_llm::lmcache::LMCacheStore::IsReady);
}
