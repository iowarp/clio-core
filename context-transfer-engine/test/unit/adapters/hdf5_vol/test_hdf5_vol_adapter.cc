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
 * Unit tests for the Clio HDF5 VOL connector (clio_hdf5_vol).
 *
 * Exercises the connector through its public surface only: the HDF5 plugin
 * discovery entry points and the H5VL registration / property-list APIs.
 * The connector's initialize callback is nullptr, so none of this requires
 * a running Clio runtime.
 *
 * These tests also serve as a link-time regression for issue #518: both the
 * connector DLL and this executable consume ${HDF5_LIBRARIES}, which was
 * empty when HDF5 was found in CONFIG mode (vcpkg exports only namespaced
 * hdf5::hdf5-shared targets), leaving every H5* symbol unresolved.
 */

#include "simple_test.h"

#include "clio_vol.h"

#include <cstring>

/* The plugin discovery entry points are the only symbols the connector
   library is guaranteed to export on Windows (H5PLextern.h marks them
   dllexport inside the plugin). Declare them locally instead of including
   H5PLextern.h, which would re-declare them dllexport in this consumer TU.
   H5VL_clio_cls / H5VL_clio_register carry no export decoration, so
   they must not be referenced directly here. */
extern "C" H5PL_type_t H5PLget_plugin_type(void);
extern "C" const void *H5PLget_plugin_info(void);

namespace {

const H5VL_class_t *GetClioVolClass() {
  return static_cast<const H5VL_class_t *>(H5PLget_plugin_info());
}

}  // namespace

TEST_CASE("HDF5 VOL - Plugin Entry Points", "[hdf5_vol]") {
  REQUIRE(H5PLget_plugin_type() == H5PL_TYPE_VOL);

  const H5VL_class_t *cls = GetClioVolClass();
  REQUIRE(cls != nullptr);
  /* The same class pointer must be returned on every call (HDF5 caches it) */
  REQUIRE(GetClioVolClass() == cls);
}

TEST_CASE("HDF5 VOL - Connector Class Definition", "[hdf5_vol]") {
  const H5VL_class_t *cls = GetClioVolClass();
  REQUIRE(cls != nullptr);

  SECTION("identification");
  REQUIRE(std::strcmp(cls->name, CLIO_VOL_CONNECTOR_NAME) == 0);
  REQUIRE(cls->value == CLIO_VOL_CONNECTOR_VALUE);
  REQUIRE(cls->version == H5VL_VERSION);
  REQUIRE(cls->conn_version == CLIO_VOL_CONNECTOR_VERSION);

  SECTION("capability flags");
  REQUIRE((cls->cap_flags & H5VL_CAP_FLAG_FILE_BASIC) != 0);
  REQUIRE((cls->cap_flags & H5VL_CAP_FLAG_DATASET_BASIC) != 0);
  REQUIRE((cls->cap_flags & H5VL_CAP_FLAG_GROUP_BASIC) != 0);
  REQUIRE((cls->cap_flags & H5VL_CAP_FLAG_ATTR_BASIC) != 0);

  SECTION("required callbacks are wired");
  REQUIRE(cls->info_cls.size == sizeof(clio_vol_info_t));
  REQUIRE(cls->info_cls.copy != nullptr);
  REQUIRE(cls->info_cls.free != nullptr);
  REQUIRE(cls->file_cls.create != nullptr);
  REQUIRE(cls->file_cls.open != nullptr);
  REQUIRE(cls->file_cls.close != nullptr);
  REQUIRE(cls->dataset_cls.create != nullptr);
  REQUIRE(cls->dataset_cls.open != nullptr);
  REQUIRE(cls->dataset_cls.read != nullptr);
  REQUIRE(cls->dataset_cls.write != nullptr);
  REQUIRE(cls->dataset_cls.close != nullptr);
  REQUIRE(cls->group_cls.create != nullptr);
  REQUIRE(cls->group_cls.open != nullptr);
  REQUIRE(cls->group_cls.close != nullptr);
  REQUIRE(cls->attr_cls.create != nullptr);
  REQUIRE(cls->attr_cls.read != nullptr);
  REQUIRE(cls->attr_cls.write != nullptr);
  REQUIRE(cls->attr_cls.close != nullptr);
}

TEST_CASE("HDF5 VOL - Registration Lifecycle", "[hdf5_vol]") {
  REQUIRE(H5open() >= 0);

  const H5VL_class_t *cls = GetClioVolClass();
  REQUIRE(cls != nullptr);

  SECTION("register");
  hid_t connector_id = H5VLregister_connector(cls, H5P_DEFAULT);
  REQUIRE(connector_id >= 0);
  REQUIRE(H5VLis_connector_registered_by_name(CLIO_VOL_CONNECTOR_NAME) > 0);
  REQUIRE(H5VLis_connector_registered_by_value(
              static_cast<H5VL_class_value_t>(CLIO_VOL_CONNECTOR_VALUE)) > 0);

  SECTION("lookup by name and value");
  hid_t by_name = H5VLget_connector_id_by_name(CLIO_VOL_CONNECTOR_NAME);
  REQUIRE(by_name >= 0);
  hid_t by_value = H5VLget_connector_id_by_value(
      static_cast<H5VL_class_value_t>(CLIO_VOL_CONNECTOR_VALUE));
  REQUIRE(by_value >= 0);
  REQUIRE(H5VLclose(by_name) >= 0);
  REQUIRE(H5VLclose(by_value) >= 0);

  SECTION("unregister");
  REQUIRE(H5VLunregister_connector(connector_id) >= 0);
  REQUIRE(H5VLis_connector_registered_by_name(CLIO_VOL_CONNECTOR_NAME) == 0);
}

TEST_CASE("HDF5 VOL - File Access Property List", "[hdf5_vol]") {
  REQUIRE(H5open() >= 0);

  const H5VL_class_t *cls = GetClioVolClass();
  hid_t connector_id = H5VLregister_connector(cls, H5P_DEFAULT);
  REQUIRE(connector_id >= 0);

  SECTION("install connector on a FAPL");
  hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  REQUIRE(fapl >= 0);

  clio_vol_info_t info;
  info.under_vol_id = H5I_INVALID_HID;
  info.under_vol_info = nullptr;
  info.chunk_size = CLIO_VOL_DEFAULT_CHUNK_SIZE;
  REQUIRE(H5Pset_vol(fapl, connector_id, &info) >= 0);

  SECTION("read the connector back from the FAPL");
  hid_t fapl_vol_id = H5I_INVALID_HID;
  REQUIRE(H5Pget_vol_id(fapl, &fapl_vol_id) >= 0);
  REQUIRE(fapl_vol_id >= 0);
  REQUIRE(H5VLclose(fapl_vol_id) >= 0);

  /* Closing the FAPL routes through clio_info_free for the copied info */
  REQUIRE(H5Pclose(fapl) >= 0);
  REQUIRE(H5VLunregister_connector(connector_id) >= 0);
}

TEST_CASE("HDF5 VOL - Introspect Callbacks", "[hdf5_vol]") {
  const H5VL_class_t *cls = GetClioVolClass();
  REQUIRE(cls != nullptr);

  SECTION("get_conn_cls returns this connector");
  REQUIRE(cls->introspect_cls.get_conn_cls != nullptr);
  const H5VL_class_t *conn_cls = nullptr;
  REQUIRE(cls->introspect_cls.get_conn_cls(nullptr, H5VL_GET_CONN_LVL_CURR,
                                           &conn_cls) == 0);
  REQUIRE(conn_cls == cls);

  SECTION("get_cap_flags reports the basic capabilities");
  REQUIRE(cls->introspect_cls.get_cap_flags != nullptr);
  uint64_t cap_flags = 0;
  REQUIRE(cls->introspect_cls.get_cap_flags(nullptr, &cap_flags) == 0);
  REQUIRE((cap_flags & H5VL_CAP_FLAG_FILE_BASIC) != 0);
  REQUIRE((cap_flags & H5VL_CAP_FLAG_DATASET_BASIC) != 0);
  REQUIRE((cap_flags & H5VL_CAP_FLAG_GROUP_BASIC) != 0);
  REQUIRE((cap_flags & H5VL_CAP_FLAG_ATTR_BASIC) != 0);
  REQUIRE((cap_flags & H5VL_CAP_FLAG_LINK_BASIC) != 0);
  REQUIRE((cap_flags & H5VL_CAP_FLAG_OBJECT_BASIC) != 0);

  SECTION("opt_query reports no optional operations");
  REQUIRE(cls->introspect_cls.opt_query != nullptr);
  uint64_t opt_flags = ~static_cast<uint64_t>(0);
  REQUIRE(cls->introspect_cls.opt_query(nullptr, H5VL_SUBCLS_DATASET, 0,
                                        &opt_flags) == 0);
  REQUIRE(opt_flags == 0);
}

TEST_CASE("HDF5 VOL - Info Copy And Free", "[hdf5_vol]") {
  const H5VL_class_t *cls = GetClioVolClass();
  REQUIRE(cls != nullptr);

  clio_vol_info_t info;
  info.under_vol_id = H5I_INVALID_HID;
  info.under_vol_info = nullptr;
  info.chunk_size = 4 * 1024;

  SECTION("copy produces an independent equal info");
  void *copied = cls->info_cls.copy(&info);
  REQUIRE(copied != nullptr);
  REQUIRE(copied != static_cast<void *>(&info));
  auto *copy = static_cast<clio_vol_info_t *>(copied);
  REQUIRE(copy->under_vol_id == info.under_vol_id);
  REQUIRE(copy->under_vol_info == nullptr);
  REQUIRE(copy->chunk_size == info.chunk_size);

  SECTION("free releases the copy");
  REQUIRE(cls->info_cls.free(copied) == 0);
}

SIMPLE_TEST_MAIN()
