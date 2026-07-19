# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.package import *


class GoogleCloudCpp(CMakePackage):
    """Google Cloud C++ client libraries — IOWarp storage-only overlay.

    This is a deliberately narrow, storage-only build of google-cloud-cpp for
    the CAE GCS assimilator (``gs://`` / ``gcs://`` import, Issue #626). It
    shadows the Spack builtin recipe on purpose:

    The builtin declares ``depends_on("grpc")`` unconditionally, which drags in
    ``grpc`` + a hard-pinned ``re2@2023-09-01``. Both fail to compile against
    the 2026-era abseil/libstdc++ toolchain on our target systems (missing
    transitive ``<cstring>``/``<string>``/``<limits>``/``<algorithm>`` includes).
    The **storage** component is a pure libcurl+REST client — it never links
    grpc or re2 — so this recipe omits them entirely and builds only storage.

    If you ever need the grpc-based libraries (bigtable, pubsub, spanner, …),
    use the Spack builtin recipe with per-node force-include flags instead;
    this overlay intentionally does not support them.
    """

    homepage = "https://github.com/googleapis/google-cloud-cpp"
    git = "https://github.com/googleapis/google-cloud-cpp.git"

    maintainers("iowarp")

    license("Apache-2.0")

    # Git-tag versions (no sha256 needed; pin `commit=` when promoting to a
    # release build). 2.30.0 is the version validated on Ares (spack hash
    # zjblapm) for the CAE GCS assimilator.
    version("main", branch="main")
    version("2.30.0", tag="v2.30.0")
    version("2.29.0", tag="v2.29.0")

    variant("shared", default=True,
            description="Build shared libraries (required to link into the "
                        "CLIO CAE runtime .so: the static archive has a "
                        "non-PIC thread_local that fails R_X86_64_TPOFF32 "
                        "relocation into a shared object)")
    variant("cxxstd", default="17", values=("14", "17", "20"), multi=False,
            description="C++ standard")

    # Storage-only dependency set: libcurl (HTTPS/REST + OAuth token refresh),
    # abseil (base/strings/time), google-crc32c (upload/download checksums),
    # nlohmann-json (REST payloads). protobuf is pulled by the common layer's
    # build but storage does not link grpc/re2 — those are intentionally absent.
    depends_on("cmake@3.16:", type="build")
    depends_on("curl")
    depends_on("abseil-cpp")
    depends_on("google-crc32c")
    depends_on("nlohmann-json")
    depends_on("protobuf")

    # C++17 is the floor for modern google-cloud-cpp.
    conflicts("cxxstd=14", when="@2.20.0:")

    def cmake_args(self):
        return [
            # Build ONLY the storage component -> its CMake never calls
            # find_package(gRPC), so grpc/re2 are unnecessary at configure too.
            self.define("GOOGLE_CLOUD_CPP_ENABLE", "storage"),
            self.define("BUILD_TESTING", False),
            self.define("GOOGLE_CLOUD_CPP_ENABLE_EXAMPLES", False),
            self.define("GOOGLE_CLOUD_CPP_WITH_MOCKS", False),
            self.define_from_variant("BUILD_SHARED_LIBS", "shared"),
            self.define_from_variant("CMAKE_CXX_STANDARD", "cxxstd"),
        ]
