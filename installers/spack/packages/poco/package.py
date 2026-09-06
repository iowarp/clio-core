# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.package import *


class Poco(CMakePackage):
    """POCO C++ Libraries — IOWarp overlay for the cloud bdev transports.

    No Spack builtin repo ships a ``poco`` package, so the S3 (``kS3``) and GCS
    (``kGcs``) bdev transports — both pure Poco::Net HTTPS clients — have no way
    to build without one. This overlay supplies it.

    Verified absent on Ares 2026-08-24 (Spack 1.3.0.dev0, builtin repo v2.2):
    ``spack info builtin.poco`` reports the package is not in builtin, and the
    builtin ``packages/`` directory contains no match. Namespace-qualify that
    check — a plain ``spack info poco`` is answered by THIS file, since a
    registered overlay shadows builtin, which makes the obvious check circular.

    The build is deliberately narrow. CLIO's root ``CMakeLists.txt`` does::

        find_package(Poco COMPONENTS Net NetSSL Crypto JSON QUIET)

    so only Foundation, Net, NetSSL, Crypto, JSON, XML and Util are enabled
    here; the Data/MongoDB/Redis/Zip/PDF components are large, pull extra
    dependencies, and nothing in this tree references them.

    Third-party sources (pcre2, expat, zlib) stay bundled — Poco's unbundled
    mode is fussy about versions and buys nothing for a dependency this narrow.
    OpenSSL is the one real external, needed by Crypto and NetSSL.

    If the target system already provides Poco (e.g. ``libpoco-dev``, as the
    IOWarp devcontainer does), prefer declaring it as a Spack ``external``
    rather than building this.
    """

    homepage = "https://pocoproject.org/"
    url = "https://github.com/pocoproject/poco/archive/refs/tags/poco-1.13.3-release.tar.gz"
    git = "https://github.com/pocoproject/poco.git"

    maintainers("iowarp")

    license("BSL-1.0")

    version("1.13.3", tag="poco-1.13.3-release")
    version("1.12.5", tag="poco-1.12.5-release")

    variant("shared", default=True, description="Build shared libraries")
    variant("cxxstd", default="17", values=("14", "17", "20"), multi=False,
            description="C++ standard")

    depends_on("cmake@3.15:", type="build")
    depends_on("openssl")

    def cmake_args(self):
        return [
            self.define_from_variant("BUILD_SHARED_LIBS", "shared"),
            self.define_from_variant("CMAKE_CXX_STANDARD", "cxxstd"),
            # Components CLIO's find_package() asks for.
            self.define("ENABLE_FOUNDATION", True),
            self.define("ENABLE_NET", True),
            self.define("ENABLE_NETSSL", True),
            self.define("ENABLE_CRYPTO", True),
            self.define("ENABLE_JSON", True),
            self.define("ENABLE_XML", True),
            self.define("ENABLE_UTIL", True),
            # Everything else: off. Nothing in IOWarp references these, and
            # they are the expensive half of a Poco build.
            self.define("ENABLE_DATA", False),
            self.define("ENABLE_DATA_SQLITE", False),
            self.define("ENABLE_DATA_MYSQL", False),
            self.define("ENABLE_DATA_ODBC", False),
            self.define("ENABLE_DATA_POSTGRESQL", False),
            self.define("ENABLE_MONGODB", False),
            self.define("ENABLE_REDIS", False),
            self.define("ENABLE_PROMETHEUS", False),
            self.define("ENABLE_ZIP", False),
            self.define("ENABLE_PDF", False),
            self.define("ENABLE_SEVENZIP", False),
            self.define("ENABLE_CPPPARSER", False),
            self.define("ENABLE_ENCODINGS", False),
            self.define("ENABLE_PAGECOMPILER", False),
            self.define("ENABLE_PAGECOMPILER_FILE2PAGE", False),
            self.define("ENABLE_ACTIVERECORD", False),
            self.define("ENABLE_ACTIVERECORD_COMPILER", False),
            self.define("ENABLE_JWT", False),
            self.define("ENABLE_TESTS", False),
            self.define("ENABLE_SAMPLES", False),
            self.define("POCO_UNBUNDLED", False),
        ]
