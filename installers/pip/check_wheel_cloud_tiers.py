#!/usr/bin/env python3
"""Fail if the installed iowarp-core wheel lacks the S3 or GCS storage tiers.

pyproject.toml turns on CLIO_ENABLE_AMAZON_DRIVE and CLIO_ENABLE_GOOGLE_CLOUD,
but CMake silently turns both off again when it cannot find Poco's TLS
components (Net, NetSSL, Crypto), and the wheel then rejects bdev_type s3 /
gcs at pool creation. This check runs against the *installed* wheel:

  * the bdev module (clio_bdev_runtime) must contain both transports. They are
    polymorphic classes, so their names survive in the RTTI of any build
    (Itanium "15S3BdevTransport", MSVC ".?AVS3BdevTransport@...").
  * on Linux, the module must not depend on a shared libssl / libcrypto. The
    wheel links OpenSSL 3 statically; a NEEDED libssl.so.1.1 (the build
    image's) would not load on current distributions.

Usage: python check_wheel_cloud_tiers.py   (run inside the venv with the wheel)
"""
import os
import subprocess
import sys

import iowarp_core

TRANSPORTS = (b"S3BdevTransport", b"GcsBdevTransport")


def find_bdev_libs(pkg):
    """Return every clio_bdev_runtime shared library (.so / .dylib / .dll) in the package."""
    hits = []
    for dirpath, _dirnames, filenames in os.walk(pkg):
        for f in filenames:
            if "clio_bdev_runtime" in f and (".so" in f or f.endswith((".dylib", ".dll"))):
                hits.append(os.path.join(dirpath, f))
    return hits


def shared_openssl_deps(path):
    """NEEDED entries naming libssl/libcrypto (Linux only; [] elsewhere or without readelf)."""
    if not sys.platform.startswith("linux"):
        return []
    try:
        out = subprocess.run(["readelf", "-d", path], capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        print("note: readelf unavailable; skipping the OpenSSL linkage check")
        return []
    return [line.strip() for line in out.splitlines()
            if "NEEDED" in line and ("libssl" in line or "libcrypto" in line)]


def main():
    pkg = os.path.dirname(os.path.abspath(iowarp_core.__file__))
    libs = find_bdev_libs(pkg)
    if not libs:
        sys.exit("FAIL: clio_bdev_runtime library not found under " + pkg)
    failures = []
    for lib in libs:
        with open(lib, "rb") as fh:
            data = fh.read()
        missing = [t.decode() for t in TRANSPORTS if t not in data]
        print(lib)
        print("  transports missing:", missing or "none")
        if missing:
            failures.append("%s lacks %s" % (lib, ", ".join(missing)))
        needed = shared_openssl_deps(lib)
        print("  shared OpenSSL deps:", needed or "none")
        if needed:
            failures.append("%s links OpenSSL dynamically: %s" % (lib, needed))
    if failures:
        sys.exit("FAIL: " + "; ".join(failures))
    print("S3 and GCS storage tiers present")


if __name__ == "__main__":
    main()
