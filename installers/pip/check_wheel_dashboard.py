#!/usr/bin/env python3
"""Fail if the installed iowarp-core wheel cannot serve the web dashboard.

The runtime only builds its HTTP server (VizServer) when CMake finds Poco, and
silently drops it otherwise ("Web dashboard (viz): DISABLED"). Wheels shipped
without it for several releases. This check runs against the *installed* wheel
and needs both halves of the dashboard:

  * Poco linked into the runtime library (clio_run_cxx). Linux wheels link Poco
    statically and Windows/macOS link it dynamically, so instead of looking for
    a Poco library file this looks for Poco inside clio_run_cxx itself: RTTI
    names for a static link, the PocoNet import / load command for a dynamic
    one. A build without Poco contains neither.
  * the dashboard pages, installed by clio_add_viz_assets() under
    <package>/share/clio/viz/<module>/.

Usage: python check_wheel_dashboard.py   (run inside the venv with the wheel)
"""
import os
import sys

import iowarp_core


def find_runtime_libs(pkg):
    """Return every clio_run_cxx shared library (.so / .dylib / .dll) in the package."""
    hits = []
    for dirpath, _dirnames, filenames in os.walk(pkg):
        for f in filenames:
            if "clio_run_cxx" in f and (".so" in f or f.endswith((".dylib", ".dll"))):
                hits.append(os.path.join(dirpath, f))
    return hits


def links_poco(path):
    """True if the library at `path` statically or dynamically links Poco Net."""
    with open(path, "rb") as fh:
        data = fh.read()
    # 4Poco3Net: Itanium-mangled RTTI (static link, Linux/macOS).
    # PocoNet: the dependent library name (dynamic link, Windows/macOS).
    return b"4Poco3Net" in data or b"PocoNet" in data


def main():
    pkg = os.path.dirname(os.path.abspath(iowarp_core.__file__))
    libs = find_runtime_libs(pkg)
    if not libs:
        sys.exit("FAIL: clio_run_cxx library not found under " + pkg)
    with_poco = [lib for lib in libs if links_poco(lib)]
    index = os.path.join(pkg, "share", "clio", "viz", "clio_admin", "index.html")

    print("Runtime libraries:", libs)
    print("Linked with Poco: ", with_poco or "NONE")
    print("Dashboard pages:  ", index if os.path.isfile(index) else "MISSING")
    if not with_poco:
        sys.exit("FAIL: clio_run_cxx was built without Poco, so clio_run has "
                 "no web dashboard")
    if not os.path.isfile(index):
        sys.exit("FAIL: dashboard pages are not in the wheel (" + index + ")")
    print("Dashboard support present")


if __name__ == "__main__":
    main()
