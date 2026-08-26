#!/usr/bin/env python3
"""Verify that a manylinux wheel's real glibc requirement matches its tag.

Background (iowarp/clio-core#973): the wheels are repaired by
``installers/pip/repair_wheel.sh``, which stamps the platform tag with
``python -m wheel tags``.  That command rewrites the filename and the WHEEL
metadata and nothing else -- it never looks at the binaries.  auditwheel,
which *would* check, is deliberately not used (see repair_wheel.sh).

So nothing in the pipeline connects the tag the wheel advertises to the
glibc symbol versions its ELF files actually import.  Get that pairing
wrong in either direction and users lose:

  * tag too new (the #973 bug): every glibc-2.28 host -- Rocky/Alma/RHEL 8,
    i.e. most HPC login nodes -- is told "no wheels with a matching platform
    tag" and there is no sdist to fall back to.
  * tag too old: pip installs the wheel happily and ``import iowarp_core``
    then dies with "version `GLIBC_2.34' not found".

This script closes that gap: it unpacks the wheel, reads the versioned
symbol requirements out of every ELF file it contains, and fails if any of
them needs a newer glibc than the wheel's own ``manylinux_<x>_<y>`` tag
promises.

The same check applies to the conda package, which had the identical defect
for the identical reason (nothing tied the built binaries to a declared
glibc floor), so this also accepts ``.conda`` archives and plain directories
together with an explicit ``--max-glibc``.

Usage:
    python3 check_wheel_glibc.py <wheel> [<wheel> ...]
    python3 check_wheel_glibc.py --max-glibc 2.28 <pkg.conda|directory> ...
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

# manylinux_<major>_<minor>_<arch> -- the glibc version the wheel promises.
TAG_RE = re.compile(r"manylinux_(\d+)_(\d+)_(?:x86_64|aarch64|i686|ppc64le|s390x)")
# `objdump -T` prints undefined imports as e.g. "... (GLIBC_2.34) pthread_create".
SYMVER_RE = re.compile(r"\(GLIBC_(\d+)\.(\d+)\)")

ELF_MAGIC = b"\x7fELF"

# Cap the per-package failure listing (see check_package).
MAX_REPORTED = 15


def wheel_glibc_tag(wheel: Path) -> tuple[int, int] | None:
    """The (major, minor) glibc version claimed by the wheel filename."""
    matches = TAG_RE.findall(wheel.name)
    if not matches:
        return None
    # A wheel may carry several platform tags; the oldest glibc is the
    # weakest promise, and that is the one that has to hold.
    return min((int(a), int(b)) for a, b in matches)


def elf_files(root: Path) -> list[Path]:
    found = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            with path.open("rb") as fh:
                if fh.read(4) == ELF_MAGIC:
                    found.append(path)
        except OSError:
            continue
    return found


def max_glibc_requirement(elf: Path) -> tuple[tuple[int, int], list[str]]:
    """Highest GLIBC_x.y this ELF imports, plus the symbols that need it."""
    out = subprocess.run(
        ["objdump", "-T", str(elf)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    worst = (0, 0)
    culprits: list[str] = []
    for line in out.splitlines():
        m = SYMVER_RE.search(line)
        if not m:
            continue
        ver = (int(m.group(1)), int(m.group(2)))
        symbol = line.split()[-1]
        if ver > worst:
            worst, culprits = ver, [symbol]
        elif ver == worst:
            culprits.append(symbol)
    return worst, culprits


def unpack(archive: Path, dest: Path) -> None:
    """Extract a wheel, a .conda package, or a legacy .tar.bz2 into dest."""
    if archive.name.endswith(".tar.bz2"):
        # conda-build still emits this format when told to.
        subprocess.run(["tar", "-xf", str(archive), "-C", str(dest)], check=True)
        return

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    # A .conda package is a zip of zstd tarballs; the payload one holds the
    # installed files. Unpack it so the ELF scan below sees real binaries.
    # tarfile only learned zstd in 3.14, so shell out to GNU tar instead of
    # requiring a new interpreter on the runner.
    for inner in list(dest.glob("pkg-*.tar.zst")):
        subprocess.run(
            ["tar", "--use-compress-program=unzstd", "-xf", str(inner), "-C", str(dest)],
            check=True,
        )
        inner.unlink()


def check_package(pkg: Path, max_glibc: tuple[int, int] | None) -> bool:
    claimed = max_glibc or wheel_glibc_tag(pkg)
    if claimed is None:
        print(f"SKIP {pkg.name}: not a manylinux wheel and no --max-glibc given")
        return True

    print(f"=== {pkg.name} (must need glibc <= {claimed[0]}.{claimed[1]}) ===")
    ok = True
    with TemporaryDirectory() as tmp:
        if pkg.is_dir():
            root = pkg
        else:
            root = Path(tmp)
            unpack(pkg, root)

        files = elf_files(root)
        if not files:
            print("  ERROR: contains no ELF files -- nothing was built?")
            return False

        worst_overall = (0, 0)
        # A package that is built wrong is built wrong for every binary in
        # it (157 of them, in the case that motivated this script), so cap
        # the listing -- the pattern is clear long before the end.
        failures = 0
        for elf in sorted(files):
            need, symbols = max_glibc_requirement(elf)
            worst_overall = max(worst_overall, need)
            if need > claimed:
                ok = False
                failures += 1
                if failures <= MAX_REPORTED:
                    rel = elf.relative_to(root)
                    print(
                        f"  FAIL {rel}: needs GLIBC_{need[0]}.{need[1]} "
                        f"(e.g. {', '.join(sorted(set(symbols))[:5])})"
                    )
        if failures > MAX_REPORTED:
            print(f"  ... and {failures - MAX_REPORTED} more failing ELF files")
        print(
            f"  highest glibc actually required: "
            f"{worst_overall[0]}.{worst_overall[1]} across {len(files)} ELF files"
        )
    if ok:
        print("  OK")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "packages",
        nargs="+",
        type=Path,
        help="wheels, .conda packages, or directories of built files",
    )
    ap.add_argument(
        "--max-glibc",
        metavar="X.Y",
        help="glibc floor to enforce. Required for .conda packages and "
        "directories; wheels default to the version in their manylinux tag.",
    )
    args = ap.parse_args()

    max_glibc = None
    if args.max_glibc:
        major, _, minor = args.max_glibc.partition(".")
        max_glibc = (int(major), int(minor or 0))

    if not all(check_package(p, max_glibc) for p in args.packages):
        print(
            "\nERROR: at least one package requires a newer glibc than it "
            "promises. Either build against an older glibc (manylinux image "
            "for pip, c_stdlib_version for conda) or raise the declared "
            "floor -- which drops support for older distros, see #973."
        )
        return 1
    print("\nAll packages are consistent with their declared glibc floor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
