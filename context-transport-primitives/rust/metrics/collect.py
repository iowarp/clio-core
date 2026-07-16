# SPDX-License-Identifier: BSD-3-Clause
# Part of IOWarp Core - CTP Rust adaptation (issue #756).
#
# Migration metrics harness: tracks how COMPILATION TIME and CODE QUALITY
# evolve as modules move from C++ to Rust. Appends one row per crate per run
# to history.csv so the trend is visible over the life of the migration.
#
# Per Rust crate:  LOC, unsafe-block count, clippy warnings, cold build time,
#                  warm (incremental) build time, test count.
# C++ baseline:    pass --cpp-cmd "<command>" to time an equivalent C++
#                  build (e.g. a clean clio_ctp_host build in the
#                  devcontainer); recorded as the pseudo-crate "cpp-baseline".
#
# Usage:
#   py metrics/collect.py                 # all crates, default features
#   py metrics/collect.py --features cuda --crates ctp-gpu
#   py metrics/collect.py --cpp-cmd "docker run ... cmake --build ..."

import argparse
import csv
import datetime
import pathlib
import re
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
HISTORY = pathlib.Path(__file__).resolve().parent / "history.csv"
FIELDS = [
    "date", "git_sha", "crate", "loc", "unsafe_count", "clippy_warnings",
    "cold_build_s", "warm_build_s", "tests_passed",
]


def sh(cmd, cwd=ROOT, timeout=3600):
    t0 = time.monotonic()
    p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                       text=True, timeout=timeout)
    return time.monotonic() - t0, p


def git_sha():
    _, p = sh("git rev-parse --short HEAD")
    return p.stdout.strip() or "unknown"


def crate_features(crate, requested):
    """Filter the requested feature list to those the crate declares —
    features are per-crate in cargo, so a blanket --features cuda must not
    be passed to crates that lack it."""
    if not requested:
        return ""
    toml = (ROOT / crate / "Cargo.toml").read_text(encoding="utf-8")
    m = re.search(r"^\[features\]\n(.*?)(?:\n\[|\Z)", toml, re.DOTALL | re.MULTILINE)
    declared = set(re.findall(r"^([\w-]+)\s*=", m.group(1), re.MULTILINE)) if m else set()
    kept = [f for f in requested.split(",") if f.strip() in declared]
    return ",".join(kept)


def crate_loc_and_unsafe(crate):
    loc = unsafe_count = 0
    for f in (ROOT / crate).rglob("*.rs"):
        text = f.read_text(encoding="utf-8", errors="replace")
        loc += len(text.splitlines())
        unsafe_count += len(re.findall(r"\bunsafe\b", text))
    # Count wrapped C++ shim/kernel lines too - they are part of the crate.
    for pat in ("*.cc", "*.cu"):
        for f in (ROOT / crate).rglob(pat):
            loc += len(f.read_text(encoding="utf-8", errors="replace").splitlines())
    return loc, unsafe_count


def clippy_warnings(crate, features):
    feat = f"--features {features}" if features else ""
    _, p = sh(f"cargo clippy -p {crate} {feat} 2>&1")
    return len(re.findall(r"^warning", p.stdout + p.stderr, re.MULTILINE))


def build_times(crate, features):
    feat = f"--features {features}" if features else ""
    sh(f"cargo clean -p {crate}")
    cold, p = sh(f"cargo build --release -p {crate} {feat}")
    if p.returncode != 0:
        raise RuntimeError(f"cold build failed for {crate}:\n{p.stderr[-2000:]}")
    # Warm: touch the lib root and rebuild incrementally.
    lib = ROOT / crate / "src" / "lib.rs"
    lib.touch()
    warm, p = sh(f"cargo build --release -p {crate} {feat}")
    if p.returncode != 0:
        raise RuntimeError(f"warm build failed for {crate}:\n{p.stderr[-2000:]}")
    return round(cold, 2), round(warm, 2)


def test_count(crate, features):
    feat = f"--features {features}" if features else ""
    _, p = sh(f"cargo test -p {crate} {feat} 2>&1")
    return sum(int(m) for m in re.findall(r"(\d+) passed", p.stdout))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crates", nargs="*", default=None,
                    help="crates to measure (default: all workspace members)")
    ap.add_argument("--features", default="",
                    help="cargo features, e.g. 'cuda' or 'boost-fibers'")
    ap.add_argument("--cpp-cmd", default=None,
                    help="C++ baseline build command to time (recorded as "
                         "crate 'cpp-baseline')")
    args = ap.parse_args()

    crates = args.crates
    if not crates:
        crates = [d.name for d in ROOT.iterdir()
                  if (d / "Cargo.toml").exists() and d.name != "metrics"]

    rows = []
    stamp = datetime.date.today().isoformat()
    sha = git_sha()
    for crate in crates:
        feats = crate_features(crate, args.features)
        loc, unsafe_count = crate_loc_and_unsafe(crate)
        cold, warm = build_times(crate, feats)
        rows.append({
            "date": stamp, "git_sha": sha, "crate": crate, "loc": loc,
            "unsafe_count": unsafe_count,
            "clippy_warnings": clippy_warnings(crate, feats),
            "cold_build_s": cold, "warm_build_s": warm,
            "tests_passed": test_count(crate, feats),
        })

    if args.cpp_cmd:
        dur, p = sh(args.cpp_cmd, timeout=7200)
        rows.append({
            "date": stamp, "git_sha": sha, "crate": "cpp-baseline",
            "loc": "", "unsafe_count": "", "clippy_warnings": "",
            "cold_build_s": round(dur, 2), "warm_build_s": "",
            "tests_passed": "" if p.returncode == 0 else "BUILD_FAILED",
        })

    new_file = not HISTORY.exists()
    with open(HISTORY, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerows(rows)

    hdr = f"{'crate':<16}{'loc':>7}{'unsafe':>8}{'clippy':>8}{'cold_s':>9}{'warm_s':>9}{'tests':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['crate']:<16}{r['loc']:>7}{r['unsafe_count']:>8}"
              f"{r['clippy_warnings']:>8}{r['cold_build_s']:>9}"
              f"{r['warm_build_s']:>9}{r['tests_passed']:>7}")


if __name__ == "__main__":
    main()
