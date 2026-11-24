#!/usr/bin/env python3
"""Print the sanitizer reports of a `ctest -T MemCheck` run.

ctest deletes each MemoryChecker.<test#>.log.<pid> file once it has parsed
it, so after the run the reports survive only in the DynamicAnalysis.xml
it writes under Testing/<tag>/ (and on CDash). This reads that file.

Usage:
  memcheck_reports.py BUILD_DIR stacks    # each defective test's report
  memcheck_reports.py BUILD_DIR messages  # one line per distinct message
"""

import base64
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET
import zlib

ERROR_RE = re.compile(r"ERROR: [A-Za-z]+Sanitizer|runtime error:")
MESSAGE_RE = re.compile(
    r"(runtime error: .*|ERROR: [A-Za-z]+Sanitizer: .*"
    r"|SUMMARY: [A-Za-z]+Sanitizer: .*)")


def newest_analysis(build_dir):
    files = glob.glob(os.path.join(build_dir, "Testing", "*",
                                   "DynamicAnalysis.xml"))
    return max(files, key=os.path.getmtime) if files else None


def log_text(log):
    """The <Log> text, which ctest gzips and base64-encodes by default."""
    text = log.text or ""
    if log.get("compression") == "gzip":
        raw = base64.b64decode(text)
        try:
            text = zlib.decompress(raw).decode("utf-8", "replace")
        except zlib.error:
            text = zlib.decompress(raw, 16 + zlib.MAX_WBITS).decode(
                "utf-8", "replace")
    return text


def defective_tests(path):
    """Yield (test name, defect counts, log text) for tests with defects."""
    for test in ET.parse(path).getroot().iter("Test"):
        defects = {d.get("type") or d.get("Type"): d.text
                   for d in test.iter("Defect")}
        if not any((n or "0").strip() not in ("", "0")
                   for n in defects.values()):
            continue
        log = test.find("Log")
        yield (test.findtext("Name") or "?", defects,
               log_text(log) if log is not None else "")


def print_stacks(name, defects, text):
    counts = ", ".join(f"{k}: {v}" for k, v in defects.items())
    print(f"==== sanitizer report: {name} ({counts})")
    lines = text.splitlines()
    if "ERROR: LeakSanitizer" in text:
        # Direct leaks name the allocation that was never freed; the
        # indirect ones hang off them. Print up to 8 direct-leak stacks,
        # then the summary.
        n, on, out = 0, False, []
        for line in lines:
            if line.startswith("Direct leak of"):
                n += 1
                on = n <= 8
            elif line.startswith(("Indirect leak of", "-----", "SUMMARY: ")):
                on = False
            if on or line.startswith("SUMMARY: "):
                out.append(line)
        print("\n".join(out[:400]))
    else:
        shown = 0
        for i, line in enumerate(lines):
            if ERROR_RE.search(line):
                print("\n".join(lines[i:i + 40]))
                shown += 1
                if shown == 2:
                    break


def main():
    if len(sys.argv) != 3 or sys.argv[2] not in ("stacks", "messages"):
        sys.exit(__doc__)
    path = newest_analysis(sys.argv[1])
    if path is None:
        print(f"No DynamicAnalysis.xml under {sys.argv[1]}/Testing")
        return
    tests = list(defective_tests(path))
    if sys.argv[2] == "stacks":
        for name, defects, text in tests:
            print_stacks(name, defects, text)
        return
    counts = {}
    for _, _, text in tests:
        for m in MESSAGE_RE.findall(text):
            counts[m] = counts.get(m, 0) + 1
    for m, c in sorted(counts.items(), key=lambda kv: -kv[1])[:40]:
        print(f"{c:7d} {m}")


if __name__ == "__main__":
    main()
