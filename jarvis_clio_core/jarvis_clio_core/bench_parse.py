"""Shared results parser for the S3 benchmark packages.

Every driver in the sweep -- the C++ ``clio_s3_read_bench`` /
``clio_s3_write_bench`` and the Python ``zarr_s3_read.py`` /
``zarr_s3_write.py`` / ``s3_raw_put.py`` -- emits the same two block types:

    === <Label> Benchmark Results ===     (clio_bench::PrintResults wording)
    ...
    ===========================

    === <Label> Fairness ===              (equivalence caveats)
    ...
    ===================

The throughput table is byte-identical to the contract documented in
``context-transfer-engine/benchmark/bench_common.h`` and already parsed by
``clio_cte_bench/pkg.py``; keeping the wording identical is what lets one table
serve every benchmark in the repo. The ``Fairness`` header deliberately does
NOT match the legacy ``=== (\\w+) Benchmark Results ===`` regex, so adding it
cannot disturb the existing clio_cte_bench parser.

Emitted stat keys are ``<pkg_id>.<label>.<metric>``. A driver may emit several
labels in one run (the Zarr driver emits ``Read`` and ``Readzstd``), and the two
block types share a label so their metrics merge into one namespace.

NOTHING HERE MAY RAISE. jarvis calls ``_get_stat`` inside a bare
``except Exception: warn`` -- one exception silently drops EVERY column the
package would have contributed, which surfaces as a green row with a blank
throughput column rather than as an error.
"""

import re

_ANSI = re.compile(r'\033\[[0-9;]*m')
# Deliberately UNANCHORED. The C++ driver emits through HLOG, which prefixes
# every line with "<file>:<line> <LEVEL> <pid> <func> ", so the banner is never
# at column 0; the Python driver prints bare. Anchoring would silently match
# only the Python side. (This is also why clio_cte_bench's regexes are
# unanchored.)
_HEADER = re.compile(r'=== (\w+) (?:Benchmark Results|Fairness) ===')

# Byte-identical to the bench_common.h PrintResults contract.
THROUGHPUT_PATTERNS = {
    'time_min_us': r'Time \(min\):\s+([\d.e+\-]+)\s+us',
    'time_max_us': r'Time \(max\):\s+([\d.e+\-]+)\s+us',
    'time_avg_us': r'Time \(avg\):\s+([\d.e+\-]+)\s+us',
    'bw_per_thread_min_mbps':
        r'Bandwidth per thread \(min\):\s+([\d.e+\-]+)\s+MB/s',
    'bw_per_thread_max_mbps':
        r'Bandwidth per thread \(max\):\s+([\d.e+\-]+)\s+MB/s',
    'bw_per_thread_avg_mbps':
        r'Bandwidth per thread \(avg\):\s+([\d.e+\-]+)\s+MB/s',
    'agg_bw_mbps': r'Aggregate bandwidth:\s+([\d.e+\-]+)\s+MB/s',
    'agg_ops_per_sec': r'Aggregate IOPS:\s+([\d.e+\-]+)',
    'ops_per_thread_avg_per_sec': r'IOPS per thread \(avg\):\s+([\d.e+\-]+)',
    'avg_latency_per_op_us': r'Avg latency per op:\s+([\d.e+\-]+)\s+us',
    'latency_stddev_us': r'Latency stddev:\s+([\d.e+\-]+)\s+us',
    'total_data_mb': r'Total data:\s+([\d.e+\-]+)\s+MB',
    'total_ops': r'Total ops:\s+(\d+)',
}

# Equivalence caveats: what each stack actually did to produce its number.
FAIRNESS_NUMERIC = {
    'objects_read': r'Objects read:\s+(\d+)',
    # Write-side counterparts of objects_read / get_count. Both tables are
    # applied to every results block, so a read row simply leaves these unset
    # and a write row leaves the read keys unset -- no label dispatch needed.
    'objects_written': r'Objects written:\s+(\d+)',
    'put_count': r'PUT count:\s+(\d+)',
    'bytes_moved': r'Bytes moved:\s+(\d+)',
    'logical_bytes': r'Logical bytes:\s+(\d+)',
    'get_count': r'GET count:\s+(\d+)',
    'requested_concurrency': r'Requested concurrency:\s+(\d+)',
    'effective_concurrency': r'Effective concurrency:\s+(\d+)',
    'runtime_worker_threads': r'Runtime worker threads:\s+(\d+)',
    'wall_time_us': r'Wall time us:\s+([\d.e+\-]+)',
    'wire_bw_mbps': r'Wire bandwidth:\s+([\d.e+\-]+)\s+MB/s',
    'subprocess_spawns': r'Subprocess spawns:\s+(\d+)',
    'temp_file_bytes': r'Temp file bytes:\s+(\d+)',
    'transport_chunk_bytes': r'Transport chunk bytes:\s+(\d+)',
    'checksum': r'Checksum:\s+(-?\d+)',
}

FAIRNESS_TEXT = {
    'compression': r'Compression:\s+(\S+)',
    'decode_step': r'Decode step:\s+(\S+)',
}

_RSS = re.compile(r'Maximum resident set size \(kbytes\):\s+(\d+)')


def parse_bench_output(text, pkg_id, stat_dict):
    """Scrape every results block in ``text`` into ``stat_dict``.

    Splits on both block headers, then runs all three metric tables against
    each block body. Blocks sharing a label merge into one namespace.

    :param text: Raw driver output (stdout+stderr, ANSI codes tolerated).
    :param pkg_id: Jarvis package id, used as the stat key prefix.
    :param stat_dict: Dict to populate with ``<pkg_id>.<label>.<metric>`` keys.
    :return: Number of metrics extracted (0 on any failure).
    """
    try:
        text = _ANSI.sub('', text)
        marks = list(_HEADER.finditer(text))
        if not marks:
            return 0
        found = 0
        for i, m in enumerate(marks):
            label = m.group(1).lower()
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            body = text[m.end():end]
            for table, cast in ((THROUGHPUT_PATTERNS, float),
                                (FAIRNESS_NUMERIC, float),
                                (FAIRNESS_TEXT, str)):
                for metric, pattern in table.items():
                    hit = re.search(pattern, body)
                    if hit:
                        stat_dict[f'{pkg_id}.{label}.{metric}'] = cast(
                            hit.group(1))
                        found += 1
        return found
    except Exception:
        return 0


def parse_time_v(path, pkg_id, label, stat_dict):
    """Scrape peak RSS out of a ``/usr/bin/time -v -o <path>`` report.

    :param path: Path to the time(1) report file.
    :param pkg_id: Jarvis package id, used as the stat key prefix.
    :param label: Results namespace this measurement belongs to.
    :param stat_dict: Dict to populate with ``<pkg_id>.<label>.max_rss_kb``.
    :return: True when a value was recorded.
    """
    try:
        with open(path, 'r') as f:
            hit = _RSS.search(f.read())
        if hit:
            stat_dict[f'{pkg_id}.{label}.max_rss_kb'] = float(hit.group(1))
            return True
    except Exception:
        pass
    return False
