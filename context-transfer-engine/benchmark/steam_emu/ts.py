"""ts.py -- prefix each stdin line with the milliseconds since start.

Used to timestamp a WinFsp request trace for steam_trace.py:
  clio_cte_fuse X: -f -d 2>&1 | python -u ts.py > trace.log
"""
import sys
import time

t0 = time.perf_counter()
for line in sys.stdin:
  sys.stdout.write("%.3f %s" % ((time.perf_counter() - t0) * 1000.0, line))
