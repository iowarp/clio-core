"""steam_trace.py -- turn a WinFsp request trace of a Steam install into a
manifest for steam_emu.py, and summarize the access pattern per phase.

The trace is clio_cte_fuse's WinFsp debug output (-d) with each line
prefixed by a millisecond timestamp (clio_cte_fuse -d ... | ts), so its
lines look like:
  733443.806 clio_cte_fuse[TID=13c44]: FFFF...: >>Read <ctx>, ..., Offset=0:37000, Length=65536

Usage:
  python steam_trace.py TRACE OUT.json.gz --app-id 1466860 \
      --game-dir "Age of Empires IV" --cuts T_PRE,T_STAGE,T_COMMIT,T_END \
      [--anonymize]

--cuts are the trace times (ms) at which Steam's content log says each
phase began (Preallocating, Staging, Committing) and the update finished.
Every handle is followed back to its path (the UserContext of the reply to
its Create), so each file's calls are seen in order. The output holds:
  manifest  [relpath, size, preallocated] in the order Steam first touched
            the files (relpath under the app's install directory)
  phases    per phase: files, threads, and the commonest per-file call
            signatures with their counts
--anonymize replaces every path component with a stable hash (keeping file
extensions), so a manifest can be shared without the game's file names.
"""
import argparse
import collections
import gzip
import hashlib
import json
import re

PHASES = ["prealloc", "staging", "commit"]
LINE = re.compile(r'^([0-9.]+) clio_cte_fuse\[TID=([0-9a-f]+)\]: '
                  r'([0-9A-F]+): (>>|<<)(\w+)(.*)$')
CREATE_REQ = re.compile(r'"([^"]*)", (\w+), CreateOptions=([0-9a-f]+)'
                        r'.*DesiredAccess=([0-9a-f]+)')
CREATE_REPLY = re.compile(r'IoStatus=([0-9a-f]+)\[\d+\]'
                          r'(?: UserContext=([0-9A-F]+:[0-9A-F]+))?')
CONTEXT = re.compile(r' ([0-9A-F]{16}:[0-9A-F]{16})')
RW = re.compile(r'Offset=([0-9a-f]+):([0-9a-f]+), Length=(\d+)')
EOF = re.compile(r'FileSize = ([0-9a-f]+):([0-9a-f]+)')
NEW_NAME = re.compile(r'NewFileName="([^"]*)"')


def hilo(hi, lo):
  """WinFsp prints 64-bit values as hex 'hi:lo'."""
  return (int(hi, 16) << 32) | int(lo, 16)


class Trace:
  """Per-file call sequences of one Steam install, by phase."""

  def __init__(self, app_id, game_dir, cuts):
    self.marks = ("\\downloading\\%s\\" % app_id,
                  "\\common\\%s\\" % game_dir.lower())
    self.cuts = cuts
    self.pending = {}                          # IRP -> request
    self.handle = {}                           # UserContext -> relpath
    self.calls = collections.defaultdict(list)  # (phase, relpath) -> calls
    self.threads = collections.defaultdict(set)
    self.size, self.extent, self.prealloc = {}, collections.Counter(), set()

  def phase(self, t):
    """Phase of a request issued at trace time t (ms), or None."""
    for i, name in enumerate(PHASES):
      if self.cuts[i] <= t < self.cuts[i + 1]:
        return name
    return None

  def relpath(self, path):
    """Path under the staging or install directory (lower case), or None."""
    low = path.lower()
    for mark in self.marks:
      k = low.find(mark)
      if k >= 0:
        return low[k + len(mark):]
    return None

  def feed(self, line):
    """Consume one trace line."""
    m = LINE.match(line)
    if not m:
      return
    t, tid, irp, arrow, op, rest = m.groups()
    if arrow == ">>":
      self.pending[irp] = (float(t), tid, op, rest)
      return
    req = self.pending.pop(irp, None)
    if req is None:
      return
    t0, tid, op, args = req
    ph = self.phase(t0)
    if ph is None:
      return
    self.threads[ph].add(tid)
    if op == "Create":
      self.on_create(ph, args, rest)
    else:
      self.on_handle_op(ph, op, args)

  def on_create(self, ph, args, reply):
    """Record an open of a file (directories are skipped)."""
    c, r = CREATE_REQ.search(args), CREATE_REPLY.search(reply)
    if not c or not r:
      return
    path, disposition, options, access = c.groups()
    rel = self.relpath(path)
    if rel is None or int(options, 16) & 1:
      return
    where = "staged" if self.marks[0] in path.lower() else "installed"
    ok = r.group(1) == "0"
    self.calls[(ph, rel)].append("open %s %s %s %s" % (
        where, disposition, access, "ok" if ok else r.group(1)))
    if ok and r.group(2):
      self.handle[r.group(2)] = rel

  def on_handle_op(self, ph, op, args):
    """Record a call made through an open handle."""
    c = CONTEXT.search(args)
    rel = self.handle.get(c.group(1)) if c else None
    if rel is None:
      return
    calls = self.calls[(ph, rel)]
    if op in ("Read", "Write"):
      m = RW.search(args)
      off, n = hilo(m.group(1), m.group(2)), int(m.group(3))
      calls.append("%s %s" % (op.lower(), "1MiB" if n == 1 << 20 else "small"))
      if op == "Write":
        self.extent[rel] = max(self.extent[rel], off + n)
    elif op == "SetInformation":
      kind = args.split("[", 1)[1].split("]", 1)[0]
      e = EOF.search(args)
      if e:
        self.size[rel] = hilo(e.group(1), e.group(2))
        if ph == "prealloc":
          self.prealloc.add(rel)
      calls.append("set" + kind)
    elif op in ("QueryInformation", "FlushBuffers", "Cleanup", "Close"):
      calls.append(op[0].lower() + op[1:])

  def manifest(self):
    """[relpath, size, preallocated] in first-touch order."""
    out, seen = [], set()
    for _, rel in self.calls:
      size = self.size.get(rel, self.extent.get(rel))
      if rel in seen or size is None:
        continue
      seen.add(rel)
      out.append([rel, size, rel in self.prealloc])
    return out

  def signatures(self, top):
    """Per phase: file count, thread count and the commonest call runs."""
    out = {}
    for ph in PHASES:
      sigs = collections.Counter()
      for (p, _), calls in self.calls.items():
        if p == ph:
          sigs[collapse(calls)] += 1
      out[ph] = {"files": sum(sigs.values()), "threads": len(self.threads[ph]),
                 "top": sigs.most_common(top)}
    return out


def collapse(calls):
  """Join calls, folding runs of the same call into 'call *N'."""
  runs = []
  for c in calls:
    if runs and runs[-1][0] == c:
      runs[-1][1] += 1
    else:
      runs.append([c, 1])
  return " | ".join(c if n == 1 else "%s *%d" % (c, n) for c, n in runs)


def anonymize(rel):
  """Replace each path component by a stable hash, keeping the extension."""
  parts = rel.split("\\")
  out = []
  for i, p in enumerate(parts):
    stem, dot, ext = p.rpartition(".") if i == len(parts) - 1 else (p, "", "")
    if not dot:
      stem, ext = p, ""
    h = hashlib.sha256(stem.encode()).hexdigest()[:10]
    out.append(("d" if i < len(parts) - 1 else "f") + h + (dot + ext if dot else ""))
  return "/".join(out)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("trace")
  ap.add_argument("out")
  ap.add_argument("--app-id", required=True)
  ap.add_argument("--game-dir", required=True)
  ap.add_argument("--cuts", required=True,
                  help="ms: prealloc,staging,commit start and update end")
  ap.add_argument("--anonymize", action="store_true")
  ap.add_argument("--top", type=int, default=12)
  a = ap.parse_args()
  tr = Trace(a.app_id, a.game_dir, [float(x) for x in a.cuts.split(",")])
  with open(a.trace, encoding="utf-8", errors="replace") as f:
    for line in f:
      tr.feed(line)
  man = tr.manifest()
  for m in man:
    m[0] = anonymize(m[0]) if a.anonymize else m[0].replace("\\", "/")
  sig = tr.signatures(a.top)
  for ph, s in sig.items():
    print("== %s: %d files, %d service threads" % (ph, s["files"], s["threads"]))
    for text, n in s["top"][:6]:
      print("  %6d  %s" % (n, text[:300]))
  print("manifest: %d files, %.1f GB, %d preallocated" % (
      len(man), sum(m[1] for m in man) / 1e9, sum(1 for m in man if m[2])))
  opener = gzip.open if a.out.endswith(".gz") else open
  with opener(a.out, "wt") as f:
    json.dump({"app_id": a.app_id, "manifest": man, "phases": sig}, f)


if __name__ == "__main__":
  main()
