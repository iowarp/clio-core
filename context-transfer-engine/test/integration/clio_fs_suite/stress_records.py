"""Self-verifying block records for the clio-fs data-integrity stress tests.

Every BLK-byte block a stress test writes is a record that names itself:

  [ 8 B magic "CLIOREC1" | u64 file_id | u64 block | u32 writer | u32 gen |
    u32 crc of the header | 28 B reserved (zero) | payload ]

The payload is the SHA-256 of the 64-byte header repeated to fill the block.
A read therefore classifies every block on its own, with no side table:

  (writer, gen)  an intact record written for exactly this file and block
  ZERO           all zero bytes (a hole, or never written)
  CORRUPT        anything else: a torn mix of two writes, garbage, a
                 truncated block, a bad checksum
  FOREIGN        an intact record of ANOTHER file or block -- data that
                 leaked (a freed extent reused without clearing, a tier
                 migration that copied the wrong blob, a misplaced page)

The tests model which (writer, gen) each block may legally hold; anything
CORRUPT or FOREIGN is data corruption, and a block holding an older version
than one whose fsync completed is data loss.
"""

import hashlib
import os
import random
import struct
import threading
import time
import zlib

BLK = 4096
MAGIC = b'CLIOREC1'
_HDR = struct.Struct('<8sQQIII28x')  # 64 bytes
ZERO = -1
CORRUPT = -2
FOREIGN = -3


def make_block(file_id, block, writer, gen, blk=BLK):
  """Encode one self-verifying record.

  Args:
    file_id: 64-bit id of the file (distinguishes files in a leak).
    block: block index within the file.
    writer: id of the writing process.
    gen: version written by that writer (monotonic per writer).
    blk: block size in bytes (a multiple of 32, >= 128).
  Returns:
    blk bytes.
  """
  head = struct.pack('<8sQQII', MAGIC, file_id, block, writer, gen)
  crc = zlib.crc32(head) & 0xffffffff
  hdr = _HDR.pack(MAGIC, file_id, block, writer, gen, crc)
  dig = hashlib.sha256(hdr).digest()
  return hdr + dig * ((blk - len(hdr)) // len(dig))


def classify(data, file_id, block, blk=BLK):
  """Classify one block read back.

  Args:
    data: the bytes read (may be short at EOF).
    file_id: the file it was read from.
    block: its block index.
    blk: block size.
  Returns:
    (writer, gen) for an intact record of this block, else
    (ZERO, 0), (CORRUPT, 0) or (FOREIGN, 0).
  """
  if len(data) == blk and data.count(0) == blk:
    return (ZERO, 0)
  if len(data) != blk or data[:8] != MAGIC:
    return (CORRUPT, 0)
  magic, fid, blkno, writer, gen, crc = _HDR.unpack(data[:_HDR.size])
  head = struct.pack('<8sQQII', magic, fid, blkno, writer, gen)
  if zlib.crc32(head) & 0xffffffff != crc:
    return (CORRUPT, 0)
  dig = hashlib.sha256(data[:_HDR.size]).digest()
  if data[_HDR.size:] != dig * ((blk - _HDR.size) // len(dig)):
    return (CORRUPT, 0)
  if fid != file_id or blkno != block:
    return (FOREIGN, 0)
  return (writer, gen)


def file_id_of(name):
  """Stable 64-bit id for a file name."""
  return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'little')


def write_runs(path, file_id, runs, writer, gen, fsync, blk=BLK,
               chunk_blocks=256):
  """pwrite records for block runs [[start, count], ...] of one file.

  Args:
    path: file (created if missing, never truncated).
    file_id: record file id.
    runs: block runs to write.
    writer: record writer id.
    gen: record version.
    fsync: fsync before closing.
    blk: block size.
    chunk_blocks: blocks per write(2).
  Returns:
    bytes written.
  """
  fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)
  total = 0
  try:
    for start, count in runs:
      b = start
      while b < start + count:
        n = min(chunk_blocks, start + count - b)
        buf = b''.join(make_block(file_id, b + i, writer, gen, blk)
                       for i in range(n))
        w = os.pwrite(fd, buf, b * blk)
        if w != len(buf):
          raise OSError(5, f'short write {w}/{len(buf)} at block {b}')
        total += w
        b += n
    if fsync:
      os.fsync(fd)
  finally:
    os.close(fd)
  return total


def scan(path, file_id, nblocks, blk=BLK, chunk_blocks=256):
  """Classify every block of a file.

  Args:
    path: file to read.
    file_id: its record file id.
    nblocks: blocks to classify (past EOF reads as short: CORRUPT unless the
      caller treats it as absent -- see the returned size).
    blk: block size.
    chunk_blocks: blocks per read(2).
  Returns:
    {'size': file size, 'runs': [[start, count, writer, gen], ...]}
    with consecutive blocks of the same (writer, gen) merged.
  """
  size = os.stat(path).st_size
  runs = []
  corrupt = []
  torn = {}
  foreign = {}
  fd = os.open(path, os.O_RDONLY)
  try:
    b = 0
    while b < nblocks:
      n = min(chunk_blocks, nblocks - b)
      data = os.pread(fd, n * blk, b * blk)
      for i in range(n):
        piece = data[i * blk:(i + 1) * blk]
        if not piece and (b + i) * blk >= size:
          w, g = ZERO, 0  # past EOF: never written
        else:
          w, g = classify(piece, file_id, b + i, blk)
          if w == FOREIGN and len(foreign) < 64:
            # Whose record is it: [file id, its block, writer, gen].
            _, ffid, fblk, fw, fg, _ = _HDR.unpack(piece[:_HDR.size])
            foreign[b + i] = [hex(ffid), fblk, fw, fg]
          if w == CORRUPT:
            d = describe_corrupt(piece, blk)
            if isinstance(d.get('head'), list) and d.get('rest_is'):
              # A write torn mid-block: [head writer, head gen, rest writer,
              # rest gen] -- the caller decides whether that tear is legal.
              torn[b + i] = d['head'][2:4] + d['rest_is']
            if len(corrupt) < 8:
              corrupt.append(d | {'block': b + i})
        if runs and runs[-1][2] == w and runs[-1][3] == g and \
            runs[-1][0] + runs[-1][1] == b + i:
          runs[-1][1] += 1
        else:
          runs.append([b + i, 1, w, g])
      b += n
  finally:
    os.close(fd)
  return {'size': size, 'runs': runs, 'corrupt': corrupt,
          'torn': {str(k): v for k, v in torn.items()},
          'foreign': {str(k): v for k, v in foreign.items()}}


def describe_corrupt(piece, blk=BLK):
  """Describe a CORRUPT block for a failure report.

  Args:
    piece: the block's bytes.
    blk: block size.
  Returns:
    {'len', 'nonzero', 'head' (header fields if the magic is there),
     'tail_zero_from' (offset where an all-zero tail starts, or None),
     'pattern_breaks_at' (first offset past the header where the record's
     digest pattern no longer matches, or None)}.
  """
  out = {'len': len(piece), 'nonzero': len(piece) - piece.count(0)}
  if piece[:8] == MAGIC and len(piece) >= _HDR.size:
    _, fid, blkno, writer, gen, _ = _HDR.unpack(piece[:_HDR.size])
    out['head'] = [hex(fid), blkno, writer, gen]
    dig = hashlib.sha256(piece[:_HDR.size]).digest()
    want = dig * ((blk - _HDR.size) // len(dig))
    got = piece[_HDR.size:]
    out['pattern_breaks_at'] = next(
        (_HDR.size + k for k in range(min(len(got), len(want)))
         if got[k] != want[k]), None)
  else:
    out['head'] = piece[:24].hex()
  stripped = piece.rstrip(b'\0')
  out['tail_zero_from'] = len(stripped) if len(stripped) < len(piece) else None
  brk = out.get('pattern_breaks_at')
  if brk is not None and 'head' in out and isinstance(out['head'], list):
    # Which record does the rest of the block belong to? Try this file and
    # block with other writers / gens (a torn write keeps an older version).
    fid, blkno = int(out['head'][0], 16), out['head'][1]
    for w in range(1, 17):
      for g in range(0, 129):
        ref = make_block(fid, blkno, w, g, blk)
        if piece[brk:] == ref[brk:]:
          out['rest_is'] = [w, g]
          return out
    out['rest_is'] = None
  return out


def _now():
  return time.time()


class SharedFileStress:
  """Concurrent single-block overwriters and readers on one shared file.

  Each writer thread overwrites random blocks with (writer, gen) records,
  gen increasing, and fsyncs after each write so the version is visible to
  every node on its next open (close-to-open). It logs
  [block, gen, t_issue, t_acked]. Each reader thread repeatedly opens the
  file, reads one random block, closes it, and logs
  [block, writer, gen, t_open, t_done] -- the orchestrator checks every
  observation against all writers' logs.
  """

  def __init__(self, path, file_id, nblocks, writer_base, writers, readers,
               secs, seed, blk=BLK, max_obs=40000, sync=True, truncater=False):
    self.path = path
    self.file_id = file_id
    self.nblocks = nblocks
    self.writer_base = writer_base
    self.nwriters = writers
    self.nreaders = readers
    self.secs = secs
    self.seed = seed
    self.blk = blk
    self.max_obs = max_obs
    # sync=False: no fsync after a write -- the version is visible on this
    # node once write(2) returns, elsewhere only after the final fsync.
    self.sync = sync
    # truncater: one extra thread shrinking/regrowing the file at random.
    self.truncater = truncater
    self.truncs = []    # [new_size_blocks, t_issue, t_done]
    self.writes = {}   # writer id -> [[block, gen, t_issue, t_ack], ...]
    self.inflight = {}  # writer id -> [block, gen, t_issue] or None
    self.obs = []
    self.errors = []
    self.lock = threading.Lock()

  def _writer(self, k):
    wid = self.writer_base + k
    rng = random.Random(f'{self.seed}:w{wid}')
    log = []
    self.writes[wid] = log
    gen = 0
    t_end = _now() + self.secs
    fd = os.open(self.path, os.O_WRONLY)
    try:
      while _now() < t_end:
        b = rng.randrange(self.nblocks)
        gen += 1
        rec = make_block(self.file_id, b, wid, gen, self.blk)
        t0 = _now()
        self.inflight[wid] = [b, gen, t0]
        try:
          if os.pwrite(fd, rec, b * self.blk) != self.blk:
            raise OSError(5, 'short write')
          if self.sync:
            os.fsync(fd)
        except OSError as e:
          with self.lock:
            self.errors.append(f'writer {wid} block {b} gen {gen}: {e}')
          # Unknown outcome: stays in flight (may or may not have landed).
          self.inflight[wid] = [b, gen, t0]
          break
        log.append([b, gen, t0, _now()])
        self.inflight[wid] = None
      if not self.sync:
        os.fsync(fd)  # publish everything before the final comparison
    finally:
      os.close(fd)

  def _truncator(self):
    rng = random.Random(f'{self.seed}:t{self.writer_base}')
    t_end = _now() + self.secs
    while _now() < t_end:
      nb = rng.randrange(self.nblocks // 2, self.nblocks + 1)
      t0 = _now()
      try:
        os.truncate(self.path, nb * self.blk)
      except OSError as e:
        with self.lock:
          self.errors.append(f'truncate to {nb}: {e}')
        continue
      self.truncs.append([nb, t0, _now()])
      time.sleep(rng.uniform(0.05, 0.5))
    # End at full size so every block is addressable for the final scan.
    os.truncate(self.path, self.nblocks * self.blk)
    self.truncs.append([self.nblocks, _now(), _now()])

  def _reader(self, k):
    rng = random.Random(f'{self.seed}:r{self.writer_base}:{k}')
    t_end = _now() + self.secs
    while _now() < t_end:
      b = rng.randrange(self.nblocks)
      t0 = _now()
      try:
        fd = os.open(self.path, os.O_RDONLY)
        try:
          data = os.pread(fd, self.blk, b * self.blk)
        finally:
          os.close(fd)
      except OSError as e:
        with self.lock:
          self.errors.append(f'reader block {b}: {e}')
        continue
      if not data:
        w, g = ZERO, 0  # past EOF (a racing truncate): absent, not corrupt
      else:
        w, g = classify(data, self.file_id, b, self.blk)
      with self.lock:
        if len(self.obs) < self.max_obs or w < 0 and w != ZERO:
          self.obs.append([b, w, g, t0, _now()])

  def run(self):
    """Run all threads to completion; return the logs."""
    ts = [threading.Thread(target=self._writer, args=(k,))
          for k in range(self.nwriters)]
    ts += [threading.Thread(target=self._reader, args=(k,))
           for k in range(self.nreaders)]
    if self.truncater:
      ts.append(threading.Thread(target=self._truncator))
    for t in ts:
      t.start()
    for t in ts:
      t.join()
    return {'writes': {str(k): v for k, v in self.writes.items()},
            'inflight': {str(k): v for k, v in self.inflight.items() if v},
            'obs': self.obs, 'errors': self.errors[:50],
            'nerrors': len(self.errors), 'truncs': self.truncs}


class FileSetWriter:
  """Keeps writing whole files of records until told to stop or killed.

  Used by the crash test: each file is written, fsynced and then logged as
  durable (with its version), so after a crash of every daemon the test
  knows exactly which versions must have survived. Files cycle through a
  fixed set of names so later rounds overwrite earlier (fsynced) versions.
  """

  def __init__(self, dirpath, writer, nfiles, blocks_per_file, secs, seed,
               blk=BLK, log_path=None, retry=False):
    self.dirpath = dirpath
    self.writer = writer
    self.nfiles = nfiles
    self.bpf = blocks_per_file
    self.secs = secs
    self.seed = seed
    self.blk = blk
    self.log_path = log_path
    # retry: keep going after a failed round (a node was lost; the next
    # round may succeed once its files fail over) instead of stopping.
    self.retry = retry

  def run(self):
    """Write rounds until secs elapse (or an error); return the log."""
    durable = {}   # name -> last fsynced gen
    started = {}   # name -> gen being written when we stopped / failed
    errors = []
    gen = 0
    t_end = _now() + self.secs
    logf = open(self.log_path, 'a') if self.log_path else None
    try:
      while _now() < t_end:
        gen += 1
        name = f'w{self.writer}_f{(gen - 1) % self.nfiles}'
        path = os.path.join(self.dirpath, name)
        started[name] = gen
        try:
          write_runs(path, file_id_of(name), [[0, self.bpf]], self.writer,
                     gen, fsync=True, blk=self.blk)
        except OSError as e:
          errors.append(f'{name} gen {gen}: {e}')
          if not self.retry:
            break
          time.sleep(1.0)
          continue
        durable[name] = gen
        started.pop(name, None)
        if logf:
          # Durable log on the (NFS) results dir: survives the crash.
          logf.write(f'{name} {gen}\n')
          logf.flush()
          os.fsync(logf.fileno())
    finally:
      if logf:
        logf.close()
    return {'durable': durable, 'started': started, 'errors': errors[:20],
            'nerrors': len(errors), 'rounds': gen}


class SafeSaveStress:
  """Concurrent write-temp / fsync / rename-over-target savers and readers.

  Each saver thread repeatedly writes a whole new version of one of the
  target files to a private temp name, fsyncs it and renames it over the
  target -- the pattern editors and databases rely on to never expose a
  half-written file. Reader threads open a target, read it whole and
  classify it: every block must be an intact record of that target with
  one single (writer, gen) throughout, and a target that existed must never
  be missing, short or mixed.
  """

  def __init__(self, dirpath, targets, nblocks, writer_base, savers,
               readers, secs, seed, blk=BLK):
    self.dirpath = dirpath
    self.targets = targets
    self.nblocks = nblocks
    self.writer_base = writer_base
    self.nsavers = savers
    self.nreaders = readers
    self.secs = secs
    self.seed = seed
    self.blk = blk
    self.saves = []    # [target, writer, gen, t_issue, t_renamed]
    self.reads = 0
    self.bad = []      # problems seen by readers
    self.errors = []
    self.lock = threading.Lock()

  def _saver(self, k):
    wid = self.writer_base + k
    rng = random.Random(f'{self.seed}:s{wid}')
    t_end = _now() + self.secs
    gen = 0
    while _now() < t_end:
      tgt = rng.choice(self.targets)
      gen += 1
      tmp = os.path.join(self.dirpath, f'.{tgt}.tmp.{wid}')
      t0 = _now()
      try:
        write_runs(tmp, file_id_of(tgt), [[0, self.nblocks]], wid, gen,
                   fsync=True, blk=self.blk)
        os.rename(tmp, os.path.join(self.dirpath, tgt))
      except OSError as e:
        with self.lock:
          self.errors.append(f'saver {wid} {tgt} gen {gen}: {e}')
        continue
      with self.lock:
        self.saves.append([tgt, wid, gen, t0, _now()])

  def _reader(self, k):
    rng = random.Random(f'{self.seed}:r{self.writer_base}:{k}')
    t_end = _now() + self.secs
    while _now() < t_end:
      tgt = rng.choice(self.targets)
      path = os.path.join(self.dirpath, tgt)
      step = 'open'
      try:
        fd = os.open(path, os.O_RDONLY)
        try:
          step = 'fstat'
          size = os.fstat(fd).st_size
          step = 'read'
          # CLIO_SUITE_READ_CHUNK_BLOCKS splits the whole-file read into
          # several pread(2)s (diagnostic: separates page-cache mixing
          # from a file whose bytes change between two reads).
          ck = int(os.environ.get('CLIO_SUITE_READ_CHUNK_BLOCKS', '0'))
          if ck <= 0:
            data = os.pread(fd, self.nblocks * self.blk, 0)
          else:
            data = b''.join(
                os.pread(fd, ck * self.blk, b * self.blk)
                for b in range(0, self.nblocks, ck))
          size2 = os.fstat(fd).st_size
        finally:
          os.close(fd)
      except FileNotFoundError:
        with self.lock:
          self.bad.append(f'{tgt}: missing (ENOENT at {step}) after it was '
                          f'created')
        continue
      except OSError as e:
        with self.lock:
          self.errors.append(f'{step} {tgt}: {e}')
        continue
      runs = []
      for b in range(self.nblocks):
        piece = data[b * self.blk:(b + 1) * self.blk]
        w, g = classify(piece, file_id_of(tgt), b, self.blk) if piece \
            else (ZERO, 0)
        if runs and runs[-1][2:] == [w, g]:
          runs[-1][1] += 1
        else:
          runs.append([b, 1, w, g])
      with self.lock:
        self.reads += 1
        full = self.nblocks * self.blk
        if size != full or len(data) != full:
          self.bad.append(f'{tgt}: short: fstat {size} (after {size2}), '
                          f'read {len(data)} of {full} '
                          f'at {time.strftime("%H:%M:%S")}')
        elif len(runs) != 1 or runs[0][2] < 0:
          first_bad = next((r for r in runs if r[2] < 0), None)
          sample = ''
          if first_bad is not None:
            piece = data[first_bad[0] * self.blk:(first_bad[0] + 1) *
                         self.blk]
            nz = len(piece) - piece.count(0)
            sample = (f' first bad block {first_bad[0]}: {nz} nonzero bytes, '
                      f'head {piece[:24].hex()}')
          self.bad.append(f'{tgt}: not one intact version: '
                          f'{[r[:4] for r in runs[:4]]}{sample} '
                          f'at {time.strftime("%H:%M:%S")}')

  def run(self):
    """Run savers and readers; return their logs."""
    ts = [threading.Thread(target=self._saver, args=(k,))
          for k in range(self.nsavers)]
    ts += [threading.Thread(target=self._reader, args=(k,))
           for k in range(self.nreaders)]
    for t in ts:
      t.start()
    for t in ts:
      t.join()
    return {'saves': self.saves, 'reads': self.reads, 'bad': self.bad[:50],
            'nbad': len(self.bad), 'errors': self.errors[:20],
            'nerrors': len(self.errors)}
