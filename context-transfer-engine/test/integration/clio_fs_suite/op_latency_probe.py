import os, sys, time
p = sys.argv[1]
def t(f):
    a = time.time(); r = f(); return (time.time() - a) * 1000, r
res = {'open': [], 'pread': [], 'close': [], 'pwrite': [], 'fsync': []}
for k in range(20):
    ms, fd = t(lambda: os.open(p, os.O_RDWR)); res['open'].append(ms)
    ms, _ = t(lambda: os.pread(fd, 4096, (k * 7 % 256) * 4096)); res['pread'].append(ms)
    ms, _ = t(lambda: os.pwrite(fd, b'x' * 4096, (k * 7 % 256) * 4096)); res['pwrite'].append(ms)
    ms, _ = t(lambda: os.fsync(fd)); res['fsync'].append(ms)
    ms, _ = t(lambda: os.close(fd)); res['close'].append(ms)
for k, v in res.items():
    v.sort(); print(f'{k:7} median {v[len(v)//2]:8.2f} ms  max {v[-1]:8.2f} ms')
