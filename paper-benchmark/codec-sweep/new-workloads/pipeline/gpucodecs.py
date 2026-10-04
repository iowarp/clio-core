"""ctypes wrapper over libgpucodecs.so (gpucodecs_capi.cu): compress and
decompress uint8 CUDA tensors with any gpu_codecs setting string, e.g.
"zstd shuffle=byte elem=8". Times are the codec's own GPU time (CUDA events),
shuffle and un-shuffle included, exactly as corpus_sweep measures them."""
import ctypes
import os

import torch

_LIB = ctypes.CDLL(os.path.expanduser("~/np-build/codec-sweep/libgpucodecs.so"))
_LIB.gc_create.restype = ctypes.c_void_p
_LIB.gc_create.argtypes = [ctypes.c_char_p]
_LIB.gc_destroy.argtypes = [ctypes.c_void_p]
_LIB.gc_bound.restype = ctypes.c_size_t
_LIB.gc_bound.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_LIB.gc_compress.restype = ctypes.c_size_t
_LIB.gc_compress.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                             ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_float)]
_LIB.gc_decompress.restype = ctypes.c_int
_LIB.gc_decompress.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                               ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_float)]

STORE = "store"  # no codec: the chunk is kept as is (a device-to-device copy)


class GpuCodec:
    """One built codec setting. Not thread safe; build one per setting."""

    def __init__(self, spec):
        self.spec = spec
        self._h = None if spec == STORE else _LIB.gc_create(spec.encode())
        if spec != STORE and not self._h:
            raise ValueError(f"cannot build codec {spec!r}")

    def __del__(self):
        if getattr(self, "_h", None):
            _LIB.gc_destroy(self._h)

    def compress(self, x):
        """@param x contiguous uint8 CUDA tensor
        @return (compressed uint8 CUDA tensor or None if the codec cannot take
                 it, GPU ms)"""
        n = x.numel()
        if self._h is None:
            t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
            t0.record()
            out = x.clone()
            t1.record()
            t1.synchronize()
            return out, t0.elapsed_time(t1)
        cap = _LIB.gc_bound(self._h, n)
        if cap == 0:
            return None, 0.0
        out = torch.empty(cap, dtype=torch.uint8, device=x.device)
        ms = ctypes.c_float(0.0)
        got = _LIB.gc_compress(self._h, x.data_ptr(), n, out.data_ptr(), cap, ctypes.byref(ms))
        if got == 0:
            return None, ms.value
        return out[:got], ms.value

    def decompress(self, c, n):
        """@param c compressed uint8 CUDA tensor, n original bytes
        @return (uint8 CUDA tensor of n bytes, GPU ms)"""
        if self._h is None:
            t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
            t0.record()
            out = c.clone()
            t1.record()
            t1.synchronize()
            return out, t0.elapsed_time(t1)
        out = torch.empty(n, dtype=torch.uint8, device=c.device)
        ms = ctypes.c_float(0.0)
        if not _LIB.gc_decompress(self._h, c.data_ptr(), c.numel(), out.data_ptr(), n,
                                  ctypes.byref(ms)):
            raise RuntimeError(f"decompress failed: {self.spec}")
        return out, ms.value
