// Does a TRIVIAL kernel making an indirect call inherit the register cost of
// the HEAVIEST address-taken function in the module?
__device__ void CalleeLight(void *p) { *(int *)p = 1; }

__device__ __noinline__ void CalleeHeavy(void *p) {   // address-taken, heavy
  int *o = (int *)p;
  float a[24];
#pragma unroll
  for (int i = 0; i < 24; ++i) a[i] = o[i] * 1.5f + i;
#pragma unroll
  for (int i = 0; i < 24; ++i) a[i] = a[i] * a[(i + 5) & 23] + a[(i + 11) & 23];
#pragma unroll
  for (int i = 0; i < 24; ++i) a[i] = a[i] * a[(i + 9) & 23] - a[(i + 3) & 23];
  float s = 0;
#pragma unroll
  for (int i = 0; i < 24; ++i) s += a[i] * a[23 - i];
  o[0] = (int)s;
}

// Both pointers escape, so the indirect callee set = {light, heavy}.
__device__ void (*volatile g_fp)(void *) = CalleeLight;
__device__ void (*volatile g_fp2)(void *) = CalleeHeavy;

__global__ void K_TrivialBodyIndirect(int *out) {   // body is one store
  g_fp(out + threadIdx.x);
}
__global__ void K_TrivialNoIndirect(int *out) {     // identical body, no call
  out[threadIdx.x] = (int)threadIdx.x;
}
