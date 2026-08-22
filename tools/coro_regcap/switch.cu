// Does a SWITCH over direct calls avoid paying for the heavy function,
// where an indirect call does not?
__device__ __noinline__ void Light(void *p)  { *(int *)p = 1; }

__device__ __noinline__ void Medium(void *p) {
  int *o = (int *)p; float a[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) a[i] = o[i] * 1.5f + i;
#pragma unroll
  for (int i = 0; i < 8; ++i) a[i] = a[i] * a[(i + 3) & 7] + a[(i + 5) & 7];
  float s = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) s += a[i];
  o[0] = (int)s;
}

__device__ __noinline__ void Heavy(void *p) {     // never reachable below
  int *o = (int *)p; float a[24];
#pragma unroll
  for (int i = 0; i < 24; ++i) a[i] = o[i] * 1.5f + i;
#pragma unroll
  for (int i = 0; i < 24; ++i) a[i] = a[i] * a[(i + 5) % 24] + a[(i + 11) % 24];
#pragma unroll
  for (int i = 0; i < 24; ++i) a[i] = a[i] * a[(i + 9) % 24] - a[(i + 3) % 24];
  float s = 0;
#pragma unroll
  for (int i = 0; i < 24; ++i) s += a[i] * a[23 - i];
  o[0] = (int)s;
}

// SWITCH form: Light/Medium called DIRECTLY, so their addresses never escape.
__global__ void K_Switch(int *out, int id) {
  switch (id) {
    case 0: Light(out);  break;
    case 1: Medium(out); break;
  }
}
// Someone still needs Heavy, called directly from its own kernel.
__global__ void K_UsesHeavy(int *out) { Heavy(out); }

// INDIRECT form: all three escape through a pointer table.
__device__ void (*volatile g_tab[3])(void *) = {Light, Medium, Heavy};
__global__ void K_Indirect(int *out, int id) { g_tab[id](out); }
