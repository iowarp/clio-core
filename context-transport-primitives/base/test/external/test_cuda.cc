#include "clio_ctp/util/gpu_api.h"

CTP_GPU_KERNEL static void TestKernel(int x) { printf("ASFASDFAS: %d\n", x); }

int main() { TestKernel<<<1, 1>>>(252); }
