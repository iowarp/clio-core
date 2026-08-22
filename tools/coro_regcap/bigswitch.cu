// Does a LARGE switch over direct calls stay direct, or become a
// function-pointer table (= indirect call, = the bug returns)?
__device__ __noinline__ void C0(void*p){int*o=(int*)p;float a[6];
#pragma unroll
 for(int k=0;k<6;++k)a[k]=o[k]*1.0f+k;
#pragma unroll
 for(int k=0;k<6;++k)a[k]=a[k]*a[(k+0+1)%6]+a[(k+2)%6];
 float s=0;
#pragma unroll
 for(int k=0;k<6;++k)s+=a[k];o[0]=(int)s;}
__device__ __noinline__ void C1(void*p){int*o=(int*)p;float a[6];
#pragma unroll
 for(int k=0;k<6;++k)a[k]=o[k]*1.25f+k;
#pragma unroll
 for(int k=0;k<6;++k)a[k]=a[k]*a[(k+1+1)%6]+a[(k+2)%6];
 float s=0;
#pragma unroll
 for(int k=0;k<6;++k)s+=a[k];o[0]=(int)s;}
__device__ __noinline__ void C2(void*p){int*o=(int*)p;float a[6];
#pragma unroll
 for(int k=0;k<6;++k)a[k]=o[k]*1.5f+k;
#pragma unroll
 for(int k=0;k<6;++k)a[k]=a[k]*a[(k+2+1)%6]+a[(k+2)%6];
 float s=0;
#pragma unroll
 for(int k=0;k<6;++k)s+=a[k];o[0]=(int)s;}
__device__ __noinline__ void C3(void*p){int*o=(int*)p;float a[6];
#pragma unroll
 for(int k=0;k<6;++k)a[k]=o[k]*1.75f+k;
#pragma unroll
 for(int k=0;k<6;++k)a[k]=a[k]*a[(k+3+1)%6]+a[(k+2)%6];
 float s=0;
#pragma unroll
 for(int k=0;k<6;++k)s+=a[k];o[0]=(int)s;}
__device__ __noinline__ void C4(void*p){int*o=(int*)p;float a[6];
#pragma unroll
 for(int k=0;k<6;++k)a[k]=o[k]*2.0f+k;
#pragma unroll
 for(int k=0;k<6;++k)a[k]=a[k]*a[(k+4+1)%6]+a[(k+2)%6];
 float s=0;
#pragma unroll
 for(int k=0;k<6;++k)s+=a[k];o[0]=(int)s;}
__device__ __noinline__ void C5(void*p){int*o=(int*)p;float a[6];
#pragma unroll
 for(int k=0;k<6;++k)a[k]=o[k]*2.25f+k;
#pragma unroll
 for(int k=0;k<6;++k)a[k]=a[k]*a[(k+0+1)%6]+a[(k+2)%6];
 float s=0;
#pragma unroll
 for(int k=0;k<6;++k)s+=a[k];o[0]=(int)s;}
__device__ __noinline__ void C6(void*p){int*o=(int*)p;float a[6];
#pragma unroll
 for(int k=0;k<6;++k)a[k]=o[k]*2.5f+k;
#pragma unroll
 for(int k=0;k<6;++k)a[k]=a[k]*a[(k+1+1)%6]+a[(k+2)%6];
 float s=0;
#pragma unroll
 for(int k=0;k<6;++k)s+=a[k];o[0]=(int)s;}
__device__ __noinline__ void C7(void*p){int*o=(int*)p;float a[6];
#pragma unroll
 for(int k=0;k<6;++k)a[k]=o[k]*2.75f+k;
#pragma unroll
 for(int k=0;k<6;++k)a[k]=a[k]*a[(k+2+1)%6]+a[(k+2)%6];
 float s=0;
#pragma unroll
 for(int k=0;k<6;++k)s+=a[k];o[0]=(int)s;}
__device__ __noinline__ void C8(void*p){int*o=(int*)p;float a[6];
#pragma unroll
 for(int k=0;k<6;++k)a[k]=o[k]*3.0f+k;
#pragma unroll
 for(int k=0;k<6;++k)a[k]=a[k]*a[(k+3+1)%6]+a[(k+2)%6];
 float s=0;
#pragma unroll
 for(int k=0;k<6;++k)s+=a[k];o[0]=(int)s;}
__device__ __noinline__ void C9(void*p){int*o=(int*)p;float a[6];
#pragma unroll
 for(int k=0;k<6;++k)a[k]=o[k]*3.25f+k;
#pragma unroll
 for(int k=0;k<6;++k)a[k]=a[k]*a[(k+4+1)%6]+a[(k+2)%6];
 float s=0;
#pragma unroll
 for(int k=0;k<6;++k)s+=a[k];o[0]=(int)s;}
__device__ __noinline__ void C10(void*p){int*o=(int*)p;float a[6];
#pragma unroll
 for(int k=0;k<6;++k)a[k]=o[k]*3.5f+k;
#pragma unroll
 for(int k=0;k<6;++k)a[k]=a[k]*a[(k+0+1)%6]+a[(k+2)%6];
 float s=0;
#pragma unroll
 for(int k=0;k<6;++k)s+=a[k];o[0]=(int)s;}
__device__ __noinline__ void C11(void*p){int*o=(int*)p;float a[6];
#pragma unroll
 for(int k=0;k<6;++k)a[k]=o[k]*3.75f+k;
#pragma unroll
 for(int k=0;k<6;++k)a[k]=a[k]*a[(k+1+1)%6]+a[(k+2)%6];
 float s=0;
#pragma unroll
 for(int k=0;k<6;++k)s+=a[k];o[0]=(int)s;}
__device__ __noinline__ void C12(void*p){int*o=(int*)p;float a[6];
#pragma unroll
 for(int k=0;k<6;++k)a[k]=o[k]*4.0f+k;
#pragma unroll
 for(int k=0;k<6;++k)a[k]=a[k]*a[(k+2+1)%6]+a[(k+2)%6];
 float s=0;
#pragma unroll
 for(int k=0;k<6;++k)s+=a[k];o[0]=(int)s;}
__device__ __noinline__ void C13(void*p){int*o=(int*)p;float a[6];
#pragma unroll
 for(int k=0;k<6;++k)a[k]=o[k]*4.25f+k;
#pragma unroll
 for(int k=0;k<6;++k)a[k]=a[k]*a[(k+3+1)%6]+a[(k+2)%6];
 float s=0;
#pragma unroll
 for(int k=0;k<6;++k)s+=a[k];o[0]=(int)s;}
__device__ __noinline__ void C14(void*p){int*o=(int*)p;float a[6];
#pragma unroll
 for(int k=0;k<6;++k)a[k]=o[k]*4.5f+k;
#pragma unroll
 for(int k=0;k<6;++k)a[k]=a[k]*a[(k+4+1)%6]+a[(k+2)%6];
 float s=0;
#pragma unroll
 for(int k=0;k<6;++k)s+=a[k];o[0]=(int)s;}
__device__ __noinline__ void C15(void*p){int*o=(int*)p;float a[6];
#pragma unroll
 for(int k=0;k<6;++k)a[k]=o[k]*4.75f+k;
#pragma unroll
 for(int k=0;k<6;++k)a[k]=a[k]*a[(k+0+1)%6]+a[(k+2)%6];
 float s=0;
#pragma unroll
 for(int k=0;k<6;++k)s+=a[k];o[0]=(int)s;}
__global__ void K_BigSwitch(int*out,int id){switch(id){
case 0: C0(out); break;
case 1: C1(out); break;
case 2: C2(out); break;
case 3: C3(out); break;
case 4: C4(out); break;
case 5: C5(out); break;
case 6: C6(out); break;
case 7: C7(out); break;
case 8: C8(out); break;
case 9: C9(out); break;
case 10: C10(out); break;
case 11: C11(out); break;
case 12: C12(out); break;
case 13: C13(out); break;
case 14: C14(out); break;
case 15: C15(out); break;
default: __builtin_unreachable();}}
