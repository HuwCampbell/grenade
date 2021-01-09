
#ifdef USE_FLOAT
typedef float RealNum;
#else
typedef double RealNum;
#endif

extern "C" __global__ void sum_vectors_gpu(int inLen, int outLen, RealNum *inVec, RealNum *outVec) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // if (i==0) {
  //   printf("%d\n", inLen);
  //   printf("%d\n", outLen);
  // }
  if (i < outLen) {
    RealNum res = 0;
    for (int iVec = i; iVec < inLen; iVec += outLen) {
      res += inVec[iVec];
    }
    outVec[i] = res;
  }
}
