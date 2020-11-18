
#ifdef USE_FLOAT
typedef float RealNum;
#else
typedef double RealNum;
#endif

extern "C" __global__ void descend_adam_gpu(int len, int t, RealNum alpha, RealNum beta1, RealNum beta2, RealNum epsilon, RealNum lambda,
  RealNum* weights,
  const RealNum* gradient,
  RealNum* m,
  RealNum* v) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // if (i==0) {
  //   printf("%d\n", len);
  //   printf("%d\n", t);
  //   printf("%f\n", alpha);
  //   printf("%f\n", beta1);
  //   printf("%f\n", beta2);
  //   printf("%f\n", epsilon);
  //   printf("%f\n", lambda);
  // }
  if (i < len) {
    // printf("i: %d\n", i);
    m[i] = beta1 * m[i] + (1 - beta1) * gradient[i];
    v[i] = beta2 * v[i] + (1 - beta2) * gradient[i] * gradient[i];

    // Clear version (as in Algoritm 1 of the paper)
    /* RealNum mHat = outputM[i] / (1 - pow(beta1, t)); */
    /* RealNum vHat = outputV[i] / (1 - pow(beta2, t)); */
    /* outputWeights[i] = weights[i] - alpha * mHat / (sqrt(vHat) + epsilon); */

    // Slightly more performant version (as described in Section 2 of the paper)
    RealNum alphaT = alpha * sqrt(1 - pow(beta2, t)) / ( 1 - pow(beta1, t));
    weights[i] = weights[i] - alphaT * (m[i] / (sqrt(v[i]) + epsilon) + lambda * weights[i]);
  }
}
