#include "gradient_descent.h"

void descend_sgd_cpu(int len, F rate, F momentum, F regulariser,
    const F* weights,
    const F* gradient,
    const F* last,
    F* outputWeights, F* outputMomentum) {

  for (int i = 0; i < len; i++) {
      outputMomentum[i] = momentum * last[i] - rate * gradient[i];
      outputWeights[i] = weights[i] + outputMomentum[i] - (rate * regulariser) * weights[i];
  }
}


void descend_adam_cpu(int len, int t, F alpha, F beta1, F beta2, F epsilon,
  const F* weights,
  const F* gradient,
  const F* m,
  const F* v,
  F* outputWeights, F* outputM, F* outputV) {
  t = t + 1;
  for (int i = 0; i < len; i++) {
    outputM[i] = beta1 * m[i] + (1 - beta1) * gradient[i];
    outputV[i] = beta2 * v[i] + (1 - beta2) * gradient[i] * gradient[i];

    // Clear version (as in Algoritm 1 of the paper)
    /* F mHat = outputM[i] / (1 - pow(beta1, t)); */
    /* F vHat = outputV[i] / (1 - pow(beta2, t)); */
    /* outputWeights[i] = weights[i] - alpha * mHat / (sqrt(vHat) + epsilon); */

    // Slightly more performant version (as described in Section 2 of the paper)
    F alphaT = alpha * sqrt(1 - pow(beta2, t)) / ( 1 - pow(beta1, t));
    outputWeights[i] = weights[i] - alphaT * outputM[i] / (sqrt(outputV[i]) + epsilon);
  }

}
