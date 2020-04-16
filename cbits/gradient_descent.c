#include "gradient_descent.h"

void descend_sgd_cpu(int len, double rate, double momentum, double regulariser,
    const double* weights,
    const double* gradient,
    const double* last,
    double* outputWeights, double* outputMomentum) {

  for (int i = 0; i < len; i++) {
      outputMomentum[i] = momentum * last[i] - rate * gradient[i];
      outputWeights[i] = weights[i] + outputMomentum[i] - (rate * regulariser) * weights[i];
  }
}


void descend_adam_cpu(int len, int t, double alpha, double beta1, double beta2, double epsilon,
  const double* weights,
  const double* gradient,
  const double* m,
  const double* v,
  double* outputWeights, double* outputM, double* outputV) {
  t = t + 1;
  for (int i = 0; i < len; i++) {
    outputM[i] = beta1 * m[i] + (1 - beta1) * gradient[i];
    outputV[i] = beta2 * v[i] + (1 - beta2) * gradient[i] * gradient[i];

    // Clear version (as in Algoritm 1 of the paper)
    /* double mHat = outputM[i] / (1 - pow(beta1, t)); */
    /* double vHat = outputV[i] / (1 - pow(beta2, t)); */
    /* outputWeights[i] = weights[i] - alpha * mHat / (sqrt(vHat) + epsilon); */

    // Slightly more performant version (as described in Section 2 of the paper)
    double alphaT = alpha * sqrt(1 - pow(beta2, t)) / ( 1 - pow(beta1, t));
    outputWeights[i] = weights[i] - alphaT * outputM[i] / (sqrt(outputV[i]) + epsilon);
  }

}
