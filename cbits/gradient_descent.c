#include "gradient_descent.h"

void descend_cpu(int len, double rate, double momentum, double regulariser,
    const double* weights,
    const double* gradient,
    const double* last,
    double* outputWeights, double* outputMomentum) {

  for (int i = 0; i < len; i++) {
      outputMomentum[i] = momentum * last[i] - rate * gradient[i];
      outputWeights[i] = weights[i] + outputMomentum[i] - (rate * regulariser) * weights[i];
  }
}
