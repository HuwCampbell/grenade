#include <stdio.h>
#include <stdint.h>
#include <math.h>

void descend_sgd_cpu(int len, double rate, double momentum, double regulariser,
    const double* weights,
    const double* gradient,
    const double* last,
    double* outputWeights, double* outputMomentum);


void descend_adam_cpu(int len, int t, double alpha, double beta1, double beta2, double epsilon,
  const double* weights,
  const double* gradient,
  const double* m,
  const double* v,
  double* outputWeights, double* outputM, double* outputV);

