#include <stdio.h>
#include <stdint.h>

void descend_cpu(int len, double rate, double momentum, double regulariser,
    const double* weights,
    const double* gradient,
    const double* last,
    double* outputWeights, double* outputMomentum);

