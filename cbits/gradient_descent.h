#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "type.h"

void descend_sgd_cpu(int len, F rate, F momentum, F regulariser,
    const F* weights,
    const F* gradient,
    const F* last,
    F* outputWeights, F* outputMomentum);


void descend_adam_cpu(int len, int t, F alpha, F beta1, F beta2, F epsilon,
  const F* weights,
  const F* gradient,
  const F* m,
  const F* v,
  F* outputWeights, F* outputM, F* outputV);

