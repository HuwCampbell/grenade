#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "type.h"

void descend_sgd_cpu(int len, RealNum rate, RealNum momentum, RealNum regulariser,
    const RealNum* weights,
    const RealNum* gradient,
    const RealNum* last,
    RealNum* outputWeights, RealNum* outputMomentum);


void descend_adam_cpu(int len, int t, RealNum alpha, RealNum beta1, RealNum beta2, RealNum epsilon, RealNum lambda,
  const RealNum* weights,
  const RealNum* gradient,
  const RealNum* m,
  const RealNum* v,
  RealNum* outputWeights, RealNum* outputM, RealNum* outputV);

