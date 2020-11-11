#include <math.h>
#include <stdio.h>
#include <omp.h>
#include "type.h"


void map_vector(RealNum (*fun)(RealNum), RealNum* vec, int len) {

  #pragma omp parallel for
  for (int i = 0; i < len; i++) {
    vec[i] = fun(vec[i]);
  }
}

void zip_with_vector_repl_snd(RealNum (*fun)(RealNum, RealNum), const RealNum* vec1, RealNum* vec2, int len) {

  #pragma omp parallel for
  for (int i = 0; i < len; i++) {
    vec2[i] = fun(vec1[i], vec2[i]);
  }
}

/* Functions */

RealNum c_times(RealNum x, RealNum y) {
  return x * y;
}

/* Activiations */


// Relu

RealNum c_relu(RealNum a) {
  return a <= 0 ? 0 : a;
}

RealNum c_relu_dif(RealNum a) {
  return (a <= 0 ? 0 : 1);
}

RealNum c_relu_dif_fast(RealNum a, RealNum g) { /* takes the value and the derivative */
  return (a <= 0 ? 0 : 1) * g;
}

// Dropout


// Leaky Relu
RealNum alpha = 0.02;
RealNum c_leaky_relu(RealNum a) {
  return (a <= 0 ? alpha * a : a);
}

RealNum c_leaky_relu_dif(RealNum a) {
  return (a <= 0 ? alpha : 1);
}

RealNum c_leaky_relu_dif_fast(RealNum a, RealNum g) { /* takes the value and the derivative */
  return (a <= 0 ? alpha : 1) * g;
}


// Sigmoid
RealNum c_sigmoid(RealNum x) {
  return (1 / (1 + exp (-x)));
}

RealNum c_sigmoid_dif(RealNum sig) { /* takes the sigmoid value */
  return (sig * (1 - sig));
}

RealNum c_sigmoid_dif_fast(RealNum sig, RealNum g) { /* takes the sigmoid value and the derivative */
  return (sig * (1 - sig)) * g;
}
