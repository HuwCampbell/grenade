#include "im2col.h"

inline int is_a_ge_zero_and_a_lt_b(int a, int b) {
  return a >= 0 && a < b;
}

void im2col_cpu(const double* data_im, int dataOffset, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    double* data_col) {

  data_im += dataOffset;
  const int output_h = (height - kernel_h) / stride_h + 1;
  const int output_w = (width - kernel_w) / stride_w + 1;

  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int fitting_height = 0; fitting_height <= (height - kernel_h); fitting_height += stride_h) {
      for (int fitting_width = 0; fitting_width <= (width - kernel_w); fitting_width += stride_w) {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
          for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
            int input_row = fitting_height + kernel_row;
            int input_col = fitting_width + kernel_col;
            *(data_col++) = data_im[input_row * width + input_col];
          }
        }
      }
    }
  }
}
