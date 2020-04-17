#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "type.h"

void im2col_cpu(const F* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    F* data_col);

void col2im_cpu(const F* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    F* data_im);

void pool_forwards_cpu(const F* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    F* data_pooled);

void pool_backwards_cpu(const F* data_im, const F* data_pooled,
    const int channels, const int height, const int width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w,
    F* data_backgrad );
