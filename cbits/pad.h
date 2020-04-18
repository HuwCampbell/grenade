#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "type.h"

void pad_cpu(RealNum* data_im, const int channels,
    const int height, const int width, const int pad_left, const int pad_top,
    const int pad_right, const int pad_bottom,
    RealNum* data_col);

void crop_cpu(RealNum* data_im, const int channels,
    const int height, const int width, const int crop_left, const int crop_top,
    const int crop_right, const int crop_bottom,
    RealNum* data_col);
