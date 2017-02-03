#include "pad.h"

void pad_cpu(double* data, const int channels,
    const int height, const int width, const int pad_left, const int pad_top,
    const int pad_right, const int pad_bottom,
    double* data_padded) {

  const int pad_width  = width + pad_left + pad_right;
  const int pad_height = height + pad_top + pad_bottom;

  memset(data_padded, 0, pad_height * pad_width * channels * sizeof(double));

  for (int channel = 0; channel < channels; channel++) {
    double* px = data_padded + (pad_width * pad_top + pad_left) + channel * (pad_width * pad_height);
    for (int y = 0; y < height; y++) {
      memcpy(px, data, sizeof(double) * width);
      px += pad_width;
      data += width;
    }
  }
}

void crop_cpu(double* data, const int channels,
    const int height, const int width, const int crop_left, const int crop_top,
    const int crop_right, const int crop_bottom,
    double* data_cropped) {

  const int crop_width  = width + crop_left + crop_right;
  const int crop_height = height + crop_top + crop_bottom;

  for (int channel = 0; channel < channels; channel++) {
    double* px = data + (crop_width * crop_top + crop_left) + channel * (crop_width * crop_height);
    for (int y = 0; y < height; y++) {
      memcpy(data_cropped, px, sizeof(double) * width);
      px += crop_width;
      data_cropped += width;
    }
  }
}
