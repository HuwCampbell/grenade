#include "pad.h"

void pad_cpu(const double* data, const int channels,
    const int height, const int width, const int pad_left, const int pad_top,
    const int pad_right, const int pad_bottom,
    double* data_padded) {

  const int pad_width  = width + pad_left + pad_right;
  const int pad_height = height + pad_top + pad_bottom;
  const int channel_size = height * width;

  memset(data_padded, 0, pad_height * pad_width * channels * sizeof(double));

  for (int channel = 0; channel < channels; channel++) {
    double* px = data_padded + (pad_width * pad_top + pad_left) + channel * (pad_width * pad_height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        *(px++) = data[y * width + x + channel_size * channel];
      }
      px += pad_left + pad_right;
    }
  }
}

void crop_cpu(double* data, const int channels,
    const int height, const int width, const int crop_left, const int crop_top,
    const int crop_right, const int crop_bottom,
    double* data_cropped) {

  const int crop_width  = width + crop_left + crop_right;
  const int crop_height = height + crop_top + crop_bottom;

  const int channel_size = height * width;

  for (int channel = 0; channel < channels; channel++) {
    double* px = data + (crop_width * crop_top + crop_left) + channel * (crop_width * crop_height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        data_cropped[y * width + x + channel_size * channel] = *(px++);
      }
      px += crop_left + crop_right;
    }
  }
}
