// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-16 20:58

#include "model/blob.h"
namespace lapis {

Blob::Blob(int num, int channels, int height, int width) {
  Reshape(num, channels, height, width);
}
void Blob::Reshape(int length) {
  Reshape(1, length);
}

void Blob::Reshape(int size, int width) {
  Reshape(size, 1, width);
}

void Blob::Reshape(int size, int height, int width) {
  Reshape(size, 1, height, width);
}

void Blob::Reshape(int size, int channels, int height, int width) {
  if (num_ != size || channels_ != channels || height_ != height
      || width_ != width) {
    num_ = size;
    channels_ = channels;
    height_ = height;
    width_ = width;
    record_length_=channels_ * height_ * width_;
    length_ = num_ * record_length_;
    if (data_ != nullptr)
      delete data_;
    data_ = new float[length_];
  }
}

void Blob::set_data(const float *other) {
  memcpy(data_, other, sizeof(float)*length_);
}

}  // namespace lapis
