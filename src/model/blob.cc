// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-16 20:58

#include "model/blob.h"
namespace lapis {

void Blob::Reshape(int length) {
  Reshape(0, length);
}

void Blob::Reshape(int height, int width) {
  Reshape(0, height, width);
}

void Blob::Reshape(int num, int height, int width) {
  Reshape(num, 0, height, width);
}

void Blob::Reshape(int num, int channels, int height, int width) {
  num_ = num;
  channels_ = channels;
  height_ = height;
  width_ = width;
  size_ = num * channels * height * width;
  content_ = new float[size_];
}

}  // namespace lapis
