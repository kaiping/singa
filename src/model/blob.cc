// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-16 20:58

#include "model/blob.h"
namespace lapis {

void Blob::Reshape(int length) {
  Reshape(1, length);
}

void Blob::Reshape(int height, int width) {
  Reshape(1, height, width);
}

void Blob::Reshape(int num, int height, int width) {
  Reshape(num, 1, height, width);
}

void Blob::Reshape(int num, int channels, int height, int width) {
  if (num_ != num || channels_ != channels || height_ != height
      || width_ != width) {
    num_ = num;
    channels_ = channels;
    height_ = height;
    width_ = width;
    int length = Length();
    if (content_ != nullptr)
      delete content_;
    content_ = new float[length];
  }
}

}  // namespace lapis
