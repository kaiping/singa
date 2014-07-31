// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-22 19:53
#include "model/lapis.h"
namespace lapis {
void Blob::Resize(int length) {
  Resize(length, 1, 1, 1);
}

void Blob::Resize(int width, int height) {
  Resize(width, height, 1, 1);
}

void Blob::Resize(int width, int height, int channels) {
  Resize(width, height, channels, 1);
}

void Blob::Resize(int width, int height, int channels, int num) {
  if (num_ != num || channels_ != channels || height_ != height
      || width_ != width) {
    num_ = num;
    channels_ = channels;
    height_ = height;
    width_ = width;
    record_length_ = channels_ * height_ * width_;
    length_ = num_ * record_length_;
    if (dptr != nullptr)
      delete dptr;
    dptr = new float[length_];
  }
}

std::shared_ptr<Lapis> Lapis::instance_;
}  // namespace lapis

