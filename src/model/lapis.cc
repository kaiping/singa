// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-22 19:53
#include <glog/logging.h>

#include "model/lapis.h"
namespace lapis {
int Blob::count_=0;
void Blob::Resize(int num, int channels, int height, int width) {
  if (num_ != num || channels_ != channels || height_ != height
      || width_ != width) {
    num_ = num;
    channels_ = channels;
    height_ = height;
    width_ = width;

    record_length_ = channels_ * height_ * width_;
    if(length_!=num_*record_length_) {
      count_-=length_;
      length_ = num_ * record_length_;
      count_+=length_;
      if (dptr != nullptr) {
        LOG(INFO)<<"DELETE BLOB DPTR!!";
        delete dptr;
      }
      dptr = new float[length_];
    }
  }
}

std::shared_ptr<Lapis> Lapis::instance_;
}  // namespace lapis

