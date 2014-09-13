// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-22 19:53
#include <glog/logging.h>

#include "net/lapis.h"
namespace lapis {
int Blob::count_=0;
Blob::Blob(int num, int channels, int height, int width, const bool alloc) {
  num_ = num;
  channels_ = channels;
  height_ = height;
  width_ = width;
  record_length_ = channels_ * height_ * width_;
  length_=num_*record_length_;
  if(alloc)
    dptr=new float[length_];
}
Blob::Blob(const Shape& shape) {
  Blob(shape.num(),shape.channels(), shape.height(),shape.width(), true);
}

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

void Blob::Resize(int num, int channels, int height,
                  int width, const bool alloc) {
  if (num_ != num || channels_ != channels || height_ != height
      || width_ != width) {
    num_ = num;
    channels_ = channels;
    height_ = height;
    width_ = width;

    record_length_ = channels_ * height_ * width_;
    if(length_!=num_*record_length_||alloc) {
      count_-=length_;
      length_ = num_ * record_length_;
      count_+=length_;
      if (dptr != nullptr) {
        LOG(INFO)<<"DELETE BLOB DPTR!!";
        delete dptr;
      }
      dptr = new float[length_];
			VLOG(3) << "Allcoate blob length: " << length_ << " record len "
					<< record_length_ << " to pointer " << dptr;
    }
  }
}
int Blob::Gt(float v, int n) const {
  int ret=0;
  int s=0, e=length_;
  if(n>=0){
    s=n*record_length_;
    e=(n+1)*record_length_;
  }
  for(unsigned int i=s;i<e;i++)
    if(dptr[i]>v)
      ret++;
  return ret;
}
int Blob::Lt(float v,int n) const {
  int ret=0;
  int s=0, e=length_;
  if(n>=0){
    s=n*record_length_;
    e=(n+1)*record_length_;
  }
  for(unsigned int i=s;i<e;i++)
    if(dptr[i]<v)
      ret++;
  return ret;
}
float Blob::Norm() const {
  float norm=0.0;
  for(int i=0;i<length_;i++)
    norm+=dptr[i]*dptr[i];
  return std::sqrt(norm/length_);
}

bool Blob::Nan() const {
  for(unsigned int i=0;i<length_;i++)
    if(isnan(dptr[i]))
      return true;
  return false;
}
std::shared_ptr<Lapis> Lapis::instance_;
}  // namespace lapis

