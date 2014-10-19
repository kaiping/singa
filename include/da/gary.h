 Copyright © 2014 Wei Wang. All Rights Reserved.
 2014-10-17 16:36
#ifndef INCLUDE_DA_LARY_H_
#define INCLUDE_DA_LARY_H_
#include "ary.h"

namespace lapis {

class GAry:public Ary{
 public:
  ~GAry();
  GAry():Ary(){}
  void Destroy();
  /**
    * init based on the shape, alloc memory
    */
  float* Setup(const Shape& shape, int pdim);
  const Range IndexRange(int k);
  int local_size(){return shape_.size/groupsize;}
  int offset() {return offset_;}
 private:
  Shape shape_;
  int lo_[2], hi_[2];
  float** dptrs_;
  int id_, groupsize_;
  int offset_;
};
}   namespace lapis
#endif   INCLUDE_DA_LARY_H_
