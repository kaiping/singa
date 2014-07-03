// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-03 19:24

#include "worker/row_param.h"

namespace lapis {
// current implementation only considers matrix and vector params
// both params' shapes have length of 2.
// for vector, it is <1, n>; for matrix, it is <m, n>
// TODO(wangwei) support other paramters, e.g., scalar and tensor
bool Parameter::next_split(string* k, string* v) {
  // vector param
  if (row_ >= shape_[0])
    return false;
  if (shape_.size() == 1) {
    *k = name_;
    *v = content_;
  } else {  // matrix param
    *k = name_ + string(row);
    *v = new vector<float>(content_.begin() + row_ * shape_[1],
                           content_.begin() + (row_ + 1) * shape[1]);
    row + =1;
  }

  return true;
}
}  // namespace lapis


