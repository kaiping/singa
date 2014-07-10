// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-03 17:35

#ifndef INCLUDE_WORKER_PARAM_H_
#define INCLUDE_WORKER_PARAM_H_
#include <vector>
#include <string>
#include "proto/lapis.pb.h"

// base paramter class, split based on the last axis, i.e., row for matrix;
// if the parameter is a vector, then itself is the only splitter
namespace lapis {
class Param{
 public:
  virtual void init(const ParamProto& param_proto);
  virtual bool next(std::string* k, std::string* v) = 0;
  virtual void fetch() = 0;

 protected:
  float* content_, *grad_;
  std::vector<int> shape_;
  std::string initializer_;
  std::string name_;  // name of the parameter, e.g., 'weight', 'bias'
};
}  // namespace lapis


#endif  // INCLUDE_WORKER_PARAM_H_
