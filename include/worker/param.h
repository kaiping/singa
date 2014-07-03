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
class Parameter {
 public:
  explicit Parameter(ParamProto& param_proto): shape_(param_proto.shape()),
      initializer_(param_proto.initializer()), name_(param_proto.name()){}
  void init();
  virtual bool next_split(std::string* k, std::string* v);

 protected:
  vector<float> content_;
  std::vector<int> shape_;
  std::string initializer_;
  std::string name_; // name of the parameter, e.g., 'weight', 'bias'
};
}  // namespace lapis


#endif  // INCLUDE_WORKER_PARAM_H_
