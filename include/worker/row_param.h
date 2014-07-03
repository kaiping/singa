// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-03 19:20

#ifndef INCLUDE_WORKER_ROW_PARAM_H_
#define INCLUDE_WORKER_ROW_PARAM_H_

#include <string>
#include "worker/param.h"

// child paramter class, split based on the last axis, i.e., row for matrix;
// if the parameter is a vector, then itself is the only splitter
namespace lapis {
class RowParameter : public Parameter {
 public:
  explicit RowParameter(const ParamProto& param_proto): Parameter(param_proto) {
    row_ = 0;
  }

  virtual bool next_split(std::string* k, std::string* v);

 private:
  int row_;
};
}  // namespace lapis

#endif  // INCLUDE_WORKER_ROW_PARAM_H_
