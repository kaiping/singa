// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-03 19:20

#ifndef INCLUDE_WORKER_ROW_PARAM_H_
#define INCLUDE_WORKER_ROW_PARAM_H_

#include <string>
#include "worker/param.h"

// child paramter class, split based on the last axis, i.e., row for matrix;
// if the parameter is a vector, then itself is the only splitter
namespace lapis {
class RowParam: public Param{
 public:
  virtual void init(const ParamProto& param_proto);
  virtual bool next(std::string* k, std::string* v);
  virtual void fetch();

 private:
  int row_;
};
}  // namespace lapis

#endif  // INCLUDE_WORKER_ROW_PARAM_H_
