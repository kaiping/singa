// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-11 13:53

#ifndef INCLUDE_WORKER_SGD_H_
#define INCLUDE_WORKER_SGD_H_
#include <vector>
#include "worker/param.h"
#include "proto/lapis.pb.h"

namespace lapis {
class SGD {
 public:
  explicit SGD(const ModelConfProto& model_conf);
  ~SGD();

  void ComputeParamUpdate(std::vector<Param*> * params);
 private:
  float learning_rate_, mometum_, weight_decay_;
};

}  // namespace lapis
#endif  // INCLUDE_WORKER_SGD_H_
