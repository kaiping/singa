// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-04 19:58

#ifndef INCLUDE_WORKER_MODEL_CONTROLLER_H_
#define INCLUDE_WORKER_MODEL_CONTROLLER_H_
#include <vector>
#include "model/param.h"
namespace lapis {
class ModelController {
 public:
  void GetParams(const std::vector<Param*>& params);
  void UpdateParams(const std::vector<Param*>& params);
 private:
  ModelConfProto model_conf_proto_;
};
}  // namespace lapis
#endif  // INCLUDE_WORKER_MODEL_CONTROLLER_H_
