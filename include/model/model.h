// Copyright © 2014 Jinyang Gao. All Rights Reserved.
// 2014-07-18 20:23

#ifndef INCLUDE_WORKER_MODEL_CONTROLLER_H_
#define INCLUDE_WORKER_MODEL_CONTROLLER_H_
#include <vector>
#include "worker/layer.h"
#include "core/common.h"
#include "core/table-registry.h"
#include "core/global-table.h"
#include "core/table.h"
#include "core/distributed-memory.h"
#include "core/memory-server.h"

namespace lapis {
class ModelController {
 public:
  void GetNextInput(Layer *layer);// to be done
  void Update(const std::vector<Param*> *params);
  void GetParam(std::vector<Param*> *params);
  void Put(const std::vector<Param*> *params);
  //set split type to 0 and split size to 2
  void Init(int machine_num, int split_tpye=0,int split_size=2);
 private:
  int my_split_tpye_,my_machine_num_,my_split_size_;
  TypedGlobalTable<int, float_vector_message>* distributed_store_;
  //ModelConfProto model_conf_proto_;
};
}  // namespace lapis
#endif  // INCLUDE_WORKER_MODEL_CONTROLLER_H_
