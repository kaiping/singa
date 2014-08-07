// Copyright © 2014 Jinyang Gao. All Rights Reserved.
// 2014-07-18 20:23

#ifndef INCLUDE_MODEL_CONTROLLER_MODEL_H_
#define INCLUDE_MODEL_CONTROLLER_MODEL_H_

#include <vector>
#include <google/protobuf/message.h>

#include "net/param.h"
#include "utils/global_context.h"
#include "core/common.h"
#include "core/table-registry.h"
#include "core/global-table.h"
#include "core/table.h"
#include "core/table_server.h"
#include "proto/model.pb.h"


using google::protobuf::Message;
namespace lapis {

class ModelController {
 public:
  //void GetNextInput(Layer *layer);// to be done
  void Update(const std::vector<Param *> &params);
  void Get(const std::vector<Param *> &params);
  void Put(const std::vector<Param *> &params);
  //set split type to 0 and split size to 2
  void Init();
 private:
  int my_split_tpye_,my_machine_num_,my_split_size_,my_rank_;
  TypedGlobalTable<int, float_vector_message>* distributed_store_;
  bool issinglemachine_,iscoordinator_,isdmm_;
  //ModelConfProto model_conf_proto_;
};
}  // namespace lapis
#endif  // INCLUDE_MODEL_CONTROLLER_MODEL_H_
