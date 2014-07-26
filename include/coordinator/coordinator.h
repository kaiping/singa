// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 15:26

#ifndef INCLUDE_COORDINATOR_COORDINATOR_H_
#define INCLUDE_COORDINATOR_COORDINATOR_H_
#include "utils/global_context.h"
#include "model_controller/model.h"

#include "proto/model.pb.h"

namespace lapis {
/**
 * The coordinator class.
 * Its taks is to initialize the distributed memory/table, by puting the
 * initialized parameters of the Net into the distributed memory. Then it calls
 * works to start work. Finally, it waits and exits until all works finish.
 * It runs in a single process.
 */
class Coordinator {
 public:
  Coordinator(ModelController *mc);
  ~Coordinator();

  // TODO(wangwei) load all data into distributed disk
  int LoadData();

  // init and partition parameters of the model,
  // then put it into the distributed memory.
  // Currently, only do initailization. TODO(wangwei), model partition
  int InitModel(const ModelProto &model_proto);

  void Run();
 private:
  ModelController *model_controller_;
  ModelProto model_proto_;
};

}  // namespace lapis

#endif  // INCLUDE_COORDINATOR_COORDINATOR_H_
