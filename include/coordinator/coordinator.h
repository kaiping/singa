// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 15:26

#ifndef INCLUDE_COORDINATOR_COORDINATOR_H_
#define INCLUDE_COORDINATOR_COORDINATOR_H_
#include "utils/global_context.h"
#include "memory/distributed_memory.h"

namespace lapis {
class Coordinator
{
 public:
  Coordinator(const GlobalContext& global_context,
              const DistributedMemory& distibuted_memory);
  ~Coordinator();

  // load all data into distributed disk
  int LoadData();

  // init and partition parameters of the model,
  // then put it into the distributed memory.
  // Currently, only do initailization. TODO(wangwei), model partition
  int PartitionInitModel();

  void Run();
 private:
  GlobalContext global_context_;
  DistributedMemory distibuted_memory_;
  ModelConfProto model_conf_proto;
};


}  // namespace lapis

#endif  //INCLUDE_COORDINATOR_COORDINATOR_H_
