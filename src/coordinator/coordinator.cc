// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 16:01

#include "coordinator/coordinator.h"
#incldue "utils/proto_helper.h"
#include "proto/lapis.pb.h"

namespace lapis {
Coordinator::Coordinator(const GlobalContext& global_context,
                         const DistributedMemory& distributed_memory)
    :global_context_(global_context), distibuted_memory_(distibuted_memory) {
  LOG(INFO) << "starting coordinator...\n";
  ReadProtoFromTextFile(global_context_.model_conf_path, &model_conf_proto_);
}

int Coordinator::LoadData() {
  for (DataProto& data_source : model_conf_proto.data())
    distributed_disk_.LoadData(data_source);

  return 0;
}

int PartitionInitModel() {
  for (LayerProto& layer_proto : model_conf_proto_.layers()){
    for (ParameterProto& param_proto : layer_proto.parameters()) {
      Parameter param(param_proto);
      for (auto& record: param.partition())
        distributed_memory_.put(record.first, record.second);
    }
  }
  return 0;
}

// we do not create a thread for the Coordinator, because workers have to wait
// the coordinator to finish the initialization work, i.e. what Run() does
void Coordinator::Run() {
  LoadData();
  PartitionInitModel();
}
}  // namespace lapis
