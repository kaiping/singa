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
  // TODO(all) in this implementation, the distributed_disk has to join tables
  // on worker nodes. <filename, rgb>---<filename, label>
  for (DataMetaProto& data_source : model_conf_proto.data()) {
    // TODO(wangwei) create the factory in main.cc
    DataReaderInterface reader = data_reader_factory.get[data_source.type()];
    reader.init(data_source);
    string k, v;
    while (reader.next(k, v) > 0)
      distributed_disk_.put(k, v);
  }

  return 0;
}

// no model splitting currently
// init parameters and put them into distributed memory
// send whole copy of modelConfigProto to each worker
int InitSplitModel() {
  for (LayerProto& layer_proto : model_conf_proto_.layers()) {
    for (ParameterProto& param_proto : layer_proto.parameters()) {
      Parameter param = param_factory_generate(param_proto.splitter_name);
      string k, v;
      while (param.next_split(&k, &v))
        distributed_memory_.put(k, v);
    }
  }

  // TODO (Anh), mpi wrapper for sending message to all workers
  send2allWorkers(model_conf_proto_.serializeToString());

  return 0;
}

// we do not create a thread for the Coordinator, because workers have to wait
// the coordinator to finish the initialization work, i.e. what Run() does
void Coordinator::Run() {
  LoadData();
  InitSplitModel();
}
}  // namespace lapis
