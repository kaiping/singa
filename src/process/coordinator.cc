// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 16:01
#include <glog/logging.h>

#include "coordinator/coordinator.h"
#include "utils/proto_helper.h"
#include "disk/data_source.h"
#include "model_controller/model.h"
#include "proto/model.pb.h"
#include "model/data_layer.h"

#include "model/net.h"


namespace lapis {
Coordinator::Coordinator(ModelController *mc){
  LOG(INFO) << "starting coordinator...";
  model_controller_=mc;
}

// no model splitting currently
// init parameters and put them into distributed memory
int Coordinator::InitModel(const ModelProto &model_proto) {
  Net net;
  // set user configured fields
  net.Init(model_proto.net());
  // setup training data which is necessary to setup the DataLayer that is in
  // turn required by upper edges and layers to setup.
  std::vector<DataSource *> train_data;
  TrainerProto trainer = model_proto.trainer();
  Trainer::InitDataSource(trainer.train_data(), &train_data);
  SGDProto sgd = trainer.sgd();
  // allocate memory for parameters and init them
  for (auto layer : net.layers())
    if (layer->HasInput())
      (dynamic_cast<DataLayer*>(layer))->SetupDataSource(sgd.train_batchsize(),
                                                       train_data);
  for (auto* layer: net.layers()){
    layer->Setup(kAllocParam|kInitParam);
    for (auto *edge : layer->out_edges())
      edge->Setup(kAllocParam|kInitParam);
  }
  // put parameters into distributed memory
  model_controller_->Put(net.params());
  return 0;
}

// The coordinator run in a single process, and call Finish() to wait workers.
// It shutdown until all works have finished.
void Coordinator::Run() {
  //LoadData();
  auto gc=GlobalContext::Get();
  ModelProto model_proto;
  ReadProtoFromTextFile(gc->model_conf(), &model_proto);
  if(!gc->standalone())
    InitModel(model_proto);
  model_controller_->CommenceBroadcast(model_proto);
  model_controller_->Finish();
}
}  // namespace lapis
