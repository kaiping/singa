// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 16:01
#include <glog/logging.h>

#include "worker.h"
#include "utils/proto_helper.h"
#include "disk/data_source.h"
#include "model_controller/model.h"
#include "proto/model.pb.h"
#include "model/data_layer.h"

#include "model/net.h"


namespace lapis {
Worker::Worker(){
  LOG(INFO) << "starting coordinator...";
}

// no model splitting currently
// init parameters and put them into distributed memory
void Worker::SetupNet(const ModelProto &model_proto, Net *net) {
  // set user configured fields
  // setup training data which is necessary to setup the DataLayer that is in
  // turn required by upper edges and layers to setup.
  std::vector<DataSource *> train_data;
  TrainerProto trainer = model_proto.trainer();
  Trainer::InitDataSource(trainer.train_data(), &train_data);
  SGDProto sgd = trainer.sgd();
  // allocate memory for parameters and init them
  for (auto layer : net->layers())
    if (layer->HasInput())
      (dynamic_cast<DataLayer*>(layer))->SetupDataSource(sgd.train_batchsize(),
                                                       train_data);
  for (auto* layer: net->layers()){
    layer->Setup(kAllocData|kAllocParam|kInitParam);
    for (auto *edge : layer->out_edges())
      edge->Setup(kAllocData|kAllocParam|kInitParam);
  }
}

void Worker::Run() {
  auto gc=GlobalContext::Get();
  ModelProto model_proto;
  ReadProtoFromTextFile(gc->model_conf(), &model_proto);

  ModelController mc;
  mc.Init();

  Net net;
  net->Init(model_proto.net());

  SGDTrainer trainer;
  trainer.Init(model_proto.trainer(), &mc);
  // The first worker is responsible to init the parameters and put them into
  // distributed table
  if(gc->IsFirstWorker()) {
    SetupNet(model_proto, &net);
    // put parameters into distributed memory
    mc.Put(net->params());
    // already allocated memory for data and paraters. no need to do
    // initialization for parameters, because will Get from distributed table
    // Actually no need to Get for the first iteration. But such a check may
    // bring aditional denpendency, e.g., call gc->IsFirstWorker(). For
    // standalone mode, the Get do nothing, because the parameters are already
    // initialized.
    trainer.Run(0, &net);
  }
  else {
    // Other workers should allocate memory for data and parameters. No need to
    // Init parameters, because they Get parameters from distributed table
    trainer.Run(kAllocData|kAllocParam, &net);
  }
  // TODO(wangwei, anh) Finish by worker instead of model controller
  mc.Finish();
}
}  // namespace lapis
