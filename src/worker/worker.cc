// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-04 11:49
#include <glog/logging.h>
#include "worker/worker.h"
#include "proto/model.pb.h"
#include "model/net.h"
#include "model/sgd_trainer.h"
#include "utils/proto_helper.h"

namespace lapis {
Worker::Worker(ModelController *mc): model_controller_(mc) {
  LOG(INFO) << "Working constructor...";
}

void Worker::Run() {
  ModelProto model_proto;
  ReadProtoFromTextFile(GlobalContext::Get()->model_conf_path(), &model_proto);
  Net net;
  net.Init(model_proto.net());
  SGDTrainer trainer;
  trainer.Init(model_proto.trainer(), model_controller_);
  trainer.Run(&net);
  model_controller_->Finish();
}
}  // namespace lapis
