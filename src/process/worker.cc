// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-04 11:49
#include <glog/logging.h>
#include "worker/worker.h"
#include "proto/model.pb.h"
#include "model/net.h"
#include "model/sgd_trainer.h"
#include "utils/proto_helper.h"

namespace lapis {
Worker::Worker(ModelProto* proto, ModelController *mc) {
  LOG(INFO) << "Working constructor...";
  model_controller_=mc;
  model_proto_=proto;
}

void Worker::Run() {
  Net net;
  net.Init(model_proto_->net());
  SGDTrainer trainer;
  trainer.Init(model_proto_->trainer(), model_controller_);
  if(model_controller_->issinglemachine())
    trainer.Run(kAllocData|kAllocParam|kInitParam, &net);
  else
    trainer.Run(kAllocData|kAllocParam, &net);
  model_controller_->Finish();
}
}  // namespace lapis
