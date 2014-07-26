// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-04 11:49

#include "worker/worker.h"
#include "proto/modle.pb.h"

using std::vector;

namespace lapis {
Worker::Worker(const GlobalContext *gc, ModelController *mc)
    :global_context_(gc), model_controller_(mc) {
}

void Worker::run() {
  ModelProto model_proto;
  Net net;
  net.Init(model_proto.net());
  Trainer trainer;
  trainer.Init(model_proto.trainer(), model_controller_);
  trainer.Run(&net)
  model_controller_->Finish();
}
}  // namespace lapis
