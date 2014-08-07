// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 16:01
#include <glog/logging.h>

#include "worker.h"
#include "model_controller/model.h"
#include "proto/model.pb.h"
#include "proto/common.pb.h"

#include "net/net.h"
#include "utils/network_thread.h"
#include "utils/global_context.h"
#include "net/sgd_trainer.h"
#include "core/table_server.h"


namespace lapis {
Worker::Worker(){
  LOG(INFO) << "starting Worker...";
}
void Worker::Run() {
  TableServer *ts=nullptr;
  if(GlobalContext::Get()->AmITableServer()) {
    TableServer *ts=new TableServer();
    ts->StartTableServer();
  }
  ModelController mc;
  mc.Init();
  ModelProto proto;
  NetworkThread::Get()->Read(GlobalContext::kCoordinatorRank,
                             MTYPE_MODEL_CONFIG,
                             &proto);
  Net net;
  net.Init(proto.net());
  SGDTrainer trainer;
  trainer.Init(proto.trainer(), &mc);
  // workers should allocate memory for data and parameters. No need to
  // Init parameters, because they Get parameters from distributed table
  trainer.Run(kAllocData|kAllocParam, &net);

  if(ts!=nullptr){
    ts->ShutdownTableServer();
    delete ts;
  }
}
}  // namespace lapis
