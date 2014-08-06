// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:41

/**
 * This file is the main entrance of the program.
 * It registers user defined classes, e.g., data source, edge and layer
 * classes and start either the worker or coordinator.
 */

#include <gflags/gflags.h>
#include <glog/logging.h>
#include "utils/global_context.h"
#include "worker/worker.h"
#include "coordinator/coordinator.h"
#include "model_controller/model.h"
#include "proto/model.pb.h"



DEFINE_string(system_conf, "system.conf", "configuration file for node roles");
DEFINE_string(model_conf, "model.conf", "DL model configuration file");
DEFINE_int32(v, 3, "vlog controller");

int main(int argc, char **argv) {
  FLAGS_logtostderr=1;
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Note you can register you own layer/edge/datasource here
  //
  // TODO(all) Init memory servers here?

  // Init GlobalContext
  VLOG(3)<<"before global context";
  lapis::GlobalContext::Get()->Init(FLAGS_system_conf, FLAGS_model_conf);
  VLOG(3)<<"after global context";
  lapis::ModelController mc;
  lapis::ModelProto *model_proto= mc.Init();
  VLOG(3)<<"after model controller";
  // There are two type of working units: coordinator, worker
  if (mc.iscoordinator()) {
    lapis::Coordinator coordinator(&mc);
    coordinator.Run();
  } else {
    lapis::Worker worker(model_proto, &mc);
    worker.Run();
  }
  return 0;
}
