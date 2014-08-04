// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:41

/**
 * This file is the main entrance of the program. It register user defined
 * classes, e.g., data source, edge and layer classes.
 */

#include <gflags/gflags.h>
#include <glog/logging.h>
#include "utils/global_context.h"
#include "model_controller/model.h"
#include "worker/worker.h"
#include "coordinator/coordinator.h"


DEFINE_string(system_conf, "system.conf", "configuration file for node roles");
DEFINE_string(model_conf, "model.conf", "DL model configuration file");

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // TODO(by wangwei) register partition/fileparser/layer classes
  lapis::GlobalContext::Get()->Init(FLAGS_system_conf, FLAGS_model_conf);
  // TODO(Anh) DistributedDisk
  // lapis::DistributedDisk distributed_disk(global_context);
  lapis::ModelController model_controller;
  model_controller.Init();
  // There are two type of processes, one for coordinator, one for worker
  if (model_controller.IsCoordinatorProcess()) {
    lapis::Coordinator coordinator(&model_controller);
    coordinator.Run();
  } else {
    lapis::Worker worker(&model_controller);
    worker.Run();
  }
  return 0;
}
