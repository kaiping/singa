// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:41

/**
 * This file is the main entrance of the program. It register user defined
 * classes, e.g., data source, edge and layer classes.
 */

#include <gflag/gflags.h>
#include <glog/logging.h>
#include "utils/global_context.h"
#include "model/model.h"

DEFINE_string(system_conf, "system.conf", "configuration file for node roles");
DEFINE_string(model_conf, "model.conf", "DL model configuration file");

int main(int argc, char **argv) {
  google::InitGoogleLoggin(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // TODO(by wangwei) register partition/fileparser/layer classes
  lapis::GlobalContext global_context(system_conf_path, model_conf_path);

	// InitServers(argc, argv);
  // TODO(Anh) DistributedDisk
  // lapis::DistributedDisk distributed_disk(global_context);
  ModelController model_controller;
  model_controller.Init(&global_context);
  // There are two type of processes, one for coordinator, one for worker
  if (model_controller.IsCoordinatorProcess()) {
    lapis::Coodinator coordinator(&global_cotext, &model_controller);
    coordinator.run();
  } else {
    lapis::Worker worker(&global_context, &model_controller);
    worker.run();
  }
  return 0;
}
