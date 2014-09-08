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
#include "utils/network_thread.h"
#include "proto/model.pb.h"
#include "coordinator.h"
#include "worker.h"

DEFINE_string(system_conf, "examples/imagenet12/system.conf", "configuration file for node roles");
DEFINE_string(model_conf, "examples/imagenet12/model.conf", "DL model configuration file");
DEFINE_bool(load_data, false, "Load data to distributed tables");
DEFINE_bool(run, true,  "run training algorithm");
// for debugging use
#ifndef FLAGS_v
  DEFINE_int32(v, 3, "vlog controller");
#endif
int main(int argc, char **argv) {
  FLAGS_logtostderr=1;
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  VLOG(3)<<"load data "<<FLAGS_load_data<<" run "<<FLAGS_run;

  // Note you can register you own layer/edge/datasource here
  //
  // Init GlobalContext
  VLOG(3)<<"before global context";
  auto gc=lapis::GlobalContext::Get(FLAGS_system_conf, FLAGS_model_conf);
  VLOG(3)<<"after global context";
  if(gc->AmICoordinator()) {
    lapis::Coordinator coordinator;
    coordinator.Run(FLAGS_load_data, FLAGS_run);
  }else {
    lapis::Worker worker;
    worker.Run(FLAGS_load_data, FLAGS_run);
  }
  return 0;
}
