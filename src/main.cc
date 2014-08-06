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
#include "proto/model.pb.h"

DEFINE_string(system_conf, "system.conf", "configuration file for node roles");
DEFINE_string(model_conf, "model.conf", "DL model configuration file");
// for debugging use
DEFINE_int32(v, 3, "vlog controller");
DEFINE_int32(logtostderr, 1 "log to stderr");

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Note you can register you own layer/edge/datasource here
  //
  // TODO(all) Init memory servers here?

  // Init GlobalContext
  VLOG(3)<<"before global context";
  auto gc=lapis::GlobalContext::Get(FLAGS_system_conf, FLAGS_model_conf);
  VLOG(3)<<"after global context";
  // TODO(wangwei, anh) InitServers
  lapis::Worker worker();
  worker.Run();
  return 0;
}
