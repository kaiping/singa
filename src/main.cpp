// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:41
//
// This file is the main entrance of the program. It register user defined
// classes, e.g., data file parser, data partitioner, and layer classes.
// Then it calls the startDaemon function to start coordinator/worker/server.

#include <gflag/gflags.h>
#include <glog/logging.h>
#include "utils/start_deamon.h"

DEFINE_string(system_conf, "system.conf", "configuration file for node roles");
DEFINE_string(model_conf, "model.conf", "DL model configuration file");

int main(int argc, char** argv) {
  google::InitGoogleLoggin(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // TODO(by wangwei) register partition/fileparser/layer classes

  startDaemon(FLAGS_system_conf.c_str(), FLAGS_model_conf.c_str());

  return 0;
}
