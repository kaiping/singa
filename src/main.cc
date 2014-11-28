// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:41

#include <gflags/gflags.h>
#include <glog/logging.h>
#include "utils/global_context.h"
#include "utils/proto_helper.h"
#include "proto/model.pb.h"
#include "proto/cluster.pb.h"
#include "coordinator.h"
#include "worker.h"
#include "da/gary.h"

/**
 * \file main.cc is the main entry of SINGA.
 */

DEFINE_string(cluster_conf, "examples/imagenet12/cluster.conf", "configuration file for node roles");
DEFINE_string(model_conf, "examples/imagenet12/model.conf", "DL model configuration file");
DEFINE_bool(restore, false, "restore from checkpoint file");
DEFINE_string(mode, "hybrid",  "partition mode");
// for debugging use
#ifndef FLAGS_v
  DEFINE_int32(v, 3, "vlog controller");
#endif


int main(int argc, char **argv) {
  int provided;
  // TODO input args check and display usage.
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  //FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Init GlobalContext
  lapis::Cluster cluster;
  lapis::ReadProtoFromTextFile(FLAGS_cluster_conf.c_str(), &cluster);
  auto gc=lapis::GlobalContext::Get(cluster);
  lapis::Model model;
  lapis::ReadProtoFromTextFile(FLAGS_model_conf.c_str(), &model);
  lapis::GAry::Init(gc->rank(), gc->groups());
  if(FLAGS_restore){
    /*
     if(gc->AmICoordinator()) {
      lapis::Coordinator coordinator(gc);
      coordinator.Resume(model);
    }else {
      lapis::Worker worker(gc);
      worker.Start(model.data(), model.solver());
    }
    */
  }else{
    if(gc->AmICoordinator()) {
      lapis::Coordinator coordinator;
      coordinator.Run(model);
    }else {
      // worker or table server
      lapis::Worker worker(gc);
      worker.Start(model);
    }
  }
  lapis::GAry::Finalize();
  gc->Finalize();
  MPI_Finalize();
  LOG(ERROR)<<"shutdown";
  return 0;
}
