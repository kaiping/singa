// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:41

#include <gflags/gflags.h>
#include <glog/logging.h>
#include "utils/global_context.h"
#include "utils/proto_helper.h"
#include "proto/model.pb.h"
#include "proto/cluster.pb.h"
#include "server.h"
#include "worker.h"
#include "da/gary.h"

/**
 * \file main.cc is the main entry of SINGA.
 */

DEFINE_string(cluster_conf, "examples/imagenet12/cluster.conf", "configuration file for node roles");
DEFINE_string(model_conf, "examples/imagenet12/model.conf", "DL model configuration file");
DEFINE_bool(restore, false, "restore from checkpoint file");
// for debug use
#ifndef FLAGS_v
  DEFINE_int32(v, 3, "vlog controller");
#endif


int main(int argc, char **argv) {
  int provided;
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
  if(!FLAGS_restore){
    if(gc->AmITableServer()) {
      lapis::TableServer server;
      server.Start(model.solver().sgd());
    }else {
    	// TODO: comment out the below to execute training at the workers.
    	// for now, this is not necessary to test table servers
    	// (use test_tuple.cc instead).

    	/*
      lapis::GAry::Init(gc->rank(), gc->groups());
      // worker or table server
      lapis::Worker worker;
      worker.Start(model);
      lapis::GAry::Finalize();
      */
    }
  }else{
    // restore
  }
  gc->Finalize();
  MPI_Finalize();
  LOG(ERROR)<<"shutdown";
  return 0;
}
