#include <gflags/gflags.h>
#include <glog/logging.h>
#include "utils/cluster.h"
#include "utils/common.h"
#include "proto/model.pb.h"
#include "proto/cluster.pb.h"
#include "server.h"
#include "worker.h"

/**
 * \file main.cc is the main entry of SINGA.
 */

DEFINE_string(cluster_conf, "examples/imagenet12/cluster.conf",
    "configuration file for the cluster");
DEFINE_string(model_conf, "examples/imagenet12/model.conf",
    "Deep learning model configuration file");

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

  // Init Cluster
  singa::ClusterProto cluster_proto;
  singa::ReadProtoFromTextFile(FLAGS_cluster_conf.c_str(), &cluster_proto);
  auto cluster =singa::Cluster::Get(cluster_proto);
  singa::ModelProto model;
  singa::ReadProtoFromTextFile(FLAGS_model_conf.c_str(), &model);
  LOG(INFO)<<"The cluster config is\n"<<cluster_proto.DebugString()
    <<"\nThe model config is\n"<<model.DebugString();

  if(cluster->AmITableServer()) {
    singa::TableServer server;
    server.Start(model.solver().sgd());
    //    singa::Debug();
  }else {
    singa::Worker worker;
    worker.Start(model);
  }
  cluster->Finalize();
  MPI_Finalize();
  //LOG(ERROR)<<"SINA has shutdown successfully";
  return 0;
}
