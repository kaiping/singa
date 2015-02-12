#include <gflags/gflags.h>
#include <glog/logging.h>
#include "utils/cluster.h"
#include "utils/common.h"
#include "proto/model.pb.h"
#include "proto/cluster.pb.h"
//#include "server.h"
#include "worker/worker.h"

/**
 * \file main.cc is the main entry of SINGA.
 */
DEFINE_int32(procsID, 0, "global process ID");
DEFINE_string(hostfile, "examples/imagenet12/hostfile", "hostfile");
DEFINE_string(cluster_conf, "examples/imagenet12/cluster.conf",
    "configuration file for the cluster");
DEFINE_string(model_conf, "examples/imagenet12/model.conf",
    "Deep learning model configuration file");

// for debug use
#ifndef FLAGS_v
  DEFINE_int32(v, 3, "vlog controller");
#endif

int main(int argc, char **argv) {
  //FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Init Cluster
  singa::ClusterProto cluster;
  singa::ReadProtoFromTextFile(FLAGS_cluster_conf.c_str(), &cluster);
  singa::Cluster::Get(cluster, FLAGS_hostfile, FLAGS_procsID);
  singa::ModelProto model;
  singa::ReadProtoFromTextFile(FLAGS_model_conf.c_str(), &model);
  LOG(INFO)<<"The cluster config is\n"<<cluster.DebugString()
    <<"\nThe model config is\n"<<model.DebugString();

  if(singa::Cluster::Get()->AmIServer()) {
    //singa::Server server;
    //server.Start();
    //    singa::Debug();
  }else {
    singa::Worker worker(Cluster::Get());
    worker.Start(model);
  }
  //LOG(ERROR)<<"SINA has shutdown successfully";
  return 0;
}
