#include <gflags/gflags.h>
#include <glog/logging.h>
#include "utils/cluster.h"
#include "utils/common.h"
#include "proto/model.pb.h"
#include "proto/cluster.pb.h"
#include "server/server.h"
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

/**
 * Registry Layer sub-classes and Param sub-classes.
 * User implemented Layer or Param sub-classes should be registryed here.
 */
void RegistryClasses(const singa::NetProto& proto){
  singa::NeuralNet::RegistryLayers();
  singa::NeuralNet::RegistryParam(proto.param_type());
}

// for debug use
#ifndef FLAGS_v
  DEFINE_int32(v, 3, "vlog controller");
#endif

int main(int argc, char **argv) {
  //FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Init Cluster
  singa::ClusterProto pcluster;
  singa::ReadProtoFromTextFile(FLAGS_cluster_conf.c_str(), &pcluster);
  auto cluster=singa::Cluster::Get(pcluster, FLAGS_hostfile, FLAGS_procsID);
  singa::ModelProto model;
  singa::ReadProtoFromTextFile(FLAGS_model_conf.c_str(), &model);
  LOG(INFO)<<"The cluster config is\n"<<pcluster.DebugString()
    <<"\nThe model config is\n"<<model.DebugString();

  RegistryClasses(model.neuralnet());
  if(cluster->AmIServer()) {
    singa::Server server(cluster);
    server.Run();
  }else {
    singa::Worker worker(cluster);
    worker.Start(model);
  }
  LOG(ERROR)<<cluster->hostname()<<" has shut down";
  return 0;
}
