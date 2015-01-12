#include <gflags/gflags.h>
#include <glog/logging.h>
#include "utils/global_context.h"
#include "utils/common.h"
#include "proto/model.pb.h"
#include "proto/cluster.pb.h"
#include "server.h"
#include "worker.h"
#include "da/gary.h"

/**
 * \file main.cc is the main entry of SINGA.
 */

DEFINE_string(cluster_conf, "examples/imagenet12/cluster.conf",
    "configuration file for node roles");
DEFINE_string(model_conf, "examples/imagenet12/model.conf",
    "DL model configuration file");

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
  singa::Cluster cluster;
  singa::ReadProtoFromTextFile(FLAGS_cluster_conf.c_str(), &cluster);
  auto gc=singa::GlobalContext::Get(cluster);
  singa::Model model;
  singa::ReadProtoFromTextFile(FLAGS_model_conf.c_str(), &model);
  if(gc->AmITableServer()) {
    auto factory=Singleton<TableServerHandler>::Instance();
    RegisterCreateFunction("SGD",
        CreateInstance(TSHandlerForSGD, TableServerHandler));
    RegisterCreateFunction("AdaGrad",
        CreateInstance(TSHandlerForAda, TableServerHandler));

    singa::TableServer server;
    server.Start(model.solver().sgd());
  }else {
    // TODO: comment out the below to execute training at the workers.

    /*
        singa::GAry::Init(gc->rank(), gc->groups());
    // worker or table server
    singa::Worker worker;
    worker.Start(model);
    singa::GAry::Finalize();
    */
  }
  gc->Finalize();
  MPI_Finalize();
  LOG(ERROR)<<"shutdown";
  return 0;
}
