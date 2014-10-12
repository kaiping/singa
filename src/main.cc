// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:41

/**
 * This file is the main entrance of the program.
 * User can register their own defined  classes, e.g., layers
 */

#include <gflags/gflags.h>
#include <glog/logging.h>
#include "utils/global_context.h"
#include "utils/network_thread.h"
#include "utils/proto_helper.h"
#include "proto/model.pb.h"
#include "proto/system.pb.h"
#include "core/table_delegate.h"
//#include "coordinator.h"
#include "datasource/data_loader.h"
//#include "worker.h"

DEFINE_string(system_conf, "examples/imagenet12/system.conf", "configuration file for node roles");
DEFINE_string(model_conf, "examples/imagenet12/model.conf", "DL model configuration file");
DEFINE_bool(load, false, "Load data to distributed tables");
DEFINE_bool(run, true,  "Run training algorithm");
DEFINE_bool(time, true,  "time training algorithm");
// for debugging use
#ifndef FLAGS_v
  DEFINE_int32(v, 3, "vlog controller");
#endif
/**
 * need to know the tuple type to create parameter table
 */
lapis::TableDelegate* CreateTableDelegate(const lapis::SolverProto& solver){
  lapis::TableDelegate* delegate;
  if(solver.method()==lapis::SolverProto::kSGD){
     delegate=new lapis::TypedTableDelegate<int, lapis::SGDValue>();
  }
  else{
    delegate= new lapis::TypedTableDelegate<int, lapis::AdaGradValue>();
  }
  delegate->CreateTables(solver);
  return delegate;
}

int main(int argc, char **argv) {
  FLAGS_logtostderr=1;
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  VLOG(3)<<"load data "<<FLAGS_load<<" run "<<FLAGS_run;

  // Note you can register you own layer/edge/datasource here
  //
  // Init GlobalContext
  VLOG(3)<<"before global context";
  auto gc=lapis::GlobalContext::Get(FLAGS_system_conf, FLAGS_model_conf);
  lapis::ModelProto model;
  lapis::ReadProtoFromTextFile(gc->model_conf(), &model);
  //lapis::TableDelegate* delegate=CreateTableDelegate(model.solver());
  VLOG(3)<<"after global context";

  lapis::SystemProto system;
  lapis::ReadProtoFromTextFile(gc->system_conf(), &system);
  if(FLAGS_load) {
    LOG(INFO)<<"Loading Data...";
    lapis::DataLoader loader(gc->rank(), system.cluster());
    if(gc->AmICoordinator())
      loader.ShardData(model.data());
    else
      loader.CreateLocalShards(model.data());
    LOG(INFO)<<"Finish Load Data";
  }
  /*
  if(FLAGS_run){
    if(gc->AmICoordinator()) {
      lapis::Coordinator coordinator(delegate);
      coordinator.Run(FLAGS_load, FLAGS_run, model);
    }else {
      lapis::Worker worker(delegate);
      worker.Run(FLAGS_run, FLAGS_time, model.solver());
    }
  }
  delete delegate;
  */
  return 0;
}
