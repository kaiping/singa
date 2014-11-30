// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 16:01
#include <glog/logging.h>
#include <memory>
#include "worker.h"
#include "proto/model.pb.h"
#include "utils/global_context.h"
#include "net/solver.h"
#include "net/net.h"

namespace lapis {
void Worker::Shutdown() {
}
void Worker::Resume() {
  /* todo get solverproto from hdfs or local disk

  ModelProto model;
  model.mutable_data()->CopyFrom(dp);
  model.mutable_solver()->CopyFrom(sp);
  NetProto np;
  mpi_->Read(GlobalContext::kCoordinator, MTYPE_NET_PARTITION, &np);
  model.mutable_net()->CopyFrom(np);
  Run(model);

  Solver solver(model.solver());
  LOG(ERROR)<<"setup solver";
  solver.Setup(model.data(), model.net());
   */
}
void Worker::Start(const Model& model){
    Solver solver(model.solver());
    solver.Setup(model.net());
    if(context_->group_id()==0)
      solver.InitParams();
    // todo, two ways to syn all workers
    // 1. create MPI communicator for all workers, and call MPI_Barrier for
    // this communicator
    // 2. handle_get returns false if the key of get() is not found in the
    // table, i.e., parameters have not been inserted
    LOG(ERROR)<<"Worker starting...";
    //solver.Train();
    solver.TimeOneBatch();
  LOG(ERROR)<<"Worker shutting down";
}
}  // namespace lapis
