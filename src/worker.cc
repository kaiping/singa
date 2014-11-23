// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 16:01
#include <glog/logging.h>
#include <memory>
#include "worker.h"
#include "proto/model.pb.h"
#include "proto/common.pb.h"
#include "utils/common.h"
#include "utils/network_thread.h"
#include "utils/global_context.h"
#include "net/solver.h"
#include "net/net.h"

namespace lapis {
Worker::Worker(const std::shared_ptr<GlobalContext>& gc){
  LOG(INFO) << "starting Worker...";
  context_=gc;
}

Worker::~Worker() {
}

void Worker::Shutdown() {
}



void Worker::Resume() {
  // TODO get solverproto from hdfs or local disk
  /* std::shared_ptr<Solver>
  TableDelegate* delegate=CreateTableDelegate(sp);
  if(context_->AmITableServer()){
    table_server_=new TableServer();
    table_server_->StartTableServer(delegate->tables());
    LOG(ERROR)<<"table server tarted";
  }

  ModelProto model;
  model.mutable_data()->CopyFrom(dp);
  model.mutable_solver()->CopyFrom(sp);
  NetProto np;
  mpi_->Read(GlobalContext::kCoordinator, MTYPE_NET_PARTITION, &np);
  model.mutable_net()->CopyFrom(np);
  Run(model);


  Solver solver(model.solver());
  LOG(ERROR)<<"setup solver";
  solver.Setup(delegate, model.data(), model.net());
   */

}
void Worker::Start(const Model& model){
  TableDelegate* delegate=CreateTableDelegate(model.solver());
  if(context_->AmITableServer()){
    table_server_=new TableServer();
    table_server_->StartTableServer(delegate->tables());
    LOG(ERROR)<<"table server tarted";
  }

  if(context_->AmIWorker()){
    Solver solver(model.solver());
    solver.Setup(delegate, model.net());
    if(context_->group_id()==0)
      solver.InitParams();
    // TODO, two ways to syn all workers
    // 1. create MPI communicator for all workers, and call MPI_Barrier for
    // this communicator
    // 2. handle_get returns false if the key of get() is not found in the
    // table, i.e., parameters have not been inserted
    LOG(ERROR)<<"Worker starting...";
    //solver.Train();
    solver.TimeOneBatch();
  }else{
    int src, cdntor=GlobalContext::kCoordinator;
    EmptyMessage dummy_msg;
    while(true){
      if(NetworkThread::Get()->TryRead(cdntor, MTYPE_SHUTDOWN, &dummy_msg, &src))
        break;
      else
        sleep(0.01);
    }
    LOG(ERROR)<<"Table Server shutting down";
  }
  if(table_server_!=nullptr){
    table_server_->ShutdownTableServer();
  }
  delete delegate;

  LOG(ERROR)<<"Worker shutting down";
}

/*
void* SolverThread(void *context) {
  Solver* solver=static_cast<Solver*>(context);
  solver->Train();
}

void Worker::Run(Solver* solver){
  pthread_t solver_thread;
  pthread_create(&solver_thread, NULL, &SolverThread, &solver);
  auto val_perf=solver.val_perf();
  auto train_perf=solver.train_perf();
  while(true) {
    //    if(NetworkThread::Get()->TryRead())
    //    send and recv msgs

    if(solver.ValidateNow()){
      solver.Validate();
      ReportPerformance(val_perf.Avg());
      solver.Continue();
    }
    if(solver.DisplayNow()){
      ReportPerformance(train_perf.Avg());
      train_perf.Reset();
      solver.Continue();
    }
  }
  pthread_join(solver_thread, NULL);
}
*/
}  // namespace lapis
