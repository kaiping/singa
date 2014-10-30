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
  mpi_=NetworkThread::Get();
  context_=gc;
  table_server_=nullptr;
}

Worker::~Worker() {
  //Shutdown();
  if(table_server_!=nullptr)
    delete table_server_;
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
void Worker::Start(const DataProto& dp, const SolverProto& sp){
  TableDelegate* delegate=CreateTableDelegate(sp);
  if(context_->AmITableServer()){
    table_server_=new TableServer();
    table_server_->StartTableServer(delegate->tables());
    LOG(ERROR)<<"table server tarted";
  }
  int src, cdntor=GlobalContext::kCoordinator;
  EmptyMessage dummy_msg;

  if(context_->AmIWorker()){
    NetProto np;
    mpi_->Read(cdntor, MTYPE_NET_PARTITION, &np);
    Solver solver(sp);
    solver.Setup(delegate, dp, np);

    while(true){
      if(mpi_->TryRead(cdntor, MTYPE_INIT_PARAMS, &dummy_msg, &src)){
        solver.InitParams();
        mpi_->Send(cdntor, MTYPE_FINISH_INIT_PARAMS, dummy_msg);
      }
      if(mpi_->TryRead(cdntor, MTYPE_WORKER_START, &dummy_msg, &src)){
        LOG(ERROR)<<"Worker starting...";
        // read start step from coordinator's msg
        solver.Train();
        //solver.TimeOneBatch();
        break;
      }
      sleep(0.1);
    }
    LOG(ERROR)<<"Worker Finish Training";
    mpi_->Flush();
    mpi_->Send(cdntor, MTYPE_WORKER_END, dummy_msg);
    mpi_->Read(cdntor, MTYPE_SHUTDOWN, &dummy_msg, &src);
  }else{
    while(true){
      if(mpi_->TryRead(cdntor, MTYPE_SHUTDOWN, &dummy_msg, &src))
        break;
      else
        sleep(5);
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
