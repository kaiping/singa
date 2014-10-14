// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 16:01
#include <glog/logging.h>

#include "worker.h"
#include "proto/model.pb.h"
#include "proto/common.pb.h"
#include "utils/common.h"
#include "utils/network_thread.h"
#include "utils/global_context.h"
#include "net/solver.h"
#include "net/net.h"

namespace lapis {
Worker::Worker(const shared_proto<GlobalContext>& gc){
  LOG(INFO) << "starting Worker...";
  mpi_=NetworkThread::Get();
  context_=gc;
}

Worker::~Worker() {
  Shutdown();
  if(table_server_!=nullptr)
    delete table_server_;
}

void Worker::Shutdown() {
	VLOG(3) << "Worker is shutting down ...";
  mpi_->Flush();
  mpi_->Send(GlobalContext::kCoordinator, MTYPE_WORKER_END, EmptyMessage());
  EmptyMessage msg;
  int src = 0;
  mpi_->Read(GlobalContext::kCoordinator, MTYPE_WORKER_SHUTDOWN, &msg, &src);
  VLOG(3) << "Worker received MTYPE_WORKER_SHUTDOWN";
  table_server_->ShutdownTableServer();
  context_->Finish();
}


void Worker::ReportPerformance(Performance perf) {
  if(mpi_->AmIGroupLeader()){
    StateQueue<int> q(context_->MembersOfGroup(context_->group_id()));
    while(q.HasValid()){
      Performance p;
      if(mpi_->TryRead(q.Next(), MTYPE_PERFORMANCE, &p)){
        perf.Aggregate(p);
        q.Invalide();
      }
    }
    //context_->Send(GlobalContext::kCoordinator, MTYPE_PERFORMANCE, perf);
    LOG(INFO)<<perf.ToString();
  }else{
    mpi_->Send(context_->leader(), MTYPE_PERFORMANCE, perf);
  }
}

void Worker::Resume() {
  // TODO get solverproto from hdfs or local disk
  /* std::shared_ptr<Solver>   */
}
void Worker::Start(const DataProto& dp, const SolverProto& sp){
  NetProto np;
  context_->Read(GlobalContext::kCoordinator, MTYPE_NET_PARTITION, &np);
  ModelProto model;
  model.mutable_data()->CopyFrom(dp);
  model.mutable_solver()->CopyFrom(sp);
  model.mutable_net()->CopyFrom(np);
  Run(model);
}
void Worker::Run(const ModelProto& model){
  TableDelegate* delegate=CreateTableDelegate(model.sovler);
  if(context_->AmITableServer()){
    table_server_=new TableServer();
    table_server_->StartTableServer(delegate->tables());
    VLOG(3)<<"table server tarted";
  }

  Solver solver(model.solver());
  solver.Setup(delegate_, model.data(), model.net());

  pthread_t solver_thread;
  pthread_create(&solver_thread, NULL, solver.Train, NULL);
  while(true) {
    //    if(NetworkThread::Get()->TryRead())
    //    send and recv msgs

    if(solver.ValidateNow()){
      Performance perf=solver.Validate();
      ReportPerformance(perf);
      solver.ContinueTrain();
    }
    if(solver.DisplayNow()){
      ReportPerformance(train_perf.Avg());
      train_perf.Reset();
      solver.ContinueTrain();
    }
  }
  pthread_join(solver_thread, NULL);
  delete delegate;
}
}  // namespace lapis
