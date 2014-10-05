// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 16:01
#include <glog/logging.h>

#include "worker.h"
#include "proto/model.pb.h"
#include "proto/common.pb.h"
#include "utils/proto_helper.h"

#include "net/net.h"
#include "net/layer.h"

#include "utils/network_thread.h"
#include "utils/global_context.h"
#include "net/solver.h"

namespace lapis {
Worker::Worker(TableDelegate* delegate){
  LOG(INFO) << "starting Worker...";
  mpi_=NetworkThread::Get();
  id_=mpi_->id();
  delegate_=delegate;
}

Worker::~Worker() {
  Shutdown();
  if(table_server_!=nullptr)
    delete table_server_;
}

bool Worker::ShouldIDoValidation(int step) {
  ShortMsg msg;
  msg.set_step(step);
  mpi_->Send(GlobalContext::kCoordinator, MTYPE_VALIDATION, msg);
  mpi_->Read(GlobalContext::kCoordinator, MTYPE_INSTRUCTION, &msg);
  return msg.answer();
}

void Worker::InitDistributedStorage(){
  if(GlobalContext::Get()->AmITableServer()){
    table_server_=new TableServer();
    table_server_->StartTableServer(delegate_->tables());
    VLOG(3)<<"table server tarted";
  }
  VLOG(3)<<"finish init storage";
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
  mpi_->Shutdown();
}

void Worker::PrefetchData(int phase, Net *net) {
  Record record;
  int k;
  // local batchsize
  int num=net->input_layer(0)->data().local_shape(0);
  for(int n=0;n<num;++n){
    delegate_->Next(phase, &k, &record);
    for(auto* layer:net->input_layer())
      layer->AddInputRecord(record);
  }
}

Performance Worker::Validate(Solver* solver, Net* net){
  Solver::phase=kVal;
  PrefetchData(kVal, net);
  Performance perf;
  for(int b=0;b<solver->validation_steps();b++){
    // TODO join prefetch thread
    for(auto* layer:net->input_layer())
      layer->SetInputData(nullptr);
    // TODO start prefetch thread
    PrefetchData(kVal, net);
    perf.Aggregate(solver->ValidateOneBatch(net));
  }
  return perf;
}

void Worker::ReportPerformance(Performance perf) {
  if(AmIGroupLeader()){
    StateQueue<int> q(member_list_);
    while(q.HasValid()){
      Performance p;
      if(mpi_->TryRead(q.Next(), MTYPE_PERFORMANCE, &p)){
        perf.Aggregate(p);
        q.Invalide();
      }
    }
    mpi_->Send(GlobalContext::kCoordinator, MTYPE_PERFORMANCE, perf);
  }else{
    mpi_->Send(leader_, MTYPE_PERFORMANCE, perf);
  }
}
void Worker::InitGroupInfo() {
  GroupConfig conf;
  mpi_->Read(GlobalContext::kCoordinator, MTYPE_GROUP_CONFIG,&conf);
  bool find=false;
  for(auto& group: conf.group()){
    leader_=group.leader();
    bool find=leader_==id_;
    if(!find)
      for(auto member: group.member())
        find|=member==id_;
    if(find){
      for(auto member: group.member())
        member_list_.push_back(member);
      break;
    }
  }
  CHECK(find);
}
void Worker::Run(bool do_train, const SolverProto& solver_proto) {
  InitDistributedStorage();
  if(!do_train)
      return;
  InitGroupInfo();
  NetProto net_proto;
  mpi_->Read(GlobalContext::kCoordinator, MTYPE_NET_PARTITION, &net_proto);
  Net net(net_proto);
  net.Setup();
  Solver solver(solver_proto);
  PrefetchData(kTrain, &net);
  Performance train_perf;
  while (!solver.HasFinished()) {
    LOG(INFO)<<solver.step();
    delegate_->Get(net.params());
    /*
    if(solver.ValidateNow()){
      if(ShouldIDoValidation(solver.step())){
        Performance perf=Validate(&solver,net,tables[kVal]);
        ReportPerformance(perf);
      }
    }
    */
    for(auto* layer:net.input_layer())
      layer->SetInputData(nullptr);
    // TODO start prefetch thread
    PrefetchData( kTrain, &net);
    train_perf.Aggregate(solver.TrainOneBatch(&net));
    if(solver.DisplayNow()){
      ReportPerformance(train_perf.Avg());
      train_perf.Reset();
    }
    delegate_->Update(net.params());
  }
}
}  // namespace lapis
