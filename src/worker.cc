// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 16:01
#include <glog/logging.h>

#include "worker.h"
#include "model_controller/model.h"
#include "proto/model.pb.h"
#include "proto/common.pb.h"
#include "utils/proto_helper.h"

#include "net/net.h"
#include "utils/network_thread.h"
#include "utils/global_context.h"
#include "net/sgd_solver.h"

namespace lapis {
Worker::Worker(){
  LOG(INFO) << "starting Worker...";
  mpi_=NetworkThread::Get();
  table_server_=nullptr;
  id_=mpi_->id();
}

Worker::~Worker() {
  Shutdown();
  if(table_server_!=nullptr)
    delete table_server_;
}

bool Worker::ShouldIDoValidation(int step) {
  ShortMsg msg;
  msg.set_step(step);
  mpi_->Send(GlobalContext::kCoordinatorRank, MTYPE_VALIDATION, msg);
  mpi_->Read(GlobalContext::kCoordinatorRank, MTYPE_INSTRUCTION, &msg);
  return msg.answer();
}

std::map<int, GlobalTable*> Worker::InitDistributedStorage(){
  ModelProto model;
  ReadProtoFromTextFile(GlobalContext::Get()->model_conf(), &model);
  delegate_=CreateTableDelegate(model.solver().method());
  std::map<int, GlobalTable*> tables= delegate_.CreateTables();
  if(GlobalContext::Get()->AmITableServer()){
    table_server_=new TableServer();
    table_server_->StartTableServer(tables);
    VLOG(3)<<"table server tarted";
  }
  VLOG(3)<<"finish init storage";
  return tables;
}
void Worker::Shutdown() {
	VLOG(3) << "Worker is shutting down ...";
  mpi_->Flush();
  mpi_->Send(GlobalContext::kCoordinatorRank, MTYPE_WORKER_END, EmptyMessage());
  EmptyMessage msg;
  int src = 0;
  mpi_->Read(GlobalContext::kCoordinatorRank, MTYPE_WORKER_SHUTDOWN, &msg, &src);
  VLOG(3) << "Worker received MTYPE_WORKER_SHUTDOWN";
  table_server_->ShutdownTableServer();
  mpi_->Shutdown();
}

void Worker::PrefetchData(Phase phase, Net *net) {
  Record record;
  int k;
  // local batchsize
  int num=net->input_layers(0)->data().local_shape(0);
  if(!table->has_loaded())
    table->Load();
  for(int n=0;n<num;++n){
    if(table->done())
      table->Load();
    table->get(&k, &record);
    for(auto* layer:net->input_layers())
      layer->AddInputRecord(record);
    table->Next();
  }
}

Performance Woker::Validate(solver* solver, Net* net){
  PrefetchData(kVal, net);
  Performance perf;
  for(int b=0;b<solver->num_valbatches();b++){
    // TODO join prefetch thread
    for(auto* layer:net->input_layers())
      layer->SetInputData(nullptr);
    // TODO start prefetch thread
    PrefetchData(kVal, net);
    perf.Aggregate(solver->ValidateOneBatch(net));
  }
  return perf;
}

void ReportPerformance(Performance perf) {
  if(AmIGroupLeader()){
    StateQueue q(member_list_);
    while(q.HasValid()){
      Performance p;
      if(mpi_->TryRead(q.Next(), MTYPE_PERFORMANCE, &p)){
        perf.Aggregate(p);
        q.Invalide();
      }
    }
    mpi_->Send(kCoordinatorRank, MTYPE_PERFORMANCE, perf);
  }else{
    mpi_->Send(leader_, MTYPE_PERFORMANCE, perf);
  }
}
void Worker::InitGroupInfo() {
  GroupConfig conf;
  mpi_->Read(GlobalContext::Get()->kCoordinatorRank, MTYPE_GROUP_CONFIG,&conf);
  bool find=false;
  for(auto& group: conf.group()){
    leader_=group.leader();
    bool find=leader_==rank_;
    if(!find)
      for(auto member: group.member())
        find|=member==rank_;
    if(find){
      for(auto member: group.member())
        member_list_.push_back(member);
      break;
    }
  }
  CHECK(find);
}
void Worker::Run(bool load_data, bool do_train) {
  InitDistributedStorage();
  if(!do_train)
      return;
  InitGroupInfo();
  ModelProto model;
  mpi_->Read(GlobalContext::kCoordinatorRank, MTYPE_MODEL_PARTITION, &model);
  Net net(model.net());
  net->Setup()
  Solver solver;
  solver.Init(model.solver());
  const SGDProto sgd=model.solver().sgd();
  PrefetchData( kTrain, net);
  Performance train_perf;
  while (!solver.HasFinished()) {
    LOG(INFO)<<solver.step();
    mc_.Get(net->params());
    if(solver.ValidateNow()){
      if(ShouldIDoValidation(solver.step())){
        Performance perf=Validate(&solver,net,tables[kVal]);
        Report(perf);
      }
    }
    for(auto* layer:net->input_layers())
      layer->SetInputData(nullptr);
    // TODO start prefetch thread
    PrefetchData( kTrain, net);
    train_perf.Aggregate(solver.TrainOneBatch(&net, batch));
    if(solver.DisplayerNow()){
      Report(train_perf.Avg());
      train_perf.Reset();
    }
    mc_.Update(net->params());
  }
}
}  // namespace lapis
