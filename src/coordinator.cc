// Copyright © 2014 Wei Wang, Anh. All Rights Reserved.
// 2014-08-07 11:32
#include "coordinator.h"
#include "proto/model.pb.h"
#include "net/net.h"
#include "net/sgd_trainer.h"
#include "net/trainer.h"
#include "model_controller/model.h"
#include "core/table-registry.h"
#include "utils/network_thread.h"
#include "utils/proto_helper.h"

namespace lapis {
Coordinator::Coordinator() {
  context_=GlobalContext::Get();
  net_=NetworkThread::Get();
}

Coordinator::~Coordinator() {
  for (auto* state: server_states_) {
    for (auto* taskid : state->local_shards)
      delete taskid;
    delete state;
  }
}

void Coordinator::InitTableServers() {
  for (int i = 0; i < context_->num_processes() - 1; ++i) {
    VLOG(3)<<"in start mem manger "<<i;
    RegisterWorkerRequest req;
    int src = 0;
    VLOG(3)<<"before read msg ";
    net_->Read(MPI::ANY_SOURCE, MTYPE_REGISTER_WORKER, &req, &src);
    VLOG(3)<<"after read msg ";
    //  adding memory server.
    if (context_->IsTableServer(i)) {
      server_states_.push_back(new ServerState(i));
    }
  }
  LOG(INFO) << " All servers registered and started up";
  //  set itself as the current worker for the table
  for (auto &entry: TableRegistry::Get()->tables())
    entry.second->worker_id_ = net_->id();

  // memory servers are specified in global context. Round-robin assignment
  int server_idx = 0;
  for (auto &entry: TableRegistry::Get()->tables()){
    VLOG(3)<<"num of shards "<<entry.second->num_shards();
    int table=entry.first;
    for (int shard = 0; shard < entry.second->num_shards(); ++shard) {
      ServerState &server = *server_states_[server_idx];
      LOG(INFO) << "Assigning table ("<<table<<","<<shard<<") to server "
                <<server_states_[server_idx]->server_id;
      server.shard_id = shard;
      server.local_shards.insert(new TaskId(table, shard));
      server_idx = (server_idx + 1) % server_states_.size();
    }
  }
  //  then send table assignment
  ShardAssignmentRequest req;
  for (size_t i = 0; i < server_states_.size(); ++i) {
    ServerState &server = *server_states_[i];
    for (auto * task: server.local_shards) {
      ShardAssignment *s  = req.add_assign();
      s->set_new_worker(server.server_id);
      s->set_table(task->table);
      s->set_shard(task->shard);
      //  update local tables
      GlobalTable *t = TableRegistry::Get()->table(task->table);
      t->get_partition_info(task->shard)->owner = server.server_id;
      delete task;
    }
  }
  net_->SyncBroadcast(MTYPE_SHARD_ASSIGNMENT, MTYPE_SHARD_ASSIGNMENT_DONE, req);
}

void Coordinator::StartWorkers(ModelProto &proto){
  net_->Broadcast(MTYPE_MODEL_CONFIG,proto);
}
//  wait for MTYPE_WORKER_END from other servers
//  send MTYPE_WORKER_SHUTDOWN messages to other
//  do not have to wait, simply exit.
void Coordinator::WaitWorkersFinish() {
  for (int i = 0; i < net_->size() - 1; i++) {
    EmptyMessage end_msg;
    int src = 0;
    net_->Read(MPI::ANY_SOURCE, MTYPE_WORKER_END, &end_msg, &src);
  }
  EmptyMessage shutdown_msg;
  for (int i = 0; i < net_->size() - 1; i++) {
    net_->Send(i, MTYPE_WORKER_SHUTDOWN, shutdown_msg);
  }
  net_->Flush();
}
void Coordinator::Run() {
  ModelProto model_proto;
  ReadProtoFromTextFile(context_->model_conf(), &model_proto);
  Net net;
  net.Init(model_proto.net());
  ModelController mc;
  mc.Init();

  if(context_->standalone()){
    SGDTrainer trainer;
    trainer.Init(model_proto.trainer(), &mc);
    // workers should allocate memory for data and parameters. No need to
    // Init parameters, because they Get parameters from distributed table
    trainer.Run(kAllocData|kAllocParam|kInitParam, &net);
  }else {
    InitTableServers();
    // setup training data which is necessary to setup the DataLayer that is in
    // turn required by upper edges and layers to setup.
    TrainerProto trainer = model_proto.trainer();
    std::vector<DataSource *> train_data;
    Trainer::InitDataSource(trainer.train_data(), &train_data);
    // allocate memory for parameters and init them
    net.Setup(trainer.sgd().train_batchsize(),
        kAllocParam|kInitParam,
        train_data);
    mc.Put(net.params());
    StartWorkers(model_proto);
    WaitWorkersFinish();
  }
}
}  // namespace lapis
