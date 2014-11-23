// Copyright © 2014 Wei Wang, Anh. All Rights Reserved.
// 2014-08-07 11:32
#include "coordinator.h"
#include "utils/network_thread.h"
#include "utils/common.h"
#include "proto/common.pb.h"
#include "proto/model.pb.h"

using std::string;
using std::vector;

namespace lapis {

Coordinator::~Coordinator() {
  for (auto* state: server_states_) {
    for (auto* taskid : state->local_shards)
      delete taskid;
    delete state;
  }
}

void Coordinator::InitTableServers(const std::map<int, GlobalTable*>& tables) {
  auto mpi=NetworkThread::Get();
  for (int i = context_->server_start();i<context_->server_end();++i){
    VLOG(3)<<"in table server "<<i;
    RegisterWorkerRequest req;
    int src = 0;
    VLOG(3)<<"before read msg ";
    mpi->Read(MPI::ANY_SOURCE, MTYPE_REGISTER_WORKER, &req, &src);
    VLOG(3)<<"after read msg ";
    //  adding memory server.
    if (context_->IsTableServer(i)) {
      server_states_.push_back(new ServerState(i));
    }
  }
  LOG(INFO) << " All servers registered and started up";
  //  set itself as the current worker for the table
  for (auto &entry: tables)
    entry.second->worker_id_ = mpi->id();

  // memory servers are specified in global context. Round-robin assignment
  int server_idx = 0;
  for (auto &entry: tables){
    VLOG(3)<<"num of shards"<<entry.second->num_shards()<<" for table"<< entry.first;
    int table=entry.first;
    for (int shard = 0; shard < entry.second->num_shards(); ++shard) {
      ServerState &server = *server_states_[server_idx];
      LOG(INFO) << "Assigning table ("<<table<<","<<shard<<") to server "
                <<server_states_[server_idx]->server_id;
      // TODO(Anh) may overwrite this field if #shards>#table_servers
      server.shard_id = shard;
      server.local_shards.insert(new TaskId(table, shard));
      server_idx = (server_idx + 1) % server_states_.size();
    }
  }
  VLOG(3)<<"table assignment";
  //  then send table assignment
  ShardAssignmentRequest req;
  for (size_t i = 0; i < server_states_.size(); ++i) {
    VLOG(3)<<"server states "<<i;
    ServerState &server = *server_states_[i];
    for (auto * task: server.local_shards) {
      ShardAssignment *s  = req.add_assign();
      s->set_new_worker(server.server_id);
      s->set_table(task->table);
      s->set_shard(task->shard);
      //  update local tables
      CHECK(tables.find(task->table)!=tables.end());
      GlobalTable *t = tables.at(task->table);
      VLOG(3)<<"table id"<<t->id();
      t->get_partition_info(task->shard)->owner = server.server_id;
      delete task;
    }
  }
  LOG(ERROR)<<"finish table assignment, req size "<<req.assign_size();
  mpi->Broadcast(MTYPE_SHARD_ASSIGNMENT, req);
  mpi->WaitForSync(MTYPE_SHARD_ASSIGNMENT_DONE, GlobalContext::kCoordinator);//context_->num_table_servers());
  LOG(ERROR)<<"finish table server init";
}


//  wait for MTYPE_WORKER_END from other servers
//  send MTYPE_WORKER_SHUTDOWN messages to other
//  do not have to wait, simply exit.
void Coordinator::Shutdown() {
  /*
  EmptyMessage shutdown_msg;
  for (int i = 0; i < mpi_->size() - 1; i++) {
    mpi_->Send(i, MTYPE_SHUTDOWN, shutdown_msg);
  }
  mpi_->Flush();
  mpi_->Shutdown();
  */
}
void Coordinator::Run(const Model& model) {
  SolverProto sp(model.solver());
  sp.mutable_sgd()->set_threshold(context_->num_groups());
  sp.mutable_adagrad()->set_threshold(0);
  TableDelegate* delegate=CreateTableDelegate(sp);
  InitTableServers(delegate->tables());
}
}  // namespace lapis
