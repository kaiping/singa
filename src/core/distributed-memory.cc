//  Copyright © 2014 Anh Dinh. All Rights Reserved.

//  implementing distributed memory interface
#include "core/distributed-memory.h"
#include "core/rpc.h"
#include "proto/worker.pb.h"
#include "utils/global_context.h"
#include "core/table-registry.h"

namespace lapis {
DistributedMemoryManager *DistributedMemoryManager::dmm_;

DistributedMemoryManager::~DistributedMemoryManager() {
  for (size_t i = 0; i < server_states_.size(); i++) {
    delete server_states_[i];
  }
}

void DistributedMemoryManager::StartMemoryManager() {
  net_ = NetworkThread::Get();
  VLOG(3)<<"begining of start mem manger in manager id: "<<net_->id();
  GlobalContext *context_ = GlobalContext::Get();
  for (int i = 0; i < net_->size() - 1; ++i) {
    VLOG(3)<<"in start mem manger "<<i;
    RegisterWorkerRequest req;
    int src = 0;
    VLOG(3)<<"before read msg ";
    net_->Read(MPI::ANY_SOURCE, MTYPE_REGISTER_WORKER, &req, &src);
    VLOG(3)<<"after read msg ";
    //  adding memory server.
    if (context_->IsRoleOf(kMemoryServer, i)) {
      server_states_.push_back(new ServerState(i));
    }
  }
  LOG(INFO) << " All servers registered and started up";
  //  set itself as the current worker for the table
  TableRegistry::Map &t = TableRegistry::Get()->tables();
  for (TableRegistry::Map::iterator i = t.begin(); i != t.end(); ++i) {
    i->second->worker_id_ = NetworkThread::Get()->id();
  }
}

//  assigning which workers own which shards
void DistributedMemoryManager::AssignTables() {
  TableRegistry::Map &tables = TableRegistry::Get()->tables();
  VLOG(3)<<"in assign tables :"<<tables.size();
  for (TableRegistry::Map::iterator i = tables.begin(); i != tables.end(); ++i) {
    VLOG(3)<<"num of shards "<<i->second->num_shards();
    for (int j = 0; j < i->second->num_shards(); ++j) {
      assign_worker(i->first, j);
    }
  }
  //  then send table assignment
  send_table_assignments();
}

//  memory servers are specified in global context. Round-robin assignment
void DistributedMemoryManager::assign_worker(int table, int shard) {
  static int server_idx = 0;
  ServerState &server = *server_states_[server_idx];
  LOG(INFO) << StringPrintf("Assigning table (%d,%d) to server %d", table, shard,
                            server_states_[server_idx]->server_id);
  server.shard_id = shard;
  server.local_shards.insert(new TaskId(table, shard));
  server_idx = (server_idx + 1) % server_states_.size();
  return;
}

//  construct ShardAssignment message containing assignment of all tables
//  then broadcast this to all servers (including non-memory servers)
void DistributedMemoryManager::send_table_assignments() {
  ShardAssignmentRequest req;
  for (size_t i = 0; i < server_states_.size(); ++i) {
    ServerState &server = *server_states_[i];
    for (unordered_set<TaskId *>::const_iterator j = server.local_shards.begin();
         j != server.local_shards.end(); ++j) {
      ShardAssignment *s  = req.add_assign();
      s->set_new_worker(server.server_id);
      s->set_table((*j)->table);
      s->set_shard((*j)->shard);
      //  update local tables
      GlobalTable *t = TableRegistry::Get()->table((*j)->table);
      t->get_partition_info((*j)->shard)->owner = server.server_id;
      delete (*j);
    }
  }
  net_->SyncBroadcast(MTYPE_SHARD_ASSIGNMENT, MTYPE_SHARD_ASSIGNMENT_DONE, req);
}

//  wait for MTYPE_WORKER_END from other servers
//  send MTYPE_WORKER_SHUTDOWN messages to other
//  do not have to wait, simply exit.
void DistributedMemoryManager::ShutdownServers() {
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

void DistributedMemoryManager::Init() {
  dmm_ = new DistributedMemoryManager();
}

}
