//  Copyright © 2014 Anh Dinh. All Rights Reserved.

//  implementing distributed memory interface
#include "core/distributed-memory.h"
#include "core/rpc.h"
#include "core/worker.pb.h"
#include "utils/global_context.h";
#include "core/table-registry.h"

namespace lapis{
	DistributedMemoryManager* DistributedMemoryManager::dmm_;

	DistributedMemoryManager::~DistributedMemoryManager(){
		for (int i=0; i<server_states_.size(); i++){
			delete server_states_[i];
		}
	}

	void DistributedMemoryManager::StartMemoryManager(){
		net_ = NetworkThread::Get();
		GlobalContext* context_ = GlobalContext::Get();

		for (int i = 0; i < net_->size()-1; ++i) {
		   RegisterWorkerRequest req;
		   int src = 0;
		   net_->Read(MPI::ANY_SOURCE, MTYPE_REGISTER_WORKER, &req, &src);

		   //  adding memory server.
		   if (context_->IsRoleOf(kMemoryServer, i)){
			   server_states_.push_back(new ServerState(i));
		   }
		}
		LOG(INFO) << StringPrintf(" All servers registered and started up");
		//  set itself as the current worker for the table
		TableRegistry::Map &t = TableRegistry::Get()->tables();
		for (TableRegistry::Map::iterator i = t.begin(); i != t.end(); ++i) {
		    i->second->worker_id_ = NetworkThread::Get()->id();
		}
	}

	//  assigning which workers own which shards
	void DistributedMemoryManager::AssignTables(){
		TableRegistry::Map &tables = TableRegistry::Get()->tables();
		  for (TableRegistry::Map::iterator i = tables.begin(); i != tables.end(); ++i) {
		    for (int j = 0; j < i->second->num_shards(); ++j) {
		      assign_worker(i->first, j);
		    }
		  }

		//  then send table assignment
		send_table_assignments();

	}

	//  memory servers are specified in global context. Round-robin assignment
	void DistributedMemoryManager::assign_worker(int table, int shard){
		static int server_idx = 0;

		ServerState& server = *server_states_[server_idx];
		LOG(INFO) << StringPrintf("ASSIGNING TABLE (%d,%d) to SERVER %d", table, shard, server_states_[server_idx]->server_id);
		server.shard_id = shard;
		server.local_shards.insert(new TaskId(table, shard));
		server_idx = (server_idx+1) % server_states_.size();
		return;
	}

	//  construct ShardAssignment message containing assignment of all tables
	//  then broadcast this to all servers (including non-memory servers)
	void DistributedMemoryManager::send_table_assignments(){
		  ShardAssignmentRequest req;
		  for (int i = 0; i < server_states_.size(); ++i) {
		    ServerState& server = *server_states_[i];
		    for (unordered_set<TaskId*>::const_iterator j = server.local_shards.begin(); j != server.local_shards.end(); ++j) {
		      ShardAssignment* s  = req.add_assign();
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

	//  send MTYPE_WORKER_SHUTDOWN messages to other
	//  do not have to wait, simply exit.
	void DistributedMemoryManager::ShutdownServers(){
		EmptyMessage message;
		for (int i=0; i<net_->size()-1; i++)
			net_->Send(i, MTYPE_WORKER_SHUTDOWN, message);
	}

		void DistributedMemoryManager::Init(){
			dmm_ = new DistributedMemoryManager();
		}

}
