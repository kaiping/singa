//  Copyright © 2014 Anh Dinh. All Rights Reserved.

//  implementing distributed memory interface

#include "distributed-memory.h"

namespace lapis{


	//  init network, wait for all the non-coordinator servers
	//  to register.
	//  also init vector of ClientStates for all memory server
	void DistributedMemoryManager::Init(){
		net_ = NetworkThread::Get();
		GlobalContext* context_ = GlobalContext::Get();

		for (int i = 0; i < net_.size()-1; ++i) {
		   RegisterWorkerRequest req;
		   int src = 0;
		   network_->Read(MPI::ANY_SOURCE, MTYPE_REGISTER_WORKER, &req, &src);

		   //  adding memory server.
		   if (context_->IsRoleOf(Role.kMemoryServer, i))
			   server_states_.push_back(new ServerState(i));
		}
	}

	DistributedMemoryManager* DistributedMemoryManager::Get(){
		static DistributedMemoryManager* dm = new DistributedMemoryManager();
		return dm;
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

	//  memory servers are specified in global context
	void DistributedMemoryManager::assign_worker(int table, int shard){
		  for (int i = 0; i < server_states.size(); ++i) {
		    ServerState& server = *server_states_[i];
		    if (server.shard_id==-1){ //  this server is availabe
		    	server.shard_id = shard;
		    	server.local_shards.insert(TaskId(table, shard));
		    	return;
		    }
		  }
		  LOG(FATAL) << "Out of distributed memory servers to assign tables";
	}

	//  construct ShardAssignment message containing assignment of all tables
	//  then broadcast this to all servers (including non-memory servers)
	void DistributedMemoryManager::send_table_assignments(){
		  ShardAssignmentRequest req;
		  for (int i = 0; i < sever_states_.size(); ++i) {
		    ServerState& server = *server_states_[i];
		    for (set<TaskId>::iterator j = server.local_shards.begin(); j != server.local_shards.end(); ++j) {
		      ShardAssignment* s  = req.add_assign();
		      s->set_new_worker(server.server_id);
		      s->set_table(j->table);
		      s->set_shard(j->shard);
		    }
		  }
		  net_->SyncBroadcast(MTYPE_SHARD_ASSIGNMENT, MTYPE_SHARD_ASSIGNMENT_DONE, req);
	}

	template<class K, class V>
		TypedGlobalTable<K, V>* CreateTable(int id,
		                                            const TypedGlobaclContext<K,V>& type_context){
		  TableDescriptor *info = new TableDescriptor(id, GlobalContext::Get()->num_memory_servers());
		  info->key_marshal = type_context.key_marshal();
		  info->value_marshal = type_context.value_marshal();
		  info->sharder = type_context.sharder();
		  info->accum = type_context.accumulator();

		  info->partition_factory = new typename SparseTable<K, V>::Factory;

		  TypedGlobalTable<K, V> *t = new TypedGlobalTable<K, V>();
		  t->Init(info);
		  TableRegistry::Get()->tables().insert(make_pair(info->table_id, t));
		  return t;
		}
}
