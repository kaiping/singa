//  Copyright © 2014 Anh Dinh. All Rights Reserved.

//  distributed memory interface, exposed to higher application level
//  singleton (similar to NetworkThread)

#ifndef INCLUDE_CORE_DISTRIBUTED-MEMORY_H_
#define INCLUDE_CORE_DISTRIBUTED-MEMORY_H_

#include "table-registry.h"

namespace lapis{
	//  represent the (table, shard) tuple
	struct TaskId{
		int table;
		int shard;

		TaskId(int t, int s): table(t), shard(s){}
	};

	//  each memory server has a set of (table,shard) partitions
	//  assigned to it. shardId is the same for all partitions
	struct ServerState{
		int server_id;
		int shard_id;
		set<TaskId> local_shards;

		ServerState(int id): server_id(id), shard_id(-1){}
	};

	class DistributedMemoryManager : private boost::noncopyable{
	 public:
		~DistributedMemoryManager();

		static void Init();

		//  start the manager
		void StartMemoryManager();

		void AssignTables();  //  assign tables to clients

		//  must be called at the end
		void ShutdownServers(); //  shut down other clients

		static DistributedMemoryManager* Get(){ return dmm_; }

	 private:
		//  assign which worker owning this (table,shard)
		void assign_worker(int table, int shard);

		void send_table_assignments();

		//  keep track of the table assignments, only to the memory servers
		vector<ServerState*> server_states_;

		NetworkThread* net_;

		static DistributedMemoryManager* dmm_;

		DistributedMemoryManager();
	};

	DistributedMemoryManager::~DistributedMemoryManager(){
		for (int i=0; i<server_states_.size(); i++){
			delete server_states_[i];
		}
	}

	template<class K, class V>
			TypedGlobalTable<K, V>* CreateTable(int id, const TypedGlobalContext& typed_context);

}  //  namespace lapis

#endif  //  INCLUDE_CORE_DISTRIBUTED-MEMORY_H_
