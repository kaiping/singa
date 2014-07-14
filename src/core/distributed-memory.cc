//  Copyright © 2014 Anh Dinh. All Rights Reserved.

//  implementing distributed memory interface

#include "distributed-memory.h"

namespace lapis{

	//  init network, wait for all the non-coordinator servers
	//  to register
	void DistributedMemory::Init(){
		net_ = NetworkThread::Get();
		for (int i = 0; i < net_.size()-1; ++i) {
		   RegisterWorkerRequest req;
		   int src = 0;
		   network_->Read(MPI::ANY_SOURCE, MTYPE_REGISTER_WORKER, &req, &src);
		}
	}

	DistributedMemory* DistributedMemory::Get(){
		static DistributedMemory* dm = new DistributedMemory();
		return dm;
	}

	template<class K, class V>
	TypedGlobalTable<K, V>* DistributedMemory::CreateTable(int id,
	                                            const GlobalContext<K,V>& context){
	  TableDescriptor *info = new TableDescriptor(id, context.num_memory_servers());
	  info->key_marshal = context.key_marshal();
	  info->value_marshal = context.value_marshal();
	  info->sharder = context.sharder();
	  info->accum = context.accumulator();

	  info->partition_factory = new typename SparseTable<K, V>::Factory;

	  TypedGlobalTable<K, V> *t = new TypedGlobalTable<K, V>();
	  t->Init(info);
	  TableRegistry::Get()->tables().insert(make_pair(info->table_id, t));
	  return t;
	}

	//  assigning which workers own which shards
	void DistributedMemory::AssignTables(){
		TableRegistry::Map &tables = TableRegistry::Get()->tables();
		  for (TableRegistry::Map::iterator i = tables.begin(); i != tables.end(); ++i) {
		    for (int j = 0; j < i->second->num_shards(); ++j) {
		      assign_worker(i->first, j);
		    }
		  }
	}

	//  memory servers are specified in global context
	void DistributedMemory::assign_worker(int table, int shard){

	}
}
