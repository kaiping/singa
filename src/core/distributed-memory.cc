//  Copyright © 2014 Anh Dinh. All Rights Reserved.

//  implementing distributed memory interface

#include "distributed-memory.h"

namespace lapis{

	DistributedMemory* DistributedMemory::Get(){
		static DistributedMemory* dm = new DistributedMemory();
		return dm;
	}

	template<class K, class V>
	TypedGlobalTable<K, V>* DistributedMemory::CreateTable(int id,
	                                           int shards, const GlobalContext<K,V>& context){
	  TableDescriptor *info = new TableDescriptor(id, shards);
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


}
