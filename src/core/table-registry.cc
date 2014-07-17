#include <stdio.h>

#include "core/table-registry.h"
#include "core/global-table.h"
#include "core/local-table.h"

namespace lapis {

TableRegistry* TableRegistry::Get() {
  static TableRegistry* t = new TableRegistry;
  return t;
}

TableRegistry::Map& TableRegistry::tables() {
  return tmap_;
}

GlobalTable* TableRegistry::table(int id) {
  CHECK(tmap_.find(id) != tmap_.end());
  return tmap_[id];
}

template<class K, class V>
TypedGlobalTable<K, V>* CreateTable(int id, int num_shards, Sharder<K>* skey,
										Accumulator<V>* accum, Marshal<K>* mkey, Marshal<V>* mval){
		  TableDescriptor *info = new TableDescriptor(id, num_shards);
		  info->key_marshal = mkey;
		  info->value_marshal = mval;
		  info->sharder = skey;
		  info->accum = accum;

		  info->partition_factory = new typename SparseTable<K, V>::Factory;

		  TypedGlobalTable<K, V> *t = new TypedGlobalTable<K, V>();
		  t->Init(info);
		  TableRegistry::Get()->tables().insert(make_pair(info->table_id, t));
		  return t;
		}

}


