#ifndef INCLUDE_CORE_TABLE_REGISTRY_H_
#define INCLUDE_CORE_TABLE_REGISTRY_H_

#include "core/common.h"
#include "core/table.h"
#include "core/global-table.h"
#include "core/local-table.h"
#include "core/sparse-table.h"
#include "core/disk-table.h"

namespace lapis {

class GlobalTable;

class TableRegistry : private boost::noncopyable {
 private:
  TableRegistry() {}
 public:
  typedef map<int, GlobalTable *> Map;

  static TableRegistry *Get();

  Map &tables();
  GlobalTable *table(int id);

 private:
  Map tmap_;
};


template<class K, class V>
TypedGlobalTable<K, V> *CreateTable(int id, int num_shards, Sharder<K> *skey,
                                    Accumulator<V> *accum, Marshal<K> *mkey, Marshal<V> *mval) {
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

template<class K, class V>
TypedDiskTable<K,V>* CreateDiskTable(int id, int max_size,
		string name, Marshal<K>* mkey, Marshal<V>* mval){
	DiskTableDescriptor *info = new DiskTableDescriptor(id, name, max_size);
	info->key_marshal = mkey;
	info->value_marshal = mval;
	TypedDiskTable<K,V> *t = new TypedDiskTable<K,V>(info);
	TableRegistry::Get()->tables().insert(make_pair(info->id, t));
	return t;
}

//  one desginated server stores the data
template<class K, class V>
TypedDiskTable<K,V>* CreateDiskTable(int id, int fixed_server_id, int max_size,
		string name, Marshal<K>* mkey, Marshal<V>* mval){
	TypedDiskTable<K,V>* t = CreateDiskTable(id, max_size, name, mkey, mval);
	t->info()->fixed_server_id = fixed_server_id;
	return t;
}

/*
template<class K, class V>
TypedGlobalTable<K, V>* CreateTable(int id, int num_shards, Sharder<K>* skey,
										Accumulator<V>* accum, Marshal<K>* mkey, Marshal<V>* mval);

*/
}  // namespace lapis

#endif  // INCLUDE_CORE_TABLE_REGISTRY_H_
