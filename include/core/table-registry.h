#ifndef INCLUDE_CORE_TABLE-REGISTRY_H_
#define INCLUDE_CORE_TABLE-REGISTRY_H_

#include "core/common.h"
#include "core/table.h"
#include "global-table.h"
#include "local-table.h"
#include "sparse-table.h"

namespace lapis {

class GlobalTable;

class TableRegistry : private boost::noncopyable {
private:
  TableRegistry() {}
public:
  typedef map<int, GlobalTable*> Map;

  static TableRegistry* Get();

  Map& tables();
  GlobalTable* table(int id);

private:
  Map tmap_;
};

// Swig doesn't like templatized default arguments; work around that here.
template<class K, class V>
static TypedGlobalTable<K, V>* CreateTable(int id,
                                           int shards,
                                           Sharder<K>* sharding,
                                           Accumulator<V>* accum) {
  TableDescriptor *info = new TableDescriptor(id, shards);
  info->key_marshal = new Marshal<K>;
  info->value_marshal = new Marshal<V>;
  info->sharder = sharding;
  info->partition_factory = new typename SparseTable<K, V>::Factory;
  info->accum = accum;

  return CreateTable<K, V>(info);
}

template<class K, class V>
static TypedGlobalTable<K, V>* CreateTable(const TableDescriptor *info) {
  TypedGlobalTable<K, V> *t = new TypedGlobalTable<K, V>();
  t->Init(info);
  TableRegistry::Get()->tables().insert(make_pair(info->table_id, t));
  return t;
}

}  // namespace lapis

#endif  // INCLUDE_CORE_TABLE-REGISTRY_H_
