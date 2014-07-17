#ifndef INCLUDE_CORE_TABLE-REGISTRY_H_
#define INCLUDE_CORE_TABLE-REGISTRY_H_

#include "core/common.h"
#include "core/table.h"
#include "core/global-table.h"
#include "core/local-table.h"
#include "core/sparse-table.h"

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

template<class K, class V>
TypedGlobalTable<K, V>* CreateTable(int id, int num_shards, Sharder<K>* skey,
										Accumulator<V>* accum, Marshal<K>* mkey, Marshal<V>* mval);

}  // namespace lapis

#endif  // INCLUDE_CORE_TABLE-REGISTRY_H_
