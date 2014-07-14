#include <stdio.h>

#include "table-registry.h"
#include "global-table.h"
#include "local-table.h"

static const int kStatsTableId = 1000000;

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


static void CreateStatsTable() {
  CreateTable(
      kStatsTableId, 1, new Sharding::String, new Accumulators<string>::Replace);
}
}

REGISTER_INITIALIZER(CreateStatsTable, lapis::CreateStatsTable());
