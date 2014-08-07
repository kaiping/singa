#include <stdio.h>

#include "core/table-registry.h"
#include "core/global-table.h"
#include "core/local-table.h"

namespace lapis {

TableRegistry *TableRegistry::Get() {
  static TableRegistry *t = new TableRegistry;
  return t;
}

TableRegistry::Map &TableRegistry::tables() {
  return tmap_;
}

GlobalTable *TableRegistry::table(int id) {
  CHECK(tmap_.find(id) != tmap_.end());
  return tmap_[id];
}

}
