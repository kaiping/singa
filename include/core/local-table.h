// Copyright © 2014 Anh Dinh. All Rights Reserved.
// piccolo/local-table.h

#ifndef INCLUDE_CORE_LOCAL-TABLE_H_
#define INCLUDE_CORE_LOCAL-TABLE_H_

#include "core/table.h"
#include "core/file.h"
#include "core/rpc.h"

namespace lapis {

// Represents a single shard of a partitioned global table.
class LocalTable :
  public TableBase,
  public Checkpointable,
  public Serializable,
  public UntypedTable {
public:
  LocalTable() : delta_file_(NULL) {}
  bool empty() { return size() == 0; }

  void start_checkpoint(const string& f);
  void finish_checkpoint();
  void restore(const string& f);
  void write_delta(const TableData& put);

  virtual int64_t size() = 0;
  virtual void clear() = 0;
  virtual void resize(int64_t size) = 0;

  virtual TableIterator* get_iterator() = 0;
protected:
  friend class GlobalTable;
  TableCoder *delta_file_;
};

}  // namespace lapis

#endif  // INCLUDE_CORE_LOCAL-TABLE_H_
