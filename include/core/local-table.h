// Copyright © 2014 Anh Dinh. All Rights Reserved.
// piccolo/local-table.h

#ifndef INCLUDE_CORE_LOCAL_TABLE_H_
#define INCLUDE_CORE_LOCAL_TABLE_H_

#include "core/table.h"
#include "core/file.h"

namespace lapis {

/**
 * A single shard of a partitioned global table. Each global table maintains
 * a number of typed local table for accessing typed key-value tuples.
 * @see sparse-table.h
 */
class LocalTable: public TableBase, public UntypedTable, public Serializable {
public:
	LocalTable() :
			delta_file_(NULL) {
	}
	bool empty() {
		return size() == 0;
	}

	virtual int64_t size() = 0;
	virtual void clear() = 0;
	virtual void resize(int64_t size) = 0;
	Stats& stats() {
		return stats_;
	}

	virtual ~LocalTable() {
	}
protected:
	friend class GlobalTable;
	TableCoder *delta_file_;
	Stats stats_;
};

}  // namespace lapis

#endif  // INCLUDE_CORE_LOCAL_TABLE_H_
