// Copyright © 2014 Anh Dinh. All Rights Reserved.
// piccolo/local-table.h

#ifndef INCLUDE_CORE_LOCAL_TABLE_H_
#define INCLUDE_CORE_LOCAL_TABLE_H_

#include "core/table.h"
#include "core/file.h"

/**
 * @file local-table.h
 * A single shard of the global table. Each global table maintains
 * a number of typed local tables for accessing typed key-value tuples.
 * @see sparse-table.h, global-table.h
 *
 */
namespace lapis {

/**
 * A simple LocalTable class. Typed local tables extend this class.
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
	TableCoder *delta_file_; /**< checkpointing */
	Stats stats_; /**< table stats */
};

}  // namespace lapis

#endif  // INCLUDE_CORE_LOCAL_TABLE_H_
