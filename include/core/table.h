// Copyright © 2014 Anh Dinh. All Rights Reserved.
// piccolo/table.h

#ifndef INCLUDE_CORE_TABLE_H_
#define INCLUDE_CORE_TABLE_H_
#include <glog/logging.h>
#include <boost/thread.hpp>

/**
 * @file table.h
 * Common interfaces for table classes.
 */
namespace lapis {

struct TableBase; /**< type declaration */
class TableData; /**< type declaration */

/**
 * Struct for creating local shard. User implements this struct and passes it
 * as argument during table initialization.
 */
struct TableFactory {
	virtual TableBase *New() = 0;
};


/**
 * Global information of table. It contains the table ID, the number of shards in the table,
 * and helper structs for accumulating updates, for mapping key to shard, for creating local shards,
 * and for data marshalling.
 */
struct TableDescriptor {
public:
	TableDescriptor(int id, int shards) {
		table_id = id;
		num_shards = shards;
	}

	TableDescriptor(const TableDescriptor &t) {
		memcpy(this, &t, sizeof(t));
	}

	int table_id; /**< unique table ID */
	int num_shards;

	void *handler; /**< user-defined handler on table operations */
	TableFactory *partition_factory; /** struct for creating local table for storing shard content */
};


/**
 * Common methods for initializing and accessing table information.
 */
class TableBase {
public:
	virtual void Init(const TableDescriptor *info) {
		info_ = new TableDescriptor(*info);
	}

	const TableDescriptor &info() const {
		return *info_;
	}

	virtual int id() {
		return info().table_id;
	}

	virtual int num_shards() const {
		return info().num_shards;
	}

protected:
	TableDescriptor *info_;
};

class LogFile;

/**
 * Template for typed table classes whose data and operations are of specific types.s
 */
template<class K, class V>
class TypedTable {
public:
	virtual bool contains(const K &k) = 0;
	virtual V get(const K &k) = 0;
	virtual void put(const K &k, const V &v) = 0;
	virtual bool update(const K &k, const V &v) = 0;

	virtual bool ApplyUpdates(TableData *in, LogFile *logfile) = 0;
	virtual bool ApplyPut(TableData *in, LogFile *logfile) = 0;
	void remove(const K &k);
};
}  // namespace lapis


#endif  // INCLUDE_CORE_TABLE_H_
