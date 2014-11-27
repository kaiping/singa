// Copyright © 2014 Anh Dinh. All Rights Reserved.
// piccolo/table.h

#ifndef INCLUDE_CORE_TABLE_H_
#define INCLUDE_CORE_TABLE_H_
#include <glog/logging.h>
#include <boost/thread.hpp>
#include "core/common.h"
#include "core/file.h"
#include "proto/worker.pb.h"

/**
 * @file table.h
 * Common interfaces for table classes.
 */
namespace lapis {

struct TableBase; /**< type declaration */

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

	void *accum; /**< user-defined accumulator (BaseUpdateHandler) */
	void *sharder; /**< mapping of key to shard ID */
	void *key_marshal; /**< struct for marshalling key type */
	void *value_marshal; /**< struct for marhsalling value type */
	TableFactory *partition_factory; /** struct for creating local table for storing shard content */
};


/**
 * Common methods for initializing and accessing table information.
 */
class TableBase {
public:
	virtual void Init(const TableDescriptor *info) {
		info_ = new TableDescriptor(*info);
		CHECK(info_->key_marshal != NULL);
		CHECK(info_->value_marshal != NULL);
	}

	const TableDescriptor &info() const {
		return *info_;
	}

	virtual int id() {
		return info().table_id;
	}

	// current shard ID
	int shard() const {
		return info().shard;
	}

	virtual int num_shards() const {
		return info().num_shards;
	}

protected:
	TableDescriptor *info_;
};


/**
 * Struct for serializing tables, either to disk or for transmitting over the network.
 */
struct TableCoder {
	virtual void WriteEntry(StringPiece k, StringPiece v) = 0;
	virtual bool ReadEntry(string *k, string *v) = 0;

	virtual ~TableCoder() {
	}
};

/**
 * Serializable table interface.
 */
class Serializable {
 public:
  virtual bool ApplyUpdates(TableCoder *in, LogFile *logfile) = 0;
  virtual bool ApplyPut(TableCoder *in, LogFile *logfile) = 0;
  virtual void restore(LogFile *logfile, int desired_size)=0; 
  virtual void Serialize(TableCoder *out) = 0;
};



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
	virtual void remove(const K &k) = 0;
};

/**
 * A generic table in which key and value are of type string.
 *
 * The GlobalTable works with get/update methods of this class. The get and put
 * method also invokes user-define get/update handler --- for customizing
 * consistency model.
 *
 * Typed tables (@see sparse-table.h) override these method to convert to/from
 * correct types.
 */
class UntypedTable {
public:

	/**
	 * Return empty string if the value is not ready to be returned.
	 */
	virtual string get_str(const StringPiece &k) = 0;

	/**
	 * Not yet implemented!
	 */
	virtual void update_str(const StringPiece &k, const StringPiece &v) = 0;
};

class TableData;


/**
 * Specific encoding of table data, to be transmitted over the network.
 */
struct NetworkTableCoder : public TableCoder {
	NetworkTableCoder(const TableData *in);
  virtual void WriteEntry(StringPiece k, StringPiece v);
  virtual bool ReadEntry(string *k, string *v);

  int read_pos_;
  TableData *t_;
};

}  // namespace lapis


#endif  // INCLUDE_CORE_TABLE_H_
