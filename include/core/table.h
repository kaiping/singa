// Copyright © 2014 Anh Dinh. All Rights Reserved.
// piccolo/table.h

#ifndef INCLUDE_CORE_TABLE_H_
#define INCLUDE_CORE_TABLE_H_
#include <glog/logging.h>
#include <boost/thread.hpp>
#include "core/common.h"
#include "core/file.h"
#include "proto/worker.pb.h"

namespace lapis {


struct TableBase;

//  differnt mappings of a key to a shard
struct Sharding {
	struct String: public Sharder<string> {
		int operator()(const string &k, int shards) {
			return StringPiece(k).hash() % shards;
		}
	};

	struct Mod: public Sharder<int> {
		int operator()(const int &key, int shards) {
			return key % shards;
		}
	};

	struct UintMod: public Sharder<uint32_t> {
		int operator()(const uint32_t &key, int shards) {
			return key % shards;
		}
	};
};

//  create new local table with this facotry
struct TableFactory {
	virtual TableBase *New() = 0;
};


//  global information of the table, containing the number of shards
//  and the table ID
//  also the helper objects for marshalling/unmarshalling data
struct TableDescriptor {
public:
	TableDescriptor(int id, int shards) {
		table_id = id;
		num_shards = shards;
	}

	TableDescriptor(const TableDescriptor &t) {
		memcpy(this, &t, sizeof(t));
	}

	int table_id;
	int num_shards;

	// For local tables, the shard of the global table they represent.
	int shard;
	int default_shard_size;

	void *accum; /**< user-defined accumulator (BaseUpdateHandler) */
	void *sharder; /**< mapping of key to shard ID */
	void *key_marshal; /**< struct for marshalling key type */
	void *value_marshal; /**< struct for marhsalling value type */
	TableFactory *partition_factory; /** struct for creating local table for storing shard content */
};


// Methods common to both global table views and local shards
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


// Interface for serializing tables, either to disk or for transmitting over the network.
struct TableCoder {
	virtual void WriteEntry(StringPiece k, StringPiece v) = 0;
	virtual bool ReadEntry(string *k, string *v) = 0;

	virtual ~TableCoder() {
	}
};

// serializable interface
class Serializable {
 public:
  virtual bool ApplyUpdates(TableCoder *in, LogFile *logfile) = 0;
  virtual bool ApplyPut(TableCoder *in, LogFile *logfile) = 0;
  virtual void restore(LogFile *logfile, int desired_size)=0; 
  virtual void Serialize(TableCoder *out) = 0;
};



// Key/value typed interface.
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

	// return empty string if the value is not ready to be return
	// use this to implement consistency models
	virtual string get_str(const StringPiece &k) = 0;

	// not used yet
	virtual void update_str(const StringPiece &k, const StringPiece &v) = 0;
};

class TableData;


// encoding network table
struct NetworkTableCoder : public TableCoder {
	NetworkTableCoder(const TableData *in);
  virtual void WriteEntry(StringPiece k, StringPiece v);
  virtual bool ReadEntry(string *k, string *v);

  int read_pos_;
  TableData *t_;
};

}  // namespace lapis


#endif  // INCLUDE_CORE_TABLE_H_
