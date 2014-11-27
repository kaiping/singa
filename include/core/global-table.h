// Copyright © 2014 Anh Dinh. All Rights Reserved.
// piccolo/global-table.cc
#ifndef INCLUDE_CORE_GLOBAL_TABLE_H_
#define INCLUDE_CORE_GLOBAL_TABLE_H_

#include <glog/logging.h>
#include "core/table.h"
#include "core/local-table.h"
#include "core/file.h"
#include "core/request_dispatcher.h"

/**
 * @file global-table.h
 * A generic GlobalTable class represents the global view of the table, i.e. each process
 * treats it as a local table to which it can put/get data. Each table maintains #shards
 * partitions which are typed table. It has the mapping of which process "owns" which partition.
 * Given a key, the table knows if it is stored in one of its local shards. It invokes the
 * local table directly if yes, or sends the request over the network if no.
 *
 * A typed GlobalTable restrict data and operations to specific types.
 *
 * The current implementation assumes that table servers are different to workers, i.e.
 * put/get will be remote requests.
 */
namespace lapis {

class TableServer;


/**
 * Generic table operating on string key-value tuples.
 */
class GlobalTable: public TableBase {
public:
	virtual ~GlobalTable();

	/**
	 * Initialize the table.
	 * @param *tinfo the table descriptor. @see table.h
	 */
	virtual void Init(const TableDescriptor *tinfo);

	/**
	 * Information of a shard: in which process is the shard stored.
	 * Tables at all processes share the same shard information. They use this
	 * to decide where to send the request.
	 */
	struct PartitionInfo {
		PartitionInfo() :owner(-1) {}
		int owner;
	};

	/**
	 * Get the shard information of a given shard.
	 */
	virtual PartitionInfo *get_partition_info(int shard) {
		return &partinfo_[shard];
	}

	/**
	 * Get the process ID (rank) where the shard is maintained.
	 */
	int owner(int shard) {
		return get_partition_info(shard)->owner;
	}

	bool is_local_shard(int shard);
	bool is_local_key(const StringPiece &k);


	/**
	 * Handle remote get request from another process. There are 3 steps:
	 * (1) check that the request is for the local shard
	 * (2) invoke user-define handle-get at the local shard
	 * (3) return true + TableData if the data can be returned right away
	 *     return false -> data is not ready to be sent back (the request
	 *                       is to be re-processed)
	 */
	bool HandleGet(const HashGet &req, TableData *resp);


	/**
	 * Apply update requests from another process.
	 *
	 * @return true if the operation is applied successfully.
	 */
	bool ApplyUpdates(const TableData &req);

	/**
	 * Apply put requests from another process. Always return true.
	 */
	bool ApplyPut(const TableData &req);

	/**
	 * Restore content of the specified (local) shard from the checkpoint file.
	 * The file is scanned backward and tuples are inserted back to the shard.
	 */
	void Restore(int shard);


	Stats stats(); /**< table stats, merged from local shards' stats */

	/**
	 * Get checkpoint files of the local shards (one file per shard).
	 */
	map<int, LogFile*>* checkpoint_files() {
		return &checkpoint_files_;
	}

	void clear(int shard); /**< clear the partition. */
	bool empty(); /**< is the shard empty. */

	/**
	 * Resize all local shards to the new size.
	 */
	void resize(int64_t new_size);

	int worker_id_; /**< rank of the current process.*/

	virtual int64_t shard_size(int shard); /**< number of tuples in the shard. */

	/**
	 * Returns the shard ID of the given string key. Typed tables override
	 * this by marshalling string to the correct type.
	 */
	virtual int get_shard_str(StringPiece k) = 0;

protected:
	vector<PartitionInfo> partinfo_; /**< shard information */

	vector<LocalTable *> partitions_; /**< actual shards */

	map<int, LogFile*> checkpoint_files_;

	/*
	 * Synchronous fetch of the given key stored in the given shard.
	 * @return true if the key exists.
	 */
	bool get_remote(int shard, const StringPiece &k, string *v);

	/**
	 * Send get request to the remote shard and return immediately.
	 */
	void async_get_remote(int shard, const StringPiece &k);


	/**
	 * Collect responses of the get requests into the specified placeholders.
	 * @return true if there is a response.
	 */
	bool async_get_remote_collect(string *k, string *v);

	/**
	 * Collect response of the get request for a specific key.
	 * @return false if the response has not arrived.
	 */
	bool async_get_remote_collect_key(int shard, const string &k, string *v);

};


/**
 * A template for GlobalTable of specific types.
 *
 * Its main job is to marshall/unmarshall between user data types
 * and the generic string type of GlobalTable.
 *
 */
template<class K, class V>
class TypedGlobalTable: public GlobalTable,
		public TypedTable<K, V>,
		private boost::noncopyable {
public:
	/**
	 * Initialize the table. First, it performs normal initialization as with
	 * a generic table. Then it instantiates the local shards which are also
	 * typed tables, from a user-given factory.
	 */
	virtual void Init(const TableDescriptor *tinfo) {
		GlobalTable::Init(tinfo);
		for (size_t i = 0; i < partitions_.size(); ++i) {
			partitions_[i] = create_local(i);
		}
	}

	int get_shard(const K &k); /**< returns ID of the shard storing k */
	int get_shard_str(StringPiece k); /**< override generic table's method */

	/**
	 * Insert new tuple to the table.
	 */
	void put(const K &k, const V &v);

	/**
	 * Update table with a new tuple.
	 */
	bool update(const K &k, const V &v);

	/**
	 * Synchronous get. Block until V is received.
	 */
	V get(const K &k);

	/**
	 * Asynchronous get. Sending the request for key K then returning.
	 */
	bool async_get(const K &k, V* v);

	/**
	 * Collect responses of the get requests.
	 */
	bool async_get_collect(K* k, V* v);

	bool contains(const K &k); /** if the table contains K */

	/**
	 * Return the local shard (a typed table) of the given index.
	 */
	TypedTable<K, V> *partition(int idx) {
		return dynamic_cast<TypedTable<K, V>*>(partitions_[idx]);
	}

protected:

	LocalTable *create_local(int shard); /**< instantiates a local shard (SparseTable). */
};


/**
 * Return shard ID for a given key. The mapping is provided by the user
 * via a Sharder struct (@see common.h).
 */
template<class K, class V>
int TypedGlobalTable<K, V>::get_shard(const K &k) {
	DCHECK(this != NULL);
	DCHECK(this->info().sharder != NULL);
	Sharder<K> *sharder = (Sharder<K> *) (this->info().sharder);
	int shard = (*sharder)(k, this->info().num_shards);
	DCHECK_GE(shard, 0);
	DCHECK_LT(shard, this->num_shards());
	return shard;
}

template<class K, class V>
int TypedGlobalTable<K, V>::get_shard_str(StringPiece k) {
	return get_shard(
			unmarshal(static_cast<Marshal<K>*>(this->info().key_marshal), k));
}

/**
 * Insert new tuple to the table. It first prepares TableData message containing
 * the tuple, then sends it off to the remote shard.
 */
template<class K, class V>
void TypedGlobalTable<K, V>::put(const K &k, const V &v) {
	int shard = this->get_shard(k);
	StringPiece key = marshal(
			static_cast<Marshal<K>*>(this->info().key_marshal), k);

	int local_rank = NetworkThread::Get()->id();

	// prepare TableData object
	TableData put;
	put.set_shard(shard);
	put.set_source(local_rank);
	put.set_table(this->id());

	StringPiece value = marshal(
			static_cast<Marshal<V>*>(this->info().value_marshal), v);

	put.set_key(key.AsString());
	Arg *a = put.add_kv_data();
	a->set_key(key.data, key.len);
	a->set_value(value.data, value.len);
	put.set_done(true);

	// send over the network. No local put.
	NetworkThread::Get()->Send(owner(shard), MTYPE_PUT_REQUEST, put); //send remote
}


/**
 * Update the table with new tuple. @see put().
 */
template<class K, class V>
bool TypedGlobalTable<K, V>::update(const K &k, const V &v) {
	int shard = this->get_shard(k);
	StringPiece key = marshal(
			static_cast<Marshal<K>*>(this->info().key_marshal), k);


	int local_rank = NetworkThread::Get()->id();

	//  prepares TableData
	TableData put;
	put.set_shard(shard);
	put.set_source(local_rank);
	put.set_table(this->id());

	StringPiece value = marshal(
			static_cast<Marshal<V>*>(this->info().value_marshal), v);

	put.set_key(key.AsString());
	Arg *a = put.add_kv_data();
	a->set_key(key.data, key.len);
	a->set_value(value.data, value.len);
	put.set_done(true);

	// send via the network
	NetworkThread::Get()->Send(owner(shard), MTYPE_UPDATE_REQUEST, put); //send remotely
	return true;
}


template<class K, class V>
V TypedGlobalTable<K, V>::get(const K &k) {
	int shard = this->get_shard(k);

	V ret;
	async_get(k, &ret);

	string key = marshal(static_cast<Marshal<K>*>(this->info().key_marshal), k);

	//collect
	string v_str;
	while (!async_get_remote_collect_key(shard, key, &v_str))
		Sleep(0.001);

	return unmarshal(static_cast<Marshal<V>*>(this->info().value_marshal),
			v_str);
}

template<class K, class V>
bool TypedGlobalTable<K, V>::async_get(const K &k, V* v) {
	int shard = this->get_shard(k);

	//ALL going through Networkthread messages, including local get
	async_get_remote(shard,
			marshal(static_cast<Marshal<K>*>(this->info().key_marshal), k));
	return false;

}

template<class K, class V>
bool TypedGlobalTable<K, V>::async_get_collect(K* k, V* v) {

	//ALWAYS remote
	string tk, tv;
	bool succeed = async_get_remote_collect(&tk, &tv);
	if (succeed) {
		*k = unmarshal(static_cast<Marshal<K>*>(this->info().key_marshal), tk);
		*v = unmarshal(static_cast<Marshal<V>*>(this->info().value_marshal),
				tv);
	}
	return succeed;
}


template<class K, class V>
bool TypedGlobalTable<K, V>::contains(const K &k) {
	int shard = this->get_shard(k);
	if (is_local_shard(shard)) {
		return partition(shard)->contains(k);
	}
	string v_str;
	return get_remote(shard,
			marshal(static_cast<Marshal<K>*>(info_->key_marshal), k), &v_str);
}

template<class K, class V>
LocalTable *TypedGlobalTable<K, V>::create_local(int shard) {
	TableDescriptor *linfo = new TableDescriptor(info());
	LocalTable *t = (LocalTable *) info_->partition_factory->New();
	t->Init(linfo);
	return t;
}

}  // namespace lapis

#endif /* INCLUDE_CORE_GLOBAL_TABLE_H_ */
