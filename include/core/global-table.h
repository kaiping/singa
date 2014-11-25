// Copyright © 2014 Anh Dinh. All Rights Reserved.
// piccolo/global-table.cc
#ifndef INCLUDE_CORE_GLOBAL_TABLE_H_
#define INCLUDE_CORE_GLOBAL_TABLE_H_

#include <glog/logging.h>
#include "core/table.h"
#include "core/local-table.h"
#include "core/file.h"
#include "core/request_dispatcher.h"

namespace lapis {

class TableServer;

/**
 * This class represent the global view of the table, i.e. each process treats it as a
 * local table to which it can put/get data.
 *
 * Each table maintains #shards partitions which are typed local table. It has the mapping
 * of which process "owns" which partition.
 *
 * The global table provides common put/get operations over string/byte streams, with
 * which the typed tables can use to convert to specific key/value types.
 *
 * Given a key K, the global table knows if it is stored in one of its local shards. It invokes
 * the local table directly if yes, or sends the request over the network if no.
 *
 * The current implementation assumes that table servers are different to workers, i.e.
 * put/get will be remote requests.
 */
class GlobalTable: public TableBase {
public:
	virtual void Init(const TableDescriptor *tinfo);
	virtual ~GlobalTable();

	// each partition has an owner which may not be the current process
	struct PartitionInfo {
		PartitionInfo() :owner(-1) {}
		int owner;
	};

	virtual PartitionInfo *get_partition_info(int shard) {
		return &partinfo_[shard];
	}

	//  the process which owns the shard
	int owner(int shard) {
		return get_partition_info(shard)->owner;
	}

	bool is_local_shard(int shard);
	bool is_local_key(const StringPiece &k);


	/**
	 * handle remote get request from another process
	 * (1) check that the request is for the local shard
	 * (2) invoke user-define handle-get at the local shard
	 * (3) return true + TableData if the data can be returned right away
	 *     return false -> data is not ready to be sent back (the request
	 *                       is to be re-processed)
	 */
	bool HandleGet(const HashGet &req, TableData *resp);


	/**
	 * Apply the put and get request from another process.
	 *
	 * return true if the operation is applied successfully
	 * return false means the request should be re-processed
	 */
	bool ApplyUpdates(const TableData &req);
	bool ApplyPut(const TableData &req);

	// restore the local shard from checkpoint file
	void Restore(int shard);

	//  table stats, obtained by merging stats of the local shards
	Stats stats();


	//  checkpoint files, one for each shard
	map<int, LogFile*>* checkpoint_files() {
		return &checkpoint_files_;
	}

	// clear the local partition
	void clear(int shard);
	bool empty();

	// resize paritions to the new size
	void resize(int64_t new_size);

	int worker_id_;

	virtual int64_t shard_size(int shard);

	// returns the shard of the string key. Typed tables overrides
	// this by marshalling the key to string
	virtual int get_shard_str(StringPiece k) = 0;

protected:
	vector<PartitionInfo> partinfo_;

	vector<LocalTable *> partitions_;

	map<int, LogFile*> checkpoint_files_;

	friend class TableServer;

	void set_worker(TableServer *w);

	// synchronous fetch of key k from the node owning it.  Returns true
	// if the key exists (always).
	bool get_remote(int shard, const StringPiece &k, string *v);

	// send get request to the remote machine then return
	void async_get_remote(int shard, const StringPiece &k);

	//  true if there're responses to GET request
	bool async_get_remote_collect(string *k, string *v);

	// true if the response is of the specified key
	bool async_get_remote_collect_key(int shard, const string &k, string *v);

};


/**
 * Typed tables. Its main job is to marshall/unmarshall between user data types
 * and the generic string type of GlobalTable.
 *
 * Workers use this interface directly to put/get data
 */
template<class K, class V>
class TypedGlobalTable: public GlobalTable,
		public TypedTable<K, V>,
		private boost::noncopyable {
public:
	virtual void Init(const TableDescriptor *tinfo) {
		GlobalTable::Init(tinfo);
		for (size_t i = 0; i < partitions_.size(); ++i) {
			partitions_[i] = create_local(i);
		}
	}

	// return shard ID for the given key
	int get_shard(const K &k);
	int get_shard_str(StringPiece k);

	// put and update (K,V)
	void put(const K &k, const V &v);
	bool update(const K &k, const V &v);

	// synchronous get: block until receiving V for K
	V get(const K &k);

	//  asynchronous get. if true then V contains the value
	//  if false, user has to call async_collect() until there's data (returning true)
	bool async_get(const K &k, V* v);
	bool async_get_collect(K* k, V* v);

	// if the table contains K
	bool contains(const K &k);

	TypedTable<K, V> *partition(int idx) {
		return dynamic_cast<TypedTable<K, V>*>(partitions_[idx]);
	}

protected:

	// create shards (Sparse Table)
	LocalTable *create_local(int shard);
};


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

//  prepares the TableData object and sends off to the shard containing K
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

	// send over the network
	if (is_local_shard(shard))
		NetworkThread::Get()->Send(local_rank, MTYPE_PUT_REQUEST, put); //send locally
	else
		NetworkThread::Get()->Send(onwer(shard), MTYPE_PUT_REQUEST, put); //send remote
}


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
	if (is_local_shard(shard))
		NetworkThread::Get()->Send(local_rank, MTYPE_UPDATE_REQUEST, put); //send locally
	else
		NetworkThread::Get()->Send(owner(i), MTYPE_UPDATE_REQUEST, put); //send remotely
	return true;
}


// synchronous get
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

/*
 * asynchronous get, returns right away. use async_get_collect to wait on the
 * content.
 *
 * if the key is local, return
 * else send send (get_remote without waiting).
 */
template<class K, class V>
bool TypedGlobalTable<K, V>::async_get(const K &k, V* v) {
	int shard = this->get_shard(k);

	//ALL going through Networkthread messages, including local get
	async_get_remote(shard,
			marshal(static_cast<Marshal<K>*>(this->info().key_marshal), k));
	return false;

}

/*
 * wait on the content. The upper layer repeatedly invokes this function until
 * it returns true, then the value can be used.
 *
 * K k;
 * V v;
 * if (async_get_collect(&k,&v))
 *     do_some_thing(k,v);
 */
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


//  check if the table contains a specific key.
//  Rarely used.
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

//  initialize local partition.
//  User-specified factory to create a new LocalTable
template<class K, class V>
LocalTable *TypedGlobalTable<K, V>::create_local(int shard) {
	TableDescriptor *linfo = new TableDescriptor(info());
	linfo->shard = shard;
	LocalTable *t = (LocalTable *) info_->partition_factory->New();
	t->Init(linfo);
	return t;
}

}  // namespace lapis

#endif /* INCLUDE_CORE_GLOBAL_TABLE_H_ */
