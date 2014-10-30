#ifndef INCLUDE_CORE_GLOBAL_TABLE_H_
#define INCLUDE_CORE_GLOBAL_TABLE_H_

#include <glog/logging.h>
#include "core/table.h"
#include "core/local-table.h"
#include "core/file.h"
#include "core/request_dispatcher.h"

namespace lapis {

class TableServer;

class GlobalTable :
  public TableBase {
 public:
  virtual void Init(const TableDescriptor *tinfo);
  virtual ~GlobalTable();

  struct PartitionInfo {
    PartitionInfo() : owner(-1) {}
    int owner;
  };

  virtual PartitionInfo *get_partition_info(int shard) {
    return &partinfo_[shard];
  }

  int owner(int shard) {
    return get_partition_info(shard)->owner;
  }

  LocalTable *get_partition(int shard);

  bool is_local_shard(int shard);
  bool is_local_key(const StringPiece &k);

  // Fill in a response from a remote worker for the given key.
  bool handle_get(const HashGet &req, TableData *resp);

  // Handle updates from the master or other workers.
  void SendUpdates();
  void SendPut();

  bool ApplyUpdates(const TableData &req);
  bool ApplyPut(const TableData &req);

  void Restore(int shard); 

  void UpdatePartitions(const ShardInfo &sinfo);

  Stats stats();
  
  map<int, LogFile*>* checkpoint_files(){ return &checkpoint_files_;}
  
  int pending_write_bytes();

  // Clear any local data for which this table has ownership.
  // Updates waiting to be sent to other workers are *not* cleared.
  void clear(int shard);
  bool empty();
  void resize(int64_t new_size);

  int worker_id_;

  virtual int64_t shard_size(int shard);

  virtual int get_shard_str(StringPiece k) = 0;
 protected:
  vector<PartitionInfo> partinfo_;
  boost::recursive_mutex &mutex() {
    return m_;
  }
  vector<LocalTable *> partitions_;
  vector<LocalTable *> cache_;

  map<int, LogFile*> checkpoint_files_;

  volatile int pending_writes_;
  boost::recursive_mutex m_;

  friend class TableServer;
  TableServer *w_;


  void set_worker(TableServer *w);

  // Fetch key k from the node owning it.  Returns true if the key exists.
  bool get_remote(int shard, const StringPiece &k, string *v);

  // send get request to the remote machine.
  void async_get_remote(int shard, const StringPiece &k);

  //  true if there're responses to GET request
  bool async_get_remote_collect(string *k, string *v);

  // true if the response is of the specified key
  bool async_get_remote_collect_key(int shard, const string &k, string *v);
};

template <class K, class V>
class TypedGlobalTable :
  public GlobalTable,
  public TypedTable<K, V>,
  private boost::noncopyable {
 public:
  virtual void Init(const TableDescriptor *tinfo) {
    GlobalTable::Init(tinfo);
    for (size_t i = 0; i < partitions_.size(); ++i) {
      partitions_[i] = create_local(i);
    }
    pending_writes_ = 0;
  }

  int get_shard(const K &k);
  int get_shard_str(StringPiece k);
  V get_local(const K &k);

  // Store the given key-value pair in this hash. If 'k' has affinity for a
  // remote thread, the application occurs immediately on the local host,
  // and the update is queued for transmission to the owner.
  void put(const K &k, const V &v);
  bool update(const K &k, const V &v);

  // Return the value associated with 'k', possibly blocking for a remote fetch.
  V get(const K &k);

  //  asynchronous get. if true (local data), then V contains the value
  //  if false, use has to call async_collect() until there's data (returning true)
  bool async_get(const K &k, V* v);
  bool async_get_collect(K* k, V* v);

  bool contains(const K &k);
  void remove(const K &k);

  TypedTable<K, V> *partition(int idx) {
    return dynamic_cast<TypedTable<K, V>*>(partitions_[idx]);
  }

 protected:
  LocalTable *create_local(int shard);
};


template<class K, class V>
int TypedGlobalTable<K, V>::get_shard(const K &k) {
  DCHECK(this != NULL);
  DCHECK(this->info().sharder != NULL);
  Sharder<K> *sharder = (Sharder<K> *)(this->info().sharder);
  int shard = (*sharder)(k, this->info().num_shards);
  DCHECK_GE(shard, 0);
  DCHECK_LT(shard, this->num_shards());
  return shard;
}

template<class K, class V>
int TypedGlobalTable<K, V>::get_shard_str(StringPiece k) {
  return get_shard(unmarshal(static_cast<Marshal<K>*>(this->info().key_marshal),
                             k));
}

template<class K, class V>
V TypedGlobalTable<K, V>::get_local(const K &k) {
  int shard = this->get_shard(k);
  return partition(shard)->get(k);
}

// Store the given key-value pair in this hash. If 'k' has affinity for a
// remote thread, the application occurs immediately on the local host,
// and the update is queued for transmission to the owner.
template<class K, class V>
void TypedGlobalTable<K, V>::put(const K &k, const V &v) {
	int shard = this->get_shard(k);
	  StringPiece key = marshal(static_cast<Marshal<K>*>(this->info().key_marshal), k);

	  int local_rank = NetworkThread::Get()->id();
	  // if local, create a TableData, then Send
	  if (is_local_shard(shard)){
		  TableData put;
	      put.set_shard(shard);
	      put.set_source(local_rank);
	      put.set_table(this->id());

	      StringPiece value = marshal(static_cast<Marshal<V>*>(this->info().value_marshal), v);

	      put.set_key(key.AsString());
	      Arg *a = put.add_kv_data();
	      a->set_key(key.data, key.len);
	      a->set_value(value.data, value.len);
	      put.set_done(true);
	      NetworkThread::Get()->Send(local_rank, MTYPE_PUT_REQUEST, put);//send locally
	  }
	  else{
	    partition(shard)->put(k,v);
	    SendPut();
	  }
}

template<class K, class V>
bool TypedGlobalTable<K, V>::update(const K &k, const V &v) {
  int shard = this->get_shard(k);
  StringPiece key = marshal(static_cast<Marshal<K>*>(this->info().key_marshal), k);

  int local_rank = NetworkThread::Get()->id();
  // if local, create a TableData, then Send
  if (is_local_shard(shard)){
	  TableData put;
      put.set_shard(shard);
      put.set_source(local_rank);
      put.set_table(this->id());

      StringPiece value = marshal(static_cast<Marshal<V>*>(this->info().value_marshal), v);

      put.set_key(key.AsString());
      Arg *a = put.add_kv_data();
      a->set_key(key.data, key.len);
      a->set_value(value.data, value.len);
      put.set_done(true);
      NetworkThread::Get()->Send(local_rank, MTYPE_UPDATE_REQUEST, put);//send locally
  }
  else{
    partition(shard)->put(k,v); 
    SendUpdates();
  }
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
	while (!async_get_remote_collect_key(shard,key,&v_str))
		Sleep(0.001);

	return unmarshal(static_cast<Marshal<V>*>(this->info().value_marshal), v_str);


	/*
  int shard = this->get_shard(k);
  string key = marshal(static_cast<Marshal<K>*>(this->info().key_marshal), k);

  if (is_local_shard(shard)) {
	RequestDispatcher::Get()->sync_local_get(key);
    V val =  get_local(k);
    RequestDispatcher::Get()->event_complete(key);
    return val;
  }

  string v_str;
  get_remote(shard,key, &v_str);
  return unmarshal(static_cast<Marshal<V>*>(this->info().value_marshal), v_str);
  */
}

/*
 * asynchronous get, returns right away. use async_get_collect to wait on the
 * content.
 *
 * if the key is local, return
 * else send send (get_remote without waiting).
 */
template<class K, class V>
bool TypedGlobalTable<K,V>::async_get(const K &k, V* v){
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
bool TypedGlobalTable<K,V>::async_get_collect(K* k, V* v){

	//ALWAYS remote

	string tk, tv;
	bool succeed = async_get_remote_collect(&tk, &tv);
	if (succeed){
		*k = unmarshal(static_cast<Marshal<K>*>(this->info().key_marshal), tk);
		*v = unmarshal(static_cast<Marshal<V>*>(this->info().value_marshal), tv);
	}
	return succeed;
}


template<class K, class V>
bool TypedGlobalTable<K, V>::contains(const K &k) {
  int shard = this->get_shard(k);
  if (is_local_shard(shard)) {
    //    boost::recursive_mutex::scoped_lock sl(mutex());
    return partition(shard)->contains(k);
  }
  string v_str;
  return get_remote(shard, marshal(static_cast<Marshal<K>*>(info_->key_marshal),
                                   k), &v_str);
}

template<class K, class V>
void TypedGlobalTable<K, V>::remove(const K &k) {
  LOG(FATAL) << "Not implemented!";
}

template<class K, class V>
LocalTable *TypedGlobalTable<K, V>::create_local(int shard) {
  TableDescriptor *linfo = new TableDescriptor(info());
  linfo->shard = shard;
  LocalTable *t = (LocalTable *)info_->partition_factory->New();
  t->Init(linfo);
  return t;
}

}  // namespace lapis

#endif /* INCLUDE_CORE_GLOBAL_TABLE_H_ */
