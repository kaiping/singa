#ifndef INCLUDE_CORE_GLOBALTABLE_H_
#define INCLUDE_CORE_GLOBALTABLE_H_

#include "core/table.h"
#include "core/local-table.h"

#include "core/file.h"
#include "core/rpc.h"

namespace lapis {

class Worker;

class GlobalTable :
  public TableBase,
  public Checkpointable {
public:
  virtual void Init(const TableDescriptor* tinfo);
  virtual ~GlobalTable();

  struct PartitionInfo {
    PartitionInfo() : dirty(false), tainted(false), owner(-1) {}
    bool dirty;
    bool tainted;
    int owner;
    ShardInfo sinfo;
  };

  virtual PartitionInfo* get_partition_info(int shard) {
    return &partinfo_[shard];
  }

  bool tainted(int shard) { return get_partition_info(shard)->tainted; }
  int owner(int shard) { return get_partition_info(shard)->owner; }

  LocalTable *get_partition(int shard);
  virtual TableIterator* get_iterator(int shard) = 0;

  bool is_local_shard(int shard);
  bool is_local_key(const StringPiece &k);

  // Fill in a response from a remote worker for the given key.
  void handle_get(const HashGet& req, TableData* resp);

  // Handle updates from the master or other workers.
  void SendUpdates();
  void ApplyUpdates(const TableData& req);
  void HandlePutRequests();
  void UpdatePartitions(const ShardInfo& sinfo);

  int pending_write_bytes();

  // Clear any local data for which this table has ownership.
  // Updates waiting to be sent to other workers are *not* cleared.
  void clear(int shard);
  bool empty();
  void resize(int64_t new_size);

  int worker_id_;

  virtual void start_checkpoint(const string& f);
  virtual void write_delta(const TableData& d);
  virtual void finish_checkpoint();
  virtual void restore(const string& f);

  virtual int64_t shard_size(int shard);

  virtual int get_shard_str(StringPiece k) = 0;
protected:
  vector<PartitionInfo> partinfo_;
  boost::recursive_mutex& mutex() { return m_; }
  vector<LocalTable*> partitions_;
  vector<LocalTable*> cache_;

  volatile int pending_writes_;
  boost::recursive_mutex m_;

  friend class Worker;
  Worker *w_;


  void set_worker(Worker *w);

  // Fetch the given key, using only local information.
  void get_local(const StringPiece &k, string *v);

  // Fetch key k from the node owning it.  Returns true if the key exists.
  bool get_remote(int shard, const StringPiece &k, string* v);
};

template <class K, class V>
class TypedGlobalTable :
  public GlobalTable,
  public TypedTable<K, V>,
  private boost::noncopyable {
public:
  typedef TypedTableIterator<K, V> Iterator;
  virtual void Init(const TableDescriptor *tinfo) {
    GlobalTable::Init(tinfo);
    for (int i = 0; i < partitions_.size(); ++i) {
      partitions_[i] = create_local(i);
    }

    pending_writes_ = 0;
  }

  int get_shard(const K& k);
  int get_shard_str(StringPiece k);
  V get_local(const K& k);

  // Store the given key-value pair in this hash. If 'k' has affinity for a
  // remote thread, the application occurs immediately on the local host,
  // and the update is queued for transmission to the owner.
  void put(const K &k, const V &v);
  void update(const K &k, const V &v);

  // Return the value associated with 'k', possibly blocking for a remote fetch.
  V get(const K &k);
  bool contains(const K &k);
  void remove(const K &k);
  TableIterator* get_iterator(int shard);
  TypedTable<K, V>* partition(int idx) {
    return dynamic_cast<TypedTable<K, V>* >(partitions_[idx]);
  }

  virtual TypedTableIterator<K, V>* get_typed_iterator(int shard) {
    return static_cast<TypedTableIterator<K, V>* >(get_iterator(shard));
  }

protected:
  LocalTable* create_local(int shard);
};

static const int kWriteFlushCount = 1000000;

template<class K, class V>
class RemoteIterator : public TypedTableIterator<K, V> {
public:
  RemoteIterator(GlobalTable *table, int shard) :
    owner_(table), shard_(shard), done_(false) {
    request_.set_table(table->id());
    request_.set_shard(shard_);
    int target_worker = table->get_partition_info(shard)->owner;

    NetworkThread::Get()->Send(target_worker+1, MTYPE_ITERATOR_REQ, request_);
    NetworkThread::Get()->Read(target_worker+1, MTYPE_ITERATOR_RESP, &response_);

    request_.set_id(response_.id());
  }

  void key_str(string *out) {
    *out = response_.key();
  }

  void value_str(string *out) {
    *out = response_.value();
  }

  bool done() {
    return response_.done();
  }

  void Next() {
    int target_worker = owner_->get_partition_info(shard_)->owner;
    NetworkThread::Get()->Send(target_worker+1, MTYPE_ITERATOR_REQ, request_);
    NetworkThread::Get()->Read(target_worker+1, MTYPE_ITERATOR_RESP, &response_);
    ++index_;
  }

  const K& key() {
    ((Marshal<K>*)(owner_->info().key_marshal))->unmarshal(response_.key(), &key_);
    return key_;
  }

  V& value() {
    ((Marshal<V>*)(owner_->info().value_marshal))->unmarshal(response_.value(), &value_);
    return value_;
  }

private:
  GlobalTable* owner_;
  IteratorRequest request_;
  IteratorResponse response_;
  int id_;

  int shard_;
  int index_;
  K key_;
  V value_;
  bool done_;
};


template<class K, class V>
int TypedGlobalTable<K, V>::get_shard(const K& k) {
  DCHECK(this != NULL);
  DCHECK(this->info().sharder != NULL);

  Sharder<K> *sharder = (Sharder<K>*)(this->info().sharder);
  int shard = (*sharder)(k, this->info().num_shards);
  DCHECK_GE(shard, 0);
  DCHECK_LT(shard, this->num_shards());
  return shard;
}

template<class K, class V>
int TypedGlobalTable<K, V>::get_shard_str(StringPiece k) {
  return get_shard(unmarshal(static_cast<Marshal<K>* >(this->info().key_marshal), k));
}

template<class K, class V>
V TypedGlobalTable<K, V>::get_local(const K& k) {
  int shard = this->get_shard(k);

  CHECK(is_local_shard(shard)) << " non-local for shard: " << shard;

  return partition(shard)->get(k);
}

// Store the given key-value pair in this hash. If 'k' has affinity for a
// remote thread, the application occurs immediately on the local host,
// and the update is queued for transmission to the owner.
template<class K, class V>
void TypedGlobalTable<K, V>::put(const K &k, const V &v) {
  LOG(FATAL) << "Need to implement.";
  int shard = this->get_shard(k);

  //  boost::recursive_mutex::scoped_lock sl(mutex());
  partition(shard)->put(k, v);

  if (!is_local_shard(shard)) {
    ++pending_writes_;
  }

  if (pending_writes_ > kWriteFlushCount) {
    SendUpdates();
  }

  PERIODIC(0.1, {this->HandlePutRequests();});
}

template<class K, class V>
void TypedGlobalTable<K, V>::update(const K &k, const V &v) {

  int shard = this->get_shard(k);
  //std::cout << "Worker " << worker_id_ <<" writes " << k << " to shard " << shard << endl;
  //  boost::recursive_mutex::scoped_lock sl(mutex());
  partition(shard)->update(k, v);

//  LOG(INFO) << "local: " << k << " : " << is_local_shard(shard) << " : " << worker_id_;
  if (!is_local_shard(shard)) {
    ++pending_writes_;
    //std::cout << "... to remote worker " << endl;
  }

  if (pending_writes_ > kWriteFlushCount) {
    SendUpdates();

  }
  PERIODIC(0.1, {this->HandlePutRequests();});
}

// Return the value associated with 'k', possibly blocking for a remote fetch.
template<class K, class V>
V TypedGlobalTable<K, V>::get(const K &k) {
  int shard = this->get_shard(k);

  // If we received a get for this shard; but we haven't received all of the
  // data for it yet. Continue reading from other workers until we do.
  while (tainted(shard)) {
    this->HandlePutRequests();
    sched_yield();
  }

  PERIODIC(0.1, this->HandlePutRequests());

  if (is_local_shard(shard)) {
    //    boost::recursive_mutex::scoped_lock sl(mutex());
    return partition(shard)->get(k);
  }

  string v_str;
  get_remote(shard,
             marshal(static_cast<Marshal<K>* >(this->info().key_marshal), k),
             &v_str);
  return unmarshal(static_cast<Marshal<V>* >(this->info().value_marshal), v_str);
}

template<class K, class V>
bool TypedGlobalTable<K, V>::contains(const K &k) {
  int shard = this->get_shard(k);

  // If we received a request for this shard; but we haven't received all of the
  // data for it yet. Continue reading from other workers until we do.
  while (tainted(shard)) {
    this->HandlePutRequests();
    sched_yield();
  }

  if (is_local_shard(shard)) {
    //    boost::recursive_mutex::scoped_lock sl(mutex());
    return partition(shard)->contains(k);
  }

  string v_str;
  return get_remote(shard, marshal(static_cast<Marshal<K>* >(info_->key_marshal), k), &v_str);
}

template<class K, class V>
void TypedGlobalTable<K, V>::remove(const K &k) {
  LOG(FATAL) << "Not implemented!";
}

template<class K, class V>
LocalTable* TypedGlobalTable<K, V>::create_local(int shard) {
  TableDescriptor *linfo = new TableDescriptor(info());
  linfo->shard = shard;
  LocalTable* t = (LocalTable*)info_->partition_factory->New();
  t->Init(linfo);
  return t;
}

template<class K, class V>
TableIterator* TypedGlobalTable<K, V>::get_iterator(int shard) {
  if (this->is_local_shard(shard)) {
    return (TypedTableIterator<K, V>*) partitions_[shard]->get_iterator();
  } else {
    return new RemoteIterator<K, V>(this, shard);
  }
}

}  // namespace lapis

#endif /* INCLUDE_CORE_GLOBALTABLE_H_ */
