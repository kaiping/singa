#include "core/global-table.h"
#include "core/table_server.h"
#include "utils/network_thread.h"
#include <gflags/gflags.h>
#include <glog/logging.h>

static const int kMaxNetworkPending = 1 << 26;
static const int kMaxNetworkChunk = 1 << 20;

DEFINE_bool(checkpoint_enabled, true, "enabling checkpoint");

namespace lapis {



GlobalTable::~GlobalTable() {
  for (size_t i = 0; i < partitions_.size(); ++i) {
    delete partitions_[i];
  }
}


bool GlobalTable::is_local_shard(int shard) {
  return owner(shard) == worker_id_;
}

bool GlobalTable::is_local_key(const StringPiece &k) {
  return is_local_shard(get_shard_str(k));
}

void GlobalTable::Init(const lapis::TableDescriptor *info) {
  TableBase::Init(info);
  worker_id_ = NetworkThread::Get()->id();
  partitions_.resize(info->num_shards);
  partinfo_.resize(info->num_shards);
}

int64_t GlobalTable::shard_size(int shard) {
	return is_local_shard(shard) ? partitions_[shard]->size : 0;
}

void GlobalTable::clear(int shard) {
	CHECK(is_local_shard(shard));
	partitions_[shard]->clear();
}

bool GlobalTable::empty() {
  for (size_t i = 0; i < partitions_.size(); ++i) {
    if (is_local_shard(i) && !partitions_[i]->empty()) {
      return false;
    }
  }
  return true;
}

void GlobalTable::resize(int64_t new_size) {
  for (size_t i = 0; i < partitions_.size(); ++i) {
    if (is_local_shard(i)) {
      partitions_[i]->resize(new_size / partitions_.size());
    }
  }
}

void GlobalTable::set_worker(TableServer *w) {
  worker_id_ = w->id();
}

bool GlobalTable::get_remote(int shard, const StringPiece &k, string *v) {
  HashGet req;
  TableData resp;
  req.set_key(k.AsString());
  req.set_table(info().table_id);
  req.set_shard(shard);
  req.set_source(worker_id_);
  int peer = owner(shard); // w_->peer_for_partition(info().table_id, shard);
  //VLOG(3)<<"get remote befor send to "<<peer<<" from "<<worker_id_;
  NetworkThread::Get()->Send(peer, MTYPE_GET_REQUEST, req);
  //VLOG(3)<<"get remote ater send"<<peer<<" from "<<worker_id_;
  NetworkThread::Get()->Read(peer, MTYPE_GET_RESPONSE, &resp);
  //VLOG(3)<<"get remote ater read"<<peer<<" from "<<worker_id_;
  if (resp.missing_key()) {
    return false;
  }
  *v = resp.kv_data(0).value();
  return true;
}

void GlobalTable::async_get_remote(int shard, const StringPiece &k){
	HashGet req;
	req.set_key(k.AsString());
	req.set_table(info().table_id);
	req.set_shard(shard);
	req.set_source(worker_id_);
	int peer = owner(shard);
	NetworkThread::Get()->Send(peer, MTYPE_GET_REQUEST, req);
}

// return false when there's no response with given key
bool GlobalTable::async_get_remote_collect_key(int shard, const string &k, string *v){
	TableData resp;
	int source = owner(shard);

	  if (NetworkThread::Get()->TryRead(source, MTYPE_GET_RESPONSE, &resp)){
		  if (k.compare(resp.kv_data(0).key())==0){
			  *v = resp.kv_data(0).value();
			  return true;
		  }
		  else{
			  //put back and return false
			  NetworkThread::Get()->send_to_local_rx_queue(source, MTYPE_GET_RESPONSE, resp);
			  return false;
		  }
	  }
	  else return false;
}

bool GlobalTable::async_get_remote_collect(string *k, string *v) {
  TableData resp;

  if (NetworkThread::Get()->TryRead(MPI::ANY_SOURCE, MTYPE_GET_RESPONSE, &resp)){
	  *k = resp.kv_data(0).key();
	  *v = resp.kv_data(0).value();
	   return true;
  }
  else return false;
}

bool GlobalTable::HandleGet(const HashGet &get_req, TableData *get_resp) {
  boost::recursive_mutex::scoped_lock sl(mutex());
  int shard = get_req.shard();
  if (!is_local_shard(shard)) {
    LOG_EVERY_N(WARNING, 1000) << "Not local for shard: " << shard;
  }
  LocalTable *t = (LocalTable *)partitions_[shard];
  if (!t->contains_str(get_req.key())) {
    get_resp->set_missing_key(true);
    return false;
  } else {
    Arg *kv = get_resp->add_kv_data();
    kv->set_key(get_req.key());
    string value = t->get_str(get_req.key());
    if (!value.empty()){
    	kv->set_value(value);
    	return true;
    }
    else return false;
  }
  //VLOG(3)<<"end of handle_get local";
}



void GlobalTable::send_update() {
  TableData put;
  for (size_t i = 0; i < partitions_.size(); ++i) {
    LocalTable *t = partitions_[i];
    if (!is_local_shard(i) && !t->empty()) {
      // Always send at least one chunk, to ensure that we clear taint on
      // tables we own.
      do {
        put.Clear();
        put.set_shard(i);
        //put.set_source(w_->id());
        put.set_source(worker_id_);
        put.set_table(id());
        NetworkTableCoder c(&put);
        t->Serialize(&c);
        t->clear();
        put.set_done(true);
        NetworkThread::Get()->Send(owner(i), MTYPE_UPDATE_REQUEST, put);
      } while (!t->empty());
      t->clear();
    }
  }
  pending_writes_ = 0;
}

void GlobalTable::send_put() {
  TableData put;
  for (size_t i = 0; i < partitions_.size(); ++i) {
    LocalTable *t = partitions_[i];
    if (!is_local_shard(i) && !t->empty()) {
      // Always send at least one chunk, to ensure that we clear taint on
      // tables we own.
      do {
        put.Clear();
        put.set_shard(i);
        //put.set_source(w_->id());
        put.set_source(worker_id_);
        put.set_table(id());
        NetworkTableCoder c(&put);
        t->Serialize(&c);
        t->clear();
        put.set_done(true);
        NetworkThread::Get()->Send(owner(i), MTYPE_PUT_REQUEST, put);
      } while (!t->empty());
      t->clear();
    }
  }
  pending_writes_ = 0;
}

Stats GlobalTable::stats(){
	Stats stats;
	for (size_t i = 0; i < partitions_.size(); ++i) {
		stats.Merge(partitions_[i]->stats());
	}
	return stats;
}

bool GlobalTable::ApplyUpdates(const lapis::TableData &req) {
  boost::recursive_mutex::scoped_lock sl(mutex());
  NetworkTableCoder c(&req);
  bool ret =  partitions_[req.shard()]->ApplyUpdates(&c, checkpoint_files_[req.shard()]);

  return ret;
}

bool GlobalTable::ApplyPut(const lapis::TableData &req) {
  boost::recursive_mutex::scoped_lock sl(mutex());
  NetworkTableCoder c(&req);
  bool ret =  partitions_[req.shard()]->ApplyPut(&c, checkpoint_files_[req.shard()]);

  return ret;
}

void GlobalTable::Restore(int shard){
	//only restore local shard
	if (is_local_shard(shard)){
		LogFile *logfile = (*checkpoint_files())[shard]; 
		int restored_size = logfile->read_latest_table_size(); 
		partitions_[shard]->restore(logfile, restored_size); 
	}		
}
}
