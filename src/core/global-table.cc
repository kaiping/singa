#include "core/global-table.h"
#include "core/table_server.h"
#include "utils/network_thread.h"
#include <gflags/gflags.h>
#include <glog/logging.h>

/**
 * @file global-table.cc
 * Implement generic, string table. @see global-table.h.
 */

/**
 * Checkpointing flag. True if checkpoint is enabled.
 */
DEFINE_bool(checkpoint_enabled, false, "enabling checkpoint");


namespace lapis {

/**
 * Delete the local shards upon destruction.
 */
GlobalTable::~GlobalTable() {
	for (size_t i = 0; i < partitions_.size(); ++i) {
		delete partitions_[i];
	}
}

void GlobalTable::Init(const lapis::TableDescriptor *info) {
	TableBase::Init(info);
	worker_id_ = NetworkThread::Get()->id();
	partitions_.resize(info->num_shards);
	partinfo_.resize(info->num_shards);
}

bool GlobalTable::is_local_shard(int shard) {
	return owner(shard) == worker_id_;
}

bool GlobalTable::is_local_key(const StringPiece &k) {
	return is_local_shard(get_shard_str(k));
}


int64_t GlobalTable::shard_size(int shard) {
	return is_local_shard(shard) ? partitions_[shard]->size() : 0;
}

void GlobalTable::clear(int shard) {
	CHECK(is_local_shard(shard));
	partitions_[shard]->clear();
}

bool GlobalTable::empty() {
	for (size_t i = 0; i < partitions_.size(); ++i)
		if (is_local_shard(i) && !partitions_[i]->empty())
			return false;
	return true;
}

void GlobalTable::resize(int64_t new_size) {
	for (size_t i = 0; i < partitions_.size(); ++i)
		if (is_local_shard(i))
			partitions_[i]->resize(new_size / partitions_.size());
}


bool GlobalTable::get_remote(int shard, const StringPiece &k, string *v) {
	HashGet req;
	TableData resp;
	req.set_key(k.AsString());
	req.set_table(info().table_id);
	req.set_shard(shard);
	req.set_source(worker_id_);
	int peer = owner(shard);
	NetworkThread::Get()->Send(peer, MTYPE_GET_REQUEST, req);
	NetworkThread::Get()->Read(peer, MTYPE_GET_RESPONSE, &resp);

	if (resp.missing_key())
		return false;
	*v = resp.kv_data(0).value();
	return true;
}

void GlobalTable::async_get_remote(int shard, const StringPiece &k) {
	HashGet req;
	req.set_key(k.AsString());
	req.set_table(info().table_id);
	req.set_shard(shard);
	req.set_source(worker_id_);
	int peer = owner(shard);
	NetworkThread::Get()->Send(peer, MTYPE_GET_REQUEST, req);
}

bool GlobalTable::async_get_remote_collect(string *k, string *v) {
	TableData resp;

	if (NetworkThread::Get()->TryRead(MPI::ANY_SOURCE, MTYPE_GET_RESPONSE,
			&resp)) {
		*k = resp.kv_data(0).key();
		*v = resp.kv_data(0).value();
		return true;
	} else	return false;
}

bool GlobalTable::async_get_remote_collect_key(int shard, const string &k,
		string *v) {
	TableData resp;
	int source = owner(shard);

	if (NetworkThread::Get()->TryRead(source, MTYPE_GET_RESPONSE, &resp)) {
		if (k.compare(resp.kv_data(0).key()) == 0) {
			*v = resp.kv_data(0).value();
			return true;
		} else {
			//put back and return false
			NetworkThread::Get()->send_to_local_rx_queue(source,
					MTYPE_GET_RESPONSE, resp);
			return false;
		}
	} else
		return false;
}


bool GlobalTable::HandleGet(const HashGet &get_req, TableData *get_resp) {
	int shard = get_req.shard();
	LocalTable *t = (LocalTable *) partitions_[shard];

	Arg *kv = get_resp->add_kv_data();
	string value = t->get_str(get_req.key());

	// empty value means that the data is not ready to be returned
	if (!value.empty()) {
		kv->set_key(get_req.key());
		kv->set_value(value);
		return true;
	} else
		return false;
}


bool GlobalTable::ApplyUpdates(const lapis::TableData &req) {
	NetworkTableCoder c(&req);
	bool ret = partitions_[req.shard()]->ApplyUpdates(&c,
			checkpoint_files_[req.shard()]);
	return ret;
}

bool GlobalTable::ApplyPut(const lapis::TableData &req) {
	NetworkTableCoder c(&req);
	bool ret = partitions_[req.shard()]->ApplyPut(&c,
			checkpoint_files_[req.shard()]);

	return ret;
}

void GlobalTable::Restore(int shard) {
	//only restore local shard
	if (is_local_shard(shard)) {
		LogFile *logfile = (*checkpoint_files())[shard];
		int restored_size = logfile->read_latest_table_size();
		partitions_[shard]->restore(logfile, restored_size);
	}
}

Stats GlobalTable::stats() {
	Stats stats;
	for (size_t i = 0; i < partitions_.size(); ++i) {
		stats.Merge(partitions_[i]->stats());
	}
	return stats;
}

}
