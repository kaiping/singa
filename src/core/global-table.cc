#include "core/global-table.h"
#include "utils/network_service.h"
#include "core/shard.h"
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
	worker_id_ = NetworkService::Get()->id();
	partitions_.resize(info->num_shards);
	partinfo_.resize(info->num_shards);

	for (int i = 0; i < info->num_shards; ++i) {
		Shard *t =
				(Shard *) info->partition_factory->New();
		t->Init(info);
		partitions_[i] = t;
		partinfo_[i].owner = i;
	}
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


bool GlobalTable::HandleGet(GetRequest &get_req, TableData *get_resp) {
	int shard = get_req.shard();
	Shard *t = partitions_[shard];

	TKey *key = get_req.mutable_key();
	TVal &val = t->get(*key);

	if (((TableServerHandler*) info_->handler)->Get(*key, val, get_resp->mutable_value())) {
		(get_resp->mutable_key())->CopyFrom(*key);
		return true;
	} else
		return false;
}


bool GlobalTable::ApplyUpdates(int shard, lapis::TableData &req) {
	bool ret = partitions_[shard]->ApplyUpdates(req,
			checkpoint_files_[shard]);
	return ret;
}

bool GlobalTable::ApplyPut(int shard, lapis::TableData &req) {
	bool ret = partitions_[shard]->ApplyPut(req,
			checkpoint_files_[shard]);

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
