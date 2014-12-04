// Copyright © 2014 Anh Dinh. All Rights Reserved.
// piccolo/global-table.cc
#ifndef INCLUDE_CORE_GLOBAL_TABLE_H_
#define INCLUDE_CORE_GLOBAL_TABLE_H_

#include <glog/logging.h>

#include "core/table.h"
#include "core/file.h"
#include "core/request_dispatcher.h"
#include "utils/common.h"

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


	/**
	 * Handle remote get request from another process. There are 3 steps:
	 * (1) check that the request is for the local shard
	 * (2) invoke user-define handle-get at the local shard
	 * (3) return true + TableData if the data can be returned right away
	 *     return false -> data is not ready to be sent back (the request
	 *                       is to be re-processed)
	 */
	bool HandleGet(const GetRequest &req, TableData *resp);



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

protected:
	vector<PartitionInfo> partinfo_; /**< shard information */

	vector<TableBase*> partitions_; /**< actual shards */

	map<int, LogFile*> checkpoint_files_;
};
}  // namespace lapis

#endif /* INCLUDE_CORE_GLOBAL_TABLE_H_ */
