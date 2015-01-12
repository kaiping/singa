#ifndef INCLUDE_CORE_GLOBAL_TABLE_H_
#define INCLUDE_CORE_GLOBAL_TABLE_H_

#include <glog/logging.h>

#include "core/table.h"
#include "core/file.h"
#include "core/request_dispatcher.h"
#include "utils/common.h"
#include "proto/worker.pb.h"

/**
 * @file global-table.h
 * The GlobalTable class Represents global view of the table, i.e. each table server process
 * treats it as a local table to which it can put/get/update data. Locally, each table maintains
 * a number of partitions of Shard objects. The worker can determine by the key which shard the key
 * belongs.
 *
 * The table maintains a lists of key-value tuples where key is of type TKey and value of TVal.
 * The current implementation assumes table servers run on different processes to workers, hence
 * table access requests will be remote.
 */
namespace singa {

class Shard;

/**
 * The GlobalTable class for implementing table interface.
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
	 * to decide where to forward the request.
	 *
	 * For now, this is not used because the request will be send to the correct owner
	 * by simply hashing the key take modulo M (where M is the number of table servers).
	 */
	struct PartitionInfo {
		PartitionInfo() :owner(-1) {}
		int owner;
	};

	/**
	 * Get the process ID (rank) where the shard is maintained.
	 */
	int owner(int shard) {
		return partinfo_[shard].owner;
	}

	bool is_local_shard(int shard) {return owner(shard) == worker_id_;}


	/**
	 * Handle remote get request from another process. There are 3 steps:
	 * (1) get value from the local shard.
	 * (2) invoke user-define handle-get on the returned value.
	 * (3) return true + TableData if the data can be returned right away.
	 *     return false -> data is not ready to be sent back (the request
	 *                       is to be re-processed).
	 */
	bool HandleGet(GetRequest &req, TableData *resp);



	/**
	 * Apply update requests from another process.
	 *
	 * @return true if the operation is applied successfully.
	 */
	bool ApplyUpdates(int shard, TableData &req);

	/**
	 * Apply put requests from another process. Always return true.
	 */
	bool ApplyPut(int shard, TableData &req);

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

	vector<Shard*> partitions_; /**< actual shards */

	map<int, LogFile*> checkpoint_files_;
};
}  // namespace singa

#endif /* INCLUDE_CORE_GLOBAL_TABLE_H_ */
