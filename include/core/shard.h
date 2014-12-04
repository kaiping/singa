// Copyright © 2014 Anh Dinh. All Rights Reserved.
// modified from piccolo/spare-table.h
#ifndef INCLUDE_CORE_SPARSE_TABLE_H_
#define INCLUDE_CORE_SPARSE_TABLE_H_
#include <functional>
#include "core/common.h"
#include "proto/worker.pb.h"
#include "core/global-table.h"
#include "proto/model.pb.h"
#include <boost/noncopyable.hpp>
#include "core/file.h"
/**
 * @file shard.h
 * Represent the local data shard, storing <TKey,TVal> tuples.
 */

namespace std {

/**
 * Definition of hash value for TKey. This override the std::hash<TKey> method
 */
template<>
struct hash<lapis::TKey> {
	size_t operator()(const lapis::TKey& k) const {
		hash<int> hashint;
		return hashint(k.id());
	}
};
}  // namespace std


namespace lapis {

class Shard: public GlobalTable,
		private boost::noncopyable {
private:
	/**
	 * A bucket/slot in the hash table.
	 */
//#pragma pack(push, 1)
	struct Bucket {
		TKey k; /**< key */
		TVal v; /**< value */
		bool in_use; /**< if the current bucket is empty (can insert new data)*/
	};
//#pragma pack(pop)

public:
	/**
	 * TableFactory to instantiate the local table for storing shards.
	 * @see global-table.h
	 */
	struct Factory: public TableFactory {
		TableBase *New() {
			return new Shard();
		}
	};


	Shard(int size = 1);
	~Shard() {}

	/**
	 * Initialize the table.
	 * @param *td pointer to TableDescriptor containing the table ID and (un)marshall objects.
	 */
	void Init(const TableDescriptor *td) {
		TableBase::Init(td);
	}

	TVal get(const TKey &k);
	bool contains(const TKey &k);
	void put(const TKey &k, const TVal &v);
	bool update(const TKey &k, const TVal &v);

	void resize(int64_t size);

	bool empty() {return size() == 0;}

	int64_t size() {return entries_;} /**< current number of entries */

	/**
	 * Clear the table. All current buckets are kept, only the in_use fields
	 * are set to false.
	 */
	void clear() {
		for (int i = 0; i < size_; ++i) {
			buckets_[i].in_use = 0;
		}
		entries_ = 0;
	}

	bool ApplyUpdates(TableData &in, LogFile *logfile);
	bool ApplyPut(TableData &in, LogFile *logfile);

	/**
	 * Restore the table content from the specified checkpoint file.
	 *
	 * @param *logfile the checkpoint file storing the table content.
	 * @param desired_size how many tuples to restore
	 */
	void restore(LogFile *logfile, int desired_size);

private:

	/**
	 * Returns the hash of a given key.
	 */
	uint32_t bucket_idx(TKey k) {
		return hashobj_(k) % size_;
	}


	/**
	 * Returns the bucket index for the given key.
	 * Use linear probing.
	 * @return bucket index, -1 if cannot find the bucket.
	 */
	int bucket_for_key(const TKey &k) {
		int start = bucket_idx(k);
		int b = start;
		do {
			if (buckets_[b].in_use) {
				if (buckets_[b].k == k) {
					return b;
				}
			} else {
				return -1;
			}
			b = (b + 1) % size_;
		} while (b != start);
		return -1;
	}

	std::vector<Bucket> buckets_; /**< data buckets */

	int64_t entries_; /**< number of entries, <= size_ */
	int64_t size_; /**< table's size */

	double kLoadFactor;

	std::hash<TKey> hashobj_; /**< hash function for the key type K */
};
} // namespace lapis

#endif  // INCLUDE_CORE_SPARSE_TABLE_H_
