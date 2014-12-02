// Copyright © 2014 Anh Dinh. All Rights Reserved.
// modified from piccolo/spare-table.h
#ifndef INCLUDE_CORE_SPARSE_TABLE_H_
#define INCLUDE_CORE_SPARSE_TABLE_H_
#include <functional>
#include "core/common.h"
#include "proto/worker.pb.h"
#include "core/table.h"
#include "proto/model.pb.h"
#include "local-table.h"
#include <boost/noncopyable.hpp>

/**
 * @file sparse-table.h
 * Template definition of typed tables storing data shard. Such local tables are
 * maintained by global-table which invokes put/get/update methods on the specific types.
 *
 * The sparse table is a hash table (hash is computed on the keys, and can be specified by
 * user - example of TKey hash below).The table automatically expands to accommodate new tuples.
 *
 * When data is put/get/update, it first invokes the user-specified handler (BaseUpdateHandler),
 * and only proceeds when the handler returns true. This mechanism is used for
 * customizing consistency model.
 *
 * Checkpointing is also implemented here: new data is appended to the log-file after user-defined
 * checkpoint handler returns true.
 *
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


//DECLARE_bool(checkpoint_enabled);

namespace lapis {

static const double kLoadFactor = 0.8; /**< threshold after which the table is resized*/

/**
 * A template for generate SparseTable class for storing local data shard.
 * Template arguments are the key and value types.
 */
template<class K, class V>
class SparseTable: public LocalTable,
		public TypedTable<K, V>,
		private boost::noncopyable {
private:
	/**
	 * A bucket/slot in the hash table.
	 */
//#pragma pack(push, 1)
	struct Bucket {
		K k; /**< key */
		V v; /**< value */
		bool in_use; /**< if the current bucket is empty (can insert new data)*/
	};
//#pragma pack(pop)

public:
	/**
	 * An iterator to access all members of the table. Use Next() to move to the next
	 * element, untils done() returns true.
	 */
	struct Iterator {
		Iterator(SparseTable<K, V> &parent) :
				pos(-1), parent_(parent) {
			Next();
		}

		void Next() {
			do {
				++pos;
			} while (pos < parent_.size_ && !parent_.buckets_[pos].in_use);
		}

		bool done() {
			return pos == parent_.size_;
		}

		const K &key() {
			return parent_.buckets_[pos].k;
		}

		V &value() {
			return parent_.buckets_[pos].v;
		}

		/**
		 * Return the generic string representation of key.
		 * @param the returned key string
		 */
		void key_str(string *k) {
			return ((Marshal<K> *) parent_.info_->key_marshal)->marshal(key(),
					k);
		}

		/**
		 * Return the generic string representation of value.
		 */
		void value_str(string *v) {
			return ((Marshal<V> *) parent_.info_->value_marshal)->marshal(
					value(), v);
		}

		int pos; /**< current read position */
		SparseTable<K, V> &parent_; /**< the current table */
	};

	/**
	 * TableFactory to instantiate the local table for storing shards.
	 * @see global-table.h
	 */
	struct Factory: public TableFactory {
		TableBase *New() {
			return new SparseTable<K, V>();
		}
	};


	SparseTable(int size = 1);
	~SparseTable() {}

	/**
	 * Initialize the table.
	 * @param *td pointer to TableDescriptor containing the table ID and (un)marshall objects.
	 */
	void Init(const TableDescriptor *td) {
		TableBase::Init(td);
	}

	V get(const K &k);
	bool contains(const K &k);
	void put(const K &k, const V &v);
	bool update(const K &k, const V &v);
	void remove(const K &k) {
		LOG(FATAL) << "Not implemented.";
	}

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

	Iterator *get_iterator() {return new Iterator(*this);} /**< the table's iterator */

	void Serialize(TableCoder *out){} //not implemented
	bool ApplyUpdates(TableCoder *in, LogFile *logfile);
	bool ApplyPut(TableCoder *in, LogFile *logfile);

	/**
	 * Restore the table content from the specified checkpoint file.
	 *
	 * @param *logfile the checkpoint file storing the table content.
	 * @param desired_size how many tuples to restore
	 */
	void restore(LogFile *logfile, int desired_size);


	/**
	 * Definition of the get_str() method from TypedTable. It first marshalls the string
	 * into correct key type, then invokes the get(K) method.
	 *
	 * Before returning, it invokes user-defined get handler (BaseUpdateHandler->Get()) and
	 * returns the value from that handler (empty string if the handler returns false).
	 *
	 * @see table.h
	 */
	string get_str(const string &s) {
		K k;
		V v;
		((Marshal<K> *) info_->key_marshal)->unmarshal(s, &k);
		string out;
		v = get(k);
		V ret;
		if (((BaseUpdateHandler<K, V> *) info_->accum)->Get(k, v, &ret)) {
			((Marshal<V> *) info_->value_marshal)->marshal(ret, &out);
			return out;
		} else
			return out; //empty string
	}

	// not implemented
	void update_str(const string &kstr, const string &vstr) {}

private:

	/**
	 * Returns the hash of a given key.
	 */
	uint32_t bucket_idx(K k) {
		return hashobj_(k) % size_;
	}


	/**
	 * Returns the bucket index for the given key.
	 * Use linear probing.
	 * @return bucket index, -1 if cannot find the bucket.
	 */
	int bucket_for_key(const K &k) {
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

	std::hash<K> hashobj_; /**< hash function for the key type K */
};


/**
 * Initialize the table with specific size (default = 0).
 */
template<class K, class V>
SparseTable<K, V>::SparseTable(int size) :
		buckets_(0), entries_(0), size_(0) {
	clear();
	resize(size);
}


/**
 * Update the table with content from the TableData message.
 *
 * It first convert TableData key-value to the correct types, then
 * performs the update. If successful, it also checkpoints the new content
 * if the user-specific checkpoint handler returns true.
 *
 * @return true if the update is successfull. On returning false, the update
 * request should be re-processed.
 */
template<class K, class V>
bool SparseTable<K, V>::ApplyUpdates(TableCoder *in, LogFile *logfile) {
	K k;
	V v;
	string kt, vt;
	//only 1 entry
	while (in->ReadEntry(&kt, &vt)) {
		((Marshal<K> *) info_->key_marshal)->unmarshal(kt, &k);
		((Marshal<V> *) info_->value_marshal)->unmarshal(vt, &v);
		bool ret = update(k, v);
		//if (ret && FLAGS_checkpoint_enabled)
			if (((BaseUpdateHandler<K, V> *) info_->accum)->is_checkpointable(k,
					v)) {
				V ret_v = get(k);
				string xv;
				((Marshal<V>*) info_->key_marshal)->marshal(ret_v, &xv);
				logfile->append(kt, xv, size());
			}
		return ret;
	}
	return false;
}


/**
 * Insert data to the table. @see ApplyUpdates
 */
template <class K, class V>
bool SparseTable<K, V>::ApplyPut(TableCoder *in, LogFile *logfile) {
  K k;
  V v;
  string kt, vt;
  //only 1 entry
  while (in->ReadEntry(&kt, &vt)) {
    ((Marshal<K> *)info_->key_marshal)->unmarshal(kt, &k);
    ((Marshal<V> *)info_->value_marshal)->unmarshal(vt, &v);
    put(k, v);
    //if (FLAGS_checkpoint_enabled)
      if (((BaseUpdateHandler<K, V> *)info_->accum)->is_checkpointable(k, v)){
        logfile->append(kt, vt, size());
      }
  }
  return true;
}

/**
 * Restore the table with content from the checkpoint file.
 * The checkpoint file is scanned backward and elements are inserted to the table.
 * This process ends when the certain number of tuples is reached.
 */
template<class K, class V>
void SparseTable<K, V>::restore(LogFile *logfile, int desired_size) {
	int tmp;
	K k;
	V v;
	while (size() != desired_size) {
		logfile->previous_entry(&k, &v, &tmp);
		if (bucket_for_key(k) == -1)
			put(k, v);
	}

}

/**
 * Resize the current table. Simply copy content of the old table
 * to the new table.
 */
template<class K, class V>
void SparseTable<K, V>::resize(int64_t size) {
	if (size_ == size)
		return;
	std::vector < Bucket > old_b = buckets_;
	int old_entries = entries_;
	buckets_.resize(size);
	size_ = size;
	clear();
	for (size_t i = 0; i < old_b.size(); ++i) {
		if (old_b[i].in_use) {
			put(old_b[i].k, old_b[i].v);
		}
	}
	CHECK_EQ(old_entries, entries_);
}

template<class K, class V>
bool SparseTable<K, V>::contains(const K &k) {
	return bucket_for_key(k) != -1;
}

template<class K, class V>
V SparseTable<K, V>::get(const K &k) {
	int b = bucket_for_key(k);
	return buckets_[b].v;
}

/**
 * Update table with a new tuple. If the entry is found, it invokes the user-defined
 * update handler (BaseUpdateHandler->Update()), giving it pointer to the current V
 * and value of the new V.
 *
 * If the entry is not found, simply insert it to the table.
 *
 * @return whatever the update handler returns.
 */
template<class K, class V>
bool SparseTable<K, V>::update(const K &k, const V &v) {
	int b = bucket_for_key(k);
	if (b != -1) {
		return ((BaseUpdateHandler<K, V> *) info_->accum)->Update(
				&buckets_[b].v, v);
	} else {
		put(k, v);
		string xk, xv;
		((Marshal<K>*) info_->key_marshal)->marshal(k, &xk);
		((Marshal<V>*) info_->value_marshal)->marshal(v, &xv);
		//VLOG(3)<< "Table " << NetworkThread::Get()->id() << " put " << (xk.size()+xv.size());
		stats_["TABLE_SIZE"] += xk.size() + xv.size();
		return true;
	}
}

/**
 * Insert new tuple to the table. First, it finds the bucket index for the key
 * (via linear probing). If the key already exist (there is a valid entry in the bucket),
 * it simply replace the old value with the new one.
 *
 * If the key is new, it inserts new value into the bucket, and set the in_use flag to true.
 */
template<class K, class V>
void SparseTable<K, V>::put(const K &k, const V &v) {
	int start = bucket_idx(k);
	int b = start;
	bool found = false;
	// find the bucket index
	do {
		if (!buckets_[b].in_use) {
			break;
		}
		if (buckets_[b].k == k) {
			found = true;
			break;
		}
		b = (b + 1) % size_;
	} while (b != start);

	// Inserting a new entry:
	if (!found) {
		if (entries_ > size_ * kLoadFactor) {
			// resize to 2*current_size + 1
			resize((int) (1 + size_ * 2));
			put(k, v);
		} else {
			buckets_[b].in_use = 1;
			buckets_[b].k = k;
			buckets_[b].v = v;
			++entries_;
		}
	} else {
		// Replacing an existing entry
		buckets_[b].v = v;
	}
}

} // namespace lapis

#endif  // INCLUDE_CORE_SPARSE_TABLE_H_
