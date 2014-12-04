#include "core/file.h"
#include "core/shard.h"
#include "google/protobuf/message.h"

namespace lapis {
/**
 * Initialize the table with specific size (default = 0).
 */
Shard::Shard(int size) :
		buckets_(0), entries_(0), size_(0), kLoadFactor(0.8) {
	clear();
	resize(size);
}


/**
 * Update the table with content from the TableData message.
 *
 * It first extracts TKey and TVal objects, then performs the update.
 * If successful, it also checkpoints the new content if the user-specific
 * checkpoint handler returns true.
 *
 * @return true if the update is successfull. On returning false, the update
 * request should be re-processed.
 */
bool Shard::ApplyUpdates(TableData &in, LogFile *logfile) {
	TKey *key = in.mutable_key();
	TVal *val = in.mutable_val();
	bool ret = update(*key, *val);

	if (ret
			&& ((BaseUpdateHandler<TKey, TVal> *) info_->handler)->CheckpointNow(
					*key, *val)) {
		TVal new_val = get(*key);
		string kt, vt;
		key->SerializeToString(&kt);
		new_val.SerializeToString(&vt);
		logfile->append(kt, vt, size());
	}
	return ret;
}


/**
 * Insert data to the table. @see ApplyUpdates
 */
bool Shard::ApplyPut(TableData &in, LogFile *logfile) {
	TKey *key = in.mutable_key();
	TVal *val = in.mutable_value();
	put(*key, *val);

	if (((BaseUpdateHandler<TKey, TVal> *) info_->handler)->is_checkpointable(
			*key, *val)) {
		string kt, vt;
		key->SerializeToString(&kt);
		val->SerializeToString(&vt);
		logfile->append(kt, vt, size());
	}
	return true;
}

/**
 * Restore the table with content from the checkpoint file.
 * The checkpoint file is scanned backward and elements are inserted to the table.
 * This process ends when the certain number of tuples is reached.
 */
void Shard::restore(LogFile *logfile, int desired_size) {
	int tmp;
	TKey k;
	TVal v;
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
void Shard::resize(int64_t size) {
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
}

bool Shard::contains(const TKey &k) {
	return bucket_for_key(k) != -1;
}

TVal Shard::get(const TKey &k) {
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
bool Shard::update(const TKey &k, const TVal &v) {
	int b = bucket_for_key(k);
	if (b != -1) {
		return ((BaseUpdateHandler<TKey, TVal> *) info_->handler)->Update(
				&buckets_[b].v, v);
	} else {
		put(k, v);
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
void Shard::put(const TKey &k, const TVal &v) {
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

} //namespace lapis
