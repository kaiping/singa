#ifndef INCLUDE_CORE_COMMON_H_
#define INCLUDE_CORE_COMMON_H_

#include <time.h>
#include <vector>
#include <string>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <glog/logging.h>

#include "proto/common.pb.h"
#include "proto/worker.pb.h"
#include "utils/timer.h"

#include <unordered_map>
#include <unordered_set>
#include <deque>

#include <boost/type_traits.hpp>
#include <boost/utility/enable_if.hpp>

using std::map;
using std::vector;
using std::string;
using std::pair;
using std::make_pair;
using std::unordered_map;
using std::unordered_set;
using std::deque;

/**
 * @file common.h
 * Templates for common structs used by the tables and the network threads.
 * Stats is a simple struct for collecting and merging stats.
 *
 * BaseUpdateHandler is a template for customizing handle_get(), handle_update() and
 * handle_checkpoint().
 *
 * Marshall is the template for converting typed objects to and from string.
 *
 */
namespace singa {

/**
 * Simple wrapper for collecting stats. It contains the mapping of string->double.
 * Stats can be merged with another by merging the map keys.
 */
struct Stats {
	double& operator[](const string& key) {
		return p_[key];
	}

	/**
	 * Not implemented.
	 */
	string ToString(string prefix) {
		string out="";
		/*
		for (unordered_map<string, double>::iterator i = p_.begin();
				i != p_.end(); ++i) {
			out += StringPrintf("%s -- %s : %.2f\n", prefix.c_str(),
					i->first.c_str(), i->second);
		}*/
		return out;
	}

	/**
	 * Merge with other stats by adding the stats value of the same key.
	 */
	void Merge(Stats &other) {
		for (unordered_map<string, double>::iterator i = other.p_.begin();
				i != other.p_.end(); ++i) {
			p_[i->first] += i->second;
		}
	}
private:
	unordered_map<string, double> p_; /**< stats mapping string->double */
};

/**
 * This is used as an accumulator object in TableDescriptor.
 */
template<class K, class V>
struct BaseUpdateHandler {
	virtual bool Update(V *a, const V &b) = 0;
	virtual bool Get(const K &k, const V &v, V* ret) = 0;
	virtual bool CheckpointNow(const K&, const V&) = 0;
};

/**
 * Struct for mapping a key to the table shard ID.
 */
template<class K>
struct Sharder {
	virtual int operator()(const K &k, int shards) = 0;
};

/**
 * Implementation of Sharder struct for simple types.
 */
struct Sharding {
	struct String: public Sharder<string> {
		int operator()(const string &k, int shards) {
			std::hash<std::string> hash_fn;
			return hash_fn(k)%shards;
		}
	};

	struct Mod: public Sharder<int> {
		int operator()(const int &key, int shards) {
			return key % shards;
		}
	};

	struct UintMod: public Sharder<uint32_t> {
		int operator()(const uint32_t &key, int shards) {
			return key % shards;
		}
	};
};

/**
 * Base class, specifies the interface of request handlers of table server.
 */
class TableServerHandler{
 public:
  virtual void Setup(const SGDProto& sgd);
  virtual bool CheckpointNow(const TKey& key, const TVal& val);

  virtual bool Update(TVal* origin, const TVal& update)=0;
  virtual bool Get(const TKey& key, const TVal &from, TVal* to);
  virtual bool Put(const TKey& key, TVal* to, const TVal& from);

 protected:
  int checkpoint_after_, checkpoint_frequency_;
  bool synchronous_;
};

} // namespace singa

#endif  // INCLUDE_CORE_COMMON_H_
