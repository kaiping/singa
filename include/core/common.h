//  Copyright © 2014 Anh Dinh. All Rights Reserved.
//  piccolo/common.h
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
#include "utils/hash.h"
//#include "core/static-initializers.h"
#include "utils/stringpiece.h"
#include "utils/timer.h"

#include <unordered_map>
#include <unordered_set>

#include <boost/type_traits.hpp>
#include <boost/utility/enable_if.hpp>

using std::map;
using std::vector;
using std::string;
using std::pair;
using std::make_pair;
using std::unordered_map;
using std::unordered_set;

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
namespace lapis {

/**
 * Simple wrapper for collecting stats. It contains the mapping of string->double.
 * Stats can be merged with another by merging the map keys.
 */
struct Stats {
	double& operator[](const string& key) {
		return p_[key];
	}

	string ToString(string prefix) {
		string out;
		for (unordered_map<string, double>::iterator i = p_.begin();
				i != p_.end(); ++i) {
			out += StringPrintf("%s -- %s : %.2f\n", prefix.c_str(),
					i->first.c_str(), i->second);
		}
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
	virtual bool Get(const K k, const V &v, V* ret) = 0;
	virtual bool is_checkpointable(const K, const V) = 0;
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
			return StringPiece(k).hash() % shards;
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

template<class T, class Enable = void>
struct Marshal {
	virtual void marshal(const T &t, string *out) {
		out->assign(reinterpret_cast<const char *>(&t), sizeof(t));
	}

	virtual void unmarshal(const StringPiece &s, T *t) {
		*t = *reinterpret_cast<const T *>(s.data);
	}
};

template<class T>
struct Marshal<T, typename boost::enable_if<boost::is_base_of<string, T>>::type> {
	void marshal(const string &t, string *out) {
		*out = t;
	}
	void unmarshal(const StringPiece &s, string *t) {
		t->assign(s.data, s.len);
	}
};


template<class T>
struct Marshal<T,
		typename boost::enable_if<
				boost::is_base_of<google::protobuf::Message, T>>::type> {
	void marshal(const google::protobuf::Message &t, string *out) {
		t.SerializePartialToString(out);
	}
	void unmarshal(const StringPiece &s, google::protobuf::Message *t) {
		t->ParseFromArray(s.data, s.len);
	}
};

/**
 * Convert type T to string.
 * @param *m marshall struct.
 * @param t value of type T to be convert
 * @return string representation of t
 */
template<class T>
string marshal(Marshal<T> *m, const T &t) {
	string out;
	m->marshal(t, &out);
	return out;
}

/**
 * Convert string to type T.
 * @see marshall().
 */
template<class T>
T unmarshal(Marshal<T> *m, const StringPiece &s) {
	T out;
	m->unmarshal(s, &out);
	return out;
}

} // namespace lapis

#endif  // INCLUDE_CORE_COMMON_H_
