//  Copyright © 2014 Anh Dinh. All Rights Reserved.
//  piccolo/common.h

//  this header file is included in all upper-layer code
//  InitServers() is called before any table related operations.

#ifndef INCLUDE_CORE_COMMON_H_
#define INCLUDE_CORE_COMMON_H_

#include <time.h>
#include <vector>
#include <string>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "proto/common.pb.h"
#include "core/hash.h"
#include "core/static-initializers.h"
#include "core/stringpiece.h"
#include "core/timer.h"

#include <tr1/unordered_map>
#include <tr1/unordered_set>

#include <boost/type_traits.hpp>
#include <boost/utility/enable_if.hpp>

using std::map;
using std::vector;
using std::string;
using std::pair;
using std::make_pair;
using std::tr1::unordered_map;
using std::tr1::unordered_set;

namespace lapis {

//  start servers on MPI process, either memory server of manager
//  called once for every process, NULL is returned if
//  the current process is not the manager.
void InitServers(int argc, char** argv);

//  called at the end.
void Finish();

//  true if the current process is the memory manager,
//	false if it is a memory server
bool IsDistributedMemoryManager();

void Sleep(double t);

template <class V>
struct Accumulator {
  virtual void Accumulate(V* a, const V& b) = 0;
};

template <class K>
struct Sharder {
  virtual int operator()(const K& k, int shards) = 0;
};

//#ifndef SWIG
// Commonly used accumulation operators.
//#endif

template <class T, class Enable = void>
struct Marshal {
  virtual void marshal(const T& t, string* out) {
    GOOGLE_GLOG_COMPILE_ASSERT(std::tr1::is_pod<T>::value, Invalid_Value_Type);
    out->assign(reinterpret_cast<const char*>(&t), sizeof(t));
  }

  virtual void unmarshal(const StringPiece& s, T *t) {
    GOOGLE_GLOG_COMPILE_ASSERT(std::tr1::is_pod<T>::value, Invalid_Value_Type);
    *t = *reinterpret_cast<const T*>(s.data);
  }
};

template <class T>
struct Marshal<T, typename boost::enable_if<boost::is_base_of<string, T> >::type>  {
  void marshal(const string& t, string *out) { *out = t; }
  void unmarshal(const StringPiece& s, string *t) { t->assign(s.data, s.len); }
};


template <class T>
struct Marshal<T, typename boost::enable_if<boost::is_base_of<google::protobuf::Message, T> >::type> {
  void marshal(const google::protobuf::Message& t, string *out) { t.SerializePartialToString(out); }
  void unmarshal(const StringPiece& s, google::protobuf::Message* t) { t->ParseFromArray(s.data, s.len); }
};

template <class T>
string marshal(Marshal<T>* m, const T& t) { string out; m->marshal(t, &out); return out; }

template <class T>
T unmarshal(Marshal<T>* m, const StringPiece& s) { T out; m->unmarshal(s, &out); return out; }

}
#include "core/tuple.h"
#endif  // INCLUDE_CORE_COMMON_H_
