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

#include "proto/common.pb.h"
#include "utils/hash.h"
#include "core/static-initializers.h"
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

namespace lapis {

template <class V>
struct Accumulator {
  virtual void Accumulate(V *a, const V &b) = 0;
};

template <class K>
struct Sharder {
  virtual int operator()(const K &k, int shards) = 0;
};

//#ifndef SWIG
// Commonly used accumulation operators.
//#endif

template <class T, class Enable = void>
struct Marshal {
  virtual void marshal(const T &t, string *out) {
    //GOOGLE_GLOG_COMPILE_ASSERT(std::tr1::is_pod<T>::value, Invalid_Value_Type);
    out->assign(reinterpret_cast<const char *>(&t), sizeof(t));
  }

  virtual void unmarshal(const StringPiece &s, T *t) {
    //GOOGLE_GLOG_COMPILE_ASSERT(std::tr1::is_pod<T>::value, Invalid_Value_Type);
    *t = *reinterpret_cast<const T *>(s.data);
  }
};

template <class T>
struct Marshal<T, typename boost::enable_if<boost::is_base_of<string, T>>::type>  {
  void marshal(const string &t, string *out) {
    *out = t;
  }
  void unmarshal(const StringPiece &s, string *t) {
    t->assign(s.data, s.len);
  }
};


template <class T>
struct Marshal<T, typename boost::enable_if<boost::is_base_of<google::protobuf::Message, T>>::type> {
  void marshal(const google::protobuf::Message &t, string *out) {
    t.SerializePartialToString(out);
  }
  void unmarshal(const StringPiece &s, google::protobuf::Message *t) {
    t->ParseFromArray(s.data, s.len);
  }
};

template <class T>
string marshal(Marshal<T> *m, const T &t) {
  string out;
  m->marshal(t, &out);
  return out;
}

template <class T>
T unmarshal(Marshal<T> *m, const StringPiece &s) {
  T out;
  m->unmarshal(s, &out);
  return out;
}

}

#include "utils/tuple.h"
#endif  // INCLUDE_CORE_COMMON_H_
