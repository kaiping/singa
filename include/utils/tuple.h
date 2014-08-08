// Copyright © 2014 Anh Dinh. All Rights Reserved.
// piccolo/tuple.h

#ifndef INCLUDE_CORE_TUPLE_H_
#define INCLUDE_CORE_TUPLE_H_
#include <vector>
#include "utils/hash.h"


#ifndef SWIG

namespace lapis {
template <class A, class B>
struct tuple2 {
  A a_; B b_;
  bool operator==(const tuple2 &o) const {
    return o.a_ == a_ && o.b_ == b_;
  }
};

template <class A, class B, class C>
struct tuple3 {
  A a_; B b_; C c_;
  bool operator==(const tuple3 &o) const {
    return o.a_ == a_ && o.b_ == b_ && o.c_ == c_;
  }
};

template <class A, class B, class C, class D>
struct tuple4 {
  A a_; B b_; C c_; D d_;
  bool operator==(const tuple4 &o) const {
    return o.a_ == a_ && o.b_ == b_ && o.c_ == c_ && o.d_ == d_;
  }
};

template<class A, class B>
inline tuple2<A, B> MP(A x, B y) {
  tuple2<A, B> t = { x, y };
  return t;
}

template<class A, class B, class C>
inline tuple3<A, B, C> MP(A x, B y, C z) {
  tuple3<A, B, C> t = { x, y, z };
  return t;
}

template<class A, class B, class C, class D>
inline tuple4<A, B, C, D> MP(A x, B y, C z, D a) {
  tuple4<A, B, C, D> t = {x, y, z, a};
  return t;
}

template<class A>
inline vector<A> MakeVector(const A &x) {
  vector<A> out;
  out.push_back(x);
  return out;
}

template<class A>
inline vector<A> MakeVector(const A &x, const A &y) {
  vector<A> out;
  out.push_back(x);
  out.push_back(y);
  return out;
}

template<class A>
inline vector<A> MakeVector(const A &x, const A &y, const A &z) {
  vector<A> out;
  out.push_back(x);
  out.push_back(y);
  out.push_back(z);
  return out;
}
}  // namespace lapis

namespace std {
template <class A, class B>
struct hash<pair<A, B>> : public unary_function<pair<A, B> , size_t> {
  hash<A> ha;
  hash<B> hb;

  size_t operator()(const pair<A, B> &k) const {
    return ha(k.first) ^ hb(k.second);
  }
};

template <class A, class B>
struct hash<lapis::tuple2<A, B>> : public
                                  unary_function<lapis::tuple2<A, B> , size_t> {
  hash<A> ha;
  hash<B> hb;

  size_t operator()(const lapis::tuple2<A, B> &k) const {
    size_t res[] = { ha(k.a_), hb(k.b_) };
    return lapis::SuperFastHash((char *)&res, sizeof(res));
  }
};

template <class A, class B, class C>
struct hash<lapis::tuple3<A, B, C>> : public
                                     unary_function<lapis::tuple3<A, B, C> , size_t> {
  hash<A> ha;
  hash<B> hb;
  hash<C> hc;

  size_t operator()(const lapis::tuple3<A, B, C> &k) const {
    size_t res[] = { ha(k.a_), hb(k.b_), hc(k.c_) };
    return lapis::SuperFastHash((char *)&res, sizeof(res));
  }
};

}

namespace std {
template <class A, class B>
static ostream &operator<< (ostream &out, const std::pair<A, B> &p) {
  out << "(" << p.first << "," << p.second << ")";
  return out;
}

template <class A, class B>
static ostream &operator<< (ostream &out, const lapis::tuple2<A, B> &p) {
  out << "(" << p.a_ << "," << p.b_ << ")";
  return out;
}

template <class A, class B, class C>
static ostream &operator<< (ostream &out, const lapis::tuple3<A, B, C> &p) {
  out << "(" << p.a_ << "," << p.b_ << "," << p.c_ << ")";
  return out;
}

template <class A, class B, class C, class D>
static ostream &operator<< (ostream &out, const lapis::tuple4<A, B, C, D> &p) {
  out << "(" << p.a_ << "," << p.b_ << "," << p.c_ << "," << p.d_ << ")";
  return out;
}
}

#endif  // SWIG
#endif  // INCLUDE_CORE_TUPLE_H_
