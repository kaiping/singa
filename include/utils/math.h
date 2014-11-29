// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-11-29 17:22

#ifndef INCLUDE_UTIL_MATH_
#define INCLUDE_UTIL_MATH_
namespace lapis {

namespace Math {

inline void Add(int len, float* ret, const float* ary, const float x){
  Map([x](float v) {return x+v;});
}
inline void Add(int len, float* ret, const float* a, const float *b){
  Map([](const float p, const float q) {return p+q;});
}

/**
 * ret[i]=x*a[i]+b[i].
 */
inline void mAdd(int len, float*ret, const float x, const float*a, const float*b){
  Map([x](const float p, const float q) {return x*p+q;});
}

inline void Minus(int len, float*ret, const float* a, const float *b){
  Map([](const float x, const float y) {return x-y;});
}

inline void Map(std::function<float(float)> func, int len,
    float *ret, const float* dptr){
  for (int i = 0; i < len; i++) {
    ret[i]=func(dptr[i]);
  }
}

inline void Map( std::function<float(float, float)> func, int len,
    float* ret, const float* dptr1, const float* dptr2){
  for (int i = 0; i < len; i++) {
    ret[i]=func(dptr1[i], dptr2[i]);
  }
}

inline void Map(std::function<float(float, float, float)> func, int len,
    float* ret, const float* dptr1, const float* dptr2, const float* dptr3){
  for (int i = 0; i < len; i++) {
    ret[i]=func(dptr1[i], dptr2[i], dptr3[i]);
  }
}
} /* Math */
} /* lapis  */
#endif
