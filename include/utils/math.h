#ifndef INCLUDE_UTIL_MATH_
#define INCLUDE_UTIL_MATH_
#include <functional>
namespace singa {

namespace Math {
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

inline void Mult(int len, float* ret, const float x, const float* ary){
  Map([x](float v) {return x*v;}, len, ret, ary);
}
inline void Add(int len, float* ret, const float* ary, const float x){
  Map([x](float v) {return x+v;}, len, ret, ary);
}
inline void Add(int len, float* ret, const float* a, const float *b){
  Map([](const float p, const float q) {return p+q;}, len, ret, a, b);
}

/**
 * ret[i]=x*a[i]+b[i].
 */
inline void mAdd(int len, float*ret, const float x, const float*a, const float*b){
  Map([x](const float p, const float q) {return x*p+q;}, len ,ret, a, b);
}

inline void Minus(int len, float*ret, const float* a, const float *b){
  Map([](const float x, const float y) {return x-y;}, len ,ret, a,b);
}

} /* Math */
} /* singa  */
#endif
