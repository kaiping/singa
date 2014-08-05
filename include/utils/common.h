// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-08-05 14:38
#ifndef INCLUDE_UTILS_COMMON_H_
#define INCLUDE_UTILS_COMMON_H_

namespace lapis {

// flag to init parameter content
constexpr char  kInitParam=1;
// flag to allocate memory for parameter
constexpr char kAllocParam=2;
// flag to allocate memory for features/data
constexpr char kAllocData=4;

inline bool InitParam(const char x) {
  return (x&kInitParam)!=0;
}

inline bool AllocParam(const char x) {
  return (x&kAllocParam)!=0;
}

inline bool AllocData(const char x) {
  return (x&kAllocData)!=0;
}

}  // namespace lapis
#endif  // INCLUDE_UTILS_COMMON_H_
