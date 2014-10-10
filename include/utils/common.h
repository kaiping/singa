// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-08-05 14:38
#ifndef INCLUDE_UTILS_COMMON_H_
#define INCLUDE_UTILS_COMMON_H_

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include "utils/stringpiece.h"
using std::string;
using std::vector;
using std::unordered_map;

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
template<typename T>
class StateQueue {
  public:
    StateQueue(int size):nvalid_(size) {
      for (int i = 0; i <size; i++) {
        states_[i]=true;
        iter_=states_.begin();
      }
    }
    StateQueue(vector<T> members){
      for(auto& x:members)
        states_[x]=true;
      iter_=states_.begin();
      nvalid_=members.size();
    }
    T Next() {
      CHECK(nvalid_);
      iter_++;
      while(iter_!=states_.end()&&iter_->second==false)
        iter_++;
      if(iter_==states_.end()){
        iter_=states_.begin();
        while(iter_!=states_.end()&&iter_->second==false)
          iter_++;
      }
      return iter_->first;
    }
    void Invalid(int k) {
      states_[k]=false;
      nvalid_--;
    }
    void Invalid() {
      CHECK(iter_->second);
      iter_->second=false;
      nvalid_--;
    }

    bool HasValid() {
      return nvalid_>0;
    }
  private:
    std::map<T,bool> states_;
    int nvalid_;
    // typename to tell complier iterator is a type in map class
    typename std::map<T,bool>::iterator iter_;
};

}  // namespace lapis
#endif  // INCLUDE_UTILS_COMMON_H_
