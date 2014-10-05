// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-08-05 14:38
#ifndef INCLUDE_UTILS_COMMON_H_
#define INCLUDE_UTILS_COMMON_H_
#include <glog/logging.h>
#include <string>
#include <unordered_map>
#include "utils/stringpiece.h"
using std::string;
using std::unordered_map;

namespace lapis {
const int kTrain=1;
const int kVal=2;
const int kTest=3;

template<typename T>
class StateQueue {
 public:
   StateQueue(int size):nvalide_(size) {
     for (int i = 0; i <size; i++) {
       states_[i]=true;
       iter_=states_.begin();
     }
   }
   StateQueue(vector<T> members){
     for(auto& x:members)
      states_[x]=true;
     iter_=states_.begin();
     nvalide_=members.size();
   }

   T Next() {
     CHECK(nvalide_);
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
   void Invalide() {
     CHECK(iter_->second);
     iter_->second=false;
     nvalide_--;
   }

   bool HasValid() {
      return nvalide_>0;
   }
 private:
  std::map<T,bool> states_;
  int nvalide_;
  // typename to tell complier iterator is a type in map class
  typename std::map<T,bool>::iterator iter_;
};

class Performance: public PerformanceProto{
 public:
  void Aggregate(const Performance& other){
    if(other.has_loss())
      set_loss(other.loss()+loss());
    if(other.has_precision())
      set_precision(other.precision()+precision());
    set_count(count()+1);
  }
  void Reset() {
    set_count(0);
    set_loss(0.f);
    set_precision(0.f);
  }

  Performance Avg() {
    Performance perf;
    perf.CopyFrom(*this);
    if(perf.has_loss())
      perf.set_loss(perf.loss()/perf.count());
    if(perf.has_precision())
      perf.set_precision(perf.precision()/perf.count());
    return perf;
  }
};
}  // namespace lapis
#endif  // INCLUDE_UTILS_COMMON_H_
