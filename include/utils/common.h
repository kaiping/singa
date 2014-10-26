// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-08-05 14:38
#ifndef INCLUDE_UTILS_COMMON_H_
#define INCLUDE_UTILS_COMMON_H_
#include <glog/logging.h>
#include <string>
#include <vector>
#include <sys/stat.h>

#include <map>
#include "proto/model.pb.h"

using std::vector;
using std::string;
using std::map;
namespace lapis {
const int kBufLen=1024;
enum Phase{
  kTrain=1,
  kValidation,
  kTest
};

inline bool check_exists(const std::string& name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}

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

  string ToString(){
    char buf[1024];
    if (has_precision())
      sprintf(buf,"Precision %.3f, ", precision());
    if (has_loss())
      sprintf(buf+strlen(buf),"loss %.3f, ", loss());
    return string(buf);
  }
};

void Debug();
}  // namespace lapis
#endif  // INCLUDE_UTILS_COMMON_H_
