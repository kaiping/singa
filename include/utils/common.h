// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-08-05 14:38
#ifndef INCLUDE_UTILS_COMMON_H_
#define INCLUDE_UTILS_COMMON_H_

#include <string>
#include <unordered_map>
#include "utils/stringpiece.h"
using std::string;
using std::unordered_map;

namespace lapis {
const int kCoordinatorRank=0;
enum Phase{
  kTrain=1,
  kVal=2,
  kTest=3
};
template<typename T>
class StateQueue {
 public:
   StateQueue(vector<T> members){
     for(auto& x:members)
      state[x]=true;
     iter_=states.begin();
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

   void HasValid() {
      return nvalide_>0;
   }
 private:
  std::map<T,bool> states_;
  int nvalide_;
  std::map<T,bool>::iterator iter_;
}

}  // namespace lapis
#endif  // INCLUDE_UTILS_COMMON_H_
