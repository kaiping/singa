#ifndef INCLUDE_UTILS_COMMON_H_
#define INCLUDE_UTILS_COMMON_H_
#pragma once
#include <glog/logging.h>
#include <google/protobuf/message.h>
#include <stdarg.h>
#include <string>
#include <vector>
#include <mutex>
#include <queue>
#include <sys/stat.h>
#include <map>
#include "proto/model.pb.h"

using std::vector;
using std::string;
using std::map;
using google::protobuf::Message;
namespace singa {

void ReadProtoFromTextFile(const char* filename, Message* proto) ;
void WriteProtoToTextFile(const Message& proto, const char* filename) ;
void ReadProtoFromBinaryFile(const char* filename, Message* proto) ;
void WriteProtoToBinaryFile(const Message& proto, const char* filename);

std::string IntVecToString(const vector<int>& vec) ;
string VStringPrintf(string fmt, va_list l) ;
string StringPrintf(string fmt, ...) ;
void Debug() ;
inline bool check_exists(const std::string& name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}

template<typename T>
class SafeQueue{
 public:
  SafeQueue():q(),m(){}
  void push(const T& e){
    std::lock_guard<std::mutex> lock(m);
    q.push(e);
  }

  T front() {
    T ret;
    std::unique_lock<std::mutex> lock(m);
    if(q.size()>0){
      ret = q.front();
    }else{
      LOG(FATAL)<<"The queue is empty";
    }
    return ret;
  }

  void pop(){
    std::unique_lock<std::mutex> lock(m);
    q.pop();
  }

  bool pop(T* ret) {
    std::unique_lock<std::mutex> lock(m);
    if(q.size()>0){
      *ret = q.front();
      q.pop();
      return true;
    }else{
      return false;
    }
  }
  int size(){
    return q.size();
  }

 private:
  std::queue<T> q;
  mutable std::mutex m;
};

class Performance: public PerformanceProto{
 public:
   void Aggregate(const Performance& other){
     set_loss(other.loss()+loss());
     set_topk_precision(other.topk_precision()+topk_precision());
     set_top_precision(other.top_precision()+top_precision());
     set_count(count()+1);
   }

  void Reset() {
    set_count(0);
    set_loss(0.f);
    set_topk_precision(0.f);
    set_top_precision(0.f);
  }

  Performance Avg() {
    Performance perf;
    perf.CopyFrom(*this);
    perf.set_loss(perf.loss()/perf.count());
    perf.set_topk_precision(perf.topk_precision()/perf.count());
    perf.set_top_precision(perf.top_precision()/perf.count());
    return perf;
  }

  string ToString(){
    char buf[1024];
    sprintf(buf,"TopK Precision %.4f, ", topk_precision());
    sprintf(buf+strlen(buf),"Top1 Precision %.4f, ", top_precision());
    sprintf(buf+strlen(buf),"loss %.4f, ", loss());
    return string(buf);
  }
};


} /* singa */
#endif  // INCLUDE_UTILS_COMMON_H_
