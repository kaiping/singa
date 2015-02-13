#ifndef INCLUDE_UTILS_COMMON_H_
#define INCLUDE_UTILS_COMMON_H_
#pragma once
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <google/protobuf/message.h>
#include <stdarg.h>
#include <string>
#include <vector>
#include <mutex>
#include <queue>
#include <sys/stat.h>
#include <map>

using std::vector;
using std::string;
using std::map;
using google::protobuf::Message;

#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_


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

inline float rand_real(){
  return  static_cast<float>(rand())/(RAND_MAX+1.0f);
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

} /* singa */
#endif  // INCLUDE_UTILS_COMMON_H_
