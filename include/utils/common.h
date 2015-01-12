#ifndef INCLUDE_UTILS_COMMON_H_
#define INCLUDE_UTILS_COMMON_H_
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

void ReadProtoFromTextFile(const char *filename, Message *proto);
void WriteProtoToTextFile(const Message &proto, const char *filename);
void ReadProtoFromBinaryFile(const char *filename, Message *proto);
void WriteProtoToBinaryFile(const Message &proto, const char *filename);

const int kBufLen=1024;

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
  void pop(T* e) {
    std::unique_lock<std::mutex> lock(m);
    if(q.size()>0){
      *e = q.front();
      q.pop();
    }
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

/**
 * Formatted string.
 */
string VStringPrintf(string fmt, va_list l) {
  char buffer[32768];
  vsnprintf(buffer, 32768, fmt.c_str(), l);
  return string(buffer);
}

/**
 * Formatted string.
 */
string StringPrintf(string fmt, ...) {
  va_list l;
  va_start(l, fmt); //fmt.AsString().c_str());
  string result = VStringPrintf(fmt, l);
  va_end(l);
  return result;
}

void Debug() {
  int i = 0;
  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  printf("PID %d on %s ready for attach\n", getpid(), hostname);
  fflush(stdout);
  while (0 == i)
    sleep(5);
}
}  // namespace singa

#endif  // INCLUDE_UTILS_COMMON_H_
