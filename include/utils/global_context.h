// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:58
#ifndef INCLUDE_UTILS_GLOBAL_CONTEXT_H_
#define INCLUDE_UTILS_GLOBAL_CONTEXT_H_

#include <string>
#include <utility>
#include <memory>

#include <gflags/gflags.h>

using std::shared_ptr;

namespace lapis {
// assume the rank of coordinator is 0
const int kCoordinatorRank=0;
//  assume that the coordinator's rank is (num_servers()-1)
class GlobalContext {
 public:
  static shared_ptr<GlobalContext> Get();
  static shared_ptr<GlobalContext> Get(const string sys_conf,
                                       const string model_conf);
  const char *model_conf() { return model_conf_.c_str(); }
  // True if running in standalone mode
  bool standalone() { return standalone_; }
  // True if running in synchronous update mode
  bool synchronous() {return synchronous_;}
  // num of memory servers, default is the num of processes
  int num_memory_servers() { mem_server_end_-mem_server_start_; }
  bool IsMemoryServer(int rank) {
    return rank>=mem_server_start_&&rank<mem_server_end_;
  }
  // There is only one coordinator with rank 0
  bool AmICoordinator() { return rank_==kCoordinatorRank;}
  // Memory server should have rank [start, end)
  bool AmIMemoryServer() { return IsMemoryServer(rank_); }
  // All processes are workers except the coordinator
  bool AmIWorker() {return rank_!=kCoordinatorRank;}


 private:
  GlobalContext(const string sys_conf, const string model_conf);

 private:
  // total number of processes started by mpi
  int num_processes_;
  // start and end rank for memory server, [start, end)
  int mem_server_start_, mem_server_end_;
  // standalone or cluster mode;
  bool standalone_;
  // update in synchronous or asynchronous mode
  bool synchronous_;
  // path of model config
  std::string model_conf_;
};
}  // namespace lapis

#endif  // INCLUDE_UTILS_GLOBAL_CONTEXT_H_
