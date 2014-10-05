// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:58
#ifndef INCLUDE_UTILS_GLOBAL_CONTEXT_H_
#define INCLUDE_UTILS_GLOBAL_CONTEXT_H_

#include <string>
#include <utility>
#include <memory>

using std::shared_ptr;
using std::string;

namespace lapis {
class GlobalContext {
 public:
  static shared_ptr<GlobalContext> Get();
  static shared_ptr<GlobalContext> Get(const string &sys_conf,
                                       const string &model_conf);
  const char *model_conf() { return model_conf_.c_str(); }
  // True if running in standalone mode
  bool standalone() { return standalone_; }
  // True if running in synchronous update mode
  bool synchronous() {return synchronous_;}
  // num of memory servers, default is the num of processes
  int num_table_servers() { return table_server_end_-table_server_start_; }
  int num_processes() { return num_processes_; }
  int num_workers() {return num_processes_-1;}
  void set_num_processes(int num) ;
  bool IsTableServer(int rank) {
    return rank>=table_server_start_&&rank<table_server_end_;
  }
  // There is only one coordinator with rank 0
  bool AmICoordinator() { return rank_==kCoordinator;}
  // Memory server should have rank [start, end)
  bool AmITableServer() { return IsTableServer(rank_); }
  // All processes are workers except the coordinator
  bool AmIWorker() {return rank_!=kCoordinator;}
  void set_rank(int rank) {rank_=rank;}
  // assume the rank of coordinator is 0
  static int kCoordinator;
 private:
  GlobalContext(const string &sys_conf, const string &model_conf);

 private:
  // mpi rank of current process
  int rank_;
  // total number of processes started by mpi
  int num_processes_;
  // start and end rank for memory server, [start, end)
  int table_server_start_, table_server_end_;
  // standalone or cluster mode;
  bool standalone_;
  // update in synchronous or asynchronous mode
  bool synchronous_;
  // path of model config
  std::string model_conf_;
  // number of workers per group
  int group_size_;
  static shared_ptr<GlobalContext> instance_;
};
}  // namespace lapis

#endif  // INCLUDE_UTILS_GLOBAL_CONTEXT_H_
