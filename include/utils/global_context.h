// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:58
#ifndef INCLUDE_UTILS_GLOBAL_CONTEXT_H_
#define INCLUDE_UTILS_GLOBAL_CONTEXT_H_
#include <glog/logging.h>
#include <string>
#include <utility>
#include <memory>
#include <vector>
#include <mpi.h>

#include "utils/network_thread.h"


using std::shared_ptr;
using std::string;
using std::vector;

namespace lapis {

class GlobalContext {
 public:
  static shared_ptr<GlobalContext> Get();
  static shared_ptr<GlobalContext> Get(const string &sys_conf);
  const char *system_conf() { return system_conf_.c_str(); }
  // True if running in standalone mode
  bool standalone() { return standalone_; }
  // True if running in synchronous update mode
  bool synchronous() {return synchronous_;}
  // num of memory servers, default is the num of processes
  int num_table_servers() { return table_server_end_-table_server_start_; }
  int server_start() {return table_server_start_;}
  int server_end() {return table_server_end_;}
  int num_procs() { return num_procs_; }
  int num_workers() {return num_workers_;}
  bool IsTableServer(int rank) {
    return rank>=table_server_start_&&rank<table_server_end_;
  }
  // There is only one coordinator with rank 0
  bool AmICoordinator() { return rank_==kCoordinator;}
  // Memory server should have rank [start, end)
  bool AmITableServer() { return IsTableServer(rank_); }
  // All processes are workers except the coordinator
  bool AmIWorker() {return gid_!=-1;}
  bool AmIGroupLeader() {return  rank_==groups_[gid_][0];}
  int worker_id() {return worker_id_;}
  int group_id() {return gid_;}
  int num_groups() {return groups_.size();}
  int group_size() {CHECK(groups_.size());return groups_[0].size();}
  int rank() {return rank_;}
  int leader() {return MembersOfGroup(gid_)[0];}
  const string shard_folder() {return shard_folder_;}
  const MPI_Comm& mpicomm() {return mpicomm_;}
  vector<int> MembersOfGroup(int gid) {return groups_[gid];}
  vector<vector<int>> groups() {return groups_;}
  // assume the rank of coordinator is 0
  static int kCoordinator;
  void Finish();
 private:
  GlobalContext(const std::string &sys_conf);
  void Setup(const std::shared_ptr<NetworkThread>& nt);

 private:
  // mpi rank of current process
  int rank_;
  int worker_id_; // order in this group, starts from 0
  int num_workers_;
  // total number of processes started by mpi
  int num_procs_;
  // start and end rank for memory server, [start, end)
  int table_server_start_, table_server_end_;
  // standalone or cluster mode;
  bool standalone_;
  // update in synchronous or asynchronous mode
  bool synchronous_;
  // path of model config
  std::string  system_conf_, shard_folder_;
  // number of workers per group
  int gid_;
  MPI_Group mpigroup_;
  MPI_Comm mpicomm_;
  vector<vector<int>> groups_;
  static shared_ptr<GlobalContext> instance_;
};
}  // namespace lapis

#endif  // INCLUDE_UTILS_GLOBAL_CONTEXT_H_
