// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:58
#ifndef INCLUDE_UTILS_GLOBAL_CONTEXT_H_
#define INCLUDE_UTILS_GLOBAL_CONTEXT_H_

#include <map>
#include <string>
#include <utility>

#include <gflags/gflags.h>

using std::shared_ptr;
namespace lapis {
enum Role {
  kCoordinator,
  kWorker,
  kMemoryServer,
  kDiskServer
};

//  assume that the coordinator's rank is (num_servers()-1)
class GlobalContext {
 public:
  bool IsRoleOf(const Role &role, int rank);
  bool single(){return single_;}
  bool is_sync_update() {return sync_;}
  int num_memory_servers() {
    mem_server_end_-mem_server_start_;
  }

  const char *model_conf_path() {
    return model_conf_path_.c_str();
  }

  int StartRankOf(Role role) {
    return role_rank_.at(role).first;
  }
  int EndRankOf(Role role) {
    return role_rank_.at(role).second;
  }

  bool GlobalContext::IsMemoryServer(int rank) {
    return rank>=mem_server_start_&&rank<mem_server_end_;
  }


  static shared_ptr<GlobalContext> Get();
  static shared_ptr<GlobalContext> Get(const string sys_conf,
                                       const string model_conf);
 private:
  GlobalContext(const string sys_conf, const string model_conf);

 private:
  // start and end rank for memory server, [start, end)
  int mem_server_start_, mem_server_end_;
  // # of nodes have memory tables
  int num_memory_servers_;
  bool standalone_, sync_;
  std::string model_conf_path_;
};
}  // namespace lapis

#endif  // INCLUDE_UTILS_GLOBAL_CONTEXT_H_
