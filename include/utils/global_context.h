// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:58
#ifndef INCLUDE_UTILS_GLOBAL_CONTEXT_H_
#define INCLUDE_UTILS_GLOBAL_CONTEXT_H_

#include <map>
#include <string>
#include <utility>

#include <gflags/gflags.h>

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
  void Init(const std::string &system_conf_path,
            const std::string &model_conf_path);
  bool single(){return single_;}
  bool is_sync_update() {return sync_;}
  int num_memory_servers() {
    return num_memory_servers_;
  }
  int num_disk_servers() {
    return num_disk_servers_;
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
  static GlobalContext *Get();
 private:
  // map from role to (start_rank, end_rank) pair
  std::map<Role, std::pair<int, int>> role_rank_;
  // # of nodes have memory tables
  int num_memory_servers_;
  // # of nodes have disk tables
  int num_disk_servers_;
  bool single_, sync_;
  std::string model_conf_path_;
  GlobalContext() {}
};
}  // namespace lapis

#endif  // INCLUDE_UTILS_GLOBAL_CONTEXT_H_
