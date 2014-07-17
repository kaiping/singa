// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:58
#ifndef INCLUDE_UTILS_GLOBAL_CONTEXT_H_
#define INCLUDE_UTILS_GLOBAL_CONTEXT_H_

#include <map>
#include <utility>
#include <gflags/gflags.h>

DECLARE_int32(num_keys);
DECLARE_string(system_conf_path);
DECLARE_string(model_conf_path);

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
  explicit GlobalContext(const char* system_conf_path,
                         const char * model_conf_path);
  inline bool IsRoleOf(const Role& role, int rank);


  int num_memory_servers() { return num_memory_servers_; }
  int num_disk_servers() { return num_disk_servers_; }
  const char* model_conf_path() {return model_conf_path_; }

  //  number of keys
  int num_keys(){ return num_keys_; }
  static GlobalContext* Get();

 private:
  // map from role to (start_rank, end_rank) pair
  std::map<Role, std::pair<int, int> > role_rank_;
  // # of nodes have memory tables
  int num_memory_servers_;
  // # of nodes have disk tables
  int num_disk_servers_;

  int num_keys_;

  const char* model_conf_path_;

};
}  // namespace lapis

#endif  // INCLUDE_UTILS_GLOBAL_CONTEXT_H_
