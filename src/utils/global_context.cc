// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:40

#include "utils/global_context.h"
#include "proto/system.pb.h"
#include "utils/proto_helper.h"

DEFINE_string(system_conf_path, "", "");
DEFINE_string(model_conf_path, "", "");
namespace lapis {
GlobalContext::GlobalContext(const char *system_conf_path,
                             const char *model_conf_path)
  : model_conf_path_(model_conf_path) {

	 SystemProto system_conf;
	  ReadProtoFromTextFile(system_conf_path, &system_conf);
	  role_rank_[kCoordinator] = std::make_pair(system_conf.coordinator(),
	                             system_conf.coordinator());
	  role_rank_[kWorker] = std::make_pair(system_conf.worker_start(),
	                                      system_conf.worker_end());
	  if (system_conf.has_memory_start() && system_conf.has_memory_end())
	    role_rank_[kMemoryServer] = std::make_pair(system_conf.memory_start(),
	                                system_conf.memory_end());
	  else
	    role_rank_[kMemoryServer] = role_rank_[kWorker];
	  if (system_conf.has_disk_start() && system_conf.has_disk_end())
	    role_rank_[kDiskServer] = std::make_pair(system_conf.disk_start(),
	                              system_conf.disk_end());
	  else
	    role_rank_[kDiskServer] = role_rank_[kWorker];
	  num_memory_servers_ = role_rank_[kMemoryServer].second -
	                        role_rank_[kMemoryServer].first + 1;
	  num_disk_servers_ = role_rank_[kDiskServer].second -
	                      role_rank_[kDiskServer].first + 1;
}

bool GlobalContext::IsRoleOf(const Role& role, int rank) {
  if (rank <= role_rank_[role].second && rank >= role_rank_[role].first)
    return true;
  else
    return false;
}

GlobalContext* GlobalContext::Get(){
	static GlobalContext* gc = new GlobalContext(FLAGS_system_conf_path.c_str(), FLAGS_model_conf_path.c_str());
	return gc;
}
}  // namespace lapis
