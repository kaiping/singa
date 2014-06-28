// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:41

#include <glog/logging.h>
#include "utils/global_context.h"
#include "utils/start_deamon.h"

int startDeamon(const char* system_conf_path, const char* model_conf_path) {
  LOG(INFO) << "Starting deamon...\n";
  lapis::GlobalContext global_context(system_conf_path, model_conf_path);
  // TODO(Anh) may from a wrapper .h file for mpi functions.
  int rank = get_my_rank_from_mpi();
  // TODO(Anh) DistributedMemory
  lapis::DistributedMemory distributed_memory(global_context);
  // TODO(Anh) DistributedDisk
  lapis::DistributedDisk distributed_disk(global_context);
  if (global_context.IsRoleOf(lapis::kCoordinator, rank)) {
    LOG(INFO) << "Coordinator is of rank " << rank << "\n";
    // TODO(wangwei) Coordinator
    lapis::Coodinator coordinator(distributed_memory,
                                  distributed_disk,
                                  global_cotext);
    coordinator.run();
  }

  if (global_context.IsRoleOf(lapis::kWorker, rank)) {
    LOG(INFO) << "Start worker of rank " << rank << "\n";
    // TODO(wangwei) Worker
    lapis::Worker worker(distributed_memory, distributed_disk, global_context);
    worker.run();
  }
  return 0;
}
