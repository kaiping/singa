// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-06-28 14:41

#include <glog/logging.h>
#include "utils/global_context.h"
#include "utils/start_deamon.h"

int startDeamon(const char *system_conf_path, const char *model_conf_path) {
  LOG(INFO) << "Starting deamon...\n";
  lapis::GlobalContext global_context(system_conf_path, model_conf_path);
	InitServers(argc, argv);

  ModelController mc;
  mc.Init(global_context.num_memory_servers());
  // TODO(Anh) DistributedDisk
  // lapis::DistributedDisk distributed_disk(global_context);
  int rank = NetworkThread::Get()->id();
  if (global_context.IsRoleOf(lapis::kCoordinator, rank)) {
    LOG(INFO) << "Coordinator is of rank " << rank << "\n";
    // TODO(wangwei) Coordinator
    lapis::Coodinator coordinator(global_cotext, mc);
    coordinator.run();
  } else if (global_context.IsRoleOf(lapis::kWorker, rank)) {
    LOG(INFO) << "Start worker of rank " << rank << "\n";
    // TODO(wangwei) Worker
    lapis::Worker worker(mc, global_context);
    worker.run();
  }
  return 0;
}
