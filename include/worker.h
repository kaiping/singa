// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-04 11:14

#ifndef INCLUDE_WORKER_H_
#define INCLUDE_WORKER_H_
#include <map>
#include <memory>

#include "core/table_server.h"
#include "net/net.h"
#include "proto/model.pb.h"

namespace lapis {
/**
 * The Worker class which runs the training algorithm.
 * The first worker will initialize parameters of the Net, and put them into
 * the distributed memory/table.
 */

class Worker {
 public:
  Worker();
  ~Worker();
  void Run(bool load_data, bool do_run);

 private:
  bool ShouldIDoValidation(int worker_id);
  const DistributedStorageConfig InitDistributedStorage();

  void Barrier(int step);
  void Shutdown();
 private:
  std::shared_ptr<NetworkThread> mpi_;
  TableServer *table_server_;
  ModelController model_controller_;
};
}  // namespace lapis

#endif  // INCLUDE_WORKER_H_
