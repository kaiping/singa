// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-04 11:14

#ifndef INCLUDE_WORKER_H_
#define INCLUDE_WORKER_H_
#include <map>
#include <memory>
#include <pthread.h>

#include "core/table_server.h"
#include "net/net.h"
#include "net/solver.h"
#include "proto/model.pb.h"
#include "utils/common.h"

namespace lapis {
/**
 * The Worker class which runs the training algorithm.
 * The first worker will initialize parameters of the Net, and put them into
 * the distributed memory/table.
 */

class Worker {
 public:
  Worker(const shared_ptr<GlobalContext>& gc);
  void Start(const DataProto& proto, const SolverProto& sp);
  void Resume();
  void Shutdown();
  ~Worker();

 private:
  //void Run(const ModelProto& proto);
  //void ReportPerformance(Performance perf);
  // must provide globalcontext with Worker
  Worker(){};

  shared_ptr<GlobalContext> context_;
  std::shared_ptr<NetworkThread> mpi_;
  TableServer *table_server_;
};
}  // namespace lapis

#endif  // INCLUDE_WORKER_H_
